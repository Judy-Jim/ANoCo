"""ANoCo production inspector: stable, monitored, reproducible inference.

Startup: load model + bank + normalizer + threshold (one-time).
Per image: decode → extract → score → normalize → aggregate → decide.
Errors are caught and returned as ERROR results — never crashes the line.

Usage:
    inspector = ANoCoInspector(
        config_path="configs/production.yaml",
        bank_path="banks/my_product/bank.pt",
        normalizer_path="banks/my_product/normalizer.pt",
        threshold_path="banks/my_product/threshold.json",
    )
    result = inspector.inspect("path/to/image.jpg")
    print(result.decision, result.score)
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .config import ANoCoConfig
from .normalization import ExcessNormalizer
from .scoring import energy_to_map, image_score_aggregated
from .logger import InspectResult


class ANoCoInspector:
    """Production-ready ANoCo inspector with single-threshold decision."""

    def __init__(
        self,
        config_path: str,
        bank_path: str,
        normalizer_path: str,
        threshold_path: str,
        device: str = "cuda:0",
        warmup: int = 3,
        backbone_type: str = "pytorch",
    ):
        """
        Args:
            config_path: path to the production YAML config.
            bank_path: path to bank.pt (pre-built memory bank tensor).
            normalizer_path: path to normalizer.pt (excess statistics).
            threshold_path: path to threshold.json (calibrated threshold).
            device: CUDA device string.
            warmup: number of dummy inferences on startup.
            backbone_type: "pytorch" (default) or "tensorrt" (future).
        """
        self.backbone_type = backbone_type
        self._startup_time = time.time()

        # 1. Load config
        self.cfg = ANoCoConfig.from_yaml(config_path)
        self.cfg.device = device
        self._production = self._load_production_params(config_path)

        # 2. Load model
        if backbone_type == "pytorch":
            from .features.dinov3 import DINOv3Extractor
            self._extractor = DINOv3Extractor(self.cfg)
            # Apply runtime knobs from production params
            self._extractor.decode = self._production.get("decode", "draft")
            self._extractor.fp16 = self._production.get("fp16", True)
        elif backbone_type == "tensorrt":
            # Future: load TensorRT engine
            raise NotImplementedError("TensorRT backbone not yet implemented. Use backbone_type='pytorch'.")
        else:
            raise ValueError(f"unknown backbone_type: {backbone_type!r}")

        # 3. Load bank
        if not os.path.isfile(bank_path):
            raise FileNotFoundError(f"bank file not found: {bank_path}")
        self._bank = torch.load(bank_path, map_location=device, weights_only=False)
        self._bank = self._bank.to(device)

        # 4. Load normalizer
        if not os.path.isfile(normalizer_path):
            raise FileNotFoundError(f"normalizer file not found: {normalizer_path}")
        norm_data = torch.load(normalizer_path, map_location="cpu", weights_only=False)
        self._normalizer = ExcessNormalizer(k=self._production.get("excess_k", 3.0))
        self._normalizer._threshold = norm_data["threshold"]
        self._normalizer._fitted = True

        # 5. Load threshold
        if not os.path.isfile(threshold_path):
            raise FileNotFoundError(f"threshold file not found: {threshold_path}")
        with open(threshold_path, "r", encoding="utf-8") as f:
            thr_data = json.load(f)
        self._threshold = float(thr_data["threshold"])
        self._target_fpr = float(thr_data.get("target_fpr", 2.0))

        # 6. Pipeline params
        self._chunk = int(self._production.get("chunk", 1024))
        self._agg_method = self._production.get("agg_method", "topk_mean")
        self._agg_k = int(self._production.get("agg_k", 5))

        # 7. Warmup
        self._model = None  # ANoCo instance (lazy)
        self._do_warmup(warmup)

        self._startup_ms = (time.time() - self._startup_time) * 1000
        print(f"[ANoCoInspector] ready in {self._startup_ms:.0f}ms | "
              f"bank={tuple(self._bank.shape)} threshold={self._threshold:.6f} "
              f"backbone={backbone_type}")

    @staticmethod
    def _load_production_params(config_path: str) -> dict:
        """Load the 'production' section from the YAML config."""
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("production", {})

    def _get_model(self):
        """Lazy-init ANoCo instance."""
        if self._model is None:
            from .anoco import ANoCo
            self._model = ANoCo(self.cfg)
        return self._model

    def _do_warmup(self, n: int):
        """Run n dummy inferences to trigger CUDA compilation."""
        if n <= 0:
            return
        h, w = self.cfg.img_hw if self.cfg.img_hw else (self.cfg.img_size, self.cfg.img_size)
        dummy = torch.randn(3, h, w, device=self.cfg.device)
        try:
            for i in range(n):
                x = dummy.unsqueeze(0)
                _ = self._extractor.model.get_intermediate_layers(
                    x, n=list(self.cfg.layer_indices or [self.cfg.layer_index]),
                    reshape=False, norm=self.cfg.apply_norm
                )
            if self.cfg.device.startswith("cuda"):
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[ANoCoInspector] warmup warning: {e}")

    # ------------------------------------------------------------------ inspect
    def inspect(
        self,
        image: Union[str, np.ndarray, object],
        filename: str = "",
    ) -> InspectResult:
        """Inspect a single image.

        Args:
            image: file path, numpy array (HWC uint8), or PIL Image.
            filename: optional name for logging (auto-extracted from path if not given).

        Returns:
            InspectResult with score, decision, anomaly_map, latency.
        """
        ts = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()

        if not filename and isinstance(image, str):
            filename = os.path.basename(image)

        try:
            # Extract features
            fq, grid = self._extractor.extract(image)

            # Score against bank
            model = self._get_model()
            out = model.score_features(fq, self._bank, grid_hw=grid, chunk=self._chunk)
            raw_energy = out["patch_energy"].cpu()

            # Excess normalization
            normed_energy = self._normalizer.transform(raw_energy, grid)

            # Aggregate
            score_tensor = image_score_aggregated(
                normed_energy, method=self._agg_method, grid_hw=grid, k=self._agg_k
            )
            score = score_tensor.item() if torch.is_tensor(score_tensor) else float(score_tensor)

            # Anomaly map
            amap = energy_to_map(
                normed_energy, grid,
                out_hw=tuple(self.cfg.img_hw) if self.cfg.img_hw else None,
                kernel_size=self.cfg.gaussian_kernel,
                sigma=self.cfg.gaussian_sigma,
            )
            amap_np = amap.cpu().numpy()

            latency = (time.perf_counter() - t0) * 1000
            decision = "NG" if score > self._threshold else "OK"

            return InspectResult(
                score=score,
                decision=decision,
                anomaly_map=amap_np,
                latency_ms=latency,
                timestamp=ts,
                filename=filename,
            )

        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            err_msg = f"{type(e).__name__}: {str(e)[:200]}"
            return InspectResult(
                score=-1.0,
                decision="ERROR",
                anomaly_map=None,
                latency_ms=latency,
                timestamp=ts,
                filename=filename,
                error_msg=err_msg,
            )

    def inspect_batch(
        self, images: list, log: bool = True
    ) -> List[InspectResult]:
        """Inspect a batch of images sequentially."""
        results = []
        for i, img in enumerate(images):
            r = self.inspect(img)
            results.append(r)
            if log and (i + 1) % 50 == 0:
                n_ng = sum(1 for x in results if x.decision == "NG")
                print(f"  [batch] {i+1}/{len(images)} | NG={n_ng}")
        return results

    # ------------------------------------------------------------------ overlay
    def save_overlay(self, image, result: InspectResult, out_path: str):
        """Save a 3-panel overlay (original | heatmap | blended)."""
        from PIL import Image, ImageDraw, ImageFont
        import torch.nn.functional as F

        # Load original image
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif hasattr(image, "convert"):
            pil_img = image.convert("RGB")
        else:
            from .features.dinov3 import _load_pil
            pil_img = _load_pil(image)

        w, h = pil_img.size
        panel_w = w // 2
        panel_h = h // 2

        # Resize anomaly map
        if result.anomaly_map is not None:
            amap = torch.tensor(result.anomaly_map, dtype=torch.float32)
            amap = F.interpolate(
                amap[None, None], size=(panel_h, panel_w),
                mode="bilinear", align_corners=False
            )[0, 0].numpy()
            amap_min, amap_max = amap.min(), amap.max()
            if amap_max > amap_min:
                amap_u8 = ((amap - amap_min) / (amap_max - amap_min) * 255).astype(np.uint8)
            else:
                amap_u8 = np.zeros_like(amap, dtype=np.uint8)

            # Colorize: blue->green->yellow->red
            r = np.clip(amap_u8.astype(np.float32) * 1.5, 0, 255)
            g = np.clip(amap_u8.astype(np.float32) * 0.8, 0, 200)
            b = np.clip(80 - amap_u8.astype(np.float32) * 0.5, 0, 80)
            hm_rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
        else:
            hm_rgb = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

        orig_small = pil_img.resize((panel_w, panel_h), Image.LANCZOS)
        hm_img = Image.fromarray(hm_rgb)
        orig_arr = np.array(orig_small).astype(np.float32)
        alpha = (amap_u8.astype(np.float32) / 255.0)[:, :, None] * 0.5 if result.anomaly_map is not None else 0
        blended = (orig_arr * (1 - alpha) + hm_rgb.astype(np.float32) * alpha).astype(np.uint8)
        blend_img = Image.fromarray(blended)

        canvas = Image.new("RGB", (panel_w * 3 + 20, panel_h + 50), (30, 30, 30))
        canvas.paste(orig_small, (0, 40))
        canvas.paste(hm_img, (panel_w + 10, 40))
        canvas.paste(blend_img, (panel_w * 2 + 20, 40))

        draw = ImageDraw.Draw(canvas)
        color = (255, 60, 60) if result.decision == "NG" else (0, 220, 0) if result.decision == "OK" else (255, 255, 0)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        title = f"{result.filename}  score={result.score:.4f}  thr={self._threshold:.4f}  [{result.decision}]"
        draw.text((10, 8), title, fill=color, font=font)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        canvas.save(out_path, quality=90)

    # ------------------------------------------------------------------ health
    def health_check(self) -> dict:
        """Return system health status."""
        info = {
            "backbone_type": self.backbone_type,
            "bank_shape": list(self._bank.shape),
            "bank_device": str(self._bank.device),
            "threshold": self._threshold,
            "target_fpr": self._target_fpr,
            "config": {
                "backbone": self.cfg.backbone,
                "layer_indices": list(self.cfg.layer_indices) if self.cfg.layer_indices else [self.cfg.layer_index],
                "layer_fusion": self.cfg.layer_fusion,
                "img_hw": list(self.cfg.img_hw) if self.cfg.img_hw else [self.cfg.img_size] * 2,
            },
            "startup_ms": round(self._startup_ms, 0),
        }
        if self.cfg.device.startswith("cuda"):
            info["gpu_name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.memory_allocated(0)
            info["gpu_mem_allocated_mb"] = round(mem / 1024**2, 1)
        return info

    # ------------------------------------------------------------------ props
    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def bank_shape(self) -> tuple:
        return tuple(self._bank.shape)
