"""Offline production bank builder.

Takes an OK image folder and produces:
  - bank.pt          : memory bank tensor (N_coreset, D)
  - normalizer.pt    : excess normalizer statistics
  - threshold.json   : calibrated decision threshold
  - build_report.json: metadata for traceability

Usage:
    builder = BankBuilder()
    builder.build(
        ok_dir="path/to/ok_images",
        config_path="configs/production.yaml",
        out_dir="banks/my_product",
    )
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from .config import ANoCoConfig
from .membank import build_memory_bank
from .normalization import ExcessNormalizer
from .scoring import image_score_aggregated


class BankBuilder:
    """Build a production-ready memory bank from OK images."""

    def build(
        self,
        ok_dir: str,
        config_path: str,
        out_dir: str,
        bank_size: int = 200,
        n_clusters: int = 20,
        coreset: int = 80000,
        coreset_method: str = "greedy",
        ref_aug: str = "none",
        excess_k: float = 3.0,
        target_fpr: float = 2.0,
        seed: int = 42,
        decode: str = "draft",
        fp16: bool = True,
        chunk: int = 1024,
    ) -> dict:
        """Full bank building pipeline.

        Args:
            ok_dir: folder of OK (normal) images.
            config_path: path to production YAML config.
            out_dir: output directory for all artifacts.
            bank_size: number of images to select for the bank.
            n_clusters: K-Means cluster count for stratified sampling.
            coreset: coreset size (number of patches to keep).
            coreset_method: "greedy" or "random".
            ref_aug: reference augmentation mode.
            excess_k: excess normalizer k parameter.
            target_fpr: target false positive rate (%) for threshold.
            seed: random seed for reproducibility.
            decode: image decode mode ("draft" or "pil").
            fp16: use fp16 for backbone forward.
            chunk: chunk size for memory-capped scoring.

        Returns:
            Build report dict.
        """
        t_total = time.time()
        os.makedirs(out_dir, exist_ok=True)

        # 1. Load config and setup extractor
        cfg = ANoCoConfig.from_yaml(config_path)
        from .features.dinov3 import DINOv3Extractor
        ext = DINOv3Extractor(cfg)
        ext.decode = decode
        ext.fp16 = fp16

        # 2. Scan OK folder
        ok_files = sorted([
            f for f in os.listdir(ok_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        ok_paths = [os.path.join(ok_dir, f) for f in ok_files]
        print(f"[bank_builder] {len(ok_paths)} OK images in {ok_dir}")
        if len(ok_paths) < bank_size * 2:
            raise ValueError(
                f"Need at least {bank_size * 2} OK images, got {len(ok_paths)}"
            )

        # 3. Extract image-level features for clustering
        t0 = time.time()
        print("[bank_builder] extracting image features for clustering...")
        img_feats = []
        for i, p in enumerate(ok_paths):
            fq, _ = ext.extract(p)
            img_feats.append(fq.mean(dim=0).cpu())
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(ok_paths)}")
        img_feats = torch.stack(img_feats).numpy()
        print(f"[bank_builder] image features: {img_feats.shape} in {time.time()-t0:.0f}s")

        # 4. K-Means clustering
        t0 = time.time()
        print(f"[bank_builder] K-Means K={n_clusters}...")
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels_km = km.fit_predict(img_feats)
        cluster_sizes = np.bincount(labels_km)
        print(f"[bank_builder] cluster sizes: {sorted(cluster_sizes, reverse=True)}")

        # 5. Stratified sampling for bank
        rng = np.random.default_rng(seed)
        selected_indices = []
        for c in range(n_clusters):
            members = np.where(labels_km == c)[0]
            n_select = max(1, int(round(bank_size * len(members) / len(ok_paths))))
            n_select = min(n_select, len(members))
            chosen = rng.choice(members, size=n_select, replace=False)
            selected_indices.extend(chosen.tolist())
        # Trim or pad to exact bank_size
        if len(selected_indices) > bank_size:
            selected_indices = rng.choice(
                selected_indices, size=bank_size, replace=False
            ).tolist()
        elif len(selected_indices) < bank_size:
            remaining = [
                i for i in range(len(ok_paths)) if i not in set(selected_indices)
            ]
            extra = rng.choice(
                remaining, size=bank_size - len(selected_indices), replace=False
            )
            selected_indices.extend(extra.tolist())

        bank_paths = [ok_paths[i] for i in selected_indices]
        calib_paths = [
            ok_paths[i] for i in range(len(ok_paths))
            if i not in set(selected_indices)
        ]
        print(f"[bank_builder] bank={len(bank_paths)} calib={len(calib_paths)} "
              f"in {time.time()-t0:.0f}s")

        # 6. Build memory bank (greedy coreset)
        t0 = time.time()
        print(f"[bank_builder] building memory bank (coreset={coreset})...")
        f_r = build_memory_bank(
            ext, bank_paths, ref_aug=ref_aug, coreset=coreset,
            coreset_method=coreset_method, seed=seed,
        )
        print(f"[bank_builder] bank: {tuple(f_r.shape)} in {time.time()-t0:.0f}s")

        # 7. Extract calib energies and fit excess normalizer
        t0 = time.time()
        print(f"[bank_builder] extracting calib energies ({len(calib_paths)} images)...")
        from .anoco import ANoCo
        model = ANoCo(cfg)
        calib_energies, calib_grids = [], []
        for i, p in enumerate(calib_paths):
            fq, grid = ext.extract(p)
            out = model.score_features(fq, f_r, chunk=chunk)
            calib_energies.append(out["patch_energy"].cpu())
            calib_grids.append(grid)
            if (i + 1) % 100 == 0:
                print(f"  calib {i+1}/{len(calib_paths)}")
        print(f"[bank_builder] calib energies done in {time.time()-t0:.0f}s")

        t0 = time.time()
        print(f"[bank_builder] fitting excess normalizer (k={excess_k})...")
        normalizer = ExcessNormalizer(k=excess_k)
        normalizer.fit(calib_energies, calib_grids)
        print(f"[bank_builder] normalizer fitted in {time.time()-t0:.0f}s")

        # 8. Calibrate threshold
        agg_method = "topk_mean"
        agg_k = 5
        calib_scores = []
        for e, g in zip(calib_energies, calib_grids):
            en = normalizer.transform(e, g)
            s = image_score_aggregated(en, method=agg_method, grid_hw=g, k=agg_k)
            calib_scores.append(s.item())
        threshold = float(np.percentile(calib_scores, 100 * (1 - target_fpr / 100)))
        print(f"[bank_builder] threshold={threshold:.6f} (calib p{100-target_fpr}, "
              f"target FPR={target_fpr}%)")

        # 9. Save artifacts
        bank_path = os.path.join(out_dir, "bank.pt")
        normalizer_path = os.path.join(out_dir, "normalizer.pt")
        threshold_path = os.path.join(out_dir, "threshold.json")
        report_path = os.path.join(out_dir, "build_report.json")

        torch.save(f_r.cpu(), bank_path)
        print(f"[bank_builder] saved bank: {bank_path} ({f_r.shape})")

        torch.save(
            {"threshold": normalizer._threshold},
            normalizer_path,
        )
        print(f"[bank_builder] saved normalizer: {normalizer_path}")

        threshold_data = {
            "threshold": threshold,
            "target_fpr": target_fpr,
            "calib_stats": {
                "mean": float(np.mean(calib_scores)),
                "std": float(np.std(calib_scores)),
                "max": float(np.max(calib_scores)),
                "p90": float(np.percentile(calib_scores, 90)),
                "p95": float(np.percentile(calib_scores, 95)),
                "p99": float(np.percentile(calib_scores, 99)),
            },
            "n_calib": len(calib_scores),
            "build_time": datetime_utc_now(),
        }
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump(threshold_data, f, indent=2, ensure_ascii=False)
        print(f"[bank_builder] saved threshold: {threshold_path}")

        # 10. Build report
        report = {
            "config_hash": self._config_hash(config_path),
            "ok_dir": ok_dir,
            "n_ok_total": len(ok_paths),
            "n_bank_images": len(bank_paths),
            "n_calib_images": len(calib_paths),
            "bank_shape": list(f_r.shape),
            "coreset": coreset,
            "coreset_method": coreset_method,
            "n_clusters": n_clusters,
            "cluster_sizes": sorted(cluster_sizes.tolist(), reverse=True),
            "excess_k": excess_k,
            "threshold": threshold,
            "target_fpr": target_fpr,
            "seed": seed,
            "decode": decode,
            "fp16": fp16,
            "chunk": chunk,
            "build_time": datetime_utc_now(),
            "build_duration_s": round(time.time() - t_total, 1),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[bank_builder] saved report: {report_path}")
        print(f"[bank_builder] total build time: {time.time()-t_total:.0f}s")

        return report

    @staticmethod
    def _config_hash(config_path: str) -> str:
        with open(config_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]


def datetime_utc_now() -> str:
    from datetime import timezone, datetime
    return datetime.now(timezone.utc).isoformat()
