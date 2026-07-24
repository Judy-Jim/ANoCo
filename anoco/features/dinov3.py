"""DINOv3 patch-feature extractor (Supplementary S1).

Loads the paper's backbone (DINOv3-L/16 by default) from the *locally cloned* hub
repository so no network access to GitHub is needed, and reads patch tokens from a
chosen transformer block (the "18-th layer" -> 0-indexed 17 for ViT-L/24-blocks).

Weights:
    * If ``cfg.dinov3_weights`` is a local .pth path, it is loaded directly.
    * Otherwise the official LVD-1689M weights are fetched via torch.hub
      (``dl.fbaipublicfiles.com``); see ``scripts/download_dinov3.py``.

Only the extractor touches DINOv3; the ANoCo core stays backbone-agnostic.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import torch

from ..config import ANoCoConfig

# 0-indexed 8-char hashes used by the DINOv3 hub filenames (see dinov3/hub/backbones.py).
_DINOV3_HASH = {
    "dinov3_vits16": "08c60483",
    "dinov3_vitb16": "73cec8be",
    "dinov3_vitl16": "8aa4cbdd",
    "dinov3_vith16plus": "7c1da9a5",
}
_DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"


def expected_weight_filename(backbone: str) -> str:
    h = _DINOV3_HASH.get(backbone, "")
    suffix = f"-{h}" if h else ""
    return f"{backbone}_pretrain_lvd1689m{suffix}.pth"


def expected_weight_url(backbone: str) -> str:
    return f"{_DINOV3_BASE_URL}/{backbone}/{expected_weight_filename(backbone)}"


def _load_pil(image):
    from PIL import Image
    import numpy as np

    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = (255 * arr.clip(0, 1)).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if torch.is_tensor(image):
        t = image.detach().cpu()
        if t.dim() == 3 and t.shape[0] in (1, 3):     # CHW -> HWC
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        if arr.dtype != "uint8":
            import numpy as np

            arr = (255 * arr.clip(0, 1)).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)}")


class DINOv3Extractor:
    def __init__(self, cfg: ANoCoConfig):
        self.cfg = cfg
        self.device = cfg.device if (not cfg.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
        self.model = self._build_model(cfg)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.n_blocks = len(self.model.blocks)
        self.layer_index = int(cfg.layer_index)
        if not (0 <= self.layer_index < self.n_blocks):
            clamped = max(0, min(self.layer_index, self.n_blocks - 1))
            print(
                f"[DINOv3Extractor] layer_index={self.layer_index} out of range for "
                f"{cfg.backbone} ({self.n_blocks} blocks); clamping to {clamped}."
            )
            self.layer_index = clamped

        self._build_transform(cfg)
        # Phase-B runtime knobs (overridable by scripts after construction):
        #   decode="draft" -> JPEG reduced-scale decode (much faster, slight quality loss)
        #   fp16=True      -> run the backbone forward under fp16 autocast (speed + VRAM)
        self.decode = getattr(cfg, "decode", "pil")
        self.fp16 = bool(getattr(cfg, "fp16", False))

    # ------------------------------------------------------------------ build
    @staticmethod
    def _build_model(cfg: ANoCoConfig):
        repo = cfg.dinov3_repo_dir or os.environ.get("DINOV3_REPO", "")
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from dinov3.hub.backbones import (  # type: ignore
                Weights,
                dinov3_vitb16,
                dinov3_vith16plus,
                dinov3_vitl16,
                dinov3_vits16,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                f"Could not import DINOv3 from '{repo}'. Ensure the repository is present "
                f"(torch hub cache) and its dependencies are installed. Original error: {exc}"
            )

        builders = {
            "dinov3_vits16": dinov3_vits16,
            "dinov3_vitb16": dinov3_vitb16,
            "dinov3_vitl16": dinov3_vitl16,
            "dinov3_vith16plus": dinov3_vith16plus,
        }
        if cfg.backbone not in builders:
            raise ValueError(f"unsupported DINOv3 backbone: {cfg.backbone!r}")
        builder = builders[cfg.backbone]

        if cfg.dinov3_weights:
            if not os.path.isfile(cfg.dinov3_weights):
                raise FileNotFoundError(
                    f"dinov3_weights not found: {cfg.dinov3_weights}. "
                    f"Run scripts/download_dinov3.py or point to a local checkpoint."
                )
            # Load the local checkpoint directly (bypasses torch.hub's file:// URL cache,
            # which would otherwise return a stale copy keyed by basename).
            model = builder(pretrained=False)
            sd = torch.load(cfg.dinov3_weights, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            model.load_state_dict(sd, strict=True)
            return model

        # No explicit path: try the official cached/downloadable LVD-1689M weights.
        try:
            return builder(weights=Weights.LVD1689M)
        except Exception as exc:
            fname = expected_weight_filename(cfg.backbone)
            raise RuntimeError(
                f"Failed to obtain pretrained weights for {cfg.backbone}. "
                f"Download {expected_weight_url(cfg.backbone)} to the torch hub checkpoints "
                f"dir (file '{fname}') and set cfg.dinov3_weights to its path. "
                f"Original error: {exc}"
            )

    def _build_transform(self, cfg: ANoCoConfig):
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        h, w = cfg.img_hw if getattr(cfg, "img_hw", None) else (cfg.img_size, cfg.img_size)
        self.input_hw = (int(h), int(w))
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.input_hw, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
            ]
        )

    # ---------------------------------------------------------------- extract
    @torch.no_grad()
    def extract(self, image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Return (patch_features (N, d), grid_hw (H_p, W_p)) for one image.

        If cfg.layer_indices is set, extracts from multiple layers and fuses them
        (concat or weighted_sum). Otherwise uses the single cfg.layer_index.
        """
        if isinstance(image, str) and self.decode in ("draft", "cv2_draft", "cv2_reduced"):
            if self.decode == "draft":
                from PIL import Image
                img = Image.open(image)
                img.draft("RGB", (self.input_hw[1], self.input_hw[0]))
                pil = img.convert("RGB")
            elif self.decode.startswith("cv2"):
                import cv2
                flag = cv2.IMREAD_REDUCED_COLOR_8 if self.decode == "cv2_draft" else cv2.IMREAD_REDUCED_COLOR_4
                bgr = cv2.imread(image, flag)
                if bgr is None:
                    raise IOError(f"cv2 failed to decode: {image}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil = Image.fromarray(rgb)
        else:
            pil = _load_pil(image)
        x = self.transform(pil).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        use_fp16 = self.fp16 and str(self.device).startswith("cuda")

        # Determine which layers to extract
        layer_indices = getattr(self.cfg, "layer_indices", None)
        if layer_indices:
            indices = list(layer_indices)
        else:
            indices = [self.layer_index]

        if use_fp16:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                feats_list = self.model.get_intermediate_layers(
                    x, n=indices, reshape=False, norm=self.cfg.apply_norm
                )
        else:
            feats_list = self.model.get_intermediate_layers(
                x, n=indices, reshape=False, norm=self.cfg.apply_norm
            )

        # Fuse multi-layer features
        if len(feats_list) == 1:
            feat = feats_list[0][0]  # (N, d)
        else:
            feat = self._fuse_layers(feats_list)

        ps = self.cfg.patch_size
        grid = (self.input_hw[0] // ps, self.input_hw[1] // ps)
        assert feat.shape[0] == grid[0] * grid[1], (feat.shape, grid)
        return feat.float(), grid

    def _fuse_layers(self, feats_list) -> torch.Tensor:
        """Fuse features from multiple layers.

        feats_list: list of (1, N, d) tensors from get_intermediate_layers.
        Returns: (N, d_fused) tensor.
        """
        fusion = getattr(self.cfg, "layer_fusion", "concat")
        layers = [f[0] for f in feats_list]  # each (N, d)

        if fusion == "concat":
            # L2-normalize each layer's features, then concatenate
            from ..utils import l2_normalize
            normed = [l2_normalize(f) for f in layers]
            return torch.cat(normed, dim=1)  # (N, d*n_layers)

        elif fusion == "weighted_sum":
            # Weighted sum of L2-normalized features
            from ..utils import l2_normalize
            weights = getattr(self.cfg, "layer_weights", None)
            if weights is None:
                weights = [1.0 / len(layers)] * len(layers)
            normed = [l2_normalize(f) for f in layers]
            out = sum(w * f for w, f in zip(weights, normed))
            return out  # (N, d)

        else:
            raise ValueError(f"unknown layer_fusion: {fusion!r}")
