"""Configuration for ANoCo (Anomaly as Non-Conformity).

All tunable knobs of the method live here so that the core algorithm stays
backbone-agnostic and every paper assumption is explicit and overridable.

Paper references:
    - Backbone / preprocessing / post-processing: Supplementary S1.
    - lambda (query stabilisation coefficient Lambda_q): Section 3.5, S2.
    - Non-conformity metric (Eq. 10): Section 3.6, S4.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

# Location of the locally cloned DINOv3 hub repo (github.com/facebookresearch/dinov3).
# Set the DINOV3_REPO environment variable, or override cfg.dinov3_repo_dir directly.
DEFAULT_DINOV3_REPO = os.environ.get("DINOV3_REPO", "")


@dataclass
class ANoCoConfig:
    # ---- Backbone / feature extraction (Supplementary S1) ----
    backbone: str = "dinov3_vitl16"          # paper main backbone: DINOv3-L/16
    dinov3_repo_dir: str = DEFAULT_DINOV3_REPO
    dinov3_weights: str = ""                  # local .pth path; "" -> try hub download
    layer_index: int = 17                     # 0-indexed block; "18-th transformer layer"
    layer_indices: Optional[Tuple[int, ...]] = None  # multi-layer fusion; overrides layer_index
    layer_fusion: str = "concat"              # "concat" | "weighted_sum" (Phase-3)
    layer_weights: Optional[Tuple[float, ...]] = None  # weights for weighted_sum fusion
    n_storage_tokens: int = 4                 # DINOv3 register/storage tokens to drop
    apply_norm: bool = True                   # apply final LayerNorm to intermediate feats
    img_size: int = 768                       # square resize target (H=W) when img_hw is None
    img_hw: Optional[Tuple[int, int]] = None  # (H, W); set to preserve aspect ratio, overrides img_size
    patch_size: int = 16                      # input dims must be multiples of this
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ---- Anchor-driven retrieval (Section 3.2) ----
    # "stop_first_violation": longest prefix (paper wording).
    # "mask_all": keep every sorted-prefix ref with a_ij > tau_i (ablation variant).
    prefix_rule: str = "stop_first_violation"
    retrieve_topk: int = 200                    # topk(K) instead of full argsort, 0=full argsort

    # ---- Graph construction / solver (Sections 3.3-3.5) ----
    lam: float = 1.0                          # shared query stabilisation weight Lambda_q
    clamp_negative_weights: bool = False      # optionally clamp w_ij >= 0 for PSD safety
    eps: float = 1e-8

    # ---- Non-conformity scoring (Section 3.6, Eq. 10) ----
    score_metric: str = "product"             # {"product", "l2", "cos"}
    image_reduction: str = "max"              # image-level score = max-pool over patches

    # ---- Post-processing (Supplementary S1) ----
    gaussian_kernel: int = 7
    gaussian_sigma: float = 0.8

    # ---- Runtime ----
    device: str = "cuda"
    dtype: str = "float32"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ANoCoConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in (d or {}).items() if k in known}
        # tuples come back as lists from yaml/json
        for key in ("mean", "std", "img_hw", "layer_indices", "layer_weights"):
            if key in clean and isinstance(clean[key], list):
                clean[key] = tuple(clean[key])
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path: str) -> "ANoCoConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)
