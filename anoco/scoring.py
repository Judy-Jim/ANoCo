"""Non-conformity as anomaly evidence (Section 3.6).

The optimised features F~_q are never used as predictions. The anomaly signal is the
*magnitude of the required update* to conform to the normal manifold. The default
patchwise energy is the product form of Eq. (10):

    E_i = || f~_q^i - f_q^i ||_2^2 * ( 1 - cos(f~_q^i, f_q^i) ) .

Patch energies form the dense anomaly map; the image-level score is a max over patches.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .utils import gaussian_blur_map


def patch_energy(
    f_q: torch.Tensor,
    f_tilde_q: torch.Tensor,
    metric: str = "product",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-patch non-conformity energy E_i, shape (N_q,).

    metric:
        "product" -> ||drift||^2 * (1 - cos)   (Eq. 10, default)
        "l2"      -> ||drift||^2
        "cos"     -> 1 - cos(f~, f)
    """
    diff = f_tilde_q - f_q
    l2 = (diff * diff).sum(dim=1)
    if metric == "l2":
        return l2
    cos = F.cosine_similarity(f_tilde_q, f_q, dim=1, eps=eps)
    cos_dis = 1.0 - cos
    if metric == "cos":
        return cos_dis
    if metric == "product":
        return l2 * cos_dis
    raise ValueError(f"unknown score_metric: {metric!r}")


def energy_to_map(
    energy: torch.Tensor,
    grid_hw: Tuple[int, int],
    out_hw: Optional[Tuple[int, int]] = None,
    kernel_size: int = 7,
    sigma: float = 0.8,
) -> torch.Tensor:
    """Reshape patch energies to the grid, upsample, then Gaussian-smooth.

    energy: (N_q,) with N_q = H*W (row-major, matching ViT patch order).
    Returns a (H, W) or (out_hw) 2D anomaly map.
    """
    h, w = grid_hw
    m = energy.reshape(h, w)
    if out_hw is not None:
        m = F.interpolate(
            m[None, None], size=out_hw, mode="bilinear", align_corners=False
        )[0, 0]
    return gaussian_blur_map(m, kernel_size, sigma)


def image_score(energy: torch.Tensor, reduction: str = "max") -> torch.Tensor:
    """Image-level anomaly score S(I_q) via max-pooling over patch energies (Section 3.6)."""
    if reduction == "max":
        return energy.max()
    if reduction == "mean":
        return energy.mean()
    raise ValueError(f"unknown image_reduction: {reduction!r}")


# ---------------------------------------------------------------------------
# Phase-1: Advanced aggregation strategies for robust image-level scoring.
# Motivation: max-pool is sensitive to isolated noisy patches (OK outliers).
# Real defects are spatially contiguous; noise is often a single-patch spike.
# ---------------------------------------------------------------------------


def image_score_topk_mean(energy: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Mean of the top-K patch energies.

    Real defects activate multiple adjacent patches → top-K mean preserves signal.
    Isolated noise spikes → diluted by the K-1 lower neighbours → suppressed.
    """
    k = min(k, energy.numel())
    topk = energy.topk(k).values
    return topk.mean()


def image_score_topk_weighted(
    energy: torch.Tensor, k: int = 10, gamma: float = 0.5
) -> torch.Tensor:
    """Adaptive score: max * (topk_mean / max)^gamma.

    When max >> topk_mean (isolated spike): ratio << 1 → score heavily penalised.
    When max ≈ topk_mean (broad anomaly): ratio ≈ 1 → score ≈ max (preserved).
    gamma controls penalty strength: higher gamma → less penalty; lower → more.
    """
    e_max = energy.max()
    if e_max <= 0:
        return e_max
    k = min(k, energy.numel())
    topk_mean = energy.topk(k).values.mean()
    ratio = (topk_mean / e_max).clamp(0.0, 1.0)
    return e_max * ratio.pow(gamma)


def image_score_connected_max(
    energy: torch.Tensor,
    grid_hw: Tuple[int, int],
    min_area: int = 3,
    threshold_ratio: float = 0.3,
) -> torch.Tensor:
    """Max energy after filtering out small connected components.

    Steps:
        1. Reshape energy to (H, W) grid.
        2. Threshold at threshold_ratio * max_energy to get a binary mask.
        3. Find connected components (4-connectivity).
        4. Zero out components with area < min_area (isolated spikes).
        5. Return the max of the filtered energy.

    This removes single-patch noise while preserving spatially extended defects.
    """
    h, w = grid_hw
    emap = energy.reshape(h, w)
    e_max = emap.max()
    if e_max <= 0:
        return e_max

    # Binary mask of "active" patches
    thresh = threshold_ratio * e_max
    binary = (emap >= thresh).cpu().numpy().astype("uint8")

    # Connected component labelling (4-connectivity)
    from scipy.ndimage import label as cc_label

    labelled, n_components = cc_label(binary)
    # Filter small components
    keep_mask = np.zeros_like(binary, dtype=bool)
    for comp_id in range(1, n_components + 1):
        comp = labelled == comp_id
        if comp.sum() >= min_area:
            keep_mask |= comp

    if not keep_mask.any():
        # All components are tiny → fall back to raw max (defence)
        return e_max

    # Max over the surviving patches
    filtered = emap.cpu().numpy() * keep_mask
    return torch.tensor(filtered.max(), dtype=energy.dtype, device=energy.device)


def image_score_aggregated(
    energy: torch.Tensor,
    method: str = "max",
    grid_hw: Optional[Tuple[int, int]] = None,
    k: int = 10,
    gamma: float = 0.5,
    min_area: int = 3,
    threshold_ratio: float = 0.3,
) -> torch.Tensor:
    """Unified dispatcher for all image-level aggregation strategies.

    method:
        "max"              – original (Section 3.6)
        "mean"             – global mean
        "topk_mean"        – mean of top-K energies
        "topk_weighted"    – adaptive: max * (topk_mean/max)^gamma
        "connected_max"    – max after removing small connected components
    """
    if method == "max":
        return energy.max()
    if method == "mean":
        return energy.mean()
    if method == "topk_mean":
        return image_score_topk_mean(energy, k=k)
    if method == "topk_weighted":
        return image_score_topk_weighted(energy, k=k, gamma=gamma)
    if method == "connected_max":
        if grid_hw is None:
            raise ValueError("connected_max requires grid_hw")
        return image_score_connected_max(
            energy, grid_hw, min_area=min_area, threshold_ratio=threshold_ratio
        )
    raise ValueError(f"unknown aggregation method: {method!r}")
