"""Positional energy normalization for ANoCo (Phase-2).

Different spatial positions in the patch grid have different baseline energy levels
(e.g., workpiece edges, background regions naturally produce higher drift even when normal).
This module learns per-position statistics from calibration OK images and normalizes
query energies accordingly, so that only *locally unusual* energy contributes to the score.

Usage:
    normalizer = PositionNormalizer()
    normalizer.fit(calib_energies, calib_grids)       # from calib OK only
    normed_energy = normalizer.transform(energy, grid) # at inference
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


class PositionNormalizer:
    """Per-position z-score normalization of patch energies.

    After fitting on calibration OK images, each patch position (h, w) has a learned
    mean μ(h,w) and std σ(h,w). At inference:
        energy_norm[i] = (energy[i] - μ[pos_i]) / σ[pos_i]

    Variants:
        method="zscore"   : standard z-score (mean/std)
        method="robust"   : (energy - median) / MAD  (more robust to calib outliers)
        method="percentile": rank of energy within the calib distribution at that position
                             (approximated via normal CDF of the z-score)
    """

    def __init__(self, method: str = "zscore", clamp_min: float = 0.0, eps: float = 1e-6):
        """
        Args:
            method: normalization method ("zscore", "robust", "percentile")
            clamp_min: if > 0, clamp normalized energy to this minimum (e.g., 0 means
                       "below-normal" energy is treated as 0 — only excess matters)
            eps: minimum std to avoid division by zero
        """
        self.method = method
        self.clamp_min = clamp_min
        self.eps = eps
        self._fitted = False
        self._mean: Optional[np.ndarray] = None   # (H*W,)
        self._std: Optional[np.ndarray] = None    # (H*W,)
        self._median: Optional[np.ndarray] = None
        self._mad: Optional[np.ndarray] = None
        self._grid_hw: Optional[Tuple[int, int]] = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        energies: List[torch.Tensor],
        grids: List[Tuple[int, int]],
    ) -> "PositionNormalizer":
        """Fit per-position statistics from calibration OK patch energies.

        Args:
            energies: list of (N_q,) tensors, one per calib OK image.
            grids: list of (H, W) tuples (should all be the same for fixed input size).
        """
        # Verify all grids are the same
        grid_set = set(grids)
        if len(grid_set) != 1:
            raise ValueError(f"All grids must be identical for positional normalization; got {grid_set}")
        self._grid_hw = grids[0]
        n_pos = self._grid_hw[0] * self._grid_hw[1]

        # Stack all energies: (n_calib, n_pos)
        stack = np.stack([e.cpu().numpy() for e in energies], axis=0)  # (C, N_q)
        assert stack.shape[1] == n_pos, f"Energy length {stack.shape[1]} != grid size {n_pos}"

        if self.method in ("zscore", "percentile"):
            self._mean = stack.mean(axis=0)  # (n_pos,)
            self._std = stack.std(axis=0).clip(self.eps)  # (n_pos,)
        if self.method in ("robust", "percentile"):
            self._median = np.median(stack, axis=0)
            mad = np.median(np.abs(stack - self._median[None, :]), axis=0)
            self._mad = mad.clip(self.eps) * 1.4826  # scale to be consistent with std for normal dist

        self._fitted = True
        return self

    def transform(self, energy: torch.Tensor, grid_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Normalize patch energies using fitted per-position statistics.

        Args:
            energy: (N_q,) patch energies for one image.
            grid_hw: (H, W) — must match the grid used during fit.

        Returns:
            Normalized energy (N_q,), same dtype/device as input.
        """
        if not self._fitted:
            raise RuntimeError("PositionNormalizer not fitted; call .fit() first")

        device = energy.device
        dtype = energy.dtype
        e = energy.cpu().numpy().astype(np.float64)

        if self.method == "zscore":
            normed = (e - self._mean) / self._std
        elif self.method == "robust":
            normed = (e - self._median) / self._mad
        elif self.method == "percentile":
            # Approximate percentile via normal CDF of z-score
            from scipy.stats import norm as sp_norm
            z = (e - self._mean) / self._std
            normed = sp_norm.cdf(z)  # in [0, 1]
        else:
            raise ValueError(f"unknown method: {self.method!r}")

        if self.clamp_min is not None and self.clamp_min > 0:
            normed = normed.clip(self.clamp_min)
        elif self.clamp_min == 0.0:
            normed = normed.clip(0.0)

        return torch.tensor(normed, dtype=dtype, device=device)

    def transform_batch(
        self, energies: List[torch.Tensor], grids: List[Tuple[int, int]]
    ) -> List[torch.Tensor]:
        """Transform a batch of energies."""
        return [self.transform(e, g) for e, g in zip(energies, grids)]

    def stats_summary(self) -> dict:
        """Return summary statistics of the fitted normalizer."""
        if not self._fitted:
            return {"fitted": False}
        info = {
            "fitted": True,
            "method": self.method,
            "grid_hw": self._grid_hw,
            "n_positions": int(self._mean.size) if self._mean is not None else 0,
        }
        if self._mean is not None:
            info["mean_of_means"] = float(self._mean.mean())
            info["mean_of_stds"] = float(self._std.mean())
            info["std_range"] = [float(self._std.min()), float(self._std.max())]
        return info


# ---------------------------------------------------------------------------
# Task-3: Adaptive normalizers that fix calibration transfer.
# Key insight: pure z-score amplifies unseen normal variation. These variants
# are robust to distribution shift between calib and test OK.
# ---------------------------------------------------------------------------


class ExcessNormalizer:
    """Soft-threshold normalization: only count energy exceeding k*σ above the mean.

    energy_norm[i] = max(0, energy[i] - μ[pos] - k * σ[pos])

    This ignores "normal variation" (within k standard deviations) and only
    responds to genuinely excess energy. Very robust to distribution shift
    because small changes in the OK population don't affect the threshold.
    """

    def __init__(self, k: float = 2.0, eps: float = 1e-6):
        self.k = k
        self.eps = eps
        self._fitted = False
        self._threshold: Optional[np.ndarray] = None  # (n_pos,) = μ + k*σ

    def fit(self, energies: List[torch.Tensor], grids: List[Tuple[int, int]]) -> "ExcessNormalizer":
        grid_set = set(grids)
        if len(grid_set) != 1:
            raise ValueError(f"All grids must be identical; got {grid_set}")
        n_pos = grids[0][0] * grids[0][1]
        stack = np.stack([e.cpu().numpy() for e in energies], axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0).clip(self.eps)
        self._threshold = mean + self.k * std
        self._fitted = True
        return self

    def transform(self, energy: torch.Tensor, grid_hw=None) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("not fitted")
        e = energy.cpu().numpy().astype(np.float64)
        normed = np.maximum(0.0, e - self._threshold)
        return torch.tensor(normed, dtype=energy.dtype, device=energy.device)


class SelfNormalizer:
    """Per-image self-normalization: divide by the image's own robust statistic.

    energy_norm[i] = energy[i] / percentile(energy, q)

    This removes global scale differences between images (e.g., overall brightness,
    exposure variation) without any fitted parameters. Completely invariant to
    distribution shift between calib and test.
    """

    def __init__(self, q: float = 90.0, eps: float = 1e-8):
        """
        Args:
            q: percentile to use as the image's "scale" (default 90th).
            eps: minimum denominator to avoid division by zero.
        """
        self.q = q
        self.eps = eps

    def transform(self, energy: torch.Tensor, grid_hw=None) -> torch.Tensor:
        e = energy.cpu().numpy().astype(np.float64)
        scale = np.percentile(e, self.q)
        scale = max(scale, self.eps)
        normed = e / scale
        return torch.tensor(normed, dtype=energy.dtype, device=energy.device)

    def fit(self, *args, **kwargs) -> "SelfNormalizer":
        """No fitting needed — self-normalization is parameter-free."""
        return self


class HybridNormalizer:
    """Positional z-score + per-image rescaling.

    Step 1: Apply positional z-score (removes position bias).
    Step 2: Divide by the image's own P-th percentile of z-scored energies
            (removes image-level scale, robust to unseen OK variation).

    This combines the position-awareness of z-score with the distribution-shift
    robustness of self-normalization.
    """

    def __init__(self, q: float = 95.0, clamp_min: float = 0.0, eps: float = 1e-6):
        self.q = q
        self.clamp_min = clamp_min
        self.eps = eps
        self._pos_norm = PositionNormalizer(method="zscore", clamp_min=0.0, eps=eps)
        self._fitted = False

    def fit(self, energies: List[torch.Tensor], grids: List[Tuple[int, int]]) -> "HybridNormalizer":
        self._pos_norm.fit(energies, grids)
        self._fitted = True
        return self

    def transform(self, energy: torch.Tensor, grid_hw=None) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("not fitted")
        # Step 1: positional z-score
        z = self._pos_norm.transform(energy, grid_hw).cpu().numpy().astype(np.float64)
        # Step 2: per-image rescaling
        scale = np.percentile(z, self.q)
        scale = max(scale, self.eps)
        normed = z / scale
        if self.clamp_min is not None:
            normed = normed.clip(self.clamp_min)
        return torch.tensor(normed, dtype=energy.dtype, device=energy.device)
