"""Tensor utilities shared across ANoCo modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalisation (last dim)."""
    return F.normalize(x, p=2, dim=-1, eps=eps)


def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Cosine similarity between every row of ``a`` and every row of ``b``.

    a: (Na, d), b: (Nb, d) -> (Na, Nb).
    """
    return l2_normalize(a, eps) @ l2_normalize(b, eps).transpose(-1, -2)


def _gaussian_kernel1d(ksize: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    k = torch.exp(-(ax ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur_map(m: torch.Tensor, kernel_size: int = 7, sigma: float = 0.8) -> torch.Tensor:
    """Separable Gaussian blur on a 2D map (Supplementary S1: k=7, sigma=0.8).

    m: (H, W) -> (H, W). No-op when kernel_size <= 1 or sigma <= 0.
    """
    if kernel_size is None or kernel_size <= 1 or sigma is None or sigma <= 0:
        return m
    assert m.dim() == 2, "gaussian_blur_map expects a (H, W) tensor"
    k = _gaussian_kernel1d(int(kernel_size), float(sigma), m.device, m.dtype)
    pad = int(kernel_size) // 2
    x = m[None, None]  # (1, 1, H, W)
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kx)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), ky)
    return x[0, 0]


def as_tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    """Convert array-likes to a float tensor on the requested device."""
    if torch.is_tensor(x):
        t = x.to(dtype=dtype)
    else:
        t = torch.as_tensor(x, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t
