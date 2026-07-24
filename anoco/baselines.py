"""Simple baselines and retrieval variants for the component-wise ablation (Table 2).

These are intentionally minimal and share ANoCo's solver/scoring so that the only
thing that changes across the ablation is *which* references each query connects to:

    k-NN (L2)                -> nearest-neighbour distance, no graph.
    k-NN + Non-Bipartite     -> top-k neighbours incl. spurious matches (naive).
    k-NN + Bipartite         -> top-k neighbours, bipartite graph (no anchor filter).
    ANoCo (Anchor + Bip.)    -> anchor-consistent neighbours (the full method).
"""

from __future__ import annotations

import torch

from .graph import edge_weights, norm_compatibility
from .scoring import patch_energy
from .solver import solve_closed_form
from .utils import cosine_sim_matrix


def knn_l2_energy(f_q: torch.Tensor, f_r: torch.Tensor) -> torch.Tensor:
    """Per-patch nearest-neighbour squared-L2 distance (independent similarity)."""
    q2 = (f_q * f_q).sum(1, keepdim=True)          # (N_q, 1)
    r2 = (f_r * f_r).sum(1)[None, :]               # (1, N_r)
    dist = q2 + r2 - 2.0 * (f_q @ f_r.transpose(0, 1))
    return dist.clamp_min(0).min(dim=1).values     # (N_q,)


def topk_neighbor_mask(s: torch.Tensor, k: int) -> torch.Tensor:
    """Naive top-k retrieval mask by similarity (no anchor consistency)."""
    n_r = s.shape[1]
    k = int(min(max(k, 1), n_r))
    idx = s.topk(k, dim=1).indices
    mask = torch.zeros_like(s, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def energy_with_mask(
    f_q: torch.Tensor,
    f_r: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 1.0,
    metric: str = "product",
    eps: float = 1e-8,
    clamp_negative: bool = False,
) -> torch.Tensor:
    """Run the ANoCo solve/score for an arbitrary neighbour mask (ablation helper)."""
    s = cosine_sim_matrix(f_q, f_r)
    alpha = norm_compatibility(f_q, f_r, eps=eps)
    w = edge_weights(s, mask, alpha, clamp_negative=clamp_negative)
    f_tilde = solve_closed_form(f_q, f_r, w, lam=lam, eps=eps)
    return patch_energy(f_q, f_tilde, metric=metric, eps=eps)
