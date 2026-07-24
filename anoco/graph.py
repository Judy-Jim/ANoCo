"""Bipartite graph construction and Laplacian blocks (Sections 3.3-3.4).

Edges connect query patches only to their anchor-consistent reference neighbours,
so the Laplacian block L_qq is strictly diagonal (no query-query edges). We keep the
query-reference weight block W (N_q, N_r) explicitly; the full Laplacian is never
materialised except in tests (see ``solver.assemble_full_laplacian``).
"""

from __future__ import annotations

import torch


def norm_compatibility(f_q: torch.Tensor, f_r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Norm-compatibility factor alpha_ij (Section 3.3).

        alpha_ij = 2 * ||f_q^i|| * ||f_r^j|| / (||f_q^i||^2 + ||f_r^j||^2)

    Equals 1 when the two norms match and decays otherwise; restores magnitude
    information that plain cosine similarity discards.
    """
    nq = f_q.norm(dim=1)                       # (N_q,)
    nr = f_r.norm(dim=1)                       # (N_r,)
    num = 2.0 * nq[:, None] * nr[None, :]
    den = nq[:, None] ** 2 + nr[None, :] ** 2
    return num / den.clamp_min(eps)


def edge_weights(
    s: torch.Tensor,
    mask: torch.Tensor,
    alpha: torch.Tensor,
    clamp_negative: bool = False,
) -> torch.Tensor:
    """Query-reference edge weights w_ij = s_ij * alpha_ij for j in N(i), else 0 (Eq. 2)."""
    w = s * alpha
    if clamp_negative:
        w = w.clamp_min(0.0)
    return w * mask.to(w.dtype)


def query_degree(w: torch.Tensor) -> torch.Tensor:
    """Diagonal query degrees d_i^(q) = sum_{j in N(i)} w_ij  (=> L_qq = diag(d))."""
    return w.sum(dim=1)
