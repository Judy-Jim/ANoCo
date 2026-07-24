"""Anchored Laplacian energy minimisation, closed form (Section 3.5).

Total energy (Eq. 7):
    E(F~) = F~^T L F~ + sum_i lambda_i || f~_q^i - f_q^i ||_2^2 ,
with reference features clamped. Because L_qq = D_q is diagonal (no query-query
edges) and Lambda_q is diagonal, the optimum decouples per query patch (Eq. 8-9):

    (L_qq + Lambda_q) F~_q = Lambda_q F_q - L_qr F_r ,   with  L_qr = -W ,

    =>  f~_q^i = ( lambda_i * f_q^i + sum_{j in N(i)} w_ij * f_r^j ) / ( lambda_i + d_i ) .

The update uses the *raw* (un-normalised) reference/query features; cosine similarity
and the norm-compatibility factor only enter through the edge weights w_ij.
"""

from __future__ import annotations

from typing import Union

import torch


def solve_closed_form(
    f_q: torch.Tensor,
    f_r: torch.Tensor,
    w: torch.Tensor,
    lam: Union[float, torch.Tensor] = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the optimised query features F~_q of shape (N_q, d).

    Args:
        f_q: (N_q, d) raw query features.
        f_r: (N_r, d) raw reference features (clamped / fixed).
        w:   (N_q, N_r) query-reference edge weights (0 outside N(i)).
        lam: shared scalar or per-query (N_q,) stabilisation weight Lambda_q.
    """
    n_q = f_q.shape[0]
    if not torch.is_tensor(lam):
        lam = torch.full((n_q,), float(lam), device=f_q.device, dtype=f_q.dtype)
    else:
        lam = lam.to(device=f_q.device, dtype=f_q.dtype)
        if lam.dim() == 0:
            lam = lam.expand(n_q)

    d_q = w.sum(dim=1)                          # (N_q,) diagonal of L_qq
    agg = w @ f_r                               # (N_q, d) = sum_j w_ij f_r^j
    denom = (lam + d_q).clamp_min(eps)          # (N_q,) diagonal of (L_qq + Lambda_q)
    f_tilde = (lam[:, None] * f_q + agg) / denom[:, None]
    return f_tilde


def assemble_full_laplacian(w: torch.Tensor) -> torch.Tensor:
    """Materialise the full (N_q+N_r) x (N_q+N_r) bipartite Laplacian L = D - A.

    Only used for verification (tests); the solver itself never needs it.
    A has the query-reference block W in the top-right and W^T in the bottom-left.
    """
    n_q, n_r = w.shape
    n = n_q + n_r
    adj = torch.zeros(n, n, dtype=w.dtype, device=w.device)
    adj[:n_q, n_q:] = w
    adj[n_q:, :n_q] = w.transpose(0, 1)
    deg = torch.diag(adj.sum(dim=1))
    return deg - adj


def total_energy(
    f_tilde_q: torch.Tensor,
    f_q: torch.Tensor,
    f_r: torch.Tensor,
    w: torch.Tensor,
    lam: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """Scalar anchored energy E(F~) of Eq. (7); used to check optimality in tests."""
    n_q = f_q.shape[0]
    if not torch.is_tensor(lam):
        lam = torch.full((n_q,), float(lam), device=f_q.device, dtype=f_q.dtype)
    f_full = torch.cat([f_tilde_q, f_r], dim=0)                 # (N, d)
    lap = assemble_full_laplacian(w)
    e_lap = torch.trace(f_full.transpose(0, 1) @ lap @ f_full)
    e_reg = (lam * ((f_tilde_q - f_q) ** 2).sum(dim=1)).sum()
    return e_lap + e_reg
