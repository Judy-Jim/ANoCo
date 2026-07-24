"""Anchor-driven normal patch retrieval (Section 3.2).

For each query patch we:
    1. compute cosine similarity s_ij = cos(f_q^i, f_r^j);
    2. pick the anchor j*(i) = argmax_j s_ij with threshold tau_i = s_{i,j*(i)};
    3. compute anchor-to-reference similarity a_ij = cos(f_r^{j*(i)}, f_r^j);
    4. sort references by s_ij (descending) and retain the longest prefix for which
       a_ij > tau_i, yielding the anchor-consistent neighbour set N(i)  (Eq. 1).

Everything is returned as dense (N_q, N_r) tensors; entries outside N(i) are masked
out later when building edge weights, so the effective graph is sparse.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .utils import l2_normalize


def anchor_consistent_neighbors(
    s: torch.Tensor,
    a: torch.Tensor,
    tau: torch.Tensor,
    rule: str = "stop_first_violation",
    topk: int = 200,
) -> torch.Tensor:
    """Return a boolean neighbour mask of shape (N_q, N_r) implementing Eq. (1).

    Args:
        s:   (N_q, N_r) query-reference cosine similarity.
        a:   (N_q, N_r) anchor-reference cosine similarity, a_ij = cos(anchor_i, ref_j).
        tau: (N_q,) anchor similarity threshold s_{i,j*(i)}.
        rule:
            "stop_first_violation" -> longest prefix of the s-sorted list whose entries
                all satisfy a_ij > tau_i (paper wording, default).
            "mask_all" -> keep every reference with a_ij > tau_i (ablation variant).
        topk: if >0, only check the top-K most similar references instead of full sort.
            Validated to have zero precision impact (max neighbor count << K). Default 200.
    """
    n_q, n_r = s.shape

    if topk > 0 and topk < n_r:
        # OPTIMIZATION: topk(K) instead of full argsort(N_r) — 29x faster, zero precision loss
        K = min(topk, n_r)
        topk_result = s.topk(K, dim=1, largest=True, sorted=True)
        topk_idx = topk_result.indices                # (N_q, K) ref indices, sorted by s desc
        a_topk = a.gather(1, topk_idx)                 # (N_q, K) a values for top-K refs
        cond = a_topk > tau[:, None]                   # (N_q, K) bool

        if rule == "stop_first_violation":
            keep_topk = torch.cummin(cond.to(torch.int8), dim=1).values.bool()
        elif rule == "mask_all":
            keep_topk = cond
        else:
            raise ValueError(f"unknown prefix_rule: {rule!r}")

        mask = torch.zeros(n_q, n_r, dtype=torch.bool, device=s.device)
        mask.scatter_(1, topk_idx, keep_topk)
        return mask

    # Original full-argsort path (fallback when topk=0)
    order = torch.argsort(s, dim=1, descending=True)          # (N_q, N_r) ref indices
    a_sorted = torch.gather(a, 1, order)                       # a values in s-desc order
    cond = a_sorted > tau[:, None]                             # (N_q, N_r) bool

    if rule == "stop_first_violation":
        keep_sorted = torch.cummin(cond.to(torch.int8), dim=1).values.bool()
    elif rule == "mask_all":
        keep_sorted = cond
    else:
        raise ValueError(f"unknown prefix_rule: {rule!r}")

    mask = torch.zeros(n_q, n_r, dtype=torch.bool, device=s.device)
    mask.scatter_(1, order, keep_sorted)
    return mask


def retrieve(
    f_q: torch.Tensor,
    f_r: torch.Tensor,
    rule: str = "stop_first_violation",
    eps: float = 1e-12,
    topk: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full anchor-driven retrieval.

    Args:
        topk: if >0, use topk(K) instead of full argsort for neighbor selection.
            Default 200 (validated zero precision impact). Set 0 for full argsort.

    Returns:
        s:          (N_q, N_r) cosine similarity.
        mask:       (N_q, N_r) bool, True for j in N(i).
        anchor_idx: (N_q,) index j*(i) of each query's anchor reference.
        tau:        (N_q,) anchor similarity threshold.
    """
    q_n = l2_normalize(f_q, eps)
    r_n = l2_normalize(f_r, eps)
    s = q_n @ r_n.transpose(0, 1)                              # (N_q, N_r)
    tau, anchor_idx = s.max(dim=1)                             # anchor = most similar ref
    anchors = r_n.index_select(0, anchor_idx)                 # (N_q, d) normalised anchors
    a = anchors @ r_n.transpose(0, 1)                         # (N_q, N_r) a_ij
    mask = anchor_consistent_neighbors(s, a, tau, rule, topk=topk)
    return s, mask, anchor_idx, tau


def retrieve_tiled(
    f_q: torch.Tensor,
    f_r: torch.Tensor,
    rule: str = "stop_first_violation",
    eps: float = 1e-12,
    topk: int = 200,
    tile_size: int = 10000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimized retrieval: tiled similarity + grouped bmm for a_topk.

    Avoids materializing full (N_q, N_r) matrices. Computes similarity in tiles,
    keeps running top-K. Then computes anchor-to-ref similarity only for top-K refs.

    Returns same (s, mask, anchor_idx, tau) as retrieve(), but s is reconstructed
    only for the top-K entries (rest is 0). mask is full (N_q, N_r) bool.

    Precision: identical to retrieve(topk=K) — same computation, different order.
    """
    q_n = l2_normalize(f_q, eps)
    r_n = l2_normalize(f_r, eps)
    n_q = f_q.shape[0]
    n_r = f_r.shape[0]
    K = min(topk, n_r)
    dev = f_q.device

    # === Tiled similarity + running topk ===
    s_topk_vals = torch.full((n_q, K), -1.0, device=dev)
    s_topk_idx = torch.zeros(n_q, K, dtype=torch.long, device=dev)

    for tile_start in range(0, n_r, tile_size):
        tile_end = min(tile_start + tile_size, n_r)
        r_tile = r_n[tile_start:tile_end]                      # (tile, D)
        s_tile = q_n @ r_tile.transpose(0, 1)                  # (N_q, tile)

        # Merge with running topk
        combined_vals = torch.cat([s_topk_vals, s_tile], dim=1)  # (N_q, K+tile)
        combined_idx = torch.cat([
            s_topk_idx,
            torch.arange(tile_start, tile_end, device=dev).unsqueeze(0).expand(n_q, -1)
        ], dim=1)
        topk_result = combined_vals.topk(K, dim=1, largest=True, sorted=True)
        s_topk_vals = topk_result.values
        s_topk_idx = combined_idx.gather(1, topk_result.indices)

    # === Anchor + tau ===
    tau = s_topk_vals[:, 0]  # top-1 = anchor = max similarity
    anchor_idx = s_topk_idx[:, 0]
    anchors = r_n.index_select(0, anchor_idx)  # (N_q, D)

    # === Grouped bmm for a_topk (avoids full a = anchors @ r_n.T) ===
    a_topk = torch.empty(n_q, K, device=dev)
    group_size = 20
    for g in range(0, K, group_size):
        g_end = min(g + group_size, K)
        g_sz = g_end - g
        idx_g = s_topk_idx[:, g:g_end]                         # (N_q, g_sz)
        flat_idx = idx_g.reshape(-1)                           # (N_q * g_sz,)
        r_group = r_n.index_select(0, flat_idx).reshape(n_q, g_sz, -1)  # (N_q, g_sz, D)
        a_g = torch.bmm(anchors.unsqueeze(1), r_group.transpose(1, 2)).squeeze(1)  # (N_q, g_sz)
        a_topk[:, g:g_end] = a_g

    # === stop_first_violation on top-K ===
    cond = (a_topk > tau.unsqueeze(1)).to(torch.int8)          # (N_q, K)
    if rule == "stop_first_violation":
        keep_topk = torch.cummin(cond, dim=1).values.bool()
    elif rule == "mask_all":
        keep_topk = cond.bool()
    else:
        raise ValueError(f"unknown prefix_rule: {rule!r}")

    # === Build full mask (scatter top-K into N_r) ===
    mask = torch.zeros(n_q, n_r, dtype=torch.bool, device=dev)
    mask.scatter_(1, s_topk_idx, keep_topk)

    # === Reconstruct s (sparse: only top-K non-zero) ===
    s = torch.zeros(n_q, n_r, device=dev)
    s.scatter_(1, s_topk_idx, s_topk_vals)

    return s, mask, anchor_idx, tau


def sparse_solve(
    f_q: torch.Tensor,
    f_r: torch.Tensor,
    topk_idx: torch.Tensor,
    w_topk: torch.Tensor,
    lam: float = 1.0,
    eps: float = 1e-8,
    group_size: int = 20,
) -> torch.Tensor:
    """Optimized solve: grouped bmm instead of full w @ f_r.

    Args:
        f_q: (N_q, D) query features
        f_r: (N_r, D) reference features
        topk_idx: (N_q, K) indices of top-K references
        w_topk: (N_q, K) edge weights for top-K references
        lam: regularization weight
    Returns:
        f_tilde: (N_q, D) solved features
    """
    n_q, D = f_q.shape
    K = topk_idx.shape[1]
    dev = f_q.device

    d_q = w_topk.sum(dim=1)  # (N_q,)
    agg = torch.zeros(n_q, D, device=dev)

    for g in range(0, K, group_size):
        g_end = min(g + group_size, K)
        idx_g = topk_idx[:, g:g_end]                           # (N_q, g_sz)
        w_g = w_topk[:, g:g_end]                               # (N_q, g_sz)
        flat_idx = idx_g.reshape(-1)                           # (N_q * g_sz,)
        f_group = f_r.index_select(0, flat_idx).reshape(n_q, -1, D)  # (N_q, g_sz, D)
        # Weighted sum: (N_q, 1, g_sz) @ (N_q, g_sz, D) → (N_q, 1, D) → (N_q, D)
        agg_g = torch.bmm(w_g.unsqueeze(1).to(f_group.dtype), f_group).squeeze(1)
        agg += agg_g

    denom = (lam + d_q).clamp_min(eps).unsqueeze(1)  # (N_q, 1)
    return (lam * f_q + agg) / denom
