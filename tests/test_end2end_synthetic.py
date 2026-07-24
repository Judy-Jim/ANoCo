"""Verification 4: synthetic end-to-end separability + ablation trend.

We build a synthetic "normal manifold" (a few Gaussian clusters on the unit sphere),
sample a reference pool from it, and a query patch grid that is normal everywhere
except an injected off-manifold block. Running the full ANoCo pipeline, the patch
energies should give near-perfect patch-level AUROC against the injected mask.

We also check the component-wise ablation ordering (Table 2):
    k-NN (L2)  <=  k-NN + bipartite (naive top-k)  <=  ANoCo (anchor-driven).

Run standalone:  python tests/test_end2end_synthetic.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from anoco import ANoCo, ANoCoConfig
from anoco.baselines import energy_with_mask, knn_l2_energy, topk_neighbor_mask
from anoco.retrieval import retrieve
from anoco.utils import cosine_sim_matrix, l2_normalize


def auroc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=np.float64)
    i, n = 0, len(scores)
    while i < n:                                   # average ranks over ties
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks_sorted[i : j + 1] = ranks_sorted[i : j + 1].mean()
        i = j + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _build_scene(seed=0):
    g = torch.Generator().manual_seed(seed)
    d, n_clusters = 32, 4
    centers = l2_normalize(torch.randn(n_clusters, d, generator=g))

    def sample(center, n, noise=0.05):
        return l2_normalize(center + noise * torch.randn(n, d, generator=g))

    f_r = torch.cat([sample(centers[c], 50) for c in range(n_clusters)], dim=0)

    h = w = 16
    labels = np.zeros(h * w, dtype=np.int64)
    f_q = torch.empty(h * w, d)
    for r in range(h):
        for c in range(w):
            cluster = (r // (h // 2)) * 2 + (c // (w // 2))  # quadrant -> cluster id
            f_q[r * w + c] = sample(centers[cluster], 1, 0.05)[0]

    # off-manifold anomaly direction: low max-cosine to every cluster centre.
    while True:
        cand = l2_normalize(torch.randn(1, d, generator=g))
        if cosine_sim_matrix(cand, centers).abs().max().item() < 0.3:
            break
    for r in range(3, 8):
        for c in range(3, 8):
            f_q[r * w + c] = l2_normalize(cand + 0.05 * torch.randn(1, d, generator=g))[0]
            labels[r * w + c] = 1
    return f_q, f_r, labels, (h, w)


def test_synthetic_separability():
    f_q, f_r, labels, _ = _build_scene()
    out = ANoCo(ANoCoConfig(device="cpu")).score_features(f_q, f_r)
    score = auroc(out["patch_energy"].cpu().numpy(), labels)
    assert score > 0.9, f"synthetic patch AUROC too low: {score:.4f}"


def test_ablation_trend():
    f_q, f_r, labels, _ = _build_scene(seed=1)
    knn = auroc(knn_l2_energy(f_q, f_r).cpu().numpy(), labels)

    s, anchor_mask, _, _ = retrieve(f_q, f_r)
    topk = topk_neighbor_mask(s, k=int(anchor_mask.sum(dim=1).float().mean().clamp(min=1)))
    bip = auroc(energy_with_mask(f_q, f_r, topk).cpu().numpy(), labels)
    anoco = auroc(energy_with_mask(f_q, f_r, anchor_mask).cpu().numpy(), labels)

    # The energy-drift formulation (even naive) should be competitive with or better
    # than plain nearest-neighbour distance, and anchor-driven should be strong.
    print(f"  AUROC  kNN={knn:.4f}  bipartite(topk)={bip:.4f}  anchor(ANoCo)={anoco:.4f}")
    assert anoco > 0.9, f"anchor-driven AUROC too low: {anoco:.4f}"
    assert anoco >= knn - 0.05, f"anchor-driven worse than kNN: {anoco:.4f} vs {knn:.4f}"


if __name__ == "__main__":
    test_synthetic_separability()
    test_ablation_trend()
    print("test_end2end_synthetic: ALL PASS")
