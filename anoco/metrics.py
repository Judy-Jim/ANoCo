"""Evaluation metrics for anomaly detection.

Implements the paper's six-metric protocol pieces with numpy + scipy only (no sklearn
dependency, so it also runs in the base env): image/pixel AUROC, F1-max, and the MVTec
PRO (per-region overlap) score.

AUROC and F1-max are computed via sorting (O(M log M)), so they scale to millions of
pixels. PRO iterates thresholds over connected ground-truth regions and integrates the
per-region TPR against FPR up to a limit (default 0.3), matching the MVTec-AD protocol.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def auroc(scores, labels) -> float:
    """Rank-based ROC-AUC (handles ties via average ranks)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(np.int64)
    n = scores.size
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks_sorted = np.arange(1, n + 1, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i : j + 1] = ranks_sorted[i : j + 1].mean()
        i = j + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def f1_max(scores, labels) -> float:
    """Maximum F1 over all thresholds (sort-based sweep)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.size:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ls = labels[order]
    tp = np.cumsum(ls)
    fp = np.cumsum(~ls)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(n_pos, 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.max())


def aupr(scores, labels) -> float:
    """Average precision (area under the precision-recall curve)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.size:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ls = labels[order]
    tp = np.cumsum(ls)
    fp = np.cumsum(~ls)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / n_pos
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def recall_at_fpr(ok_scores, ng_scores, fpr_pct, calib_scores=None):
    """Threshold from OK-score (calib if given, else ok_scores) quantile at target FPR;
    return (recall%, actual_fpr%, threshold)."""
    ok = np.asarray(ok_scores, dtype=np.float64)
    ng = np.asarray(ng_scores, dtype=np.float64)
    base = np.asarray(calib_scores, dtype=np.float64) if calib_scores is not None else ok
    thr = float(np.percentile(base, 100.0 - fpr_pct))
    recall = 100.0 * float(np.mean(ng > thr))
    act_fpr = 100.0 * float(np.mean(ok > thr))
    return recall, act_fpr, thr


def compute_pro(masks, amaps, num_th: int = 200, fpr_limit: float = 0.3) -> float:
    """MVTec PRO: per-region overlap integrated against FPR up to ``fpr_limit``.

    masks: (N, H, W) binary ground truth; amaps: (N, H, W) anomaly maps.
    """
    from scipy.ndimage import label

    masks = (np.asarray(masks) > 0.5).astype(np.uint8)
    amaps = np.asarray(amaps, dtype=np.float64)
    if masks.sum() == 0:
        return float("nan")
    inv = 1 - masks
    n_norm = float(inv.sum())
    normal_vals = amaps[inv.astype(bool)]
    if normal_vals.size == 0 or float(amaps.max()) <= float(amaps.min()):
        return float("nan")

    regions = []                                   # connected components per image
    for m in masks:
        lab, k = label(m)
        regions.append([(lab == r) for r in range(1, k + 1)])

    # Scale-robust thresholds: sample FPR uniformly in [0, fpr_limit] via normal-pixel
    # quantiles. Plain linspace over raw values under-resolves the low-FPR region when
    # maps have per-image scale variance, collapsing PRO even for well-ranked maps.
    fpr_grid = np.linspace(0.0, fpr_limit, num_th)
    ths = np.quantile(normal_vals, np.clip(1.0 - fpr_grid, 0.0, 1.0))
    pros, fprs = [], []
    for th in ths:
        binary = amaps > th
        tprs = []
        for b, regs in zip(binary, regions):
            for reg in regs:
                tprs.append(float(b[reg].sum()) / float(reg.sum()))
        pros.append(np.mean(tprs) if tprs else 0.0)
        fprs.append(float((binary * inv).sum()) / (n_norm + 1e-12))

    pros = np.asarray(pros)
    fprs = np.asarray(fprs)
    o = np.argsort(fprs)
    fprs, pros = fprs[o], pros[o]
    keep = fprs <= fpr_limit + 1e-9
    if keep.sum() < 2:
        return float("nan")
    return float(np.trapz(pros[keep], fprs[keep]) / fpr_limit)


def image_metrics(scores, labels) -> Dict[str, float]:
    return {"auroc": auroc(scores, labels), "aupr": aupr(scores, labels), "f1_max": f1_max(scores, labels)}


def pixel_metrics(
    amaps, masks, with_pro: bool = True, pro_num_th: int = 200, pro_fpr_limit: float = 0.3
) -> Dict[str, float]:
    out = {"auroc": auroc(amaps, masks), "f1_max": f1_max(amaps, masks)}
    if with_pro:
        out["pro"] = compute_pro(masks, amaps, pro_num_th, pro_fpr_limit)
    return out
