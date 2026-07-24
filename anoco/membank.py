"""Normal-feature memory bank for engineering deployment.

Few-shot (K=1..4) is a *research* constraint; on a real line you usually have many
"good" samples. This module builds a reference patch pool once from many normal images,
optionally enriched with cheap, deterministic reference-side geometric augmentation
(rotations/flips) for pose robustness, and optionally coreset-subsampled to bound memory.

Key property for production: all augmentation happens on the *reference* side and is
precomputed once. Scoring a query costs the same as without augmentation (no per-query
test-time augmentation), and results are deterministic/reproducible.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from PIL import Image

REF_AUG_PRESETS = ("none", "flips", "rot90", "light")


def _aug_views(pil: Image.Image, preset: str) -> List[Image.Image]:
    """Return deterministic geometric views of a reference image.

    "rot90" uses exact 90-degree transposes (no interpolation / black corners), which is
    ideal for square, rotation-varying parts (screw, metal_nut, hazelnut, ...).
    """
    views = [pil]
    if preset == "none":
        return views
    if preset == "flips":
        views += [pil.transpose(Image.FLIP_LEFT_RIGHT), pil.transpose(Image.FLIP_TOP_BOTTOM)]
    elif preset == "rot90":
        views += [
            pil.transpose(Image.ROTATE_90),
            pil.transpose(Image.ROTATE_180),
            pil.transpose(Image.ROTATE_270),
            pil.transpose(Image.FLIP_LEFT_RIGHT),
        ]
    elif preset == "light":
        views += [
            pil.transpose(Image.FLIP_LEFT_RIGHT),
            pil.rotate(10, resample=Image.BILINEAR, expand=False),
            pil.rotate(-10, resample=Image.BILINEAR, expand=False),
        ]
    else:
        raise ValueError(f"unknown ref_aug preset: {preset!r} (choose from {REF_AUG_PRESETS})")
    return views


def _approx_greedy_indices(
    feats: torch.Tensor,
    n_samples: int,
    start_points: int = 10,
    proj_dim: int = 128,
    seed: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """Approximate greedy k-center (coreset) selection, adapted from PatchCore.

    Farthest-point sampling that maximises coverage of the feature manifold, using a
    random low-dim projection + a few random anchors for speed (avoids the full NxN
    distance matrix). Returns the selected row indices.
    """
    n = feats.shape[0]
    x = feats.to(device)
    if proj_dim and x.shape[1] > proj_dim:
        pg = torch.Generator().manual_seed(seed)
        proj = torch.randn(x.shape[1], proj_dim, generator=pg).to(device)
        x = x @ proj

    sg = torch.Generator().manual_seed(seed)
    sp = int(min(start_points, n))
    starts = torch.randperm(n, generator=sg)[:sp].to(device)

    def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:   # (Na, Nb) Euclidean
        a2 = (a * a).sum(1, keepdim=True)
        b2 = (b * b).sum(1)[None, :]
        return (a2 + b2 - 2.0 * a @ b.t()).clamp_min_(0).sqrt_()

    min_d = _dist(x, x.index_select(0, starts)).mean(dim=1)     # (N,)
    idxs = torch.empty(n_samples, dtype=torch.long)
    for i in range(n_samples):
        sel = int(torch.argmax(min_d).item())
        idxs[i] = sel
        min_d = torch.minimum(min_d, _dist(x, x[sel:sel + 1]).squeeze(1))
    return idxs


def coreset_subsample(
    feats: torch.Tensor,
    size: int,
    method: str = "random",
    seed: int = 0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Subsample rows (patches) to at most ``size`` (0 or >= N disables).

    method:
        "random" -> uniform random subset (fast).
        "greedy" -> approximate greedy k-center coreset (better manifold coverage,
                    PatchCore-style); slower but preserves diversity at fixed budget.
    """
    n = feats.shape[0]
    if not size or size <= 0 or size >= n:
        return feats
    if method == "random":
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:size]
        return feats[idx]
    if method == "greedy":
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        idx = _approx_greedy_indices(feats, size, seed=seed, device=dev)
        return feats[idx]
    raise ValueError(f"unknown coreset method: {method!r} (use 'random' or 'greedy')")


def build_memory_bank(
    extractor,
    image_paths: Sequence[str],
    ref_aug: str = "none",
    coreset: int = 0,
    coreset_method: str = "random",
    seed: int = 0,
    verbose: bool = False,
) -> torch.Tensor:
    """Build a normal reference patch pool F_r (N_r, d) on CPU.

    Args:
        extractor: a DINOv3Extractor (or any object with ``.extract(image)->(feat, grid)``).
        image_paths: normal ("good") reference images.
        ref_aug: one of REF_AUG_PRESETS; augmentations applied per reference image.
        coreset: if > 0, randomly subsample the pooled patches to this many.
    """
    feats: List[torch.Tensor] = []
    for i, p in enumerate(image_paths):
        pil = Image.open(p).convert("RGB")
        for v in _aug_views(pil, ref_aug):
            f, _ = extractor.extract(v)          # (N, d) on extractor device
            feats.append(f.detach().to("cpu"))
        if verbose and (i + 1) % 20 == 0:
            print(f"  [membank] extracted {i + 1}/{len(image_paths)} refs")
    bank = torch.cat(feats, dim=0)
    return coreset_subsample(
        bank, coreset, method=coreset_method, seed=seed,
        device=getattr(extractor, "device", None),
    )


def save_bank(path: str, bank: torch.Tensor) -> None:
    torch.save(bank.cpu(), path)


def load_bank(path: str, device: str = "cpu") -> torch.Tensor:
    return torch.load(path, map_location=device)
