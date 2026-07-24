"""Verification 3: non-conformity scoring semantics (Section 3.6).

Checks the qualitative behaviour the paper relies on:
    * a query patch identical to a reference has zero feature drift -> zero energy;
    * a query patch that lies far off the normal manifold is dragged toward it,
      producing a large drift -> high energy;
so normal patches score well below anomalous ones.

Run standalone:  python tests/test_scoring.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from anoco import ANoCo, ANoCoConfig
from anoco.utils import l2_normalize


def _build_features():
    g = torch.Generator().manual_seed(0)
    d = 16
    u = torch.zeros(d); u[0] = 1.0          # normal manifold direction
    v = torch.zeros(d); v[1] = 1.0          # anomalous direction (orthogonal to u)

    def near(base, n, noise):
        return l2_normalize(base + noise * torch.randn(n, d, generator=g))

    f_r = near(u, 8, 0.02)                  # reference pool: tight normal cluster
    normals = near(u, 5, 0.03)              # normal queries (close but not identical)
    copies = f_r[:2].clone()                # exact copies -> must score ~0
    outliers = near(v, 5, 0.02)             # off-manifold queries -> must score high

    f_q = torch.cat([normals, copies, outliers], dim=0)
    idx = {
        "normal": slice(0, 5),
        "copy": slice(5, 7),
        "outlier": slice(7, 12),
    }
    return f_q, f_r, idx


def test_energy_separates_normal_and_anomalous():
    f_q, f_r, idx = _build_features()
    model = ANoCo(ANoCoConfig(device="cpu"))
    out = model.score_features(f_q, f_r)
    e = out["patch_energy"].cpu()

    e_norm = e[idx["normal"]]
    e_copy = e[idx["copy"]]
    e_out = e[idx["outlier"]]

    assert e_copy.max().item() < 1e-6, f"exact copies should have ~0 energy: {e_copy}"
    assert e_out.mean().item() > 5.0 * (e_norm.mean().item() + 1e-12), (
        f"anomalous energy not clearly higher: out={e_out.mean()} norm={e_norm.mean()}"
    )
    assert e_out.min().item() > e_norm.max().item(), (
        f"overlap between normal and anomalous energies: "
        f"out.min={e_out.min()} norm.max={e_norm.max()}"
    )


if __name__ == "__main__":
    test_energy_separates_normal_and_anomalous()
    print("test_scoring: ALL PASS")
