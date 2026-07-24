"""ANoCo: Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy Minimization.

Training-free, closed-form few-shot anomaly detection. Public API:

    from anoco import ANoCo, ANoCoConfig
    model = ANoCo(ANoCoConfig())
    out = model.score_features(f_q, f_r, grid_hw=(48, 48))   # backbone-agnostic
    out = model.run_image(query_path, [ref_path])            # DINOv3-L/16 features
"""

from __future__ import annotations

from .config import ANoCoConfig
from .anoco import ANoCo
from . import retrieval, graph, solver, scoring, baselines, utils

__all__ = [
    "ANoCo",
    "ANoCoConfig",
    "retrieval",
    "graph",
    "solver",
    "scoring",
    "baselines",
    "utils",
]

__version__ = "0.1.0"
