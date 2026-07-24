# tests/

Run from the repo root:

```bash
pytest tests/ -q
```

These tests depend only on the `anoco` core (CPU, synthetic data — no GPU and no DINOv3
weights required), so they verify the mathematical correctness of the method itself.

| Test | Covers |
|---|---|
| `test_solver.py` | Closed-form anchored Laplacian solve (§3.5): matches a dense linear solve (bit-exact) and minimizes the convex energy; per-patch decoupling. |
| `test_retrieval.py` | Anchor-driven retrieval (§3.2): anchor = most similar reference; `stop_first_violation` longest-prefix neighbour set (Eq. 1). |
| `test_scoring.py` | Non-conformity energy (§3.6, Eq. 10): exact copies score ~0; off-manifold queries score high. |
| `test_end2end_synthetic.py` | Synthetic end-to-end separability + component ablation ordering (kNN ≤ bipartite ≤ anchor-driven). |
