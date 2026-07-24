# Engineering Notes

General techniques and design decisions layered on top of the ANoCo method for
**many-shot industrial inspection**. These are backbone- and dataset-agnostic; no
proprietary data or numbers are included here. Tune every knob on your own data.

---

## 1. Method recap

ANoCo scores a query patch by the **feature drift** required to conform to the normal
manifold, obtained in closed form (no training):

1. **Anchor-driven retrieval** (§3.2): for each query patch, the anchor is its most
   similar reference; keep the longest prefix of similarity-sorted references that stay
   consistent with the anchor (`stop_first_violation`).
2. **Bipartite edge weights** (§3.3–3.4): `w = cosine · norm_compatibility`. With
   L2-normalized features the norm-compatibility factor is ~1 and can be skipped.
3. **Closed-form solve** (§3.5): because the query–query Laplacian block is diagonal,
   the optimum decouples per patch — one division per patch, no iterations.
4. **Non-conformity energy** (§3.6): `E = ‖f̃ − f‖² · (1 − cos(f̃, f))`. Patch energies
   form the anomaly map; the image score aggregates them.

Correctness is pinned by `tests/` (bit-exact vs a dense solve; synthetic separability).

---

## 2. Many-shot memory bank (vs few-shot research setting)

Few-shot (K = 1..4) is a *research* constraint. On a real line you usually have many
"good" samples, so build a richer reference pool once, offline:

- **Cluster-stratified sampling.** Cluster OK images (K-Means on mean patch features) and
  sample the bank proportionally per cluster. This guarantees every "normal mode"
  (lighting, batch, pose) is represented, avoiding false positives on under-covered modes.
  In practice this is one of the highest-leverage changes for coverage.
- **Greedy coreset** (PatchCore-style farthest-point sampling with a random projection):
  better manifold coverage than random subsampling at the same patch budget.
- **Bank size and coreset must grow together.** A large bank with a tiny coreset can hurt
  (the coreset can't represent the added diversity). Keep the coreset roughly proportional
  to `bank_images × patches_per_image`.
- **Reference-side augmentation only** (`none | flips | rot90 | light`), precomputed once.
  Choose per part geometry — e.g. `rot90` suits rotation-varying square parts, but is wrong
  for fixed-orientation elongated parts. **No query-side test-time augmentation** is used, so
  per-query cost is unchanged and results are deterministic.

---

## 3. Robust scoring & normalization

- **Positional excess normalization** `max(0, E − μ[pos] − k·σ[pos])` (fit on calibration OK):
  removes per-position baselines (edges/background are naturally higher) and normal variation,
  keeping only genuinely excess energy. It is **robust to calib/test distribution shift** —
  unlike plain per-position z-score, which amplifies unseen normal variation and can cause a
  large gap between the nominal and the actual false-positive rate. `k ≈ 2–3` is a good start.
- **Image-level aggregation.** `max` is simple but sensitive to a single outlier patch.
  `topk_mean` (mean of the top-K patch energies) is steadier; `topk_weighted`
  (`max · (topk_mean/max)^γ`) adaptively penalizes isolated spikes.
- **Threshold calibration.** Set the decision threshold from a **separate** calibration set
  at a target false-positive (overkill) rate, never on the test set (see §5).

---

## 4. Deployment efficiency (all bit-exact unless noted)

- **Chunked matching** (`chunk`): score query patches in chunks so the large
  `(N_q, N_r)` intermediates never materialize at once — the main VRAM lever. Result identical.
- **Top-k retrieval** (`retrieve_topk`): replace a full `argsort(N_r)` with `topk(K)`.
  With anchor-consistent neighbours the kept set is tiny, so a modest K is lossless.
- **Tiled similarity + sparse solve**: compute similarity in tiles and keep a running top-K,
  then do grouped `bmm` only over the top-K references — avoids full `(N_q, N_r)` matmuls.
- **fp16 autocast** for the backbone forward (features cast back to fp32 for matching):
  faster and lighter, negligible effect on ranking.
- **Reduced-scale JPEG decode** (`decode="draft"`): decodes at ~1/4 resolution then resizes —
  a large preprocessing speedup. Validate its effect on your images (it drops high-frequency
  detail; very fine defects may be affected).

Typical ordering of cost: input/output size (`N_q`) ≈ backbone > matching (`N_r`) > map.

---

## 5. Evaluation protocol (avoid leakage)

Split OK images into **three mutually exclusive** sets, plus all NG into test:

- **bank** — builds the normal feature memory.
- **calib** — sets the threshold only (target-FPR quantile of OK scores).
- **test** — final metrics only. Never tune the threshold on it.

`scripts/build_testset.py` produces a reproducible, leak-free split. Report **AUROC**
(threshold-free, primary), **AUPR** (honest under class imbalance), **F1-max**, and
**recall @ target-FPR** operating points. A calibration set that is too small underestimates
the OK-score ceiling, so the *actual* FPR at run time can exceed the nominal target — prefer a
few hundred calibration images.

---

## 6. Tuning checklist

1. **Input resolution** = biggest quality/speed lever; match the part's aspect ratio (each side a
   multiple of `patch_size`). More pixels is not always better — feature quality can saturate.
2. **More OK images in the bank** usually beats most other changes for recall (raise the coreset with it).
3. **Multi-layer fusion** (a mid + a deep block, L2-normalize then concat) tends to give the
   largest single quality gain; three layers are often redundant vs the best two.
4. **Excess normalization + separate calibration** to make the nominal and actual overkill agree.
5. **`topk_mean` aggregation** to suppress isolated OK spikes.
6. Inspect the highest-scoring OK images: persistent outliers are often **mislabeled defects** or
   edge/fixture artifacts — cleaning labels frequently helps more than model changes.
