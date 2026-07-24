# ANoCo (Unofficial Reproduction)

**Training-free, closed-form anomaly detection** — an independent reimplementation of the CVPR 2026
paper [*Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy Minimization*](https://openaccess.thecvf.com/content/CVPR2026/html/Seo_Anomaly_as_Non-Conformity_via_Training-Free_Graph_Laplacian_Energy_Minimization_CVPR_2026_paper.html)
(Seo et al.), plus a set of engineering optimizations for many-shot industrial inspection.

> ⚠️ **Disclaimer.** This is an **unofficial** reproduction written from the paper. It is **not**
> affiliated with or endorsed by the paper's authors or Meta AI. The authors had not released official
> code at the time of writing; this implementation reflects our own reading of the method. DINOv3
> weights and any proprietary datasets are **not** included — see [Weights](#weights).

## What is ANoCo?

Like PatchCore, ANoCo is a memory-bank / retrieval anomaly detector — but the **scoring kernel is different**:

- **PatchCore** scores a query patch by its **distance** to the nearest normal patch (independent similarity).
- **ANoCo** scores a query patch by the **feature drift** needed to pull it back onto the normal
  manifold — the closed-form solution of an *anchored graph-Laplacian energy minimization*. Larger
  drift ⇒ more anomalous. No training, no gradients: one closed-form solve per patch.

Pipeline (paper Sections 3.2–3.6):

```
DINOv3 patch features
  → anchor-driven retrieval of consistent normal neighbours   (§3.2, Eq. 1)
  → bipartite edge weights  w = cos · norm-compatibility       (§3.3–3.4)
  → closed-form anchored Laplacian solve  f̃                    (§3.5, Eq. 8–9)
  → non-conformity energy  E = ‖f̃ − f‖² · (1 − cos)            (§3.6, Eq. 10)
  → anomaly map / image-level score
```

## Highlights

- **Faithful core, verified.** The closed-form solver is bit-exact vs a dense linear solve
  (`tests/`), and MVTec-AD numbers track the paper (see [Results](#results)).
- **Backbone-agnostic core** (`anoco/`): operates on patch features; DINOv3 is the only optional
  backbone dependency, isolated in `anoco/features/`.
- **Engineering optimizations** (beyond the paper, for real production lines):
  - **Multi-layer feature fusion** (e.g. L11 + L17 concat) — texture + semantic cues.
  - **Cluster-stratified memory bank** (K-Means) + **greedy coreset** (PatchCore-style) for coverage at a fixed budget.
  - **Excess normalization** `max(0, E − μ − kσ)` — robust to calib/test distribution shift.
  - **Top-k aggregation** for image scoring, less sensitive to isolated spikes than max-pool.
  - **Tiled + sparse matching** and **top-k retrieval** — same result, far less VRAM (bit-exact).
  - **Threshold calibration** at a target false-positive (overkill) rate.
- **Zero test-time augmentation** at inference — all augmentation is precomputed on the reference side.

## Install

```bash
pip install -e .
# optional extras (MVTec demo overlays, safetensors conversion):
pip install -e ".[demo]"
```

A CUDA GPU is needed for the DINOv3 backbone; the synthetic unit tests run on CPU.

## Weights

DINOv3 is released by Meta under the **DINOv3 License** (gated). Weights are **not** distributed here —
obtain them yourself and comply with Meta's license:

```bash
# Option A: official checkpoint (accept Meta's DINOv3 license first)
python scripts/download_dinov3.py --backbone dinov3_vitl16

# Option B: convert a Hugging Face safetensors checkpoint to the fb format
python scripts/convert_dinov3_safetensors.py \
    --src dinov3-vitl16-pretrain-lvd1689m.safetensors \
    --backbone dinov3_vitl16 --out weights/dinov3_vitl16.pt
```

Point the config at the DINOv3 hub repo and checkpoint via `dinov3_repo_dir` / `dinov3_weights`
in `configs/*.yaml`, or via the `DINOV3_REPO` environment variable.

## Quick start

```bash
# 1) Unit tests (CPU, synthetic data — no weights needed)
pytest tests/ -q

# 2) Reproduce MVTec-AD numbers (public data, few-shot)
export MVTEC_ROOT=/path/to/mvtec_anomaly_detection
python scripts/demo_mvtec.py --categories screw metal_nut bottle --shots 1

# 3) Your own line: build a bank, then inspect a folder
python scripts/build_bank.py     --ok-dir path/to/ok  --config configs/production.yaml --out-dir banks/my_product
python scripts/run_inspector.py  --bank-dir banks/my_product --config configs/production.yaml \
                                 --image-dir path/to/images --save-overlays results/viz/
```

Python API:

```python
from anoco import ANoCo, ANoCoConfig

model = ANoCo(ANoCoConfig())
out = model.score_features(f_q, f_r, grid_hw=(48, 48))   # backbone-agnostic
print(out["score"], out["anomaly_map"].shape)
```

## Results

### Reproduction fidelity (pixel-level, `screw`)

The core method tracks the paper — MVTec-AD `screw`, DINOv3-L/16, single-layer, many-shot:
pixel-AUROC 97.7 (paper 98.4), pixel-PRO 91.8 (93.5), pixel-F1 49.1 (53.2).

### Do the engineering optimizations help? (public-data ablation)

Image-level AUROC on **all 15 MVTec-AD categories**, DINOv3-L/16, with an **identical** 50-image
bank + greedy coreset (10k) for both columns — isolating **multi-layer fusion (L11+L17) + top-k
aggregation** against the single-layer baseline. Reproduce with:

```bash
# baseline
python scripts/demo_mvtec.py --data-root $MVTEC_ROOT --shots 0 --max-refs 50 \
    --coreset 10000 --coreset-method greedy --fp16 --layers 17 --agg max
# optimized
python scripts/demo_mvtec.py --data-root $MVTEC_ROOT --shots 0 --max-refs 50 \
    --coreset 10000 --coreset-method greedy --fp16 --layers 11,17 --agg topk_mean --agg-k 5
```

| Category | Baseline (L17, max) | + Multi-layer + top-k | Δ |
|---|---|---|---|
| toothbrush | 87.8 | 100.0 | +12.2 |
| screw | 87.7 | 92.4 | +4.7 |
| capsule | 95.9 | 98.4 | +2.5 |
| cable | 98.3 | 99.7 | +1.4 |
| hazelnut | 99.6 | 100.0 | +0.4 |
| carpet | 99.6 | 99.9 | +0.3 |
| pill | 98.8 | 99.0 | +0.2 |
| bottle | 100.0 | 100.0 | 0 |
| grid | 100.0 | 100.0 | 0 |
| leather | 100.0 | 100.0 | 0 |
| metal_nut | 100.0 | 100.0 | 0 |
| tile | 100.0 | 100.0 | 0 |
| wood | 99.4 | 99.4 | 0 |
| zipper | 100.0 | 99.8 | −0.2 |
| transistor | 96.3 | 95.4 | −0.9 |
| **Mean (15)** | **97.6** | **98.9** | **+1.3** |

Multi-layer fusion + top-k aggregation help most exactly where the single-layer baseline is weak
(toothbrush, screw, capsule); categories already at ceiling stay there. The remaining optimizations
(**excess normalization**, **threshold calibration**, **cluster-stratified banking**) target the
**industrial operating point** — the nominal-vs-actual overkill gap and multi-mode OK coverage — which
MVTec's standard AUROC does not capture; see `docs/METHOD_NOTES.md`.

## Repo structure

```
anoco/                    core library (backbone-agnostic)
  retrieval.py            §3.2 anchor-driven retrieval (+ tiled / sparse / top-k)
  graph.py                §3.3–3.4 bipartite edge weights
  solver.py               §3.5 closed-form anchored Laplacian solve
  scoring.py              §3.6 non-conformity energy + aggregations
  membank.py              memory bank + greedy coreset
  normalization.py        excess / z-score normalizers
  calibration.py          score calibration
  metrics.py              AUROC / AUPR / F1 / PRO (numpy + scipy)
  bank_builder.py         cluster-stratified bank + threshold calibration
  inspector.py            production inference wrapper
  features/dinov3.py      DINOv3 extractor (multi-layer fusion)
configs/                  L/16 (paper), B/16 (speed), production template
scripts/                  weights prep, MVTec demo, build / infer
tests/                    4 synthetic verification tests
docs/METHOD_NOTES.md      general engineering techniques
```

## Citation

Please cite the original paper ([CVPR 2026, open access](https://openaccess.thecvf.com/content/CVPR2026/html/Seo_Anomaly_as_Non-Conformity_via_Training-Free_Graph_Laplacian_Energy_Minimization_CVPR_2026_paper.html)):

```bibtex
@InProceedings{Seo_2026_CVPR,
  author    = {Seo, Jungwook and Kim, Minjeong and Lee, Younkwan and Shin, Seungho and Baik, Sungyong},
  title     = {Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy Minimization},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  pages     = {21336-21345}
}
```

This repository is an unofficial reproduction; feel free to reference its URL as well.

## License & attribution

- **Code**: Apache-2.0 (see `LICENSE`). Copyright 2026 Judy-Jim.
- **DINOv3 weights**: governed by Meta's separate **DINOv3 License** — not included here.
- Not affiliated with the ANoCo authors or Meta AI. "PatchCore", "DINOv3", and "MVTec AD" belong to
  their respective owners.
