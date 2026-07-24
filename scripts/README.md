# scripts/

All scripts run from the repo root and import the local `anoco` package
(each inserts the repo root into `sys.path`). Run with `python scripts/<name>.py ...`.

## 1. Weights preparation
| Script | Purpose |
|---|---|
| `download_dinov3.py` | Download / verify the official DINOv3 LVD-1689M checkpoint (gated; you must accept Meta's DINOv3 license). Prints manual steps on failure. |
| `convert_dinov3_safetensors.py` | Convert a Hugging Face DINOv3 `*.safetensors` to a facebookresearch-format `*.pt` (fuses q/k/v, zeroes qkv bias, bf16 RoPE; validated with `strict=True`). |

## 2. Reproduction (public data)
| Script | Purpose |
|---|---|
| `demo_mvtec.py` | Reproduction / ablation on MVTec-AD; reports per-category image AUROC vs the paper's Table S6. Flags: `--layers 11,17` (multi-layer fusion), `--agg topk_mean`, `--fp16`, `--shots 0 --max-refs N` (many-shot), `--coreset N --coreset-method greedy`. Set `MVTEC_ROOT` or pass `--data-root`. |

## 3. Production pipeline (your own data)
| Script | Purpose |
|---|---|
| `build_testset.py` | Split OK + NG folders into leak-free bank / calib / test parts (+ manifest). |
| `build_bank.py` | Build a bank from OK images: cluster-stratified sampling + greedy coreset, fit excess normalizer, calibrate threshold → `bank.pt` / `normalizer.pt` / `threshold.json`. |
| `run_inspector.py` | Load a built bank and inspect a single image or a folder; optional overlay export + CSV logging. |

## Notes
- DINOv3 weights are **not** shipped (Meta license). Prepare them with section 1.
- Set `dinov3_repo_dir` / `dinov3_weights` in the config, or the `DINOV3_REPO` env var.
