"""Small end-to-end demo of ANoCo on MVTec-AD (Verification 6 in the plan).

For each requested category it:
    * samples ``--shots`` normal reference image(s) from train/good;
    * scores every test image (image-level anomaly score S = max patch energy);
    * reports image-level AUROC and compares to the paper's Table S6 (1-shot);
    * saves an anomaly-map overlay for one anomalous sample per category.

Run (needs a CUDA GPU and DINOv3 weights; set MVTEC_ROOT or pass --data-root):
    python scripts/demo_mvtec.py \
        --categories bottle carpet grid --shots 1 --max-per-defect 10
"""

import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from anoco import ANoCo, ANoCoConfig
from anoco.membank import build_memory_bank, load_bank
from anoco.metrics import image_metrics, pixel_metrics
from anoco.scoring import image_score_aggregated

DEFAULT_ROOT = os.environ.get("MVTEC_ROOT", "data/mvtec_anomaly_detection")

# Paper Table S6: per-category image-level AUROC on MVTec-AD, 1-shot (DINOv3-L/16).
TABLE_S6_IMG_AUROC = {
    "carpet": 99.8, "grid": 100.0, "leather": 100.0, "tile": 100.0, "wood": 98.6,
    "bottle": 99.9, "cable": 96.0, "capsule": 93.8, "hazelnut": 98.2,
    "metal_nut": 100.0, "pill": 96.5, "screw": 89.7, "toothbrush": 100.0,
    "transistor": 96.7, "zipper": 99.4,
}


def auroc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=np.float64)
    i, n = 0, len(scores)
    while i < n:
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


def list_images(folder):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def load_mask(cat_dir, defect, img_path, size):
    """Load the MVTec ground-truth mask for a test image, resized to (size, size)."""
    from PIL import Image

    if defect == "good":
        return np.zeros((size, size), dtype=np.uint8)
    name = os.path.splitext(os.path.basename(img_path))[0]
    mp = os.path.join(cat_dir, "ground_truth", defect, name + "_mask.png")
    if not os.path.exists(mp):
        return np.zeros((size, size), dtype=np.uint8)
    m = Image.open(mp).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)


def save_overlay(img_path, amap, out_path, size):
    import cv2
    from PIL import Image

    img = np.array(Image.open(img_path).convert("RGB").resize((size, size)))
    m = amap.detach().cpu().numpy().astype(np.float64)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img + 0.5 * heat).astype(np.uint8)
    panel = np.concatenate([img, heat, overlay], axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(panel).save(out_path)


def build_config(args) -> ANoCoConfig:
    cfg = ANoCoConfig.from_yaml(args.config) if args.config else ANoCoConfig()
    if args.backbone:
        cfg.backbone = args.backbone
    if args.weights:
        cfg.dinov3_weights = args.weights
    if args.repo:
        cfg.dinov3_repo_dir = args.repo
    if args.device:
        cfg.device = args.device
    if args.img_size:
        cfg.img_size = args.img_size
    if args.layers:
        cfg.layer_indices = tuple(int(x) for x in args.layers.split(","))
        cfg.layer_fusion = "concat"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_ROOT)
    ap.add_argument("--categories", nargs="+", default=["bottle", "carpet", "grid"])
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--ref-seed", type=int, default=0)
    ap.add_argument("--max-per-defect", type=int, default=0, help="0 = all test images")
    ap.add_argument("--ref-aug", default="none", choices=["none", "flips", "rot90", "light"],
                    help="deterministic reference-side augmentation to enrich the memory bank")
    ap.add_argument("--coreset", type=int, default=0, help="subsample reference patches to N (0=off)")
    ap.add_argument("--max-refs", type=int, default=0, help="cap number of reference images (0=all)")
    ap.add_argument("--coreset-method", default="random", choices=["random", "greedy"])
    ap.add_argument("--bank", default="", help="load a prebuilt bank file, or a dir of <category>.pt")
    ap.add_argument("--pixel-metrics", action="store_true", help="also compute pixel AUROC/PRO/F1")
    ap.add_argument("--pixel-size", type=int, default=256, help="resolution for pixel metrics")
    ap.add_argument("--config", default="")
    ap.add_argument("--backbone", default="", help="override cfg.backbone (e.g. dinov3_vitb16)")
    ap.add_argument("--weights", default="", help="override cfg.dinov3_weights (local .pth)")
    ap.add_argument("--device", default="")
    ap.add_argument("--img-size", type=int, default=0)
    ap.add_argument("--repo", default="", help="local DINOv3 hub repo dir (else the DINOV3_REPO env var)")
    ap.add_argument("--layers", default="", help="comma-sep layers for multi-layer fusion, e.g. 11,17")
    ap.add_argument("--agg", default="max", help="image aggregation: max | topk_mean | topk_weighted")
    ap.add_argument("--agg-k", type=int, default=5, help="K for topk aggregation")
    ap.add_argument("--fp16", action="store_true", help="fp16 autocast for the backbone forward")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = build_config(args)
    layers_str = ",".join(map(str, cfg.layer_indices)) if cfg.layer_indices else str(cfg.layer_index)
    print(f"[config] backbone={cfg.backbone} img_size={cfg.img_size} layers={layers_str} "
          f"agg={args.agg} lam={cfg.lam} metric={cfg.score_metric} device={cfg.device}")
    print(f"[refs] shots={args.shots} (0=all) ref_aug={args.ref_aug} coreset={args.coreset} max_refs={args.max_refs}")

    model = ANoCo(cfg)
    ext = model._get_extractor()  # load backbone once, reuse across all images
    if args.fp16:
        ext.fp16 = True
    print(f"[backbone] loaded {cfg.backbone} with {ext.n_blocks} blocks; "
          f"using layer_index={ext.layer_index}; device={ext.device}")

    results = {}
    for cat in args.categories:
        cat_dir = os.path.join(args.data_root, cat)
        train_good = list_images(os.path.join(cat_dir, "train", "good"))
        if not train_good:
            print(f"[skip] {cat}: no train/good images at {cat_dir}")
            continue
        bank_path = ""
        if args.bank:
            bank_path = os.path.join(args.bank, f"{cat}.pt") if os.path.isdir(args.bank) else args.bank
        if bank_path and os.path.isfile(bank_path):
            refs = []
            f_r = load_bank(bank_path, model.device)
        else:
            if args.shots and args.shots > 0:
                rng = random.Random(args.ref_seed)
                refs = rng.sample(train_good, k=min(args.shots, len(train_good)))
            else:
                refs = list(train_good)                   # many-shot: use all good images
            if args.max_refs and len(refs) > args.max_refs:
                refs = random.Random(args.ref_seed).sample(refs, args.max_refs)
            f_r = build_memory_bank(
                ext, refs, ref_aug=args.ref_aug, coreset=args.coreset,
                coreset_method=args.coreset_method, seed=args.ref_seed,
            ).to(model.device)

        scores, labels = [], []
        amaps_list, masks_list = [], []
        saved = False
        test_dir = os.path.join(cat_dir, "test")
        for defect in sorted(os.listdir(test_dir)):
            imgs = list_images(os.path.join(test_dir, defect))
            if args.max_per_defect > 0:
                imgs = imgs[: args.max_per_defect]
            for p in imgs:
                f_q, grid = ext.extract(p)
                out = model.score_features(
                    f_q, f_r, grid_hw=grid, out_hw=(cfg.img_size, cfg.img_size)
                )
                score = image_score_aggregated(
                    out["patch_energy"], method=args.agg, grid_hw=grid, k=args.agg_k
                )
                scores.append(float(score))
                labels.append(0 if defect == "good" else 1)
                if args.pixel_metrics:
                    ps = args.pixel_size
                    am = F.interpolate(out["anomaly_map"][None, None], size=(ps, ps),
                                       mode="bilinear", align_corners=False)[0, 0]
                    amaps_list.append(am.cpu().numpy())
                    masks_list.append(load_mask(cat_dir, defect, p, ps))
                if defect != "good" and not saved:
                    save_overlay(
                        p, out["anomaly_map"],
                        os.path.join(args.out_dir, f"{cat}_{defect}.png"), cfg.img_size,
                    )
                    saved = True

        a = 100.0 * auroc(scores, labels)
        ref = TABLE_S6_IMG_AUROC.get(cat)
        ref_s = f"{ref:.1f}" if ref is not None else "  -  "
        results[cat] = a
        print(f"[{cat:>10}] refs={len(refs)} bank={f_r.shape[0]}p test={len(scores)}  "
              f"image-AUROC={a:5.1f}  (paper S6 1-shot: {ref_s})")
        if args.pixel_metrics and masks_list:
            pm = pixel_metrics(np.stack(amaps_list), np.stack(masks_list))
            imf = image_metrics(scores, labels)
            print(f"             image-F1={100 * imf['f1_max']:.1f}  "
                  f"pixel-AUROC={100 * pm['auroc']:.1f}  pixel-PRO={100 * pm['pro']:.1f}  "
                  f"pixel-F1={100 * pm['f1_max']:.1f}")

    if results:
        mean = sum(results.values()) / len(results)
        print(f"\nMean image-AUROC over {len(results)} categories: {mean:.1f}")
        print(f"Overlays saved under: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
