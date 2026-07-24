"""CLI: run ANoCoInspector on a folder of images.

Usage:
    # Inspect a single image
    python scripts/run_inspector.py \
        --bank-dir banks/my_product \
        --config configs/production.yaml \
        --image path/to/image.jpg

    # Inspect a folder (batch mode)
    python scripts/run_inspector.py \
        --bank-dir banks/my_product \
        --config configs/production.yaml \
        --image-dir path/to/images \
        --save-overlays results/viz/
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anoco.inspector import ANoCoInspector
from anoco.logger import InspectionLogger


def main():
    ap = argparse.ArgumentParser(description="Run ANoCoInspector")
    ap.add_argument("--bank-dir", required=True, help="directory with bank.pt, normalizer.pt, threshold.json")
    ap.add_argument("--config", default="configs/production.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--image", default="", help="single image path")
    ap.add_argument("--image-dir", default="", help="folder of images to inspect")
    ap.add_argument("--log-dir", default="logs", help="CSV log directory")
    ap.add_argument("--save-overlays", default="", help="save overlay images to this dir")
    ap.add_argument("--save-ng-only", action="store_true", help="only save NG overlays")
    args = ap.parse_args()

    # Resolve paths
    bank_path = os.path.join(args.bank_dir, "bank.pt")
    normalizer_path = os.path.join(args.bank_dir, "normalizer.pt")
    threshold_path = os.path.join(args.bank_dir, "threshold.json")

    # Init inspector
    inspector = ANoCoInspector(
        config_path=args.config,
        bank_path=bank_path,
        normalizer_path=normalizer_path,
        threshold_path=threshold_path,
        device=args.device,
        warmup=args.warmup,
    )

    # Health check
    health = inspector.health_check()
    print(f"[health] {health}")

    # Init logger
    logger = InspectionLogger(log_dir=args.log_dir)

    # Collect images
    if args.image:
        images = [args.image]
    elif args.image_dir:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        images = sorted([
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(exts) and os.path.isfile(os.path.join(args.image_dir, f))
        ])
    else:
        print("[error] must specify --image or --image-dir")
        sys.exit(1)

    print(f"[inspect] {len(images)} images")

    # Inspect
    n_ok = n_ng = n_err = 0
    t0 = time.time()
    for i, img_path in enumerate(images):
        result = inspector.inspect(img_path)
        logger.log(result)

        if result.decision == "OK":
            n_ok += 1
        elif result.decision == "NG":
            n_ng += 1
        else:
            n_err += 1

        # Save overlay
        if args.save_overlays:
            if result.decision == "NG" or not args.save_ng_only:
                fname = os.path.basename(img_path)
                out_name = f"{result.decision}_{result.score:.4f}_{fname}"
                out_path = os.path.join(args.save_overlays, out_name)
                inspector.save_overlay(img_path, result, out_path)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(images)}] OK={n_ok} NG={n_ng} ERR={n_err} "
                  f"({(i+1)/(time.time()-t0):.1f} img/s)")

    elapsed = time.time() - t0
    print(f"\n===== Results =====")
    print(f"  Total: {len(images)} | OK: {n_ok} | NG: {n_ng} | ERROR: {n_err}")
    print(f"  Time: {elapsed:.0f}s ({len(images)/elapsed:.1f} img/s)")
    print(f"  Mean latency: {1000*elapsed/len(images):.0f}ms/img")

    # Stats
    stats = logger.get_stats(last_n=len(images))
    if stats.get("count", 0) > 0:
        print(f"  Score: mean={stats['mean_score']:.4f} max={stats['max_score']:.4f}")
        print(f"  NG rate: {stats['ng_rate']:.1%}")

    logger.close()
    print(f"\n[logs] {args.log_dir}/")


if __name__ == "__main__":
    main()
