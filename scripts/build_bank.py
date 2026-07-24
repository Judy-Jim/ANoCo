"""CLI: build a production memory bank from a folder of OK (normal) images.

Produces bank.pt + normalizer.pt + threshold.json + build_report.json.

Usage:
    python scripts/build_bank.py \
        --ok-dir path/to/ok_images \
        --config configs/production.yaml \
        --out-dir banks/my_product
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anoco.bank_builder import BankBuilder


def main():
    ap = argparse.ArgumentParser(description="Build production memory bank")
    ap.add_argument("--ok-dir", required=True, help="folder of OK (normal) images")
    ap.add_argument("--config", default="configs/production.yaml")
    ap.add_argument("--out-dir", default="banks/my_product")
    ap.add_argument("--bank-size", type=int, default=200)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--coreset", type=int, default=80000)
    ap.add_argument("--coreset-method", default="greedy")
    ap.add_argument("--ref-aug", default="none")
    ap.add_argument("--excess-k", type=float, default=3.0)
    ap.add_argument("--target-fpr", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode", default="draft")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()

    builder = BankBuilder()
    report = builder.build(
        ok_dir=args.ok_dir,
        config_path=args.config,
        out_dir=args.out_dir,
        bank_size=args.bank_size,
        n_clusters=args.n_clusters,
        coreset=args.coreset,
        coreset_method=args.coreset_method,
        ref_aug=args.ref_aug,
        excess_k=args.excess_k,
        target_fpr=args.target_fpr,
        seed=args.seed,
        decode=args.decode,
        fp16=args.fp16,
        chunk=args.chunk,
    )
    print(f"\n===== Build Report =====")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
