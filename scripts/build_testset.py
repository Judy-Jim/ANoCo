"""Build a real production-style test set from the engineering OK/NG folders.

Splits OK into: bank (memory bank), calib (threshold calibration), test; NG all go to test.
Copies test images into one folder with ok__/ng__ prefixes (source folders reuse the same
0.jpg,1.jpg names, so prefixing is required to avoid collisions) and writes a manifest +
the bank/calib lists so the split is reproducible and provably leak-free.

Example:
    python scripts/build_testset.py --ok-dir path/to/ok --ng-dir path/to/ng \
        --out-dir data/testset --bank-size 100 --calib-size 100 --seed 0
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok-dir", required=True, help="folder of OK (normal) images")
    ap.add_argument("--ng-dir", required=True, help="folder of NG (defect) images")
    ap.add_argument("--out-dir", default="data/testset", help="output dir for the split")
    ap.add_argument("--bank-size", type=int, default=100)
    ap.add_argument("--calib-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ok = [p for p in list_images(args.ok_dir) if "ng" not in os.path.basename(p).lower()]
    ng = list_images(args.ng_dir)
    if not ok or not ng:
        raise SystemExit(f"OK={len(ok)} NG={len(ng)} - check paths")

    rng = random.Random(args.seed)
    ok_shuf = ok[:]
    rng.shuffle(ok_shuf)                       # independent parts -> random split is safe
    bank = sorted(ok_shuf[: args.bank_size])
    calib = sorted(ok_shuf[args.bank_size: args.bank_size + args.calib_size])
    test_ok = sorted(ok_shuf[args.bank_size + args.calib_size:])

    os.makedirs(args.out_dir, exist_ok=True)
    # sanity: no overlap
    assert not (set(bank) & set(calib) & set(test_ok))

    def _copy(src_list, prefix, label, rows):
        for src in src_list:
            stem, ext = os.path.splitext(os.path.basename(src))
            name = f"{prefix}{stem}{ext}"
            shutil.copy2(src, os.path.join(args.out_dir, name))
            rows.append({"filename": name, "label": label, "split": "test", "orig_path": src})

    rows = []
    _copy(test_ok, "ok__", 0, rows)
    _copy(ng, "ng__", 1, rows)

    with open(os.path.join(args.out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "label", "split", "orig_path"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "bank_list.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(bank))
    with open(os.path.join(args.out_dir, "calib_list.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(calib))
    meta = {
        "ok_total": len(ok), "ng_total": len(ng), "seed": args.seed,
        "bank": len(bank), "calib": len(calib), "test_ok": len(test_ok), "test_ng": len(ng),
        "ok_dir": args.ok_dir, "ng_dir": args.ng_dir,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"[build] OK_total={len(ok)} NG_total={len(ng)}")
    print(f"[split] bank={len(bank)}  calib={len(calib)}  test_ok={len(test_ok)}  test_ng={len(ng)}")
    print(f"[out  ] {args.out_dir}  (images + manifest.csv + bank_list.txt + calib_list.txt + meta.json)")


if __name__ == "__main__":
    main()
