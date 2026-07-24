"""Fetch / verify the DINOv3 checkpoint used by ANoCo (default: DINOv3-L/16).

The paper's main backbone is DINOv3-L/16. Its LVD-1689M checkpoint is not in the
local torch-hub cache by default, so this script tries to download it and, on
failure (offline / gated), prints exact manual steps.

Usage:
    python scripts/download_dinov3.py                      # dinov3_vitl16
    python scripts/download_dinov3.py --backbone dinov3_vitb16
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from anoco.features.dinov3 import expected_weight_filename, expected_weight_url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov3_vitl16")
    args = ap.parse_args()

    fname = expected_weight_filename(args.backbone)
    url = expected_weight_url(args.backbone)
    ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    target = os.path.join(ckpt_dir, fname)

    print(f"backbone      : {args.backbone}")
    print(f"expected file : {target}")
    print(f"download url  : {url}")

    if os.path.isfile(target):
        try:
            sd = torch.load(target, map_location="cpu")
            n = len(sd) if hasattr(sd, "__len__") else "?"
            mb = os.path.getsize(target) / 1e6
            print(f"[OK] already present ({mb:.1f} MB, {n} tensors).")
            return 0
        except Exception as exc:
            print(f"[WARN] file present but failed to load ({exc}); re-downloading.")

    print("[..] attempting download via torch.hub ...")
    try:
        torch.hub.load_state_dict_from_url(
            url, model_dir=ckpt_dir, map_location="cpu", progress=True
        )
        print(f"[OK] downloaded to {target}")
        return 0
    except Exception as exc:
        print(f"[FAIL] automatic download failed: {exc}")
        print("\nManual steps:")
        print(f"  1) Download the checkpoint (accept the DINOv3 license if prompted):")
        print(f"       {url}")
        print(f"  2) Place the file at:")
        print(f"       {target}")
        print(f"  3) Re-run this script to verify, or set dinov3_weights to that path")
        print(f"     in configs/anoco_dinov3l.yaml.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
