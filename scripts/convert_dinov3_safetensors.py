"""Convert a Hugging Face DINOv3 *.safetensors checkpoint to a facebookresearch-DINOv3
state_dict *.pt that ANoCo's DINOv3Extractor can load with strict=True.

Hugging Face (`transformers`) and facebookresearch/dinov3 use different state_dict key
names. This script detects the source layout and, if it is HF-style, remaps keys to the
fb layout (fusing q/k/v, renaming layer_scale/mlp/patch_embeddings, etc.), then validates
by loading into the freshly-built fb backbone.

Usage:
    # 1) inspect keys of both sides (no writing):
    python scripts/convert_dinov3_safetensors.py --src model.safetensors --backbone dinov3_vitb16 --inspect
    # 2) convert + validate + save:
    python scripts/convert_dinov3_safetensors.py --src model.safetensors --backbone dinov3_vitb16 --out weights/dinov3_vitb16.pt
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from anoco.config import DEFAULT_DINOV3_REPO


def load_any(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    return obj


def build_reference(backbone, repo=DEFAULT_DINOV3_REPO):
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    import dinov3.hub.backbones as B  # noqa: N812

    builder = getattr(B, backbone)
    model = builder(pretrained=False)
    return model, model.state_dict()


def strip_prefix(sd):
    """Drop a common leading prefix like 'backbone.' or 'model.' if present on all keys."""
    for pref in ("backbone.", "model.", "module."):
        if sd and all(k.startswith(pref) for k in sd):
            return {k[len(pref):]: v for k, v in sd.items()}
    return sd


def looks_like_hf(sd):
    keys = list(sd.keys())
    return any("embeddings.patch_embeddings" in k or k.startswith("layer.") or ".attention." in k
               for k in keys)


def inspect(src_sd, ref_sd):
    def summarize(name, sd):
        print(f"\n=== {name}: {len(sd)} tensors ===")
        for k in list(sd.keys())[:32]:
            print(f"  {k:60s} {tuple(sd[k].shape)}")
    summarize("SOURCE", src_sd)
    summarize("REFERENCE (fb)", ref_sd)
    common = set(src_sd) & set(ref_sd)
    shape_ok = sum(1 for k in common if tuple(src_sd[k].shape) == tuple(ref_sd[k].shape))
    print(f"\nname-overlap: {len(common)}/{len(ref_sd)} ; shape-matching among overlap: {shape_ok}")
    print(f"source looks HF-style: {looks_like_hf(src_sd)}")


def convert_hf_to_fb(src, ref):
    """Best-effort HF -> fb remap. Uses reference shapes to fuse qkv correctly."""
    out = {}

    def put(k, v):
        out[k] = v

    # top-level tokens / patch embed / final norm
    direct = {
        "embeddings.cls_token": "cls_token",
        "embeddings.register_tokens": "storage_tokens",
        "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
        "embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
        "layernorm.weight": "norm.weight",
        "layernorm.bias": "norm.bias",
    }
    for hf, fb in direct.items():
        if hf in src:
            put(fb, src[hf])
    # mask_token: HF stores (1, 1, D); fb expects (1, D).
    if "embeddings.mask_token" in src:
        mt = src["embeddings.mask_token"]
        target = tuple(ref["mask_token"].shape) if "mask_token" in ref else (1, mt.shape[-1])
        put("mask_token", mt.reshape(*target))

    layer_re = re.compile(r"^layer\.(\d+)\.(.*)$")
    per_layer = {}
    for k, v in src.items():
        m = layer_re.match(k)
        if m:
            per_layer.setdefault(int(m.group(1)), {})[m.group(2)] = v

    for i, sub in per_layer.items():
        p = f"blocks.{i}."

        def g(*names):
            for n in names:
                if n in sub:
                    return sub[n]
            return None

        # norms
        for a, b in (("norm1", "norm1"), ("norm2", "norm2")):
            if f"{a}.weight" in sub:
                put(p + f"{b}.weight", sub[f"{a}.weight"])
                put(p + f"{b}.bias", sub[f"{a}.bias"])

        # attention: fuse q/k/v -> qkv (fb order [q; k; v] along out-dim); o_proj -> proj.
        q_w = g("attention.q_proj.weight", "attention.query.weight")
        k_w = g("attention.k_proj.weight", "attention.key.weight")
        v_w = g("attention.v_proj.weight", "attention.value.weight")
        qkv_w = None
        if q_w is not None and k_w is not None and v_w is not None:
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        elif g("attention.qkv.weight") is not None:
            qkv_w = g("attention.qkv.weight")
        if qkv_w is not None:
            put(p + "attn.qkv.weight", qkv_w)
            # DINOv3's released fb checkpoints apply NO qkv bias: both qkv.bias and its
            # bias_mask are zero (verified against the official B/16 checkpoint). HF stores
            # nonzero q/v biases that the canonical model masks out, so we zero them here to
            # faithfully reproduce the facebookresearch backbone the paper uses.
            zeros = torch.zeros(qkv_w.shape[0], dtype=qkv_w.dtype)
            put(p + "attn.qkv.bias", zeros.clone())
            put(p + "attn.qkv.bias_mask", zeros.clone())

        o_w = g("attention.o_proj.weight", "attention.output.dense.weight",
                "attention.proj.weight", "attention.out_proj.weight")
        if o_w is not None:
            put(p + "attn.proj.weight", o_w)
            o_b = g("attention.o_proj.bias", "attention.output.dense.bias",
                    "attention.proj.bias", "attention.out_proj.bias")
            if o_b is not None:
                put(p + "attn.proj.bias", o_b)

        # layer scale
        ls1 = g("layer_scale1.lambda1", "lambda1", "ls1.gamma")
        if ls1 is not None:
            put(p + "ls1.gamma", ls1)
        ls2 = g("layer_scale2.lambda1", "lambda2", "ls2.gamma")
        if ls2 is not None:
            put(p + "ls2.gamma", ls2)

        # mlp
        fc1_w = g("mlp.fc1.weight", "mlp.up_proj.weight", "mlp.gate_proj.weight")
        fc2_w = g("mlp.fc2.weight", "mlp.down_proj.weight")
        if fc1_w is not None:
            put(p + "mlp.fc1.weight", fc1_w)
            if g("mlp.fc1.bias", "mlp.up_proj.bias") is not None:
                put(p + "mlp.fc1.bias", g("mlp.fc1.bias", "mlp.up_proj.bias"))
        if fc2_w is not None:
            put(p + "mlp.fc2.weight", fc2_w)
            if g("mlp.fc2.bias", "mlp.down_proj.bias") is not None:
                put(p + "mlp.fc2.bias", g("mlp.fc2.bias", "mlp.down_proj.bias"))

    # keep any keys that already match the reference verbatim (e.g. rope buffers)
    for k, v in src.items():
        if k in ref and k not in out:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="model.safetensors")
    ap.add_argument("--backbone", default="dinov3_vitb16")
    ap.add_argument("--out", default="")
    ap.add_argument("--repo", default=DEFAULT_DINOV3_REPO)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    print(f"[load] {args.src}")
    src_sd = strip_prefix(load_any(args.src))
    print(f"[build] fb reference: {args.backbone}")
    model, ref_sd = build_reference(args.backbone, args.repo)

    if args.inspect:
        inspect(src_sd, ref_sd)
        return

    # Decide conversion path.
    common = set(src_sd) & set(ref_sd)
    shape_ok = sum(1 for k in common if tuple(src_sd[k].shape) == tuple(ref_sd[k].shape))
    if shape_ok >= int(0.95 * len(ref_sd)):
        print(f"[convert] source already fb-style ({shape_ok}/{len(ref_sd)} match); using as-is.")
        converted = {k: src_sd[k] for k in ref_sd if k in src_sd}
    else:
        print("[convert] remapping HF -> fb layout ...")
        converted = convert_hf_to_fb(src_sd, ref_sd)

    missing, unexpected = model.load_state_dict(converted, strict=False)
    ignorable = ("rope", "periods", "bias_mask")  # deterministic buffers set at build time
    missing = [m for m in missing if not any(tok in m for tok in ignorable)]
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}")
        for m in missing[:20]:
            print("   missing   :", m)
        for u in unexpected[:20]:
            print("   unexpected:", u)
        if missing:
            raise SystemExit("[FAIL] conversion incomplete; see missing keys above.")
    print(f"[ok] loaded into {args.backbone} (missing={len(missing)}, unexpected={len(unexpected)})")

    # Released fb checkpoints store RoPE periods in bfloat16 precision; reproduce that so
    # extracted features are bit-faithful to the official facebookresearch model.
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name.endswith("periods"):
                buf.copy_(buf.to(torch.bfloat16).to(buf.dtype))

    out = args.out or f"weights/{args.backbone}.pt"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"[saved] {os.path.abspath(out)}")
    print(f"\nUse it via:")
    print(f"  --backbone {args.backbone} --weights {os.path.abspath(out)}")
    print(f"or set dinov3_weights in configs/anoco_dinov3l.yaml.")


if __name__ == "__main__":
    main()
