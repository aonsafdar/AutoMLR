#!/usr/bin/env python3
"""
Compute MACs / FLOPs and parameter count for CvT models in the
Efficient_ViT_with_DW repo using fvcore (FlopCountAnalysis).

- Convention: 1 MAC = 1 FLOP (fvcore; matches CvT, ViT, DeiT, Swin).
- The script reads the same DW yaml configs you use for training
  (expects MODEL.SPEC for cls_cvt).

Example:

  python compute_flops_ptflops.py \
      --repo /path/to/repo \
      --yaml configs/advit_minimal.yaml
"""

import argparse
import os
import sys
import yaml

import torch
from fvcore.nn import FlopCountAnalysis


def load_cfg_and_img_size(yaml_path):
    """Minimal yaml loader: returns (cfg_dict, img_h, img_w)."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Image size: prefer DATA.IMG_SIZE (DW harness), then TRAIN/TEST.IMAGE_SIZE.
    if "DATA" in cfg and "IMG_SIZE" in cfg["DATA"]:
        img = cfg["DATA"]["IMG_SIZE"]
        img_h = img_w = int(img) if isinstance(img, (int, float)) else int(img[0])
    elif "TRAIN" in cfg and "IMAGE_SIZE" in cfg["TRAIN"]:
        val = cfg["TRAIN"]["IMAGE_SIZE"]
        img_h, img_w = (val, val) if isinstance(val, int) else (val[0], val[1])
    elif "TEST" in cfg and "IMAGE_SIZE" in cfg["TEST"]:
        val = cfg["TEST"]["IMAGE_SIZE"]
        img_h, img_w = (val, val) if isinstance(val, int) else (val[0], val[1])
    else:
        img_h = img_w = 224

    return cfg, img_h, img_w


def build_cvt_from_yaml(repo_root, cfg):
    """
    Build a ConvolutionalVisionTransformer from a DW yaml config.

    This assumes:
      MODEL.TYPE == 'cls_cvt'
      MODEL.SPEC contains the CvT spec dict used by your DW harness.
    """
    if "MODEL" not in cfg or cfg["MODEL"].get("TYPE", "") != "cls_cvt":
        raise ValueError("This helper currently supports only MODEL.TYPE == 'cls_cvt'.")

    spec = cfg["MODEL"]["SPEC"]
    num_classes = cfg["MODEL"].get("NUM_CLASSES", 1000)
    init = spec.get("INIT", "trunc_norm")

    # Make repo importable
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Import CvT implementation from DW repo
    from models.cls_cvt import (  # type: ignore
        ConvolutionalVisionTransformer,
        LayerNorm,
        QuickGELU,
    )

    model = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=num_classes,
        act_layer=QuickGELU,
        norm_layer=LayerNorm,
        init=init,
        spec=spec,
    )
    return model


def pretty(n: float) -> str:
    if n >= 1e12:
        return f"{n/1e12:.3f}T"
    if n >= 1e9:
        return f"{n/1e9:.3f}G"
    if n >= 1e6:
        return f"{n/1e6:.3f}M"
    if n >= 1e3:
        return f"{n/1e3:.3f}K"
    return str(int(n))


def main():
    ap = argparse.ArgumentParser(
        description="fvcore FLOPs + params for CvT (DW yaml)."
    )
    ap.add_argument(
        "--repo",
        required=True,
        help="Path to Efficient_ViT_with_DW repo root (for importing models).",
    )
    ap.add_argument(
        "--yaml",
        required=True,
        help="Path to DW yaml config (expects MODEL.TYPE=cls_cvt & MODEL.SPEC).",
    )
    ap.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Input channels (default: 3 for RGB).",
    )
    args = ap.parse_args()

    cfg, img_h, img_w = load_cfg_and_img_size(args.yaml)
    model = build_cvt_from_yaml(args.repo, cfg)

    model.eval()
    device = torch.device("cpu")
    model.to(device)

    print(f"Model.TYPE  : {cfg['MODEL'].get('TYPE')}")
    print(f"Model.NAME  : {cfg['MODEL'].get('NAME')}")
    print(f"Input shape : ({args.channels}, {img_h}, {img_w})")
    print("Running fvcore FlopCountAnalysis...\n")

    dummy = torch.randn(1, args.channels, img_h, img_w)
    flop_counter = FlopCountAnalysis(model, dummy)
    flops = flop_counter.total()
    params = sum(p.numel() for p in model.parameters())

    print("===== fvcore summary =====")
    print(f"Params : {pretty(params)} ({params:.3e})")
    print(f"FLOPs  : {pretty(flops)} ({flops:.3e})  # 1 MAC = 1 FLOP (fvcore convention)")
    print(f"GFLOPs : {flops / 1e9:.3f}")
    print("==========================")


if __name__ == "__main__":
    main()