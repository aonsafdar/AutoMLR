#!/usr/bin/env python3
"""
Channel Attention Visualization v2: heatmaps only (no attention matrix, no spectral clustering).

For each of 4 images, shows one row with 6 columns:
  Col 1: Input image
  Cols 2-5: Top-4 co-attending channel pair heatmaps overlaid on input
  Col 6: Union overlay (sum of 4 heatmaps, normalized, overlaid on input)

Output: figs/channel_attn_vis_v2.pdf and figs/channel_attn_vis_v2.png

Usage:
  python visualize_channel_attn_v2.py \\
    --model-type mod_cvt \\
    --attn_module_name stage2.blocks.4.attn \\
    --images img1.jpg img2.jpg img3.jpg img4.jpg

Dependencies: torch, torchvision, matplotlib, numpy, PIL, sklearn. No others.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add repo to path for model imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# Add ade20k-segmentation for mod_cvt (CvT13Encoder)
ADE20K_DIR = os.path.join(SCRIPT_DIR, '..', 'ade20k-segmentation')
if os.path.isdir(ADE20K_DIR) and ADE20K_DIR not in sys.path:
    sys.path.insert(0, ADE20K_DIR)


# -----------------------------------------------------------------------------
# ImageNet preprocessing
# -----------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])


# -----------------------------------------------------------------------------
# Hook: capture A (post-softmax [C,C]) and V ([C,H*W]) from SHSA block
# -----------------------------------------------------------------------------
def _recompute_attn_and_v(module, x, h, w):
    """Recompute A and V from the attention module (SDPA doesn't expose them)."""
    with torch.no_grad():
        # Get Q, K, V from forward_conv
        q, k, v = module.forward_conv(x, h, w)

        # Channel mode: LN + bias
        if getattr(module, 'token_mode', 'channel') == 'channel':
            q = F.layer_norm(q, (q.shape[-1],))
            k = F.layer_norm(k, (k.shape[-1],))
            q = q + module.channel_bias_scale * module.channel_bias
            k = k + module.channel_bias_scale * module.channel_bias

        # Project and split heads: [B, T, feat_dim] -> [B, num_heads, T, head_dim]
        nh = module.num_heads
        q = module.proj_q(q)
        k = module.proj_k(k)
        v = module.proj_v(v)
        B, T, feat_dim = q.shape
        hd = feat_dim // nh
        q = q.view(B, T, nh, hd).permute(0, 2, 1, 3)
        k = k.view(B, T, nh, hd).permute(0, 2, 1, 3)
        v = v.view(B, T, nh, hd).permute(0, 2, 1, 3)

        head_dim = q.shape[-1]
        scale = head_dim ** -0.5
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)

        # A: [C, C], V: [C, H*W]
        A = attn.mean(dim=0).mean(dim=0).cpu().numpy()
        V = v.mean(dim=0).mean(dim=0).cpu().numpy()

        module._captured_A = A
        module._captured_V = V


def _make_capture_hook():
    def hook(module, args):
        x, h, w = args[0], args[1], args[2]
        _recompute_attn_and_v(module, x, int(h.item()) if torch.is_tensor(h) else h,
                              int(w.item()) if torch.is_tensor(w) else w)
        return args
    return hook


# -----------------------------------------------------------------------------
# Top-k off-diagonal pairs: global max across entire matrix (i < j)
# -----------------------------------------------------------------------------
def top_k_off_diagonal_pairs(A_off, k=4):
    """Find global top-k off-diagonal entries with i < j. A_off must have diagonal already zeroed.
    Returns [(i, j, value), ...]. Uses raw attention values (not row-norm) to avoid bias toward low row indices."""
    n = A_off.shape[0]
    candidates = [(A_off[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
    candidates.sort(key=lambda x: -x[0])
    pairs = [(i, j, v) for v, i, j in candidates[:k]]
    return pairs


# -----------------------------------------------------------------------------
# Heatmap: V[i]*V[j], reshape, upsample to 224, normalize to [0,1]
# -----------------------------------------------------------------------------
def compute_heatmap(V, i, j, h_map, w_map, target_size=224):
    """Compute heatmap for channel pair (i, j), upsample to target_size."""
    heat = (V[i] * V[j]).astype(np.float32)
    heat = np.maximum(heat, 0)
    heat = heat.reshape(h_map, w_map)
    heat_pil = Image.fromarray(heat, mode='F')
    heat_224 = np.array(heat_pil.resize((target_size, target_size), Image.BICUBIC))
    hmin, hmax = heat_224.min(), heat_224.max()
    if hmax - hmin > 1e-12:
        heat_224 = (heat_224 - hmin) / (hmax - hmin)
    else:
        heat_224 = np.zeros_like(heat_224)
    return heat_224.astype(np.float32)


# -----------------------------------------------------------------------------
# Overlay heatmap on image with jet colormap at 60% opacity
# -----------------------------------------------------------------------------
def heatmap_overlay(img_rgb, heatmap_2d, alpha=0.6, cmap='jet'):
    """Overlay heatmap on image. img_rgb: [H,W,3] uint8, heatmap_2d: [H,W] float [0,1]."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h = np.clip(heatmap_2d.astype(np.float32), 0, 1)
    cm = plt.get_cmap(cmap)
    h_color = cm(h)[:, :, :3]
    h_color = (h_color * 255).astype(np.uint8)
    overlay = (alpha * h_color + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay


def tensor_to_vis(img_tensor):
    """Convert normalized tensor [3,H,W] to [H,W,3] uint8."""
    x = img_tensor.cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model_mod_cvt(checkpoint_path):
    """Load CvT13Encoder (mod_cvt) with pretrained weights."""
    from ad_vit.cvt13_encoder import CvT13Encoder
    return CvT13Encoder(in_chans=3, pretrained_path=checkpoint_path)


def load_model_cls_cvt(config_path, checkpoint_path, num_classes=1000):
    import yaml
    from models.cls_cvt import ConvolutionalVisionTransformer, LayerNorm, QuickGELU
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    spec = cfg['MODEL']['SPEC']
    num_classes = cfg['MODEL'].get('NUM_CLASSES', num_classes)
    model = ConvolutionalVisionTransformer(
        in_chans=3, num_classes=num_classes,
        act_layer=QuickGELU, norm_layer=LayerNorm,
        init=spec.get('INIT', 'trunc_norm'), spec=spec,
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    return model


def load_model_advit(checkpoint_path, num_classes=1000):
    from models.advit import ADViT_S
    model = ADViT_S(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    default_mod_cvt_ckpt = os.path.join(SCRIPT_DIR, '..', 'ade20k-segmentation', 'weights', 'model_best.pth')
    ap = argparse.ArgumentParser(description='Channel attention heatmap visualization v2')
    ap.add_argument('--checkpoint', default=None, help='Path to checkpoint')
    ap.add_argument('--attn_module_name', default='stage2.blocks.4.attn',
                    help='SHSA block path: stage2=final stage (384ch), blocks.4=mid-depth CA block (good semantic level)')
    ap.add_argument('--images', nargs=4, required=True, help='Four image paths')
    ap.add_argument('--output', default='figs/channel_attn_vis_v2',
                    help='Output path without extension')
    ap.add_argument('--model-type', default='mod_cvt', choices=['cls_cvt', 'advit', 'mod_cvt'])
    ap.add_argument('--config', default=None, help='YAML config for cls_cvt')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.model_type == 'mod_cvt':
        args.checkpoint = args.checkpoint or default_mod_cvt_ckpt
    if args.checkpoint is None:
        ap.error('--checkpoint required')
    if args.model_type == 'cls_cvt' and not args.config:
        ap.error('--config required when --model-type=cls_cvt')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load model
    print('Loading model...')
    if args.model_type == 'mod_cvt':
        model = load_model_mod_cvt(args.checkpoint)
    elif args.model_type == 'cls_cvt':
        model = load_model_cls_cvt(args.config, args.checkpoint)
    else:
        model = load_model_advit(args.checkpoint)
    model = model.to(args.device)
    model.eval()

    # Get attention module and register hook
    parts = args.attn_module_name.split('.')
    module = model
    for p in parts:
        module = getattr(module, p)
    hook_handle = module.register_forward_pre_hook(_make_capture_hook())

    # Load and preprocess images
    print('Loading images...')
    images = []
    for p in args.images:
        img = Image.open(p).convert('RGB')
        img_t = PREPROCESS(img).unsqueeze(0).to(args.device)
        images.append((img_t, tensor_to_vis(img_t[0])))

    # Run forward passes and collect A, V per image
    print('Running forward passes...')
    results = []
    with torch.no_grad():
        for img_t, img_vis in images:
            _ = model(img_t)
            A = module._captured_A
            V = module._captured_V
            results.append((img_vis, A, V))

    hook_handle.remove()

    # Build figure: 4 rows x 6 columns
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_rows, n_cols = 4, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    for row, (img_vis, A, V) in enumerate(results):
        # Use raw softmax A for pair selection (row-norm would make every row max=1, biasing to low row indices)
        A_off = A.copy()
        np.fill_diagonal(A_off, 0)
        pairs = top_k_off_diagonal_pairs(A_off, k=4)

        h_map = w_map = int(np.sqrt(V.shape[1]))

        # Column 1: Input
        axes[row, 0].imshow(img_vis)
        axes[row, 0].set_title('Input', fontsize=10)
        axes[row, 0].set_axis_off()

        # Columns 2-5: Top-4 heatmaps
        heatmaps = []
        for col in range(4):
            if col < len(pairs):
                i, j, _ = pairs[col]
                heat = compute_heatmap(V, i, j, h_map, w_map)
                heatmaps.append(heat)
                overlay = heatmap_overlay(img_vis, heat, alpha=0.6)
                axes[row, 1 + col].imshow(overlay)
                axes[row, 1 + col].set_title(f'Ch {i}↔{j}', fontsize=10)
            else:
                axes[row, 1 + col].imshow(img_vis)
                axes[row, 1 + col].set_title('—', fontsize=10)
            axes[row, 1 + col].set_axis_off()

        # Column 6: Union overlay (sum of 4 heatmaps, normalized to [0,1])
        if len(heatmaps) >= 1:
            union = sum(heatmaps)
            umin, umax = union.min(), union.max()
            if umax - umin > 1e-12:
                union = (union - umin) / (umax - umin)
            else:
                union = np.zeros_like(union)
            overlay_union = heatmap_overlay(img_vis, union.astype(np.float32), alpha=0.6)
            axes[row, 5].imshow(overlay_union)
        else:
            axes[row, 5].imshow(img_vis)
        axes[row, 5].set_title('Union', fontsize=10)
        axes[row, 5].set_axis_off()

    # Equal aspect, no ticks, tight layout, thin white borders
    for ax in axes.flat:
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05, wspace=0.02, hspace=0.08)

    # Add 1px white border between panels via spines
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('white')
            spine.set_linewidth(1)

    pdf_path = args.output + '.pdf'
    png_path = args.output + '.png'
    plt.savefig(pdf_path, bbox_inches='tight', dpi=150)
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved: {pdf_path}')
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    main()
