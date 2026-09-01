#!/usr/bin/env python3
"""
Channel Attention Visualization for AD-ViT / CvT models.

Generates a figure showing:
- Input images (4 rows)
- Sorted C×C attention matrices (spectral clustering, k=8)
- Top-4 co-attending channel pair activation heatmaps per image

Output: figs/channel_attn_vis.pdf and .png

Usage:
  python visualize_channel_attn.py \\
    --checkpoint path/to/advit_s.pth \\
    --attn_module_name stage2.blocks.4.attn \\
    --images img1.jpg img2.jpg img3.jpg img4.jpg

For cls_cvt:
  python visualize_channel_attn.py \\
    --model-type cls_cvt \\
    --config configs/advit_minimal.yaml \\
    --checkpoint output/.../checkpoint-best.pth \\
    --attn_module_name stage2.blocks.4.attn \\
    --images img1.jpg img2.jpg img3.jpg img4.jpg

For mod_cvt (same as ADE20K segmentation backbone):
  python visualize_channel_attn.py \\
    --model-type mod_cvt \\
    --checkpoint ../ade20k-segmentation/weights/model_best.pth \\
    --attn_module_name stage2.blocks.4.attn \\
    --images img1.jpg img2.jpg img3.jpg img4.jpg

Use --attn-mode logits (default) to show pre-softmax structure; softmax collapses to near-identity.
Use --attn-mode row_norm to stretch each row's contrast. Top pairs exclude diagonal (i==j).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import SpectralClustering
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
# ImageNet normalization
# -----------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Standard preprocessing for 224×224
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])


# -----------------------------------------------------------------------------
# Hook to capture attention matrix A and value map V from channel-attention block
# -----------------------------------------------------------------------------
def _recompute_attn_and_v(module, x, h, w):
    """
    Recompute pre-softmax logits L = Q K^T / sqrt(d), post-softmax A, and V from the attention module.
    Used when the module uses F.scaled_dot_product_attention (which doesn't expose A).
    Assumes cls_cvt-style Attention: forward_conv, proj_q/k/v (Identity for dw), LN+bias for channel.
    """
    try:
        from einops import rearrange
    except ImportError:
        raise ImportError("einops required for attention capture. pip install einops")

    with torch.no_grad():
        q, k, v = module.forward_conv(x, h, w)  # [B, T, C] or [B, C, H*W] depending on token_mode

        # Channel mode: apply LN + bias
        if getattr(module, 'token_mode', 'channel') == 'channel':
            q = F.layer_norm(q, (q.shape[-1],))
            k = F.layer_norm(k, (k.shape[-1],))
            q = q + module.channel_bias_scale * module.channel_bias
            k = k + module.channel_bias_scale * module.channel_bias

        # Project and split heads
        q = rearrange(module.proj_q(q), 'b t (h d) -> b h t d', h=module.num_heads)
        k = rearrange(module.proj_k(k), 'b t (h d) -> b h t d', h=module.num_heads)
        v = rearrange(module.proj_v(v), 'b t (h d) -> b h t d', h=module.num_heads)

        head_dim = q.shape[-1]
        scale = head_dim ** -0.5
        logits = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T] pre-softmax
        attn = F.softmax(logits, dim=-1)

        # For channel attention: T=C
        L = logits.mean(dim=0).mean(dim=0).cpu().numpy()  # [C, C] pre-softmax
        A = attn.mean(dim=0).mean(dim=0).cpu().numpy()    # [C, C] post-softmax

        # V: [B, H, C, D] -> [C, H*W]
        V = v.mean(dim=0).mean(dim=0).cpu().numpy()

        module._captured_L = L
        module._captured_A = A
        module._captured_V = V


def _make_capture_hook():
    """Return a forward_pre_hook that captures A and V before the attention forward runs."""
    def hook(module, args):
        x, h, w = args[0], args[1], args[2]
        _recompute_attn_and_v(module, x, int(h.item()) if torch.is_tensor(h) else h,
                              int(w.item()) if torch.is_tensor(w) else w)
        return args
    return hook


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model_cls_cvt(config_path, checkpoint_path, num_classes=1000):
    """Build cls_cvt from YAML config and load checkpoint."""
    import yaml
    from models.cls_cvt import ConvolutionalVisionTransformer, LayerNorm, QuickGELU

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    spec = cfg['MODEL']['SPEC']
    num_classes = cfg['MODEL'].get('NUM_CLASSES', num_classes)

    model = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=num_classes,
        act_layer=QuickGELU,
        norm_layer=LayerNorm,
        init=spec.get('INIT', 'trunc_norm'),
        spec=spec,
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    return model


def load_model_advit(checkpoint_path, num_classes=1000):
    """Load ADViT_S from models.advit."""
    from models.advit import ADViT_S
    model = ADViT_S(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    return model


def load_model_mod_cvt(checkpoint_path):
    """Load CvT13Encoder (mod_cvt) with ADE20K pretrained weights (model_best.pth)."""
    from ad_vit.cvt13_encoder import CvT13Encoder
    model = CvT13Encoder(in_chans=3, pretrained_path=checkpoint_path)
    return model


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def row_minmax_norm(A):
    """Row-wise min-max normalization: stretch each row to [0,1] to reveal off-diagonal structure."""
    out = np.zeros_like(A, dtype=np.float64)
    for i in range(A.shape[0]):
        row = A[i].astype(np.float64)
        rmin, rmax = row.min(), row.max()
        if rmax - rmin > 1e-12:
            out[i] = (row - rmin) / (rmax - rmin)
        else:
            out[i] = row
    return out.astype(np.float32)


def sort_attn_by_clusters(M, n_clusters=8):
    """Apply spectral clustering and sort rows/columns by cluster label.
    Returns (M_sorted, order, boundaries). M can be logits, softmax, or row_norm.
    """
    n = M.shape[0]
    if n < n_clusters:
        n_clusters = max(2, n // 2)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    # Non-negative symmetric affinity: for logits shift to positive; for softmax/row_norm use abs
    if M.min() < 0:
        aff = (M - M.min()) + 1e-8
    else:
        aff = np.abs(M) + 1e-8
    aff = (aff + aff.T) / 2
    labels = clustering.fit_predict(aff)
    order = np.argsort(labels)
    sorted_labels = labels[order]
    boundaries = [0] + [i + 1 for i in range(n - 1) if sorted_labels[i] != sorted_labels[i + 1]] + [n]
    return M[np.ix_(order, order)], order, boundaries


def top_k_off_diagonal_pairs(A, k=4):
    """Find k highest off-diagonal entries (excludes diagonal i==j). Returns list of (i, j, value)."""
    n = A.shape[0]
    A_off = A.copy()
    np.fill_diagonal(A_off, -np.inf)  # exclude diagonal from selection
    flat = A_off.flatten()
    idx = np.argsort(-flat)[:k]
    pairs = []
    seen = set()
    for i in idx:
        row, col = np.unravel_index(i, A.shape)
        if row != col and (row, col) not in seen and (col, row) not in seen:
            if A[row, col] > -1e9:  # skip -inf
                pairs.append((row, col, A[row, col]))
                seen.add((row, col))
                seen.add((col, row))
        if len(pairs) >= k:
            break
    return pairs[:k]


def heatmap_overlay(img_rgb, heatmap_2d, alpha=0.5, cmap='jet'):
    """Overlay heatmap on image. img_rgb: [H,W,3] uint8, heatmap_2d: [H,W] float."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Normalize heatmap to [0,1]
    h = heatmap_2d.astype(np.float32)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)

    cm = plt.get_cmap(cmap)
    h_color = cm(h)[:, :, :3]  # RGB
    h_color = (h_color * 255).astype(np.uint8)

    overlay = (alpha * h_color + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay


def tensor_to_vis(img_tensor):
    """Convert normalized tensor [3,H,W] to [H,W,3] uint8 for display."""
    x = img_tensor.cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    default_mod_cvt_ckpt = os.path.join(SCRIPT_DIR, '..', 'ade20k-segmentation', 'weights', 'model_best.pth')
    ap = argparse.ArgumentParser(description='Channel attention visualization for AD-ViT')
    ap.add_argument('--checkpoint', default=None,
                    help='Path to model checkpoint (default for mod_cvt: ade20k-segmentation/weights/model_best.pth)')
    ap.add_argument('--attn_module_name', required=True,
                    help='Module path for CA block, e.g. stage2.blocks.4.attn')
    ap.add_argument('--images', nargs=4, required=True,
                    help='Four image paths (e.g. cat, dog, car, bird)')
    ap.add_argument('--output', default='figs/channel_attn_vis',
                    help='Output path without extension (default: figs/channel_attn_vis)')
    ap.add_argument('--model-type', default='mod_cvt', choices=['cls_cvt', 'advit', 'mod_cvt'],
                    help='Model type: mod_cvt (ADE20K backbone), cls_cvt, or advit')
    ap.add_argument('--config', default=None,
                    help='YAML config for cls_cvt (required if model-type=cls_cvt)')
    ap.add_argument('--num-classes', type=int, default=1000,
                    help='Number of classes (1000 for ImageNet)')
    ap.add_argument('--n-clusters', type=int, default=8,
                    help='Spectral clustering k (default: 8)')
    ap.add_argument('--attn-mode', default='logits', choices=['logits', 'softmax', 'row_norm'],
                    help='Matrix to visualize: logits (pre-softmax, shows structure), softmax (post-softmax), row_norm (row min-max on softmax)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.model_type == 'mod_cvt':
        args.checkpoint = args.checkpoint or default_mod_cvt_ckpt
    if args.checkpoint is None:
        ap.error('--checkpoint required (or use --model-type mod_cvt for default)')
    if args.model_type == 'cls_cvt' and not args.config:
        ap.error('--config required when --model-type=cls_cvt')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load model
    print('Loading model...')
    if args.model_type == 'mod_cvt':
        model = load_model_mod_cvt(args.checkpoint)
    elif args.model_type == 'cls_cvt':
        model = load_model_cls_cvt(args.config, args.checkpoint, args.num_classes)
    else:
        model = load_model_advit(args.checkpoint, args.num_classes)

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

    # Collect L (logits), A (softmax), and V per image
    print('Running forward passes...')
    results = []
    with torch.no_grad():
        for img_t, img_vis in images:
            _ = model(img_t)
            L = module._captured_L
            A = module._captured_A
            V = module._captured_V
            results.append((img_vis, L, A, V))

    hook_handle.remove()

    # Build figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_rows = 4
    n_cols = 6  # input, attn, 4 heatmaps
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for row, (img_vis, L, A, V) in enumerate(results):
        # Select matrix to visualize based on --attn-mode
        if args.attn_mode == 'logits':
            M = L
            title_suffix = ' (logits)'
        elif args.attn_mode == 'row_norm':
            M = row_minmax_norm(A)
            title_suffix = ' (row-norm)'
        else:
            M = A
            title_suffix = ' (softmax)'

        # Column 1: input image
        axes[row, 0].imshow(img_vis)
        axes[row, 0].set_axis_off()
        if row == 0:
            axes[row, 0].set_title('Input', fontsize=10)

        # Column 2: sorted attention matrix (logits / softmax / row_norm)
        M_sorted, order, boundaries = sort_attn_by_clusters(M, n_clusters=args.n_clusters)
        im = axes[row, 1].imshow(M_sorted, cmap='viridis', aspect='equal')
        axes[row, 1].set_axis_off()
        if row == 0:
            axes[row, 1].set_title(f'A{title_suffix}', fontsize=10)

        # Add thin white lines between clusters
        n = M_sorted.shape[0]
        for pos in boundaries[1:-1]:
            axes[row, 1].axhline(pos - 0.5, color='white', lw=0.5)
            axes[row, 1].axvline(pos - 0.5, color='white', lw=0.5)
        axes[row, 1].axhline(n - 0.5, color='white', lw=0.5)
        axes[row, 1].axvline(n - 0.5, color='white', lw=0.5)

        # Top-4 co-attending pairs (off-diagonal only, indices in sorted matrix)
        pairs = top_k_off_diagonal_pairs(M_sorted, k=4)
        h_map, w_map = int(np.sqrt(V.shape[1])), int(np.sqrt(V.shape[1]))

        for col in range(4):
            if col < len(pairs):
                i, j, _ = pairs[col]
                i_orig, j_orig = order[i], order[j]
                if i_orig >= V.shape[0] or j_orig >= V.shape[0]:
                    axes[row, 2 + col].imshow(img_vis)
                else:
                    heat = (V[i_orig] * V[j_orig]).reshape(h_map, w_map)
                    heat = np.clip(heat, 0, None)
                    heat_224 = np.array(
                        Image.fromarray(heat.astype(np.float32)).resize(
                            (224, 224), Image.BICUBIC))
                    overlay = heatmap_overlay(img_vis, heat_224, alpha=0.5, cmap='jet')
                    axes[row, 2 + col].imshow(overlay)
                axes[row, 2 + col].set_title(f'Ch {i}↔{j}', fontsize=10)
            else:
                axes[row, 2 + col].imshow(img_vis)
                axes[row, 2 + col].set_title('—', fontsize=10)
            axes[row, 2 + col].set_axis_off()

    plt.tight_layout()
    pdf_path = args.output + '.pdf'
    png_path = args.output + '.png'
    plt.savefig(pdf_path, bbox_inches='tight', dpi=150)
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved: {pdf_path}')
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    main()
