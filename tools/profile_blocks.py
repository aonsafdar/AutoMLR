#!/usr/bin/env python3
"""
Per-stage / per-block params & MACs profiler for CvT models **inside the Efficient_ViT_with_DW repo** (1 MAC ≈ 2 FLOPs).

Channel Attention (CA) blocks can have a configurable MLP ratio via CA_MLP_RATIO (and no W_o), while tail SA blocks use TAIL_MLP_RATIO.

Key changes vs the older version (for the microsoft CvT tree):
- Accepts `--repo` to point to your Efficient_ViT_with_DW repo root and imports from `models.cls_cvt` there.
- YAML reader matches the DW harness: reads `MODEL.SPEC` and `DATA.IMG_SIZE`.
- Fallbacks if sanity build classes aren’t present (so the analytic math still runs).

Example:
  python tools/profile_blocks.py \
      --repo /path/to/repo \
      --yaml  /path/to/repo/configs/cvt/cls_cvt13_16_224_cifar10.yaml \
      --csv   /path/to/output/cvt13_c10_blocks.csv
"""
import argparse, os, sys, math, yaml, csv
from typing import Tuple, List

# --------------------------- small helpers ---------------------------

def conv_out(h: int, k: int, p: int, s: int, d: int = 1) -> int:
    """PyTorch Conv2d output size (floor)."""
    return int(math.floor((h + 2 * p - d * (k - 1) - 1) / s + 1))

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def pretty(n: float) -> str:
    if n >= 1e9:  return f"{n/1e9:.3f}G"
    if n >= 1e6:  return f"{n/1e6:.3f}M"
    if n >= 1e3:  return f"{n/1e3:.3f}K"
    return str(int(n))

def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if getattr(p, 'requires_grad', True))

# ---- generic conv analytics (supports groups) ----
def conv2d_params_and_macs(Cin:int, Cout:int, k:int, s:int, p:int, d:int, groups:int, H_in:int, W_in:int, batch:int=1):
    H_out = conv_out(H_in, k, p, s, d)
    W_out = conv_out(W_in, k, p, s, d)
    # params: Cout * (Cin/groups) * k*k + optional bias (ignored here for analytics)
    weight_params = Cout * (Cin // groups) * (k * k)
    # MACs per output element = (Cin/groups) * k*k
    macs_per_out = (Cin // groups) * (k * k)
    total_macs = batch * Cout * H_out * W_out * macs_per_out
    return weight_params, total_macs, H_out, W_out

# ---- tokenization (patch-embed) analytics for different methods ----

def token_embed_cost(method: str, Cin: int, Cout: int, k: int, s: int, p: int, H_in: int, W_in: int, batch: int = 1):
    """
    Return (params, macs, H_out, W_out) for patch embedding before a stage.
    Methods:
      - 'std':     single dense conv (groups=1)
      - 'dw_sep':  depthwise kxk (groups=Cin) + pointwise 1x1 (Cin->Cout)
      - 'dw_only': depthwise-only with multiplier (groups=Cin, out=Cout). Requires Cout % Cin == 0.
    """
    H_out = conv_out(H_in, k, p, s, 1)
    W_out = conv_out(W_in, k, p, s, 1)
    method = (method or 'std').lower()

    if method == 'std':
        # params: Cout * Cin * k^2
        params = Cout * Cin * (k * k)
        macs = batch * H_out * W_out * Cout * (Cin * (k * k))
        tag = f"STD"

    elif method == 'dw_sep':
        # depthwise: Cin * k^2;  pointwise: Cin * Cout; BN after each (gamma+beta)
        params_dw = Cin * (k * k) + 2 * Cin
        macs_dw = batch * H_out * W_out * Cin * (k * k)
        params_pw = Cin * Cout + 2 * Cout
        macs_pw = batch * H_out * W_out * Cin * Cout
        params = params_dw + params_pw
        macs = macs_dw + macs_pw
        tag = f"DW+PW"

    elif method == 'dw_only':
        assert Cout % Cin == 0, f"dw_only requires Cout % Cin == 0 (got Cin={Cin}, Cout={Cout})"
        # depthwise-only with multiplier: Cout*k^2 + BN (gamma+beta)
        params = Cout * (k * k) + 2 * Cout
        macs = batch * H_out * W_out * Cout * (k * k)
        m = Cout // Cin
        tag = f"DW_ONLY(m={m})"

    else:
        # Fallback to grouped conv estimator; treat as grouped dense conv
        params, macs, H_out, W_out = conv2d_params_and_macs(
            Cin=Cin, Cout=Cout, k=k, s=s, p=p, d=1, groups=1, H_in=H_in, W_in=W_in, batch=batch)
        tag = f"FALLBACK"

    return params, macs, H_out, W_out, tag

# --------------------------- FLOPs model ---------------------------

def block_budget(stage_H: int, stage_W: int, C: int, mlp_ratio: float,
                 stride_q: int, stride_kv: int, kernel: int,
                 token_mode: str, qkv_method: str = 'dw_bn') -> Tuple[int, int]:
    """
    Analytic per-block estimate for CvT under the DW harness.
    - token_mode == 'channel' → **Channel Attention (CA)** with Linear W_o and optional MLP.
    - token_mode == 'spatial' → **Spatial Attention (SA)** (tail) with standard Linear W_o and MLP.

    qkv_method controls the projection stack before attention for Q, K, V per the stage:
      'dw_bn'     → depthwise kxk + BN on each of q,k,v
      'dw_pw_bn'  → depthwise kxk + pointwise 1x1 + BN on each of q,k,v
      'dw'        → depthwise only (no BN)
      'linear'    → full Linear QKV (no conv projections)
      'none'      → no conv projections (fallback)

    Returns (params, macs) using MACs (multiply-add counted as 1).
    """
    def _has(s: str) -> bool:
        return (qkv_method or '').lower().find(s) >= 0

    # Spatial shapes after strides for q and kv
    Hq, Wq   = ceil_div(stage_H, stride_q), ceil_div(stage_W, stride_q)
    Hkv, Wkv = ceil_div(stage_H, stride_kv), ceil_div(stage_W, stride_kv)

    # --- Attention FLOPs (QK^T + AV) over aggregate head dim ---
    if token_mode == 'channel':   # CA: tokens are channels, feature dim ~ spatial
        Tq = C
        Tkv = C
        d = Hq * Wq
    else:                         # SA: tokens are spatial, feature dim ~ channels
        Tq = Hq * Wq
        Tkv = Hkv * Wkv
        d = C
    attn_flops = (Tq * Tkv * d) + (Tq * d * Tkv)

    # --- Conv Q/K/V projections (depthwise + optional pointwise), plus BN ---
    # For 'linear' (ViT-style): full Linear QKV, no conv projections
    k = kernel
    dw_params = 0
    dw_flops  = 0
    pw_params = 0
    pw_flops  = 0
    bn_params = 0
    linear_qkv_params = 0
    linear_qkv_flops  = 0

    if _has('linear'):
        # Full Linear QKV: 3 * Linear(C, C) with bias
        linear_qkv_params = 3 * (C * C + C)
        T_in = (stage_H * stage_W)
        linear_qkv_flops  = 3 * T_in * C * C
    elif _has('dw'):
        # depthwise weights: Cin=C, three paths q/k/v
        dw_params = 3 * (C * (k * k))
        dw_flops  = (C * (k * k)) * (Hq * Wq)           # q path
        dw_flops += 2 * (C * (k * k)) * (Hkv * Wkv)     # k & v paths
    if _has('pw') and not _has('linear'):
        pw_params = 3 * (C * C)
        pw_flops  = (C * C) * (Hq * Wq) + 2 * (C * C) * (Hkv * Wkv)
    if _has('bn') and not _has('linear'):
        bn_params = 3 * (2 * C)

    # --- Output projection W_o (Linear) + MLP + LayerNorms ---
    # cls_cvt uses Linear(C,C) for W_o in both CA and SA blocks
    proj_params = (C * C) + C  # +bias
    if token_mode == 'channel':
        # CA: after rearrange, tokens=C (channels), feature=C; proj applies C*C per token
        proj_flops  = C * C * C
        hidden      = int(max(0.0, mlp_ratio) * C)
        if hidden > 0:
            mlp_params  = (C * hidden + hidden) + (hidden * C + C)
            mlp_flops   = (C * hidden + hidden * C) * (stage_H * stage_W)
        else:
            mlp_params = 0
            mlp_flops = 0
        ln_params   = 2 * C
    else:
        # SA tail: W_o on spatial tokens
        proj_flops  = (stage_H * stage_W) * C * C
        hidden      = int(max(1, mlp_ratio) * C)
        mlp_params  = (C * hidden + hidden) + (hidden * C + C)
        mlp_flops   = (C * hidden + hidden * C) * (stage_H * stage_W)
        ln_params   = 4 * C  # two LNs (pre/post)

    params = dw_params + pw_params + bn_params + linear_qkv_params + proj_params + mlp_params + ln_params
    flops  = dw_flops  + pw_flops  + linear_qkv_flops  + attn_flops + proj_flops  + mlp_flops
    return params, flops


def vit_tail_block_budget(stage_H: int, stage_W: int, C: int, mlp_ratio: float,
                          num_heads: int, qkv_bias: bool = True,
                          with_cls_token: bool = True) -> Tuple[int, int]:
    """
    Analytic per-block estimate for ViTTail blocks (TAIL_TYPE='vit').
    ViTTail uses standard ViT-style blocks: Linear QKV (no conv projections),
    spatial attention with CLS token, W_o and MLP.
    T = 1 + H*W when with_cls_token, else H*W.
    """
    T = (1 + stage_H * stage_W) if with_cls_token else (stage_H * stage_W)
    # QKV: 3 * Linear(C, C)
    qkv_params = 3 * (C * C + (C if qkv_bias else 0))
    qkv_flops  = 3 * T * C * C
    # Attention: QK^T + AV
    attn_flops = 2 * T * T * C
    # W_o: Linear(C, C)
    proj_params = C * C + C
    proj_flops  = T * C * C
    # MLP
    hidden = int(max(1, mlp_ratio) * C)
    mlp_params = (C * hidden + hidden) + (hidden * C + C)
    mlp_flops  = (C * hidden + hidden * C) * T
    # Two LNs
    ln_params = 4 * C
    params = qkv_params + proj_params + mlp_params + ln_params
    flops  = qkv_flops + attn_flops + proj_flops + mlp_flops
    return params, flops

# --------------------------- YAML loader ---------------------------

def load_spec_from_yaml(path: str):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    # DW harness stores model spec under MODEL.SPEC for cls_cvt
    if 'MODEL' not in cfg or 'SPEC' not in cfg['MODEL']:
        raise KeyError("YAML must contain MODEL.SPEC for CvT profiling.")
    spec = cfg['MODEL']['SPEC']

    # Provide defaults if absent
    if 'SA_TAIL_DEPTH' not in spec:
        spec['SA_TAIL_DEPTH'] = [0] * (spec['NUM_STAGES'] - 1) + [0]
    if 'SA_TAIL_NUM_HEADS' not in spec:
        spec['SA_TAIL_NUM_HEADS'] = spec['NUM_HEADS']

    # Patch embedding methods per stage (optional); default to 'std'
    if 'PATCH_EMBED_METHOD' not in spec:
        spec['PATCH_EMBED_METHOD'] = ['std'] * spec['NUM_STAGES']

    # Image size: DW harness uses DATA.IMG_SIZE (int). Fallbacks included.
    if 'DATA' in cfg and 'IMG_SIZE' in cfg['DATA']:
        img = cfg['DATA']['IMG_SIZE']
        img_h = img_w = int(img) if isinstance(img, (int, float)) else int(img[0])
    elif 'TRAIN' in cfg and 'IMAGE_SIZE' in cfg['TRAIN']:
        val = cfg['TRAIN']['IMAGE_SIZE']
        img_h, img_w = (val, val) if isinstance(val, int) else (val[0], val[1])
    elif 'TEST' in cfg and 'IMAGE_SIZE' in cfg['TEST']:
        val = cfg['TEST']['IMAGE_SIZE']
        img_h, img_w = (val, val) if isinstance(val, int) else (val[0], val[1])
    else:
        img_h = img_w = 224

    return cfg, spec, img_h, img_w

# --------------------------- main ---------------------------

def main():
    # MACs & params profiler (plus derived FLOPs)
    ap = argparse.ArgumentParser(description='Per-stage/per-block MACs & params profiler (plus derived FLOPs) for CvT (DW harness)')
    ap.add_argument('--yaml', required=True, help='Path to model YAML (DW harness format with MODEL.SPEC)')
    ap.add_argument('--csv', default=None, help='Optional CSV output path')
    ap.add_argument('--repo', default=None, help='Path to Efficient_ViT_with_DW repo root (added to sys.path)')
    args = ap.parse_args()

    # Add repo to path if provided so we can import models.cls_cvt for sanity build
    if args.repo:
        repo_root = os.path.abspath(args.repo)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    cfg, spec, in_h, in_w = load_spec_from_yaml(args.yaml)

    S = spec['NUM_STAGES']
    tail_type = spec.get('TAIL_TYPE', 'sa')  # 'sa' (in-stage SA blocks) or 'vit' (standalone ViTTail)
    ps, st, pd     = spec['PATCH_SIZE'], spec['PATCH_STRIDE'], spec['PATCH_PADDING']
    C_list         = spec['DIM_EMBED']
    depth_list     = spec['DEPTH']
    mlp_ratios     = spec['MLP_RATIO']
    stride_q       = spec['STRIDE_Q']
    stride_kv      = spec['STRIDE_KV']
    kernels        = spec['KERNEL_QKV']
    sa_tail        = spec.get('SA_TAIL_DEPTH', [0] * (S - 1) + [0])
    sa_tail_heads  = spec.get('SA_TAIL_NUM_HEADS', spec['NUM_HEADS'])
    pe_methods     = spec.get('PATCH_EMBED_METHOD', ['std'] * S)

    qkv_methods    = spec.get('QKV_PROJ_METHOD', ['dw_bn'] * S)
    ca_mlp_ratios  = spec.get('CA_MLP_RATIO', [])
    tail_mlp_ratio = spec.get('TAIL_MLP_RATIO', [])
    qkv_bias       = spec.get('QKV_BIAS', [True] * S)
    backbone_tm    = spec.get('BACKBONE_TOKEN_MODE', 'channel')  # 'channel' | 'spatial' | list per-stage
    cls_tokens     = spec.get('CLS_TOKEN', [False] * S)

    # Infer input channels for each stage's tokenization (stage 1 takes RGB=3)
    Cin_list = [3] + C_list[:-1]

    # Stage output sizes from conv embed
    H, W = in_h, in_w
    stage_sizes: List[Tuple[int,int]] = []
    for i in range(S):
        H = conv_out(H, ps[i], pd[i], st[i])
        W = conv_out(W, ps[i], pd[i], st[i])
        stage_sizes.append((H, W))

    print(f"\nImage: {in_h}x{in_w}  tail_type={tail_type}")
    print("Per-stage / per-block budget (Params, MACs) — includes tokenization before each stage:\n")

    rows = []
    total_params, total_flops = 0, 0

    H_in, W_in = in_h, in_w  # track running input size per stage
    for i in range(S):
        Hs, Ws = stage_sizes[i]
        C      = C_list[i]
        Cin    = Cin_list[i]
        D      = depth_list[i]
        SA     = sa_tail[i]

        # Params & MACs for tokenization conv BEFORE Stage i+1 (use method-aware formula)
        k_tok, s_tok, p_tok = ps[i], st[i], pd[i]
        pe_method = (pe_methods[i] if i < len(pe_methods) else 'std')
        tok_params, tok_macs, H_tok, W_tok, tok_tag = token_embed_cost(
            pe_method, Cin, C, k_tok, s_tok, p_tok, H_in, W_in, batch=1)

        qkv_m = qkv_methods[i] if i < len(qkv_methods) else 'dw_bn'

        # Optional LayerNorm after tokenization (PATCH_NORM; BN already in token_embed_cost for dw_*)
        if cfg.get('MODEL', {}).get('PATCH_NORM', False) or spec.get('PATCH_NORM', False):
            tok_params += 2 * C  # gamma + beta

        tail_label = "ViT tail" if (i == S - 1 and tail_type == 'vit' and SA > 0) else f"SA tail={SA}"
        print(f"Stage{i+1}: {Hs}x{Ws}, C={C}, CA depth={D}, {tail_label}, qkv={pe_methods[i] if i < len(pe_methods) else 'std'}|{qkv_m}")
        print(f"  tok  TOK  {pretty(tok_params):>8}   {pretty(tok_macs):>10}   (k={k_tok}, s={s_tok}, p={p_tok}, method={pe_method}:{tok_tag})")
        print("  idx  kind   params        macs")

        # accumulate and record CSV row for tokenization
        total_params += tok_params; total_flops += tok_macs
        rows.append([i+1, -1, 'TOK', Hs, Ws, C, tok_params, tok_macs])

        # Backbone blocks: token_mode from BACKBONE_TOKEN_MODE (channel or spatial per-stage)
        backbone_tm_i = backbone_tm[i] if isinstance(backbone_tm, (list, tuple)) and i < len(backbone_tm) else (backbone_tm if isinstance(backbone_tm, str) else 'channel')
        block_kind = 'CA' if backbone_tm_i == 'channel' else 'SA'
        ca_ratio = ca_mlp_ratios[i] if (ca_mlp_ratios and i < len(ca_mlp_ratios)) else (mlp_ratios[i] if backbone_tm_i == 'spatial' else 0.0)
        for j in range(D):
            p, f = block_budget(Hs, Ws, C, ca_ratio, stride_q[i], stride_kv[i], kernels[i],
                                 token_mode=backbone_tm_i, qkv_method=qkv_m)
            total_params += p; total_flops += f
            rows.append([i+1, j, block_kind, Hs, Ws, C, p, f])
            print(f"  {j:>2}   {block_kind}   {pretty(p):>8}   {pretty(f):>10}")

        # Tail blocks: SA (in-stage) or ViT (standalone ViTTail)
        tail_ratio = (tail_mlp_ratio[i] if i < len(tail_mlp_ratio) and tail_mlp_ratio else mlp_ratios[i])
        use_vit_tail = (i == S - 1) and (tail_type == 'vit') and (SA > 0)

        if use_vit_tail:
            # ViTTail: Linear QKV, spatial attention with optional CLS token
            num_heads = sa_tail_heads[i] if i < len(sa_tail_heads) else max(4, spec['NUM_HEADS'][i])
            qkv_b = qkv_bias[i] if i < len(qkv_bias) else True
            with_cls = cls_tokens[i] if i < len(cls_tokens) else False
            # ViTTail overhead: cls_token (C) if used + final LayerNorm (2*C)
            vit_overhead_p = (3 * C) if with_cls else (2 * C)
            total_params += vit_overhead_p
            rows.append([i+1, -2, 'ViT_overhead', Hs, Ws, C, vit_overhead_p, 0])
            print(f"  --   ViT+  {pretty(vit_overhead_p):>8}   {pretty(0):>10}   (cls_token + norm)")
            for j in range(SA):
                p, f = vit_tail_block_budget(Hs, Ws, C, tail_ratio, num_heads, qkv_bias=qkv_b, with_cls_token=with_cls)
                total_params += p; total_flops += f
                rows.append([i+1, D+j, 'ViT', Hs, Ws, C, p, f])
                print(f"  {D+j:>2}   ViT  {pretty(p):>8}   {pretty(f):>10}")
        else:
            # SA tail blocks (token_mode='spatial', dw_bn QKV)
            for j in range(SA):
                p, f = block_budget(Hs, Ws, C, tail_ratio, stride_q[i], stride_kv[i], kernels[i],
                                     token_mode='spatial', qkv_method=qkv_m)
                total_params += p; total_flops += f
                rows.append([i+1, D+j, 'SA', Hs, Ws, C, p, f])
                print(f"  {D+j:>2}   SA   {pretty(p):>8}   {pretty(f):>10}")

        # Prepare H_in/W_in for next stage's tokenization
        H_in, W_in = Hs, Ws

        print("")

    # --- Classification head (global pooling + fc) ---
    num_classes = cfg.get('MODEL', {}).get('NUM_CLASSES', 1000)
    C_last = C_list[-1]
    head_params = (C_last * num_classes) + num_classes
    head_flops  = C_last * num_classes  # single token/vector to logits

    total_params += head_params; total_flops += head_flops
    rows.append([S, 999, 'HEAD', stage_sizes[-1][0], stage_sizes[-1][1], C_last, head_params, head_flops])
    print(f"HEAD: cls Linear {C_last}->{num_classes}   params {pretty(head_params)}   macs {pretty(head_flops)}\n")

    print(f"TOTAL Params: {pretty(total_params)}")
    print(f"TOTAL MACs  : {pretty(total_flops)}")
    print(f"TOTAL FLOPs : {pretty(total_flops * 2)}  # using 1 MAC ≈ 2 FLOPs\n")

    # Optional CSV output
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['stage', 'block_index', 'kind', 'H', 'W', 'C', 'params', 'macs'])
            writer.writerows(rows)
        print(f"[Saved] CSV written to: {args.csv}")

    # Optional sanity build using the DW repo's implementation of cls_cvt
    try:
        if args.repo:
            from models.cls_cvt import ConvolutionalVisionTransformer, LayerNorm, QuickGELU  # type: ignore
            model = ConvolutionalVisionTransformer(
                in_chans=3,
                num_classes=cfg.get('MODEL', {}).get('NUM_CLASSES', 1000),
                act_layer=QuickGELU,
                norm_layer=LayerNorm,
                init=spec.get('INIT', 'trunc_norm'),
                spec=spec
            )
            print(f"[Confirming] Model builds. Trainable params (PyTorch): {pretty(count_params(model))}")
        else:
            print("[Confirming] Skipped (no --repo provided). Analytic report above is valid.")
    except Exception as e:
        print(f"[Confirming] Model build failed: {e}")

if __name__ == '__main__':
    main()

# -------------- HOW TO RUN --------------
# python profile_blocks.py \
#   --repo /path/to/repo \
#   --yaml /path/to/repo/configs/cvt/cls_cvt13_tail5_224_cifar10.yaml \
#   --csv  /path/to/repo/csv/cvt13_c10_blocks.csv
