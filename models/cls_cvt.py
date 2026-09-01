from functools import partial
from itertools import repeat
try:
    from torch._six import container_abcs, string_classes, int_classes
except Exception:
    import collections.abc as container_abcs
    string_classes = (str,)
    int_classes = int

import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, trunc_normal_


try:
    from .registry import register_model
except Exception:
    # Fallback for harnesses without a registry module
    def register_model(fn):
        return fn

__all__ = ['get_cls_model', 'ConvolutionalVisionTransformer', 'VisionTransformer']

# --- Debug helpers -----------------------------------------------------------
DEBUG_SHAPES = False  # set False to silence


def nystrom_iterative_inv(mat: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
    """Approximate Moore-Penrose pseudoinverse via Newton-Schulz iteration.

    Standard recipe used in the Nyströmformer reference implementation (Xiong et al 2021,
    arXiv:2102.03902). Input: square (..., L, L) matrix. Returns (..., L, L) approximate inverse.
    Numerically stable for the small landmark-landmark matrices used in Nyström attention.
    """
    L = mat.shape[-1]
    I = torch.eye(L, device=mat.device, dtype=mat.dtype).expand_as(mat)
    # Init V = (1 / max_row_sum) * (1 / max_col_sum) * mat^T
    row_max = torch.max(torch.sum(torch.abs(mat), dim=-2), dim=-1, keepdim=True).values  # (..., 1)
    col_max = torch.max(torch.sum(torch.abs(mat), dim=-1), dim=-1, keepdim=True).values  # (..., 1)
    V = (mat.transpose(-1, -2) / (row_max.unsqueeze(-1) * col_max.unsqueeze(-1) + 1e-6))
    for _ in range(n_iter):
        KV = torch.matmul(mat, V)
        V = torch.matmul(V, 13.0 * I - torch.matmul(KV, 15.0 * I - torch.matmul(KV, 7.0 * I - KV)))
    return V


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo 2016, arXiv:1602.02068).

    Projects z onto the probability simplex (Euclidean projection). Output is
    sparse-but-differentiable: some entries become exactly 0, the rest are
    `max(z - tau, 0)` with tau chosen so the sum is 1 per row. Differentiable
    almost everywhere; gradient flows through the active support.
    """
    z = z - z.max(dim=dim, keepdim=True).values  # numerical stability
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim=dim)
    K = z.shape[dim]
    k_idx = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = K
    k_idx = k_idx.view(shape)
    support = (1.0 + k_idx * z_sorted) > z_cumsum
    k_support = support.to(z.dtype).sum(dim=dim, keepdim=True).clamp(min=1)
    idx = (k_support.long() - 1)
    tau = (z_cumsum.gather(dim, idx) - 1.0) / k_support
    return torch.clamp(z - tau, min=0)

# Use PyTorch scaled dot-product attention (Flash/SDPA) when available
# Works for both token modes ('channel' and 'spatial')
USE_SDPA = True

def _p(name, *tensors_or_vals):
    if not DEBUG_SHAPES:
        return
    parts = []
    for v in tensors_or_vals:
        if isinstance(v, torch.Tensor):
            parts.append(f"{tuple(v.shape)}")
        else:
            parts.append(str(v))
    print(f"[SHAPE] {name}: " + ", ".join(parts))


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse



to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

# --- 2D sine-cos positional embeddings (no learned params) -------------------
import math as _math

def _get_1d_sincos_pos_embed(dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Return (N, dim) tensor for positions in `pos` using sine-cos encoding."""
    assert dim % 2 == 0
    omega = torch.arange(dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 2)))
    out = pos[:, None] * omega[None, :]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb  # (N, dim)

def get_2d_sincos_pos_embed(embed_dim: int, h: int, w: int, cls_token: bool = True, device=None) -> torch.Tensor:
    """Create 2D sin-cos positional embeddings; shape (1+H*W, C) if cls_token else (H*W, C)."""
    if device is None:
        device = 'cpu'
    grid_h = torch.arange(h, device=device, dtype=torch.float32)
    grid_w = torch.arange(w, device=device, dtype=torch.float32)
    gh, gw = torch.meshgrid(grid_h, grid_w, indexing='ij')  # (H,W)
    gh = gh.reshape(-1)
    gw = gw.reshape(-1)
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, gh)
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, gw)
    pos = torch.cat([emb_h, emb_w], dim=1)  # (H*W, C)
    if cls_token:
        cls = torch.zeros(1, embed_dim, device=device)
        pos = torch.cat([cls, pos], dim=0)
    return pos  # (1+H*W, C)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, arXiv:1910.07467).

    Removes the mean-centering step of LayerNorm; only performs RMS-based rescaling.
    For channel-token mode, a channel token's spatial feature *mean* carries its global
    activation level — LayerNorm discards this by subtracting the mean, while RMSNorm
    preserves it.  Handles fp16 identically to the LayerNorm subclass above.
    """

    def __init__(self, normalized_shape, eps: float = 1e-6, **kwargs):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms * self.weight
        return x.to(orig_dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class SwiGLUMlp(nn.Module):
    """SwiGLU feed-forward network (Shazeer 2020; used in PaLM, LLaMA, etc.).
    out = W2( SiLU(W_gate * x) * W_val * x )
    Hidden dim scaled by 2/3 to keep param parity with the 2-linear GELU MLP.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Scale down hidden to keep total params ≈ standard 2-linear MLP
        hidden_swi = max(1, int(hidden_features * 2 / 3))
        self.gate = nn.Linear(in_features, hidden_swi)
        self.val  = nn.Linear(in_features, hidden_swi)
        self.proj = nn.Linear(hidden_swi, out_features)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.proj(self.act(self.gate(x)) * self.val(x)))


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 token_mode='channel',  # NEW: 'channel' (channels as tokens) or 'spatial' (positions as tokens)
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # Will compute scale at runtime from the actual head dimension
        self.scale = None
        self.with_cls_token = with_cls_token
        # Token mode selects whether we operate in channel-token space or spatial-token space
        self.token_mode = token_mode  # 'channel' or 'spatial'

        self.token_mode = token_mode  # 'channel' or 'spatial'

        # Group channel attention: divide C channel-tokens into G equal groups.
        # Within each group: (C/G)×(C/G) attention instead of full C×C.
        # G=1 (default) = standard full channel attention. G>1 reduces attention FLOPs by G.
        # Cross-group mixing is performed by the post-attention proj Linear(C,C).
        self.num_channel_groups = int(kwargs.get('num_channel_groups', 1))
        self.sparse_topk = int(kwargs.get('sparse_topk', 0))
        # Dual-resolution CA: parallel coarse pathway over G super-channel groups.
        # 0 = disabled; G > 0 = enabled. dual_ca_scale starts at 0 so init behaviour = baseline.
        self.dual_ca_groups = int(kwargs.get('dual_ca_groups', 0))
        # Agent attention (H3-4): factorised low-rank CA via M learnable agent tokens.
        # Replaces softmax CA with two cascaded softmaxes (M×C, C×M). All channel-channel
        # info flows through M agents → rank ≤ M bottleneck. M=64 → ~5× cheaper than dense.
        self.agent_tokens = int(kwargs.get('agent_tokens', 0))
        self.agent_spatial_hw = int(kwargs.get('agent_spatial_hw', 0))
        # Hierarchical local+global CA (H3-5): split C channels into G super-groups.
        # Within-group dense CA (C/G tokens per group) + cross-group dense CA (G group-summary tokens).
        # Adds cross-group attention pathway absent in prior failed grouped-only CA (H17, P2-H3).
        self.hier_groups = int(kwargs.get('hier_groups', 0))
        # H3-8: Linear attention (Katharopoulos et al. 2020, arXiv:2006.16236) — replace softmax
        # CA with phi(Q) · (phi(K)^T · V) / phi(Q) · sum(phi(K)), phi = ELU+1. Rank ≤ D (inner-feature
        # dim), no softmax stochastic-matrix constraint, O(C·D²) cost. Different inductive bias from
        # softmax: dense multiplicative interactions, not "select few" probability distributions.
        # Currently used only in the funnel tail block (additive parallel branch).
        self.linear_attn = bool(kwargs.get('linear_attn', False))
        # H3-9: Linformer-style projected channel attention (Wang et al. 2020, arXiv:2006.04768).
        # Project K, V along the CHANNEL (token) axis from C to r << C learned "supertokens" before
        # softmax. Each query channel attends to r=128 supertokens (linear combinations of all C
        # channels) instead of attending directly to all C channels. Different inductive bias from
        # full-rank softmax CA: forces a learned compact channel basis. Stable (still softmax).
        # Active when linformer_r > 0 in channel mode and not overridden by other operators.
        self.linformer_r = int(kwargs.get('linformer_r', 0))
        # H3-15: optional sparse top-k mask applied on attention scores BEFORE softmax inside the
        # use_linformer branch. Each query token keeps only its top-k of r supertokens; the rest
        # are masked to -inf. 0 (default) → dense softmax over all r supertokens (H3-12 behaviour).
        self.linformer_topk = int(kwargs.get('linformer_topk', 0))
        # H3-16: optional parallel FULL-RANK CA pathway alongside Linformer (only used when
        # use_linformer is active). Combine via a learned scalar gate init to 0 → at init equals
        # pure Linformer (H3-12). Training may ramp the gate to integrate full-rank information.
        self.linformer_fullrank_parallel = bool(kwargs.get('linformer_fullrank_parallel', False))
        # H3-17: choice of normalisation inside the use_linformer branch. 'softmax' (default,
        # = H3-12) or 'sparsemax' (Martins 2016 — Euclidean projection, sparse-but-differentiable).
        self.linformer_softmax_type = str(kwargs.get('linformer_softmax_type', 'softmax')).lower()
        assert self.linformer_softmax_type in ('softmax', 'sparsemax'), \
            f"linformer_softmax_type must be 'softmax' or 'sparsemax', got {self.linformer_softmax_type!r}"
        # H3-18: Synthesizer-Dense replaces dot-product attention. attn_score = Linear(D, C)(Q),
        # then softmax → · V. K computed but unused. Active in channel mode when synthesizer_dense=True.
        self.synthesizer_dense = bool(kwargs.get('synthesizer_dense', False))
        self.synthesizer_hw = int(kwargs.get('synthesizer_hw', 0))
        # H3-19: Nyström attention enable + landmark count. 0 → disabled (default).
        self.nystrom_L = int(kwargs.get('nystrom_L', 0))
        self.nystrom_iters = int(kwargs.get('nystrom_iters', 6))

        # --- QKV projection path depends on method + token_mode ---
        if method == 'linear' and self.token_mode == 'spatial':
            # ViT-style tail: no conv projections, use full linear QKV
            self.conv_proj_q = None
            self.conv_proj_k = None
            self.conv_proj_v = None

            self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
            self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
            self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        else:
            # Convolutional QKV projections (or avg/identity) as before
            self.conv_proj_q = self._build_projection(
                dim_in, dim_out, kernel_size, padding_q,
                stride_q, 'linear' if method == 'avg' else method
            )
            self.conv_proj_k = self._build_projection(
                dim_in, dim_out, kernel_size, padding_kv,
                stride_kv, method
            )
            self.conv_proj_v = self._build_projection(
                dim_in, dim_out, kernel_size, padding_kv,
                stride_kv, method
            )

            self.proj_q = nn.Identity()
            self.proj_k = nn.Identity()
            self.proj_v = nn.Identity()

        # Learnable channel-identity bias (used only in channel-token mode), broadcast over spatial features
        self.channel_bias = nn.Parameter(torch.zeros(1, dim_out, 1))
        # Small learnable gate so the bias cannot dominate early training
        self.channel_bias_scale = nn.Parameter(torch.tensor(1e-2))
        # In spatial-token mode these parameters are not used in forward; disable grads to avoid DDP unused-params errors
        if self.token_mode != 'channel':
            self.channel_bias.requires_grad_(False)
            self.channel_bias_scale.requires_grad_(False)

        # Learned per-head attention temperature (channel mode only).
        # log_τ=0 → τ=1 at init (no change). exp() ensures τ>0 always.
        # ndim==1 → automatically excluded from weight decay by optimizer.
        self.log_attn_temp = nn.Parameter(torch.zeros(num_heads))
        if self.token_mode != 'channel':
            self.log_attn_temp.requires_grad_(False)

        # Dual-resolution CA learned scalar (only active when dual_ca_groups > 0 in channel mode).
        # Initialized to 0 → coarse pathway starts disabled; model can ramp it up during training.
        self.dual_ca_scale = nn.Parameter(torch.zeros(1))
        if not (self.token_mode == 'channel' and self.dual_ca_groups > 0):
            self.dual_ca_scale.requires_grad_(False)

        # Agent attention parameters (H3-4): allocate M learnable tokens of shape (1, H, M, head_dim)
        # where head_dim = spatial_hw / num_heads (matches q/k/v inner dim in channel mode).
        # Init with truncated normal std=0.02 (standard for learnable tokens).
        if self.token_mode == 'channel' and self.agent_tokens > 0 and self.agent_spatial_hw > 0:
            assert self.agent_spatial_hw % num_heads == 0, \
                f"agent_spatial_hw ({self.agent_spatial_hw}) must be divisible by num_heads ({num_heads})"
            head_dim = self.agent_spatial_hw // num_heads
            self.agents = nn.Parameter(torch.zeros(1, num_heads, self.agent_tokens, head_dim))
            trunc_normal_(self.agents, std=0.02)
        else:
            self.register_parameter('agents', None)

        # Hierarchical CA (H3-5): learnable scalar weighting cross-group contribution.
        # Init = 1.0 → cross-group fully active from start (no cold-start; model learns to balance).
        # Disabled (requires_grad=False) when hier_groups == 0.
        self.hier_cross_scale = nn.Parameter(torch.ones(1))
        if not (self.token_mode == 'channel' and self.hier_groups > 0):
            self.hier_cross_scale.requires_grad_(False)

        # H3-9: Linformer-style K/V channel-axis projection (C → r supertokens).
        # E_k, E_v: (r, C) learnable. Applied as `k.transpose(-2,-1) @ E^T → transpose back`.
        # Output K_proj, V_proj have shape (B, H, r, D), so attention is (C, r) instead of (C, C).
        # Initialized with xavier_uniform for stable softmax statistics (similar magnitude to identity).
        if self.token_mode == 'channel' and self.linformer_r > 0:
            self.linformer_E_k = nn.Linear(dim_out, self.linformer_r, bias=False)
            self.linformer_E_v = nn.Linear(dim_out, self.linformer_r, bias=False)
            nn.init.xavier_uniform_(self.linformer_E_k.weight)
            nn.init.xavier_uniform_(self.linformer_E_v.weight)
            # H3-16: gate for parallel full-rank pathway. Scalar param init to 0 → at init the
            # parallel pathway contributes 0; output matches pure Linformer (H3-12).
            if self.linformer_fullrank_parallel:
                self.linformer_fullrank_gate = nn.Parameter(torch.zeros(1))
            else:
                self.linformer_fullrank_gate = None
        else:
            self.linformer_E_k = None
            self.linformer_E_v = None
            self.linformer_fullrank_gate = None

        # H3-18: Synthesizer-Dense weight. Shape (D, C) where D = spatial_hw**2, C = dim_out.
        # Applied to Q (B,H,C,D) → (B,H,C,C) attention scores.
        if self.token_mode == 'channel' and self.synthesizer_dense and self.synthesizer_hw > 0:
            D = self.synthesizer_hw * self.synthesizer_hw
            self.synthesizer_W = nn.Linear(D, dim_out, bias=False)
            nn.init.xavier_uniform_(self.synthesizer_W.weight)
        else:
            self.synthesizer_W = None

        self.attn_drop = nn.Dropout(attn_drop)
        # Post-attention projection: disable channel mixing for channel-token mode
        if self.token_mode == 'channel':
            # Keep identity so only attention performs cross-channel mixing
            #self.proj = nn.Identity()
            self.proj = nn.Linear(dim_out, dim_out)
        else:
            # Spatial-token (tail) keeps full pointwise mixing
            self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            # Convolutional projection with depthwise conv + BN; rearrange depends on token_mode
            rearr = Rearrange('b c h w -> b c (h w)') if self.token_mode == 'channel' else Rearrange('b c h w -> b (h w) c')
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearr', rearr),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'pw_bn':
            # Pointwise 1x1 conv + BN (full channel mixing for Q/K/V)
            rearr = Rearrange('b c h w -> b c (h w)') if self.token_mode == 'channel' else Rearrange('b c h w -> b (h w) c')
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)),
                ('bn', nn.BatchNorm2d(dim_out)),
                ('rearr', rearr),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        _p("Attn.forward_conv.in", x)
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)
            _p("Attn.split.cls", cls_token, x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        _p("Attn.BCHW", x)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')
        _p("Attn.q_pre_cat", q)

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')
        _p("Attn.k_pre_cat", k)

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')
        _p("Attn.v_pre_cat", v)

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)
            _p("Attn.qkv_post_cat", q, k, v)

        return q, k, v

    def forward(self, x, h, w, channel_pair_bias=None, bridge_kv=None):
        _p("Attn.forward.in", x)
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)
            _p("Attn.qkv.after_forward_conv", q, k, v)  # [b, t, c]
            # For channel-attention, stabilize dot-product geometry and add a tiny gated identity bias
            if self.token_mode == 'channel':
                q = F.layer_norm(q, (q.shape[-1],))
                k = F.layer_norm(k, (k.shape[-1],))
                q = q + self.channel_bias_scale * self.channel_bias
                k = k + self.channel_bias_scale * self.channel_bias
        else:
            # Linear QKV path (standard ViT): use incoming tokens directly
            # x is (B, T, C) where T may include a CLS token already
            q, k, v = x, x, x

        _p("self.proj_q(q)", self.proj_q(q))
        _p("self.proj_k(k)", self.proj_k(k))
        _p("self.proj_v(v)", self.proj_v(v))
        # Group channel attention: fold G groups into batch dim so standard attention runs within each group.
        G = self.num_channel_groups if (self.token_mode == 'channel' and self.num_channel_groups > 1) else 1
        if G > 1:
            # q/k/v shape in channel mode: (B, C, H*W) where C = G × (C//G)
            # After rearrange: (B*G, 1, C//G, H*W) — group in batch, 1 head, group-tokens, spatial-features
            q = rearrange(self.proj_q(q), 'b (g t) d -> (b g) 1 t d', g=G)
            k = rearrange(self.proj_k(k), 'b (g t) d -> (b g) 1 t d', g=G)
            v = rearrange(self.proj_v(v), 'b (g t) d -> (b g) 1 t d', g=G)
        else:
            q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
            k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
            v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        _p("Attn.qkv.split_heads", q, k, v)

        # Apply per-head learned temperature to Q (channel mode only).
        # Q*τ effectively scales QK^T by τ before softmax → sharper/flatter attention.
        # For group CA: temp broadcasts uniformly across all groups (shared temperature).
        if self.token_mode == 'channel':
            temp = self.log_attn_temp.exp().view(1, self.num_heads, 1, 1)
            q = q * temp

        # Cross-stage CA bridge (H3-3): concat bridge tokens (B, C, HW) as extra K/V tokens.
        # Bridge projection weights are init-zero → bridge contributes 0 scores at step 0
        # (safe baseline). Disabled for group CA (G>1) and spatial mode.
        if (self.token_mode == 'channel' and G == 1 and bridge_kv is not None):
            # bridge_kv: (B, C, HW) → (B, 1, C, HW) to match k/v shape (B, 1, C_tokens, HW)
            bridge_kv_4d = bridge_kv.unsqueeze(1)
            k = torch.cat([k, bridge_kv_4d], dim=2)  # concat along token dim: C → C + C_bridge
            v = torch.cat([v, bridge_kv_4d], dim=2)

        # Sparse top-k channel attention: each channel attends to its top-k most-similar channels only.
        # Enabled when sparse_topk > 0 AND token_mode == 'channel' AND sparse_topk < num_tokens.
        # Bypasses SDPA because SDPA does not support dynamic row-wise top-k masks efficiently.
        T_tokens = q.shape[-2]
        use_sparse = (
            self.token_mode == 'channel'
            and self.sparse_topk > 0
            and self.sparse_topk < T_tokens
            and G == 1
        )

        # H3-8: linear attention path. Bypasses softmax: phi(Q) · (phi(K)^T · V) / norm.
        # Active when linear_attn=True and channel mode, ignores sparse/hier/agent/bridge.
        use_linear = (
            self.token_mode == 'channel'
            and self.linear_attn
            and G == 1
            and bridge_kv is None
            and not use_sparse
        )

        # H3-9: Linformer projected attention path. Project K, V along channel axis to r supertokens.
        # Active when linformer_r > 0 and channel mode, no other operator active.
        use_linformer = (
            self.token_mode == 'channel'
            and self.linformer_r > 0
            and self.linformer_E_k is not None
            and G == 1
            and bridge_kv is None
            and not use_sparse
            and not use_linear
        )

        # H3-18: Synthesizer-Dense path. Replaces Q·K^T with Linear(D,C)(Q) for attention scores.
        use_synthesizer = (
            self.token_mode == 'channel'
            and self.synthesizer_dense
            and self.synthesizer_W is not None
            and G == 1
            and bridge_kv is None
            and not use_sparse
            and not use_linear
            and not use_linformer
        )

        # H3-19: Nyström landmark-based attention. Preserves Q·K^T; compresses via L landmarks.
        use_nystrom = (
            self.token_mode == 'channel'
            and self.nystrom_L > 0
            and G == 1
            and bridge_kv is None
            and not use_sparse
            and not use_linear
            and not use_linformer
            and not use_synthesizer
        )
        # Require C divisible by L (uniform avg-pool); fall through to standard CA if not.
        if use_nystrom and (q.shape[-2] % self.nystrom_L != 0):
            use_nystrom = False

        # Hierarchical local+global CA (H3-5): split C tokens into G super-groups.
        # Within-group dense CA (C/G × C/G softmax per group) + cross-group dense CA (G×G
        # softmax over group-summary tokens). Output = within + α·broadcast(cross), with
        # α = hier_cross_scale learnable scalar (init 1.0). Distinct from prior failed grouped CA
        # because of the explicit cross-group attention pathway. Inspired by BiFormer (CVPR 2023).
        use_hier = (
            self.token_mode == 'channel'
            and self.hier_groups > 0
            and G == 1
            and bridge_kv is None
            and (self.agents is None)
            and not use_linformer
            and not use_synthesizer
            and not use_nystrom
        )
        # Validate divisibility (skip silently if not divisible — fall through to standard CA)
        if use_hier and (q.shape[-2] % self.hier_groups != 0):
            use_hier = False

        if use_hier:
            G_h = self.hier_groups
            B_, H_, C_tokens, head_dim = q.shape
            c_per_g = C_tokens // G_h
            scale = head_dim ** -0.5
            # (B, H, C, D) -> (B, H, G, c, D)
            qg = q.view(B_, H_, G_h, c_per_g, head_dim)
            kg = k.view(B_, H_, G_h, c_per_g, head_dim)
            vg = v.view(B_, H_, G_h, c_per_g, head_dim)
            # Within-group attention: (B, H, G, c, c)
            a_w = torch.einsum('bhgid,bhgjd->bhgij', qg, kg) * scale
            a_w = F.softmax(a_w, dim=-1)
            a_w = self.attn_drop(a_w)
            out_w = torch.einsum('bhgij,bhgjd->bhgid', a_w, vg)  # (B, H, G, c, D)
            # Cross-group via group-mean summaries
            qm = qg.mean(dim=3)  # (B, H, G, D)
            km = kg.mean(dim=3)
            vm = vg.mean(dim=3)
            a_c = torch.einsum('bhgd,bhkd->bhgk', qm, km) * scale  # (B, H, G, G)
            a_c = F.softmax(a_c, dim=-1)
            a_c = self.attn_drop(a_c)
            out_c = torch.einsum('bhgk,bhkd->bhgd', a_c, vm)  # (B, H, G, D)
            # Broadcast cross-group output to all channels in each group
            out_c_exp = out_c.unsqueeze(3).expand(-1, -1, -1, c_per_g, -1)  # (B, H, G, c, D)
            out = out_w + self.hier_cross_scale * out_c_exp  # (B, H, G, c, D)
            x = out.reshape(B_, H_, C_tokens, head_dim)  # (B, H, C, D)
            _p("Attn.hier_out", x)

        # Agent attention path (H3-4): factorised low-rank CA via M learnable agent tokens.
        # Two cascaded softmaxes: A1 = softmax(agents @ K^T / sqrt(D)) → (B,H,M,C) gathers
        # weighted V into M agent features (B,H,M,D). A2 = softmax(Q @ agents^T / sqrt(D))
        # → (B,H,C,M) routes each query through the agents to read agent features.
        # Result: (B,H,C,D). Inspired by Han et al. "Agent Attention" (arXiv:2312.08874, ECCV 2024).
        # Bypasses SDPA + dual-CA + sparse + cpb + bridge (all skipped when agent active).
        use_agent = (
            self.token_mode == 'channel'
            and self.agents is not None
            and G == 1
            and bridge_kv is None
            and not use_hier
            and not use_linear
            and not use_linformer
            and not use_synthesizer
            and not use_nystrom
        )

        if use_hier:
            pass  # x already computed above
        elif use_linear:
            # H3-8: Linear attention. phi = ELU + 1 (>= 0 ensures stable normalization).
            # Computes phi(Q) · (phi(K)^T · V) / (phi(Q) · sum(phi(K))).
            phi_q = F.elu(q) + 1.0  # (B, H, C, D)
            phi_k = F.elu(k) + 1.0  # (B, H, C, D)
            # KV summary: sum over the C dim of K and V (treating channels-as-tokens here)
            kv_sum = torch.einsum('bhcd,bhce->bhde', phi_k, v)  # (B, H, D, D)
            k_sum = phi_k.sum(dim=-2)                            # (B, H, D)
            num = torch.einsum('bhcd,bhde->bhce', phi_q, kv_sum) # (B, H, C, D)
            denom = torch.einsum('bhcd,bhd->bhc', phi_q, k_sum).unsqueeze(-1) + 1e-6  # (B, H, C, 1)
            x = num / denom
            _p("Attn.linear_out", x)
        elif use_linformer:
            # H3-9: Linformer projected attention. Project K, V along channel (token) axis from C to r.
            # k, v shape: (B, H, C, D). Linear(C, r) operates on last dim, so transpose first.
            k_proj = self.linformer_E_k(k.transpose(-2, -1)).transpose(-2, -1)  # (B, H, r, D)
            v_proj = self.linformer_E_v(v.transpose(-2, -1)).transpose(-2, -1)  # (B, H, r, D)
            head_dim = q.shape[-1]
            # Standard softmax attention with projected K, V. attn shape: (B, H, C, r).
            attn_score = torch.einsum('bhlk,bhtk->bhlt', q, k_proj) * (head_dim ** -0.5)
            # H3-15: optional sparse top-k mask within the Linformer supertoken basis. Each query
            # keeps only its top-k of r supertokens before softmax.
            if self.linformer_topk > 0 and self.linformer_topk < attn_score.shape[-1]:
                topk_vals, _ = attn_score.topk(self.linformer_topk, dim=-1)
                threshold = topk_vals[..., -1:]  # (B, H, C, 1) — k-th largest score per query
                attn_score = attn_score.masked_fill(attn_score < threshold, float('-inf'))
            # H3-17: normalisation dispatch. sparsemax produces sparse outputs differentiably;
            # softmax (default) recovers H3-12 behaviour.
            if self.linformer_softmax_type == 'sparsemax':
                attn = sparsemax(attn_score, dim=-1)
            else:
                attn = F.softmax(attn_score, dim=-1)
            attn = self.attn_drop(attn)
            x = torch.einsum('bhlt,bhtv->bhlv', attn, v_proj)  # (B, H, C, D)
            # H3-16: optional parallel full-rank softmax(Q·K^T)·V pathway, gated by init-0 scalar.
            if self.linformer_fullrank_gate is not None:
                fr_score = torch.einsum('bhlk,bhtk->bhlt', q, k) * (head_dim ** -0.5)
                fr_attn = F.softmax(fr_score, dim=-1)
                fr_attn = self.attn_drop(fr_attn)
                fr_out = torch.einsum('bhlt,bhtv->bhlv', fr_attn, v)
                x = x + self.linformer_fullrank_gate * fr_out
            _p("Attn.linformer_out", x)
        elif use_nystrom:
            # H3-19: Nyström landmark-based attention. Pool C channel tokens to L landmarks via
            # 1D avg-pool over the channel axis. Compute three softmaxes (Q·K_lm^T, Q_lm·K_lm^T,
            # Q_lm·K^T), then approximate softmax(QK^T)V ≈ F_1 · pinv(F_2) · F_3 · V.
            head_dim = q.shape[-1]
            scale = head_dim ** -0.5
            L = self.nystrom_L
            # Landmarks: pool along the C (token) axis. Assumes C % L == 0.
            q_lm = reduce(q, 'b h (l p) d -> b h l d', 'mean', l=L)  # (B, H, L, D)
            k_lm = reduce(k, 'b h (l p) d -> b h l d', 'mean', l=L)  # (B, H, L, D)
            # Three softmax matrices.
            F1 = torch.einsum('bhcd,bhld->bhcl', q, k_lm) * scale  # (B, H, C, L)
            F1 = F.softmax(F1, dim=-1)
            F2 = torch.einsum('bhld,bhmd->bhlm', q_lm, k_lm) * scale  # (B, H, L, L)
            F2 = F.softmax(F2, dim=-1)
            F3 = torch.einsum('bhld,bhcd->bhlc', q_lm, k) * scale  # (B, H, L, C)
            F3 = F.softmax(F3, dim=-1)
            F2_inv = nystrom_iterative_inv(F2, n_iter=self.nystrom_iters)  # (B, H, L, L)
            # Multiply in the cheapest order: V→F3→F2_inv→F1
            x = torch.einsum('bhlc,bhcd->bhld', F3, v)              # (B, H, L, D)
            x = torch.einsum('bhlm,bhmd->bhld', F2_inv, x)          # (B, H, L, D)
            x = torch.einsum('bhcl,bhld->bhcd', F1, x)              # (B, H, C, D)
            x = self.attn_drop(x)
            _p("Attn.nystrom_out", x)
        elif use_synthesizer:
            # H3-18: Synthesizer-Dense. Compute attention scores as a learned linear projection of
            # the queries: F = W·Q where W ∈ R^{D,C}. K is ignored — keep conv_proj_k intact (it's
            # shared infrastructure with channel_bias/proj_k logic) but tie a 0-magnitude reference
            # to k below so its params receive gradient (otherwise DDP errors on unused params).
            head_dim = q.shape[-1]
            # q shape: (B, H, C, D). synthesizer_W: Linear(D, C). Apply to last dim.
            attn_score = self.synthesizer_W(q) * (head_dim ** -0.5)  # (B, H, C, C)
            attn = F.softmax(attn_score, dim=-1)
            attn = self.attn_drop(attn)
            x = torch.einsum('bhij,bhjd->bhid', attn, v)  # (B, H, C, D)
            # Tie K's gradient flow to the output via a zero scalar so DDP sees conv_proj_k as
            # producing gradient. k.sum() is a scalar; multiplied by 0 it adds nothing to x but
            # ensures the conv_proj_k path is included in the backward graph.
            x = x + 0.0 * k.sum()
            _p("Attn.synthesizer_out", x)
        elif use_agent:
            head_dim = q.shape[-1]
            scale = head_dim ** -0.5
            B_ = q.shape[0]
            agents = self.agents.expand(B_, -1, -1, -1)  # (B, H, M, D)
            # A1: agents-as-queries attend to keys → (B, H, M, C)
            a1 = torch.einsum('bhmd,bhtd->bhmt', agents, k) * scale
            a1 = F.softmax(a1, dim=-1)
            a1 = self.attn_drop(a1)
            agent_feat = torch.einsum('bhmt,bhtd->bhmd', a1, v)  # (B, H, M, D)
            # A2: queries attend to agents → (B, H, C, M)
            a2 = torch.einsum('bhcd,bhmd->bhcm', q, agents) * scale
            a2 = F.softmax(a2, dim=-1)
            a2 = self.attn_drop(a2)
            x = torch.einsum('bhcm,bhmd->bhcd', a2, agent_feat)  # (B, H, C, D)
            _p("Attn.agent_out", x)
        elif USE_SDPA and not use_sparse and not use_synthesizer and not use_nystrom:
            # Faster, memory-efficient attention; numerically equivalent to manual softmax attention
            # channel_pair_bias (C, C) is added to logits before softmax when in channel-token mode
            # Group CA: skip channel_pair_bias since it's defined for full C (not per-group C/G)
            # Skip cpb if bridge_kv is concatenated (shape mismatch: cpb is C×C but K is now C×(C+C_bridge)).
            _cpb_ok = (bridge_kv is None)
            cpb = channel_pair_bias if (self.token_mode == 'channel' and channel_pair_bias is not None and G == 1 and _cpb_ok) else None
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=cpb,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )  # -> [B, H, T, D]
            _p("Attn.sdpa_out", x)
        else:
            # Scale by the true head dimension (the length of the spatial feature slice per head)
            head_dim = q.shape[-1]
            attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * (head_dim ** -0.5)
            if self.token_mode == 'channel' and channel_pair_bias is not None and bridge_kv is None:
                # Add (C, C) bias; broadcast over batch and head dims. Skip if bridge active (shape mismatch).
                attn_score = attn_score + channel_pair_bias.unsqueeze(0).unsqueeze(0)
            if use_sparse:
                # Row-wise top-k mask: for each query channel, keep only its top-k most-similar keys.
                # BSFA-inspired (arXiv:2512.07011): exact QK scoring, then prune to top-k per query row.
                k_top = min(self.sparse_topk, attn_score.shape[-1])
                topk_vals, _topk_idx = torch.topk(attn_score, k=k_top, dim=-1)
                # threshold = smallest kept score per row; detach so gradient flows only through attn
                threshold = topk_vals[..., -1:].detach()
                attn_score = attn_score.masked_fill(attn_score < threshold, float('-inf'))
            _p("Attn.attn_score", attn_score)
            attn = F.softmax(attn_score, dim=-1)
            _p("Attn.attn_softmax", attn)
            attn = self.attn_drop(attn)

            x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
            _p("Attn.weighted_v", x)

        # Dual-resolution channel attention: parallel coarse pathway over G super-channel groups.
        # Fine pathway (above) already attends across all C channel-tokens. Coarse pathway group-averages
        # channels into G super-tokens, runs attention over G, then scatter-broadcasts back to C.
        # Reuses fine q/k/v (no new projections) and adds learned dual_ca_scale (init 0) for safe start.
        # MSIT-inspired (arXiv:2403.06536, Mar 2024): multi-scale parallel attention, adapted to channel tokens.
        if (self.token_mode == 'channel' and self.dual_ca_groups > 0 and G == 1 and not use_agent and not use_hier):
            G_coarse = self.dual_ca_groups
            C_tokens = q.shape[-2]
            if C_tokens % G_coarse == 0:
                group_size = C_tokens // G_coarse
                # Group-mean pool over channel dim: (B, H, C, D) -> (B, H, G, D)
                coarse_q = q.view(q.shape[0], q.shape[1], G_coarse, group_size, q.shape[-1]).mean(dim=3)
                coarse_k = k.view(k.shape[0], k.shape[1], G_coarse, group_size, k.shape[-1]).mean(dim=3)
                coarse_v = v.view(v.shape[0], v.shape[1], G_coarse, group_size, v.shape[-1]).mean(dim=3)
                c_head_dim = coarse_q.shape[-1]
                coarse_score = torch.einsum('bhlk,bhtk->bhlt', [coarse_q, coarse_k]) * (c_head_dim ** -0.5)
                coarse_attn = F.softmax(coarse_score, dim=-1)
                coarse_out = torch.einsum('bhlt,bhtv->bhlv', [coarse_attn, coarse_v])  # (B, H, G, D)
                # Scatter each group token back to its group_size channels via broadcast-expand
                coarse_out = coarse_out.unsqueeze(3).expand(-1, -1, -1, group_size, -1).reshape(x.shape)
                x = x + self.dual_ca_scale * coarse_out

        # Merge heads; for channel mode we also swap (tokens=C -> tokens=H*W), for spatial we keep tokens=H*W
        if self.token_mode == 'channel':
            if G > 1:
                # Group CA: ungroup (B*G, 1, C//G, H*W) -> (B, H*W, C); proj handles cross-group mixing
                x = rearrange(x, '(b g) h t d -> b (h d) (g t)', g=G)
            else:
                # channel tokens -> swap back to spatial tokens: (B, H*W, C)
                x = rearrange(x, 'b h t d -> b (h d) t')
        else:
            # spatial tokens -> standard head merge along feature dim: (B, H*W, C)
            x = rearrange(x, 'b h t d -> b t (h d)')
        _p("Attn.merged_heads", x)

        x = self.proj(x)
        _p("Attn.proj", x)
        x = self.proj_drop(x)

        return x
class ViTTail(nn.Module):
    """Standard ViT tail operating on (B, C, H, W) stage output.
    Adds CLS token + absolute 2D sine-cos positional embeddings, then runs
    a stack of spatial-token blocks with linear QKV.
    Returns a pooled (CLS) feature vector of shape (B, C).
    """
    def __init__(self,
                 embed_dim: int,
                 depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_dw_shortcut: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.pos_drop = nn.Dropout(drop_rate)
        # CLS token for the tail
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        # Build blocks: spatial tokens, linear projections, with CLS
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        for i in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    # Force a standard ViT-style spatial block with linear QKV and CLS token
                    token_mode='spatial',
                    method='linear',
                    with_cls_token=True,
                    use_dw_shortcut=use_dw_shortcut,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        tokens = rearrange(x, 'b c h w -> b (h w) c')  # (B, T, C), T=H*W
        # Build fixed sine-cos pos embeddings on the fly (small cost)
        pos = get_2d_sincos_pos_embed(C, H, W, cls_token=True, device=x.device)  # (1+T, C)
        pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B, 1+T, C)
        # CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+T, C)
        tokens = tokens + pos
        tokens = self.pos_drop(tokens)
        # Run ViT tail blocks
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        # Return normalized CLS vector
        cls_tok = tokens[:, :1, :]
        cls_tok = self.norm(cls_tok)
        return cls_tok.squeeze(1)  # (B, C)

    @staticmethod
    def compute_macs(module, input, output):
        """Estimate MACs for this Attention call.
        Token-mode aware:
          - channel mode  : tokens=C, feature=H*W
          - spatial mode  : tokens=H*W, feature=C
        Includes: Q/K/V depthwise convs, attention (QK^T + AV), and output proj.
        Notes:
          - Counts multiply-add as 1 MAC (common convention in this repo).
          - Ignores Dropout/Softmax/Norm/Activations.
        """
        # Unpack inputs; forward signature is (x, h, w)
        x = input[0]
        try:
            H = int(input[1])
            W = int(input[2])
        except Exception:
            # Fallback if hooks are attached elsewhere; infer from token length
            T = x.shape[1] - (1 if module.with_cls_token else 0)
            H = W = int(np.sqrt(T))

        B, T_in, C = x.shape  # x is (B, H*W{+cls}, C)

        # Utility
        def ceil_div(a, b):
            return (a + b - 1) // b

        # Spatial sizes after striding
        Hq = ceil_div(H, int(module.stride_q))
        Wq = ceil_div(W, int(module.stride_q))
        Hkv = ceil_div(H, int(module.stride_kv))
        Wkv = ceil_div(W, int(module.stride_kv))

        # Tokens & feature dims for attention math
        if getattr(module, 'token_mode', 'channel') == 'channel':
            # Channel-attention: tokens=C, features=Hq*Wq (assume q/k share d)
            Tq = C + (1 if module.with_cls_token else 0)
            Tkv = C + (1 if module.with_cls_token else 0)
            d = Hq * Wq  # total feature dim (num_heads handled implicitly)
        else:
            # Spatial-attention: tokens=H*W (with stride), features=C
            Tq = Hq * Wq + (1 if module.with_cls_token else 0)
            Tkv = Hkv * Wkv + (1 if module.with_cls_token else 0)
            d = C

        flops = 0

        # --- Depthwise conv projections (if present) ---
        def dw_conv_macs(conv, Ho, Wo, Cin):
            # conv weight: (Cin, 1, k, k), groups=Cin
            k = conv.kernel_size[0]
            return Cin * (k * k) * (Ho * Wo)

        Cin = C  # depthwise convs in this module use dim_in==dim_out==C per stage
        if hasattr(module, 'conv_proj_q') and hasattr(module.conv_proj_q, 'conv'):
            flops += dw_conv_macs(module.conv_proj_q.conv, Hq, Wq, Cin)
        if hasattr(module, 'conv_proj_k') and hasattr(module.conv_proj_k, 'conv'):
            flops += dw_conv_macs(module.conv_proj_k.conv, Hkv, Wkv, Cin)
        if hasattr(module, 'conv_proj_v') and hasattr(module.conv_proj_v, 'conv'):
            flops += dw_conv_macs(module.conv_proj_v.conv, Hkv, Wkv, Cin)

        # --- Attention math: QK^T + AV ---
        flops += Tq * Tkv * d  # QK^T
        flops += Tq * d * Tkv  # AV

        # --- Output projection W_o (per token) ---
        # For channel-token mode we set W_o = Identity → no mixing cost.
        if getattr(module, 'token_mode', 'channel') == 'channel':
            pass  # no FLOPs for identity projection
        else:
            # Spatial-token (tail): standard pointwise mixing across channels
            T_out = H * W
            flops += (C * C) * T_out

        module.__flops__ = getattr(module, '__flops__', 0) + int(flops)


#
# NOTE: MLP/FFN usage is now controlled per block via mlp_ratio.
# Set mlp_ratio <= 0 to disable the MLP for that block (common for channel‑attention backbone).
class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']
        self.token_mode = kwargs.get('token_mode', 'channel')
        # Enable/disable MLP via mlp_ratio: ≤0 disables MLP for this block
        self.use_mlp = (mlp_ratio is not None and mlp_ratio > 0.0)
        self.use_dw_shortcut = kwargs.get('use_dw_shortcut', False)
        self.dw_shortcut_after_first_residual = kwargs.get('dw_shortcut_after_first_residual', False)

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        if self.use_mlp:
            self.norm2 = norm_layer(dim_out)
            dim_mlp_hidden = int(dim_out * mlp_ratio)
            _use_swiglu = kwargs.get('use_swiglu', False)
            mlp_cls = SwiGLUMlp if _use_swiglu else Mlp
            self.mlp = mlp_cls(
                in_features=dim_out,
                hidden_features=dim_mlp_hidden,
                act_layer=act_layer,
                drop=drop,
            )
        else:
            # No FFN for channel-token mode to ensure mixing happens only via attention
            self.norm2 = None
            self.mlp = None

        # Layer Scale (CaiT-style): per-block learned scalar for stable deep CA training
        layer_scale_init = kwargs.get('layer_scale_init', None)
        if layer_scale_init is not None:
            self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim_out))
            self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim_out)) if self.use_mlp else None
        else:
            self.gamma1 = None
            self.gamma2 = None

        # Parallel DW conv shortcut (Efficient ViT with DW): gelu -> bn -> 3x3 dw_conv
        if self.use_dw_shortcut:
            self.dw_gelu = nn.GELU()
            self.dw_bn = nn.BatchNorm2d(dim_out)
            dw_k = kwargs.get('dw_shortcut_kernel_size', 3)
            self.dw_conv = nn.Conv2d(dim_out, dim_out, kernel_size=dw_k, stride=1, padding=dw_k // 2, groups=dim_out)
        else:
            self.dw_gelu = None
            self.dw_bn = None
            self.dw_conv = None

    def forward(self, x, h, w, channel_pair_bias=None, bridge_kv=None):
        _p("Block.in", x)
        res = x
        # DW shortcut from block input (before attn/mlp), added at end
        dw_shortcut = None
        if self.use_dw_shortcut and self.dw_conv is not None:
            if self.with_cls_token:
                spatial = res[:, 1:, :]
            else:
                spatial = res
            spatial = rearrange(spatial, 'b (h w) c -> b c h w', h=h, w=w)
            dw_shortcut = self.dw_gelu(spatial)
            dw_shortcut = self.dw_bn(dw_shortcut)
            dw_shortcut = self.dw_conv(dw_shortcut)
            dw_shortcut = rearrange(dw_shortcut, 'b c h w -> b (h w) c')
            if self.with_cls_token:
                cls_pad = torch.zeros(res.shape[0], 1, res.shape[2], device=res.device, dtype=res.dtype)
                dw_shortcut = torch.cat([cls_pad, dw_shortcut], dim=1)

        x = self.norm1(x)
        _p("Block.norm1", x)
        attn = self.attn(x, h, w, channel_pair_bias=channel_pair_bias, bridge_kv=bridge_kv)
        _p("Block.attn_out", attn)
        if self.gamma1 is not None:
            attn = attn * self.gamma1
        x = res + self.drop_path(attn)
        _p("Block.res1", x)
        # DW shortcut: after 1st residual (before MLP) if dw_shortcut_after_first_residual; else after MLP
        if dw_shortcut is not None and self.dw_shortcut_after_first_residual:
            x = x + self.drop_path(dw_shortcut)
        if self.use_mlp:
            x_mlp_in = self.norm2(x)
            _p("Block.norm2", x_mlp_in)
            x_mlp = self.mlp(x_mlp_in)
            _p("Block.mlp", x_mlp)
            if self.gamma2 is not None:
                x_mlp = x_mlp * self.gamma2
            x = x + self.drop_path(x_mlp)
        if dw_shortcut is not None and not self.dw_shortcut_after_first_residual:
            x = x + self.drop_path(dw_shortcut)
        _p("Block.out", x)
        return x



class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None,
                 method: str = 'std'):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.method = method  # 'std' | 'dw_sep' | 'dw_only'
        if self.method == 'dw_sep':
            # Depthwise + Pointwise (mixes channels)
            self.proj_dw = nn.Conv2d(
                in_chans, in_chans,
                kernel_size=patch_size,
                stride=stride,
                padding=padding,
                groups=in_chans,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(in_chans)
            self.proj_pw = nn.Conv2d(
                in_chans, embed_dim,
                kernel_size=1,
                bias=False,
            )
            self.bn_pw = nn.BatchNorm2d(embed_dim)
        elif self.method == 'dw_only':
            # Depthwise-only with channel multiplier (NO channel mixing here)
            # Requires: embed_dim % in_chans == 0
            assert embed_dim % in_chans == 0, (
                f"dw_only requires embed_dim % in_chans == 0; got embed_dim={embed_dim}, in_chans={in_chans}")
            self.proj_dw = nn.Conv2d(
                in_chans, embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=padding,
                groups=in_chans,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(embed_dim)
            self.proj_pw = None
            self.bn_pw = None
        else:
            # Original single conv patch embedding (mixes channels)
            self.proj = nn.Conv2d(
                in_chans, embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=padding
            )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        _p("ConvEmbed.in", x)
        if getattr(self, 'method', 'std') == 'dw_only':
            x = self.proj_dw(x)
            x = self.bn_dw(x)
            _p("ConvEmbed.dw_only", x)
        elif getattr(self, 'method', 'std') == 'dw_sep':
            x = self.proj_dw(x)
            x = self.bn_dw(x)
            _p("ConvEmbed.dw", x)
            x = self.proj_pw(x)
            x = self.bn_pw(x)
            _p("ConvEmbed.pw", x)
        else:
            x = self.proj(x)
            _p("ConvEmbed.proj", x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c') # rearrange into 2D for norm
        _p("ConvEmbed.tokens_pre_norm", x)
        if self.norm:
            x = self.norm(x)
        _p("ConvEmbed.tokens_post_norm", x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W) # rearrange back as flattened patches
        _p("ConvEmbed.out", x)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 sa_tail_depth=0,           # NEW: number of SA-only blocks appended at the tail
                 sa_tail_heads=None,        # NEW: heads for SA tail (defaults to >=4 if None)
                 patch_embed_method: str = 'std',
                 backbone_mlp_ratio: float = None,
                 tail_mlp_ratio: float = None,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            method=patch_embed_method,
        )
        # Allow separate MLP ratios for backbone (channel-attention) and tail (spatial) blocks.
        self.backbone_mlp_ratio = backbone_mlp_ratio if backbone_mlp_ratio is not None else mlp_ratio
        self.tail_mlp_ratio = tail_mlp_ratio if tail_mlp_ratio is not None else mlp_ratio

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Convolutional Position Encoding (CPVT/Twins style, arXiv:2102.10882).
        # DWConv applied to feature map before tokens to inject local spatial context.
        # CPE(x) = x + DWConv(x); init to zero output → identity at start.
        _use_cpe = kwargs.get('use_cpe', False)
        _cpe_kernel = kwargs.get('cpe_kernel_size', 3)
        if _use_cpe:
            self.cpe = nn.Conv2d(embed_dim, embed_dim, kernel_size=_cpe_kernel,
                                 stride=1, padding=_cpe_kernel // 2, groups=embed_dim, bias=True)
            nn.init.zeros_(self.cpe.weight)
            nn.init.zeros_(self.cpe.bias)
        else:
            self.cpe = None

        # Cross-stage channel attention bridge (H3-3): if enabled, this stage projects the
        # previous stage's feature map (prev_embed_dim → embed_dim, spatial pooled to this
        # stage's HW) and exposes it as extra K/V tokens in each block's attention.
        # Init bridge_proj weights to zero → bridge tokens contribute 0 scores at step 0
        # (safe start, baseline behaviour). Model learns to use the bridge during training.
        self._xstage_bridge = bool(kwargs.get('xstage_bridge', False))
        _prev_embed_dim = int(kwargs.get('prev_embed_dim', 0))
        if self._xstage_bridge and _prev_embed_dim > 0:
            self.bridge_proj = nn.Conv2d(_prev_embed_dim, embed_dim, kernel_size=1, bias=False)
            nn.init.zeros_(self.bridge_proj.weight)
            self.bridge_pool = nn.AdaptiveAvgPool2d(None)  # target set at first forward call
        else:
            self.bridge_proj = None
            self.bridge_pool = None
        self._bridge_source = None  # set from outside (CvT.forward_features) before this stage's forward

        # Append SA-only tail blocks at 14x14 for global spatial routing
        blocks = []
        total_depth = depth + (sa_tail_depth or 0)
        backbone_token_mode = kwargs.get('backbone_token_mode', 'channel')  # 'channel' or 'spatial'
        # H3-14: optional per-block Linformer rank override at this stage. Consume here so it
        # doesn't propagate as an unknown kwarg into Block/Attention.
        linformer_r_per_block = kwargs.pop('linformer_r_per_block', None) or []
        base_linformer_r = int(kwargs.get('linformer_r', 0))
        for j in range(total_depth):
            is_tail_sa = (j >= depth)
            # token_mode: tail always spatial; backbone uses backbone_token_mode (channel=dim swap, spatial=no swap)
            token_mode = 'spatial' if is_tail_sa else backbone_token_mode
            nh = (sa_tail_heads if sa_tail_heads is not None else max(4, num_heads)) if is_tail_sa else num_heads
            local_mlp_ratio = self.tail_mlp_ratio if is_tail_sa else self.backbone_mlp_ratio
            # Per-block Linformer rank: override base when per-block list is present and block is backbone.
            if linformer_r_per_block and (not is_tail_sa) and j < len(linformer_r_per_block):
                kwargs['linformer_r'] = int(linformer_r_per_block[j])
            else:
                kwargs['linformer_r'] = base_linformer_r
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=nh,
                    mlp_ratio=local_mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[min(j, len(dpr)-1)],  # reuse last drop_path for tail blocks
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    token_mode=token_mode,  # 'channel' backbone, 'spatial' tail
                    **kwargs
                )
            )
        # Restore base linformer_r in kwargs (other downstream code reads it later).
        kwargs['linformer_r'] = base_linformer_r
        self.blocks = nn.ModuleList(blocks)

        # Shared per-stage learnable channel pair bias (C×C), added to attention logits.
        # Initialized to zero so the model starts with the same behavior as without bias.
        # Excluded from weight decay (it's a bias/positional-prior term, not a weight).
        use_channel_pair_bias = kwargs.get('use_channel_pair_bias', False)
        backbone_token_mode = kwargs.get('backbone_token_mode', 'channel')
        if use_channel_pair_bias and backbone_token_mode == 'channel':
            self.channel_pair_bias = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        else:
            self.channel_pair_bias = None

        # Mid-block auxiliary supervision: save intermediate feature map after block mid_block_idx.
        # Set to -1 (default) to disable. Only meaningful for the last stage.
        self._mid_block_idx = int(kwargs.get('mid_block_idx', -1))
        self._mid_block_feat = None  # populated during forward when mid_block_idx >= 0

        # Periodic CPE re-injection: reapply the CPE DW conv every N blocks within this stage.
        # After N attention blocks, channel tokens may drift from their original spatial structure;
        # re-injecting CPE (5×5 DW conv) refreshes spatial context for channel attention Q/K.
        # 0 = disabled (default). Meaningful when CPE is also enabled (use_cpe=True).
        self._cpe_repeat_interval = int(kwargs.get('cpe_repeat_interval', 0))

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        _p("ViT.in", x)
        x = self.patch_embed(x)
        _p("ViT.after_patch_embed", x)
        B, C, H, W = x.size()
        _p("ViT.spatial_HW", torch.tensor([H, W]))

        # CPE: inject local spatial context before channel attention tokenization
        if self.cpe is not None:
            x = x + self.cpe(x)
        _p("ViT.after_cpe", x)

        x = rearrange(x, 'b c h w -> b (h w) c')
        _p("ViT.tokens", x)

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            _p("ViT.tokens_with_cls", x)

        x = self.pos_drop(x)
        _p("ViT.pos_drop", x)

        cpb = self.channel_pair_bias  # None if not enabled; shared across all blocks in stage
        self._mid_block_feat = None

        # Cross-stage bridge (H3-3): build once per forward; shared across all blocks in this stage.
        # Bridge shape: (B, embed_dim, H*W) — same spatial as current stage, same channel dim.
        bridge_kv = None
        if self.bridge_proj is not None and self._bridge_source is not None:
            src = self._bridge_source  # (B, prev_embed_dim, H_prev, W_prev)
            bridge_feat = self.bridge_proj(src)  # → (B, embed_dim, H_prev, W_prev)
            bridge_feat = F.adaptive_avg_pool2d(bridge_feat, (H, W))  # align to current HW
            bridge_kv = rearrange(bridge_feat, 'b c h w -> b c (h w)')  # (B, C, HW)
            self._bridge_source = None  # clear to avoid leaking into next forward

        for i, blk in enumerate(self.blocks):
            _p(f"ViT.block[{i}].in", x)
            x = blk(x, H, W, channel_pair_bias=cpb, bridge_kv=bridge_kv)
            _p(f"ViT.block[{i}].out", x)
            # Mid-block auxiliary supervision: capture feature map at designated block index
            if self._mid_block_idx >= 0 and i == self._mid_block_idx:
                x_spatial = x[:, 1:, :] if (self.cls_token is not None) else x
                self._mid_block_feat = rearrange(x_spatial, 'b (h w) c -> b c h w', h=H, w=W)
            # Periodic CPE re-injection: reapply spatial context every cpe_repeat_interval blocks.
            # Ensures channel tokens retain spatial structure through the deep final stage.
            # Triggered after block i (0-indexed) when (i+1) is a multiple of the interval
            # and there are still blocks remaining.
            if (self._cpe_repeat_interval > 0 and self.cpe is not None
                    and (i + 1) % self._cpe_repeat_interval == 0
                    and (i + 1) < len(self.blocks)):
                x_spatial = x[:, 1:, :] if (self.cls_token is not None) else x
                x_bchw = rearrange(x_spatial, 'b (h w) c -> b c h w', h=H, w=W)
                x_bchw = x_bchw + self.cpe(x_bchw)
                x_new = rearrange(x_bchw, 'b c h w -> b (h w) c')
                if self.cls_token is not None:
                    x = torch.cat([x[:, :1, :], x_new], dim=1)
                else:
                    x = x_new

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H*W], 1)
            _p("ViT.split.cls_token", cls_tokens)
            _p("ViT.split.tokens", x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        _p("ViT.out_feature_map", x)

        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        tail_type = spec.get('TAIL_TYPE', 'sa')  # 'sa' (default) or 'vit'
        self.tail_type = tail_type
        # SA tail defaults: append 3 spatial-attention blocks to the last stage by default
        sa_tail_depths = spec.get('SA_TAIL_DEPTH', [0] * (self.num_stages - 1) + [3])
        # Use multi-head for SA tail (>=4); default to stage heads, clamped to at least 4
        default_tail_heads = spec['NUM_HEADS'][:-1] + [max(4, spec['NUM_HEADS'][-1])]
        sa_tail_heads = spec.get('SA_TAIL_NUM_HEADS', default_tail_heads)
        patch_embed_methods = spec.get('PATCH_EMBED_METHOD', ['std'] * self.num_stages)
        def _get_ratio(key, i, fallback_key='MLP_RATIO'):
            v = spec.get(key, None)
            if v is None:
                v = spec.get(fallback_key, 4.0)
            if isinstance(v, (list, tuple)):
                return v[i]
            return float(v)

        def _get_per_stage_int(spec, key, i, default=1):
            v = spec.get(key, default)
            if isinstance(v, (list, tuple)):
                return int(v[i]) if i < len(v) else default
            return int(v)

        for i in range(self.num_stages):
            backbone_tm = spec.get('BACKBONE_TOKEN_MODE', 'channel')
            if isinstance(backbone_tm, (list, tuple)):
                backbone_tm = backbone_tm[i] if i < len(backbone_tm) else 'channel'
            # DW_SHORTCUT: enable parallel 3x3 depthwise conv shortcut (Efficient ViT with DW)
            # DW_SHORTCUT_BACKBONE / DW_SHORTCUT_TAIL override DW_SHORTCUT when set
            dw_shortcut = spec.get('DW_SHORTCUT', False)
            if isinstance(dw_shortcut, (list, tuple)):
                dw_shortcut = dw_shortcut[i] if i < len(dw_shortcut) else False
            dw_shortcut_backbone = spec.get('DW_SHORTCUT_BACKBONE', dw_shortcut)
            if isinstance(dw_shortcut_backbone, (list, tuple)):
                dw_shortcut_backbone = dw_shortcut_backbone[i] if i < len(dw_shortcut_backbone) else dw_shortcut
            dw_shortcut_tail = spec.get('DW_SHORTCUT_TAIL', dw_shortcut)
            dw_shortcut_after_first = spec.get('DW_SHORTCUT_AFTER_FIRST_RESIDUAL', False)

            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'backbone_token_mode': backbone_tm,
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
                'patch_embed_method': patch_embed_methods[i],
                'use_dw_shortcut': dw_shortcut_backbone,
                'dw_shortcut_after_first_residual': dw_shortcut_after_first,
                'layer_scale_init': spec.get('LAYER_SCALE', None),
                'use_channel_pair_bias': spec.get('CHANNEL_PAIR_BIAS', False),
                'dw_shortcut_kernel_size': spec.get('DW_SHORTCUT_KERNEL', [3, 3, 3, 3])[i],
                'use_cpe': spec.get('CPE', False),
                'cpe_kernel_size': spec.get('CPE_KERNEL_SIZE', 3),
                'use_swiglu': spec.get('SWIGLU', False),
                'num_channel_groups': _get_per_stage_int(spec, 'CHANNEL_GROUPS', i, 1),
                'sparse_topk': _get_per_stage_int(spec, 'SPARSE_TOPK', i, 0),
                'dual_ca_groups': _get_per_stage_int(spec, 'DUAL_CA_GROUPS', i, 0),
                'xstage_bridge': bool(spec.get('XSTAGE_BRIDGE', False)) and (i == self.num_stages - 1),
                'prev_embed_dim': spec['DIM_EMBED'][i - 1] if i > 0 else 0,
                'agent_tokens': _get_per_stage_int(spec, 'AGENT_TOKENS', i, 0),
                'agent_spatial_hw': _get_per_stage_int(spec, 'AGENT_SPATIAL_HW', i, 0),
                'hier_groups': _get_per_stage_int(spec, 'HIER_GROUPS', i, 0),
                # H3-12: per-stage Linformer rank for in-trunk CA blocks (>0 → Linformer; 0 → standard CA).
                'linformer_r': _get_per_stage_int(spec, 'LINFORMER_R_PER_STAGE', i, 0),
                # H3-14: per-block rank override at the final stage only. Non-empty list overrides
                # the per-stage value block-by-block. Empty everywhere else.
                'linformer_r_per_block': list(spec.get('LINFORMER_R_PER_BLOCK_S3', []))
                    if i == (self.num_stages - 1) else [],
                # H3-15: per-stage Linformer top-k mask inside the supertoken basis (0 = dense).
                'linformer_topk': _get_per_stage_int(spec, 'LINFORMER_TOPK_PER_STAGE', i, 0),
                # H3-16: per-stage flag for parallel full-rank CA pathway (init-0 gated additive).
                'linformer_fullrank_parallel': bool(_get_per_stage_int(spec, 'LINFORMER_FULLRANK_PARALLEL_PER_STAGE', i, 0)),
                # H3-17: per-stage normalisation function inside use_linformer branch.
                'linformer_softmax_type': str(
                    (spec.get('LINFORMER_SOFTMAX_TYPE_PER_STAGE', ['softmax'] * self.num_stages)[i])
                    if i < len(spec.get('LINFORMER_SOFTMAX_TYPE_PER_STAGE', []))
                    else 'softmax'
                ),
                # H3-18: Synthesizer-Dense per-stage enable + per-stage spatial HW (to size W matrix).
                'synthesizer_dense': bool(_get_per_stage_int(spec, 'SYNTHESIZER_DENSE_PER_STAGE', i, 0)),
                'synthesizer_hw': _get_per_stage_int(spec, 'SYNTHESIZER_HW_PER_STAGE', i, 0),
                # H3-19: Nyström landmark-based attention per-stage enable + iter count for pinv.
                'nystrom_L': _get_per_stage_int(spec, 'NYSTROM_L_PER_STAGE', i, 0),
                'nystrom_iters': _get_per_stage_int(spec, 'NYSTROM_ITERS_PER_STAGE', i, 6),
                'expected_spatial_hw': None,  # filled by VisionTransformer at runtime (first forward)
                # Mid-block aux: only enabled for the final stage; -1 disables in all others
                'mid_block_idx': int(spec.get('AUX_MID_BLOCK_IDX', 5)) if (i == self.num_stages - 1 and float(spec.get('AUX_LOSS_WEIGHT_MID', 0.0)) > 0) else -1,
                'cpe_repeat_interval': _get_per_stage_int(spec, 'CPE_REPEAT_INTERVAL', i, 0),
            }

            backbone_mlp_ratio_i = _get_ratio('CA_MLP_RATIO', i, 'MLP_RATIO')
            tail_mlp_ratio_i = _get_ratio('TAIL_MLP_RATIO', i, 'MLP_RATIO')

            if (i == self.num_stages - 1) and (tail_type == 'vit') and (sa_tail_depths[i] > 0):
                # Build backbone with NO SA tail; attach a separate ViT tail below
                stage = VisionTransformer(
                    in_chans=in_chans,
                    init=init,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    sa_tail_depth=0,                 # no SA blocks inside the stage
                    sa_tail_heads=sa_tail_heads[i],  # ignored here
                    backbone_mlp_ratio=backbone_mlp_ratio_i,
                    tail_mlp_ratio=0.0,
                    **kwargs
                )
                setattr(self, f'stage{i}', stage)
                # Create a standalone ViT tail operating at this stage's spatial size
                self.vit_tail = ViTTail(
                    embed_dim=spec['DIM_EMBED'][i],
                    depth=sa_tail_depths[i],
                    num_heads=sa_tail_heads[i] if sa_tail_heads[i] is not None else max(4, spec['NUM_HEADS'][i]),
                    mlp_ratio=tail_mlp_ratio_i,
                    qkv_bias=spec['QKV_BIAS'][i],
                    drop_rate=spec['DROP_RATE'][i],
                    attn_drop_rate=spec['ATTN_DROP_RATE'][i],
                    drop_path_rate=spec['DROP_PATH_RATE'][i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    use_dw_shortcut=dw_shortcut_tail,
                )
            else:
                stage = VisionTransformer(
                    in_chans=in_chans,
                    init=init,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    sa_tail_depth=sa_tail_depths[i],      # SA-only blocks in-stage
                    sa_tail_heads=sa_tail_heads[i],
                    backbone_mlp_ratio=backbone_mlp_ratio_i,
                    tail_mlp_ratio=tail_mlp_ratio_i,
                    **kwargs
                )
                setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        # If using a standalone ViT tail, forward_features() returns early and bypasses this top-level norm.
        # Disable grads to prevent DDP "unused parameters" errors.
        if self.tail_type == 'vit':
            for p in self.norm.parameters():
                p.requires_grad = False
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)

        # Cross-stage channel fusion (H4 — DuoFormer-inspired, 2024).
        # After the backbone, collect GAP features from intermediate stages (1, 2) and fuse with
        # the final stage-3 representation via learned additive projections (scale init=0 → identity at start).
        # Config key: CROSS_STAGE_FUSION: true
        self._cross_stage_fusion = bool(spec.get('CROSS_STAGE_FUSION', False))
        if self._cross_stage_fusion and self.num_stages >= 3:
            # Project stage 1 (dim[1]) and stage 2 (dim[-2]) to final dim
            self.fuse_proj_s1 = nn.Linear(spec['DIM_EMBED'][1], dim_embed, bias=False)
            self.fuse_scale_s1 = nn.Parameter(torch.zeros(dim_embed))  # LayerScale-style init=0
            if self.num_stages >= 4:
                self.fuse_proj_s2 = nn.Linear(spec['DIM_EMBED'][2], dim_embed, bias=False)
                self.fuse_scale_s2 = nn.Parameter(torch.zeros(dim_embed))
            else:
                self.fuse_proj_s2 = None
                self.fuse_scale_s2 = None
            # P2-H22: also fuse stage 0 (lowest-level features: edges/textures at 56×56).
            # DuoFormer (arXiv:2407.13920) showed all-stage aggregation > partial. scale init=0.
            if bool(spec.get('CROSS_STAGE_FUSION_S0', False)):
                self.fuse_proj_s0 = nn.Linear(spec['DIM_EMBED'][0], dim_embed, bias=False)
                self.fuse_scale_s0 = nn.Parameter(torch.zeros(dim_embed))
            else:
                self.fuse_proj_s0 = None
                self.fuse_scale_s0 = None
        else:
            self.fuse_proj_s0 = None
            self.fuse_proj_s1 = None
            self.fuse_proj_s2 = None
        self._fuse_s0_feat = None  # populated in forward_features (P2-H22)
        self._fuse_s1_feat = None
        self._fuse_s2_feat = None

        # Auxiliary classification head — applied to stage (num_stages-2) output during training.
        # Provides deep supervision (Lee et al. 2015; GoogLeNet auxiliary classifiers).
        # Set AUX_LOSS_WEIGHT > 0 in spec to enable.  Follows the LOBT side-effect pattern.
        self._aux_loss_weight = float(spec.get('AUX_LOSS_WEIGHT', 0.0))
        self._aux_stage_idx = max(0, self.num_stages - 2)  # default: second-to-last stage
        if self._aux_loss_weight > 0 and num_classes > 0:
            aux_dim = spec['DIM_EMBED'][self._aux_stage_idx]
            self.aux_head = nn.Linear(aux_dim, num_classes)
            trunc_normal_(self.aux_head.weight, std=0.02)
        else:
            self.aux_head = None
        self._aux_logits = None  # populated in forward_features during training

        # Second auxiliary head at stage (num_stages-3) — earlier supervision for deeper gradient flow.
        # Set AUX_LOSS_WEIGHT_S1 > 0 to enable.  Complementary to aux_head (H50).
        self._aux_s1_weight = float(spec.get('AUX_LOSS_WEIGHT_S1', 0.0))
        self._aux_s1_stage_idx = max(0, self.num_stages - 3)
        if self._aux_s1_weight > 0 and num_classes > 0 and self.num_stages >= 3:
            aux_s1_dim = spec['DIM_EMBED'][self._aux_s1_stage_idx]
            self.aux_head_s1 = nn.Linear(aux_s1_dim, num_classes)
            trunc_normal_(self.aux_head_s1.weight, std=0.02)
        else:
            self.aux_head_s1 = None
        self._aux_logits_s1 = None

        # Third auxiliary head at stage 0 (H64) — earliest supervision signal.
        self._aux_s0_weight = float(spec.get('AUX_LOSS_WEIGHT_S0', 0.0))
        self._aux_s0_stage_idx = max(0, self.num_stages - 4)
        if self._aux_s0_weight > 0 and num_classes > 0 and self.num_stages >= 4:
            aux_s0_dim = spec['DIM_EMBED'][self._aux_s0_stage_idx]
            self.aux_head_s0 = nn.Linear(aux_s0_dim, num_classes)
            trunc_normal_(self.aux_head_s0.weight, std=0.02)
        else:
            self.aux_head_s0 = None
        self._aux_logits_s0 = None

        # Intra-stage-3 mid-block auxiliary head (P2-H9): supervision signal at block mid_block_idx
        # within the final stage. Forces mid-stage channel representations to be discriminative,
        # providing a direct gradient shortcut through the deep final stage (12 blocks).
        self._aux_mid_weight = float(spec.get('AUX_LOSS_WEIGHT_MID', 0.0))
        self._aux_mid_block_idx = int(spec.get('AUX_MID_BLOCK_IDX', 5))
        if self._aux_mid_weight > 0 and num_classes > 0:
            aux_mid_dim = spec['DIM_EMBED'][-1]  # final stage dim
            self.aux_head_mid = nn.Linear(aux_mid_dim, num_classes)
            trunc_normal_(self.aux_head_mid.weight, std=0.02)
        else:
            self.aux_head_mid = None
        self._aux_logits_mid = None

        # H3-6: Funneling tail-block. After all stages, take the final-stage spatial feature
        # (B, C, 7, 7) and additively refine it through a SINGLE channel-attention block at
        # extreme spatial compression (AdaptiveAvgPool2d → fhw×fhw, fhw=3 → D=9 features).
        # Captures rank-≤9 global channel relationships at a qualitatively different scale than
        # stage-3's 7×7 (rank-49). Output GAP'd and gate-added (init 0) to the classifier feature.
        # Existing stages 0-3 are completely UNCHANGED — purely additive structural innovation.
        self.funnel_tail = bool(spec.get('FUNNEL_TAIL', False))
        if self.funnel_tail:
            fhw = int(spec.get('FUNNEL_TAIL_HW', 3))
            f_dim = dim_embed
            f_mlp_ratio = float(spec.get('FUNNEL_TAIL_MLP_RATIO', 0.5))
            f_kernel = int(spec.get('FUNNEL_TAIL_KERNEL', 1))
            self.funnel_pool = nn.AdaptiveAvgPool2d(fhw)
            self._funnel_hw = fhw
            funnel_block_kwargs = {
                'with_cls_token': False,
                'token_mode': 'channel',
                'use_dw_shortcut': False,
                'use_swiglu': bool(spec.get('SWIGLU', False)),
                'layer_scale_init': spec.get('LAYER_SCALE', None),
                'use_cpe': False,  # spatial too small for CPE
                'num_channel_groups': 1,
                'sparse_topk': 0,
                'dual_ca_groups': 0,
                'xstage_bridge': False,
                'agent_tokens': 0,
                'agent_spatial_hw': 0,
                'hier_groups': 0,
                'mid_block_idx': -1,
                'cpe_repeat_interval': 0,
                'linear_attn': bool(spec.get('LINEAR_FUNNEL', False)),  # H3-8
                'linformer_r': int(spec.get('LINFORMER_R', 0)),  # H3-9
            }
            f_depth = max(1, int(spec.get('FUNNEL_TAIL_DEPTH', 1)))
            # H3-10: stack f_depth funnel blocks. Each gets its own attn (incl. linformer E_k, E_v).
            self.funnel_blocks = nn.ModuleList([
                Block(
                    dim_in=f_dim,
                    dim_out=f_dim,
                    num_heads=spec['NUM_HEADS'][-1],
                    mlp_ratio=f_mlp_ratio,
                    qkv_bias=spec['QKV_BIAS'][-1],
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    method='dw_bn',
                    kernel_size=f_kernel,
                    stride_kv=1,
                    stride_q=1,
                    padding_kv=f_kernel // 2,
                    padding_q=f_kernel // 2,
                    **funnel_block_kwargs,
                )
                for _ in range(f_depth)
            ])
            # Backward-compat alias: single-block paths still resolve via .funnel_block.
            self.funnel_block = self.funnel_blocks[0]
            # H3-11: optional parallel funnel block with a DIFFERENT operator config.
            # Built when FUNNEL_PARALLEL_BLOCK is True. Its output is averaged with the sequential
            # funnel_blocks chain's output before funnel_norm. Tests whether operator DIVERSITY
            # (e.g., Linformer + sparse top-k) compounds where REPLICATION (H3-10) did not.
            if bool(spec.get('FUNNEL_PARALLEL_BLOCK', False)):
                parallel_kwargs = dict(funnel_block_kwargs)
                parallel_kwargs['linformer_r'] = int(spec.get('FUNNEL_PARALLEL_LINFORMER_R', 0))
                parallel_kwargs['sparse_topk'] = int(spec.get('FUNNEL_PARALLEL_SPARSE_TOPK', 0))
                self.funnel_block_parallel = Block(
                    dim_in=f_dim,
                    dim_out=f_dim,
                    num_heads=spec['NUM_HEADS'][-1],
                    mlp_ratio=f_mlp_ratio,
                    qkv_bias=spec['QKV_BIAS'][-1],
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    method='dw_bn',
                    kernel_size=f_kernel,
                    stride_kv=1,
                    stride_q=1,
                    padding_kv=f_kernel // 2,
                    padding_q=f_kernel // 2,
                    **parallel_kwargs,
                )
            else:
                self.funnel_block_parallel = None
            self.funnel_norm = norm_layer(f_dim)
            self.funnel_scale = nn.Parameter(torch.zeros(f_dim))  # LayerScale-style init=0 → identity at init
            # H3-7: aux supervision on funnel GAP feature via the SHARED main classifier head.
            # Provides independent gradient signal so the funnel block can learn discriminative features
            # aligned with the main head's space, even while funnel_scale stays small.
            self._funnel_aux_weight = float(spec.get('FUNNEL_AUX_WEIGHT', 0.0))
        else:
            self.funnel_pool = None
            self.funnel_block = None
            self.funnel_blocks = None
            self.funnel_block_parallel = None
            self.funnel_norm = None
            self.funnel_scale = None
            self._funnel_aux_weight = 0.0
        self._aux_logits_funnel = None  # populated in forward_features during training

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] == '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                            .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = torch.tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')
            layers.add(f'stage{i}.channel_pair_bias')  # bias/positional-prior: no weight decay
        # Cross-stage fusion scale params: 1D learnable scales, exclude from weight decay
        layers.add('fuse_scale_s0')
        layers.add('fuse_scale_s1')
        layers.add('fuse_scale_s2')
        layers.add('funnel_scale')  # H3-6: 1D learnable scale, no weight decay
        return layers

    def forward_features(self, x):
        _p("CvT.forward_features.in", x)
        self._aux_logits = None  # reset
        self._aux_logits_s1 = None  # reset stage-1 aux logits
        self._aux_logits_s0 = None  # reset stage-0 aux logits
        self._aux_logits_mid = None  # reset mid-block aux logits
        self._aux_logits_funnel = None  # H3-7: reset funnel aux logits
        self._fuse_s0_feat = None  # reset cross-stage fusion buffers
        self._fuse_s1_feat = None
        self._fuse_s2_feat = None
        prev_stage_feat = None  # H3-3: holds previous stage OUTPUT for cross-stage bridge
        for i in range(self.num_stages):
            _p(f"CvT.stage[{i}].in", x)
            stage_module = getattr(self, f'stage{i}')
            # H3-3 cross-stage bridge: if this stage has a bridge, provide previous stage's output
            if getattr(stage_module, 'bridge_proj', None) is not None and prev_stage_feat is not None:
                stage_module._bridge_source = prev_stage_feat
            x, cls_tokens = stage_module(x)
            prev_stage_feat = x  # stage-i output, becomes stage-(i+1)'s bridge source
            if cls_tokens is not None:
                _p(f"CvT.stage[{i}].cls", cls_tokens)
            _p(f"CvT.stage[{i}].out", x)
            # Cross-stage fusion: collect GAP features from stages 0, 1, and 2
            if self._cross_stage_fusion:
                if i == 0 and self.fuse_proj_s0 is not None:
                    self._fuse_s0_feat = x.mean(dim=[-2, -1])  # (B, dim[0])
                elif i == 1 and self.fuse_proj_s1 is not None:
                    self._fuse_s1_feat = x.mean(dim=[-2, -1])  # (B, dim[1])
                elif i == 2 and self.fuse_proj_s2 is not None:
                    self._fuse_s2_feat = x.mean(dim=[-2, -1])  # (B, dim[2])
            # Deep supervision: compute auxiliary logits from aux_stage output during training
            if (self.training and self.aux_head is not None and i == self._aux_stage_idx):
                # x is (B, C, H, W) at this point; apply GAP then auxiliary classifier
                aux_feat = x.mean(dim=[-2, -1])  # (B, C) global average pool
                self._aux_logits = self.aux_head(aux_feat)
            # Earlier auxiliary supervision (stage 1, H50)
            if (self.training and self.aux_head_s1 is not None and i == self._aux_s1_stage_idx):
                aux_feat_s1 = x.mean(dim=[-2, -1])
                self._aux_logits_s1 = self.aux_head_s1(aux_feat_s1)
            # Earliest auxiliary supervision (stage 0, H64)
            if (self.training and self.aux_head_s0 is not None and i == self._aux_s0_stage_idx):
                aux_feat_s0 = x.mean(dim=[-2, -1])
                self._aux_logits_s0 = self.aux_head_s0(aux_feat_s0)
            # Intra-stage mid-block auxiliary supervision (P2-H9): read from last stage's _mid_block_feat
            if (self.training and self.aux_head_mid is not None and i == self.num_stages - 1):
                last_stage = getattr(self, f'stage{i}')
                if last_stage._mid_block_feat is not None:
                    mid_feat = last_stage._mid_block_feat.mean(dim=[-2, -1])
                    self._aux_logits_mid = self.aux_head_mid(mid_feat)

        # H3-6: capture final-stage spatial feature (B, C, 7, 7) for funneling tail BEFORE GAP/norm.
        # Stored locally; consumed after the main classifier feature is computed below.
        last_stage_spatial = x if (self.funnel_tail and self.funnel_block is not None) else None

        # If a standalone ViT tail exists, use its CLS output and bypass GAP/CLS from the backbone
        if hasattr(self, 'vit_tail') and (self.vit_tail is not None):
            x_tail = self.vit_tail(x)  # (B, C)
            return x_tail

        _p("CvT.before_head_prep.in", x)
        if self.cls_token:
            x = self.norm(cls_tokens)
            _p("CvT.norm_cls", x)
            x = torch.squeeze(x)
            _p("CvT.squeeze_cls", x)
        else:
            _p("CvT.tokens_no_cls", x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            _p("CvT.tokens_flat", x)
            x = self.norm(x)
            _p("CvT.tokens_norm", x)
            x = torch.mean(x, dim=1)
            _p("CvT.tokens_gap", x)

        # Cross-stage channel fusion: add projected intermediate stage features with learned scales
        # Scale initialized to 0 → zero contribution at init (safe, doesn't disrupt baseline)
        if self._cross_stage_fusion:
            if self._fuse_s0_feat is not None:
                x = x + self.fuse_scale_s0 * self.fuse_proj_s0(self._fuse_s0_feat)
            if self._fuse_s1_feat is not None:
                x = x + self.fuse_scale_s1 * self.fuse_proj_s1(self._fuse_s1_feat)
            if self._fuse_s2_feat is not None:
                x = x + self.fuse_scale_s2 * self.fuse_proj_s2(self._fuse_s2_feat)

        # H3-6: Funneling tail. Operate on the captured (B, C, 7, 7) spatial feature, pool to 3×3,
        # run through a single channel-attention block, GAP, gate-add to classifier input.
        # H3-7: also expose funnel GAP feature as aux logits via shared main head when training.
        if last_stage_spatial is not None:
            fp = self.funnel_pool(last_stage_spatial)              # (B, C, fhw, fhw)
            B_, C_, fH, fW = fp.shape
            fp_tokens_in = rearrange(fp, 'b c h w -> b (h w) c')   # (B, fhw², C)
            # H3-10: chain through stacked funnel blocks (1+ blocks).
            fp_tokens = fp_tokens_in
            for blk in self.funnel_blocks:
                fp_tokens = blk(fp_tokens, fH, fW)
            # H3-11: parallel branch with a different operator (e.g., sparse top-k). Average outputs.
            if self.funnel_block_parallel is not None:
                fp_parallel = self.funnel_block_parallel(fp_tokens_in, fH, fW)
                fp_tokens = 0.5 * (fp_tokens + fp_parallel)
            fp_tokens = self.funnel_norm(fp_tokens)
            f_feat = torch.mean(fp_tokens, dim=1)                  # (B, C)
            # H3-7: aux supervision on funnel GAP feature (shared main head).
            if self.training and self._funnel_aux_weight > 0:
                self._aux_logits_funnel = self.head(f_feat)
            x = x + self.funnel_scale * f_feat

        return x

    def forward(self, x):
        _p("CvT.forward.in", x)
        x = self.forward_features(x)
        _p("CvT.forward.features", x)
        x = self.head(x)
        _p("CvT.head", x)

        return x


@register_model
def get_cls_model(config, **kwargs):
    msvit_spec = config.MODEL.SPEC

    # Select norm layer based on NORM_LAYER config key (default: layernorm)
    _norm_type = getattr(msvit_spec, 'NORM_LAYER', 'layernorm').lower()
    if _norm_type == 'rmsnorm':
        _norm_layer = partial(RMSNorm, eps=1e-6)
    else:
        _norm_layer = partial(LayerNorm, eps=1e-5)

    msvit = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=config.MODEL.NUM_CLASSES,
        act_layer=QuickGELU,
        norm_layer=_norm_layer,
        init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
        spec=msvit_spec
    )

    if config.MODEL.INIT_WEIGHTS:
        msvit.init_weights(
            config.MODEL.PRETRAINED,
            config.MODEL.PRETRAINED_LAYERS,
            config.VERBOSE
        )

    return msvit

# Alias used by some harness build scripts
def build_cls_cvt(config, **kwargs):
    return get_cls_model(config, **kwargs)


# --- Minimal test main -------------------------------------------------------
if __name__ == "__main__":
    # Minimal runnable demo matching your YAML
    # Prefer fast attention & matmul backends when available
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    spec = {
        'INIT': 'trunc_norm',
        'NUM_STAGES': 3,
        'PATCH_SIZE': [7, 3, 3],
        'PATCH_STRIDE': [4, 2, 2],
        'PATCH_PADDING': [2, 1, 1],
        'DIM_EMBED': [96, 192, 384],
        'NUM_HEADS': [1, 1, 1],
        # Append SA-only tail blocks per stage (0,0,3) → 3 extra SA blocks at Stage-3
        'SA_TAIL_DEPTH': [0, 0, 4],
        # Heads for SA tail blocks; keep stages 1–2 as is, use 8 heads for Stage-3 tail
        'SA_TAIL_NUM_HEADS': [1, 1, 4],
        # TAIL_TYPE: 'vit' (standalone ViT tail) or 'sa' (SA blocks appended to last stage)
        'DEPTH': [1, 1, 1],
        'PATCH_EMBED_METHOD': ['dw_sep', 'dw_sep', 'dw_sep'],
        'MLP_RATIO': [4.0, 4.0, 4.0],
        'CA_MLP_RATIO': [1.0, 1.0, 1.0],      # per-stage override for CA blocks
        'TAIL_MLP_RATIO': [4.0, 4.0, 4.0],    # per-stage override for tail blocks
        'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
        'DROP_RATE': [0.0, 0.0, 0.0],
        'DROP_PATH_RATE': [0.0, 0.0, 0.1],
        'QKV_BIAS': [True, True, True],
        'CLS_TOKEN': [False, False, False],
        'POS_EMBED': [False, False, False],
        'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
        'KERNEL_QKV': [3, 3, 3],
        'PADDING_KV': [1, 1, 1],
        'STRIDE_KV': [1, 1, 1],
        'PADDING_Q': [1, 1, 1],
        'STRIDE_Q': [1, 1, 1],
        'PATCH_EMBED_METHOD': ['dw_only', 'dw_only', 'dw_only'],
    }
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=1000,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        init=spec.get('INIT', 'trunc_norm'),
        spec=spec
    )
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    _p("MAIN.input", x)
    with torch.no_grad():
        y = model(x)
    _p("MAIN.output", y)
