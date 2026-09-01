# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Tianxiao Zhang
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Support for Microsoft-style CvT init helpers
_C.MODEL.INIT_WEIGHTS = False
_C.MODEL.PRETRAINED_LAYERS = ['*']
# Verbosity flag used by some loaders
_C.VERBOSE = False
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True


# ViT parameters
_C.MODEL.ViT = CN()
_C.MODEL.ViT.PATCH_SIZE = 16
_C.MODEL.ViT.DIM = 192
_C.MODEL.ViT.DEPTHS = 12
_C.MODEL.ViT.NUM_HEADS = 3
_C.MODEL.ViT.MLP_DIM = 768
_C.MODEL.ViT.DIM_HEAD = 192 // 3

# ViT-S parameters
_C.MODEL.ViT_S = CN()
_C.MODEL.ViT_S.PATCH_SIZE = 16
_C.MODEL.ViT_S.DIM = 384
_C.MODEL.ViT_S.DEPTHS = 12
_C.MODEL.ViT_S.NUM_HEADS = 6
_C.MODEL.ViT_S.MLP_DIM = 1536
_C.MODEL.ViT_S.DIM_HEAD = 384 // 6


# CaiT XXS parameters
_C.MODEL.CaiT_XXS = CN()
_C.MODEL.CaiT_XXS.PATCH_SIZE = 16
_C.MODEL.CaiT_XXS.DIM = 192
_C.MODEL.CaiT_XXS.DEPTHS = 12
_C.MODEL.CaiT_XXS.CLS_DEPTHS = 2
_C.MODEL.CaiT_XXS.NUM_HEADS = 4
_C.MODEL.CaiT_XXS.MLP_DIM = 192 * 4

_C.MODEL.CaiT_XXS.DIM_HEAD = 192 // 4

# CvT parameters (lucidrains implementation; MODEL_NAME kept for completeness)
_C.MODEL.CVT = CN()
_C.MODEL.CVT.MODEL_NAME = 'cvt_13'
# Microsoft-style CvT SPEC subtree (used by cls_cvt builder)
_C.MODEL.SPEC = CN()
_C.MODEL.SPEC.INIT = 'trunc_norm'
_C.MODEL.SPEC.NUM_STAGES = 3
_C.MODEL.SPEC.PATCH_SIZE = [7, 3, 3]
_C.MODEL.SPEC.PATCH_STRIDE = [4, 2, 2]
_C.MODEL.SPEC.PATCH_PADDING = [2, 1, 1]
_C.MODEL.SPEC.DIM_EMBED = [64, 192, 384]
_C.MODEL.SPEC.NUM_HEADS = [1, 1, 1]
_C.MODEL.SPEC.SA_TAIL_DEPTH = [0, 0, 3]
_C.MODEL.SPEC.SA_TAIL_NUM_HEADS = [1, 1, 8]
_C.MODEL.SPEC.DEPTH = [1, 2, 10]
_C.MODEL.SPEC.MLP_RATIO = [4.0, 4.0, 4.0]
# Optional per-stage overrides:
# If left empty, the code should fall back to MLP_RATIO for that stage.
_C.MODEL.SPEC.CA_MLP_RATIO = []      # list[float], length = NUM_STAGES; for Channel-Attention (backbone) blocks
_C.MODEL.SPEC.TAIL_MLP_RATIO = []    # list[float], length = NUM_STAGES; for tail (spatial/ViT) blocks
_C.MODEL.SPEC.ATTN_DROP_RATE = [0.0, 0.0, 0.0]
_C.MODEL.SPEC.DROP_RATE = [0.0, 0.0, 0.0]
_C.MODEL.SPEC.DROP_PATH_RATE = [0.0, 0.0, 0.1]
_C.MODEL.SPEC.QKV_BIAS = [True, True, True]
_C.MODEL.SPEC.CLS_TOKEN = [False, False, False]
_C.MODEL.SPEC.POS_EMBED = [False, False, False]
_C.MODEL.SPEC.QKV_PROJ_METHOD = ['dw_bn', 'dw_bn', 'dw_bn']
_C.MODEL.SPEC.PATCH_EMBED_METHOD = ['dw_sep', 'dw_sep', 'dw_sep']  
_C.MODEL.SPEC.KERNEL_QKV = [3, 3, 3]
_C.MODEL.SPEC.PADDING_KV = [1, 1, 1]
_C.MODEL.SPEC.STRIDE_KV = [1, 1, 1]
_C.MODEL.SPEC.PADDING_Q = [1, 1, 1]
_C.MODEL.SPEC.STRIDE_Q = [1, 1, 1]
# DW shortcut (Efficient ViT with DW): parallel 3x3 depthwise conv per block
_C.MODEL.SPEC.DW_SHORTCUT = False
_C.MODEL.SPEC.DW_SHORTCUT_BACKBONE = False  # override for backbone; falls back to DW_SHORTCUT if unset
_C.MODEL.SPEC.DW_SHORTCUT_TAIL = False     # override for ViT tail; falls back to DW_SHORTCUT if unset
_C.MODEL.SPEC.DW_SHORTCUT_AFTER_FIRST_RESIDUAL = False  # add shortcut after attn residual (before MLP)
_C.MODEL.SPEC.LAYER_SCALE = None  # float init value for CaiT-style layer scale (None = disabled)
_C.MODEL.SPEC.CHANNEL_PAIR_BIAS = False  # learnable C×C bias added to channel attention logits (shared per stage)
_C.MODEL.SPEC.DW_SHORTCUT_KERNEL = [3, 3, 3, 3]  # per-stage DW shortcut kernel size (default 3×3 at all stages)
_C.MODEL.SPEC.CHANNEL_GROUPS = [1, 1, 1, 1]  # per-stage channel groups for group CA (1=full CA, G>1=group CA)
_C.MODEL.SPEC.SPARSE_TOPK = [0, 0, 0, 0]  # per-stage top-k sparse channel attention (0=dense; k>0: each channel attends to its top-k most-similar channels). EXHAUSTED (H3-1 -0.17pp).
_C.MODEL.SPEC.DUAL_CA_GROUPS = [0, 0, 0, 0]  # per-stage dual-resolution CA (0=disabled; G>0: parallel coarse pathway with G super-channel groups, must divide DIM_EMBED[i]). EXHAUSTED (H3-2 -0.15pp).
_C.MODEL.SPEC.XSTAGE_BRIDGE = False  # cross-stage channel attention bridge: last stage adds projected previous-stage features as extra K/V tokens (640 own + 640 bridge = 1280 keys). EXHAUSTED (H3-3 -0.10pp).
_C.MODEL.SPEC.AGENT_TOKENS = [0, 0, 0, 0]  # per-stage agent attention (0=disabled; M>0: replaces softmax CA with two cascaded softmaxes via M learnable agent tokens). EXHAUSTED (H3-4 -0.27pp).
_C.MODEL.SPEC.AGENT_SPATIAL_HW = [0, 0, 0, 0]  # per-stage agent feature dim = expected H*W of stage output (e.g. 49 for ImageNet 224 stage-3). Required when AGENT_TOKENS[i]>0.
_C.MODEL.SPEC.HIER_GROUPS = [0, 0, 0, 0]  # per-stage hierarchical local+global CA (0=disabled; G>0: G super-channel groups, must divide DIM_EMBED[i]; within-group dense CA + cross-group dense CA). EXHAUSTED (H3-5 -0.20pp).
# H3-6: Funneling tail-block. Adds an EXTRA single CA block AFTER stage-3 at extreme spatial compression (AdaptiveAvgPool2d → tail_hw×tail_hw).
# Operates on the final-stage feature dim, uses standard channel attention (same Attention class), gated additively into the classifier
# input via a learnable scalar (init 0). Existing stages 0-3 are completely UNCHANGED — purely additive structural innovation.
_C.MODEL.SPEC.FUNNEL_TAIL = False           # enable funneling tail block
_C.MODEL.SPEC.FUNNEL_TAIL_HW = 3            # spatial pool target (3 → 3×3=9 features per channel)
_C.MODEL.SPEC.FUNNEL_TAIL_MLP_RATIO = 0.5   # MLP ratio inside funneling block (small to fit budget)
_C.MODEL.SPEC.FUNNEL_TAIL_KERNEL = 1        # Q/K/V kernel size in funneling block (1=plain linear, cheaper)
# H3-7: Funneling tail aux supervision. Without an independent gradient signal, the funnel block at
# init-0 gate gets no gradient from the main loss (g=0 ⇒ ∂L/∂f = 0). Adding a shared-head aux loss
# on the funnel GAP feature gives the block direct discriminative supervision and aligns its output
# with the main classifier's feature space, so when the gate ramps up it routes USEFUL info.
_C.MODEL.SPEC.FUNNEL_AUX_WEIGHT = 0.0       # >0 enables aux supervision on funnel GAP feature (shared main head)
# H3-8: Linear-attention funneling tail. When True, the funnel block's Attention uses linear
# attention (phi(Q)·(phi(K)^T·V)/norm, phi=ELU+1) instead of softmax CA. Different inductive bias
# (no row-stochastic constraint, dense multiplicative interactions, rank ≤ D inner-feature dim).
_C.MODEL.SPEC.LINEAR_FUNNEL = False
# H3-9: Linformer-style projected channel attention in the funnel tail block. When > 0, K and V are
# projected along the channel (token) axis from C to r supertokens before softmax. Each query channel
# attends to r learned channel-basis vectors instead of all C channels. Stable softmax operator,
# different inductive bias from full-rank softmax CA. r=128 is typical (vs C=640).
_C.MODEL.SPEC.LINFORMER_R = 0
# H3-12: Per-stage Linformer rank for IN-TRUNK CA blocks (separate from the head funnel scalar above).
# When > 0 for a stage, all blocks in that stage use Linformer-style K/V channel-axis projection from
# C_stage to r supertokens. 0 = standard full-rank softmax CA. Default [0,0,0,0] = baseline.
# Tests whether the validated head-side Linformer inductive bias (H3-9) generalizes to in-trunk.
_C.MODEL.SPEC.LINFORMER_R_PER_STAGE = [0, 0, 0, 0]
# H3-14: Per-block Linformer rank override at the FINAL stage only (stage-3). When non-empty list,
# overrides LINFORMER_R_PER_STAGE[-1] per block: blocks[j] uses rank linformer_r_per_block_s3[j].
# Length should match DEPTH[-1] (10 for the current architecture). Empty list = no override.
# Tests depth-varying attention rank within stage-3 — does rank need to grow with depth?
_C.MODEL.SPEC.LINFORMER_R_PER_BLOCK_S3 = []
# H3-15: Per-stage Linformer top-k. When > 0 for a stage that uses Linformer (linformer_r > 0),
# each query token keeps only its top-k supertokens (of r) before softmax. Tests adding
# data-dependent sparsity ON TOP OF the validated rank-r learned basis. 0 = dense softmax over
# all r supertokens (H3-12 default).
_C.MODEL.SPEC.LINFORMER_TOPK_PER_STAGE = [0, 0, 0, 0]
# H3-16: Per-stage enable for a PARALLEL FULL-RANK CA pathway inside each Linformer block.
# When 1 for a stage that uses Linformer, each block computes BOTH (i) Linformer output and
# (ii) original full-rank softmax(Q·K^T)·V output, then combines via a learned scalar gate
# initialised to 0. At init, behaviour is identical to pure Linformer (H3-12). Training may
# ramp the gate to integrate full-rank information if useful. 0 = pure Linformer (default).
_C.MODEL.SPEC.LINFORMER_FULLRANK_PARALLEL_PER_STAGE = [0, 0, 0, 0]
# H3-17: Per-stage Linformer normalisation function. 'softmax' (default, = H3-12) or 'sparsemax'
# (Martins & Astudillo 2016, arXiv:1602.02068 — Euclidean projection onto simplex, sparse but
# differentiable). Applied inside the use_linformer branch instead of F.softmax. Only honoured
# where linformer_r > 0. Tests soft sparsity in the supertoken basis without H3-15's hard-mask
# gradient pathology.
_C.MODEL.SPEC.LINFORMER_SOFTMAX_TYPE_PER_STAGE = ['softmax', 'softmax', 'softmax', 'softmax']
# H3-18: Per-stage Synthesizer-Dense channel attention enable. When 1 for a stage, REPLACES the
# standard dot-product Q·K^T attention with a Synthesizer-Dense layer (Tay et al 2020,
# arXiv:2005.00743): attn_score = Linear(D, C)(Q) where D is the spatial-feature dim. The C×C
# attention matrix is then softmaxed and applied to V. K is computed but ignored. Tests
# operator-family pivot from Q·K^T attention to LEARNED channel-pair patterns. 0 = standard CA.
_C.MODEL.SPEC.SYNTHESIZER_DENSE_PER_STAGE = [0, 0, 0, 0]
# H3-18: Spatial dimension (H×W = D) per stage, needed at init to size the Synthesizer W matrix.
# Only honoured for stages with SYNTHESIZER_DENSE_PER_STAGE[i] == 1.
_C.MODEL.SPEC.SYNTHESIZER_HW_PER_STAGE = [0, 0, 0, 0]
# H3-19: Per-stage Nyström attention enable. When > 0 for a stage, replaces standard dot-product
# CA with Nyström landmark-based approximation (Xiong et al 2021, arXiv:2102.03902): L landmark
# tokens are obtained by 1D avg-pool over the C channel tokens, and softmax(QK^T)V is approximated
# as F_1 · pinv(F_2) · F_3 · V where F_1, F_2, F_3 are three smaller softmax matrices. Preserves
# data-dependent Q·K^T matching, structurally different from Linformer's learned projection.
# 0 = standard CA (default).
_C.MODEL.SPEC.NYSTROM_L_PER_STAGE = [0, 0, 0, 0]
# H3-19: Iterations of Newton-Schulz for the F_2 pseudoinverse. 6 is the Nyströmformer default.
_C.MODEL.SPEC.NYSTROM_ITERS_PER_STAGE = [0, 0, 0, 0]
# H3-10: Funnel-tail block depth. >=1 stacks multiple Linformer/CA blocks at the head, each with its
# own E_k, E_v projections. Tests whether the low-rank channel pathway has more capacity to give
# beyond a single block.
_C.MODEL.SPEC.FUNNEL_TAIL_DEPTH = 1
# H3-11: Parallel hybrid funnel block. When True, build a SECOND funnel block in PARALLEL with the
# sequential funnel_blocks chain (H3-9/H3-10). Its output is averaged with the sequential chain's
# output before funnel_norm. Allows mixing two DIFFERENT operator families (e.g., Linformer +
# sparse top-k) at the head — tests whether operator diversity compounds where operator replication
# (H3-10 stacked Linformer at depth=2) did not.
_C.MODEL.SPEC.FUNNEL_PARALLEL_BLOCK = False
# Linformer rank for the parallel funnel block (0 = no Linformer projection, use full-rank softmax).
_C.MODEL.SPEC.FUNNEL_PARALLEL_LINFORMER_R = 0
# Sparse top-k for the parallel funnel block (each query keeps top-K of all C source channels).
# 0 = disabled. Typical: 32-128 for C=640.
_C.MODEL.SPEC.FUNNEL_PARALLEL_SPARSE_TOPK = 0
_C.MODEL.SPEC.AUX_LOSS_WEIGHT_MID = 0.0  # intra-stage mid-block aux supervision weight (P2-H9); 0=disabled
_C.MODEL.SPEC.AUX_MID_BLOCK_IDX = 5     # block index (0-based) within final stage for mid-block aux supervision
_C.MODEL.SPEC.CPE_REPEAT_INTERVAL = [0, 0, 0, 0]  # per-stage periodic CPE re-injection interval (0=disabled)
_C.MODEL.SPEC.CROSS_STAGE_FUSION = False  # DuoFormer-inspired: fuse GAP features from stages 1+2 into final repr
_C.MODEL.SPEC.CROSS_STAGE_FUSION_S0 = False  # extend fusion to also include stage 0 (low-level edges/textures)

_C.MODEL.SPEC.TAIL_TYPE = 'sa'  # 'sa' (default) or 'vit' — controls last-stage tail kind

# LOBT (Low-Rank Orthogonal Basis Tokenization) settings
_C.MODEL.LOBT = CN()
_C.MODEL.LOBT.ENABLE = False
_C.MODEL.LOBT.K = 16                # number of channel tokens
_C.MODEL.LOBT.TAU = 0.1             # temperature / softmax sharpness
_C.MODEL.LOBT.ORTHO_LAMBDA = 1e-4   # regularization weight
_C.MODEL.LOBT.POOL = 'attn'         # 'attn' | 'gap'
_C.MODEL.LOBT.TYPE_EMBED = True     # add type embedding for LOBT tokens

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 100
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    # Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
_C.MODEL.SPEC.NORM_LAYER = 'layernorm'  # normalization type: 'layernorm' (default) or 'rmsnorm'
_C.MODEL.SPEC.AUX_LOSS_WEIGHT = 0.0  # auxiliary deep-supervision loss weight (0 = disabled); applied to stage (N-2) output
_C.MODEL.SPEC.AUX_LOSS_WEIGHT_S1 = 0.0  # second auxiliary loss weight (H50); applied to stage (N-3) output
_C.MODEL.SPEC.AUX_LOSS_WEIGHT_S0 = 0.0  # third auxiliary loss weight (H64); applied to stage (N-4) = stage 0 output
_C.MODEL.SPEC.CPE = False               # Convolutional Position Encoding (CPVT style): DWConv before tokenization (H51)
_C.MODEL.SPEC.CPE_KERNEL_SIZE = 3       # CPE depthwise conv kernel size
_C.MODEL.SPEC.SWIGLU = False            # Use SwiGLU MLP instead of GELU MLP (H54)
