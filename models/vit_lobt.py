import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# -----------------------------------------------------------------------------
# Learned Orthonormal Basis Tokenizer (LOBT)
# Produces K semantic tokens by rotating channel space with a 1x1 conv (near-orthogonal)
# and attention-pooling the original feature map per basis. Intended to be called on
# post-patch features shaped (B, D, H, W). Returns (B, K, D) tokens and a regularization loss.
# -----------------------------------------------------------------------------
class LOBTTokenizer(nn.Module):
    def __init__(self, dim: int, K: int = 16, tau: float = 0.1,
                 ortho_lambda: float = 1e-4, pool: str = 'attn',
                 type_embed: bool = True):
        super().__init__()
        assert K > 0, 'K must be > 0 for LOBT'
        assert pool in {'attn', 'gap'}
        self.dim = dim
        self.K = K
        self.tau = tau
        self.ortho_lambda = ortho_lambda
        self.pool = pool
        self.type_embed = type_embed

        # 1x1 conv acting as channel rotation (start near-identity for stability)
        self.W = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        with torch.no_grad():
            I = torch.eye(dim)
            self.W.weight.copy_(I.view(dim, dim, 1, 1) + 0.01 * torch.randn_like(self.W.weight))

        # Small linear to post-process pooled vectors (can be Identity but learnable is useful)
        self.proj = nn.Linear(dim, dim)

        # Optional token type embeddings
        if type_embed:
            self.e_basis = nn.Parameter(torch.zeros(1, 1, dim))
            self.e_spatial = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.e_basis, std=0.02)
            nn.init.trunc_normal_(self.e_spatial, std=0.02)
        else:
            self.register_parameter('e_basis', None)
            self.register_parameter('e_spatial', None)

    def orthogonality_loss(self) -> torch.Tensor:
        # ||W^T W - I||_F^2 computed on flattened (C x C) weight
        W = self.W.weight.squeeze(-1).squeeze(-1)  # (C, C)
        gram = W.t() @ W
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return (gram - eye).pow(2).sum() * self.ortho_lambda

    def forward(self, X: torch.Tensor):
        """X: (B, D, H, W) -> returns (tokens, reg)
        tokens: (B, K, D), reg: scalar tensor
        """
        B, D, H, W = X.shape
        Z = self.W(X)                       # (B, D, H, W) rotated features
        Z_flat = Z.view(B, D, -1)           # (B, D, H*W)
        X_flat = X.view(B, D, -1)           # (B, D, H*W) original features for pooling

        K = min(self.K, D)
        idx = torch.arange(K, device=X.device)  # choose first-K bases (simple & stable)

        tokens = []
        if self.pool == 'attn':
            for i in idx:
                alpha = F.softmax(Z_flat[:, i, :] / self.tau, dim=-1)         # (B, H*W)
                f = torch.einsum('bs,bcs->bc', alpha, X_flat)                 # (B, D)
                tokens.append(self.proj(f))
        else:  # 'gap'
            f = X_flat.mean(dim=-1)                                           # (B, D)
            for _ in idx:
                tokens.append(self.proj(f))

        T_basis = torch.stack(tokens, dim=1)                                  # (B, K, D)
        reg = self.orthogonality_loss()
        return T_basis, reg

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
                 lobt_enable: bool = False, lobt_k: int = 16, lobt_tau: float = 0.1,
                 lobt_ortho_lambda: float = 1e-4, lobt_pool: str = 'attn', lobt_type_embed: bool = True):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Compute feature grid size for LOBT
        self.grid_h = image_height // patch_height
        self.grid_w = image_width // patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # --- Optional LOBT tokenizer ---
        self.lobt_enable = bool(lobt_enable and lobt_k > 0)
        self._lobt_reg = torch.tensor(0.0)
        if self.lobt_enable:
            self.lobt = LOBTTokenizer(dim=dim, K=lobt_k, tau=lobt_tau,
                                      ortho_lambda=lobt_ortho_lambda,
                                      pool=lobt_pool, type_embed=lobt_type_embed)
            # separate positional embedding for basis tokens so patch pos_embed remains unchanged
            self.pos_embedding_basis = nn.Parameter(torch.zeros(1, lobt_k, dim))
            nn.init.trunc_normal_(self.pos_embedding_basis, std=0.02)
        else:
            self.lobt = None
            self.register_parameter('pos_embedding_basis', None)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # Patch embedding → tokens (B, T, D)
        x = self.to_patch_embedding(img)
        b, n, d = x.shape

        if self.lobt_enable:
            # Rebuild feature grid (B, D, H, W) for LOBT
            X = rearrange(x, 'b (h w) d -> b d h w', h=self.grid_h, w=self.grid_w)
            T_basis, reg = self.lobt(X)                    # (B, K, D), scalar
            self._lobt_reg = reg
            # Add optional token-type embeddings
            if self.lobt.type_embed:
                T_basis = T_basis + self.lobt.e_basis.expand(b, T_basis.size(1), -1)
                x = x + self.lobt.e_spatial.expand(b, n, -1)

            # Build sequence: [CLS | K basis | T patches]
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, T_basis, x), dim=1)   # (B, 1+K+T, D)

            # Positional embeddings: cls (from existing), basis (separate), patches (slice)
            pos_cls = self.pos_embedding[:, :1]
            pos_patches = self.pos_embedding[:, 1:(n + 1)]
            pos = torch.cat((pos_cls, self.pos_embedding_basis, pos_patches), dim=1)
            x = x + pos
            x = self.dropout(x)
        else:
            # Original path: [CLS | T patches]
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
            self._lobt_reg = x.new_tensor(0.0)

        # Transformer encoder
        x = self.transformer(x)

        # Pooling
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
