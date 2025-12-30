import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from torch.nn.attention import sdpa_kernel, SDPBackend

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Create 2D sin/cos positional embeddings.
    
    Args:
        embed_dim (int): The embedding dimension.
        grid_size (int): The height and width of the grid.
    
    Returns:
        (np.ndarray): Positional embedding table, shape (grid_size*grid_size, embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # H, W
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    # use half of dimensions to encode grid_h and the other half for grid_w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return pos_embed


class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class Timesteps(nn.Module):
    """Creates sinusoidal timestep embeddings."""
    def __init__(self, num_channels, max_period=10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    MODIFIED to accept an additional conditioning signal.
    """
    def __init__(self, hidden_size, frequency_embedding_size=None, cond_proj_dim=None):
        super().__init__()
        if frequency_embedding_size is None:
            frequency_embedding_size = hidden_size * 4 # Default

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, hidden_size),
        )
        self.time_embed = Timesteps(hidden_size)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, hidden_size, bias=False)
        else:
            self.cond_proj = None

    def forward(self, t, condition=None):
        # t: [B]
        # condition: [B, 1, cond_proj_dim] or [B, cond_proj_dim]
        t_freq = self.time_embed(t) # [B, hidden_size]

        if self.cond_proj is not None and condition is not None:
            if len(condition.shape) == 3:
                condition = condition.squeeze(1) # [B, cond_proj_dim]
            
            cond_embedding = self.cond_proj(condition) # [B, hidden_size]
            t_freq = t_freq + cond_embedding

        t_emb = self.mlp(t_freq)
        return t_emb.unsqueeze(dim=1)
    

class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

        self.with_dca = with_decoupled_ca
        if self.with_dca:
            self.kv_proj_dca = nn.Linear(kdim, 2 * qdim, bias=qkv_bias)
            self.k_norm_dca = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
            self.dca_dim = decoupled_ca_dim
            self.dca_weight = decoupled_ca_weight


    def forward(self, x, y):
        """
        x: [B, S1, Cq]  (Cq = num_heads * head_dim)
        y: [B, S2, Ck]  (projected to K,V with same output dim as Cq)
        """
        B, S1, _ = x.shape

        # Optional decoupled CA branch
        if self.with_dca:
            token_len = y.shape[1]
            context_dca = y[:, -self.dca_dim:, :]                          # [B, Sd, Ck]
            kv_dca = self.kv_proj_dca(context_dca)                          # [B, Sd, 2*Cq]
            kv_dca = kv_dca.view(B, self.dca_dim, 2, self.num_heads, self.head_dim)
            k_dca, v_dca = kv_dca.unbind(dim=2)                             # [B, Sd, H, D]
            k_dca = self.k_norm_dca(k_dca)                                  # LN over last dim
            y = y[:, :(token_len - self.dca_dim), :]                        # remove dca tail
            Sd = self.dca_dim
        else:
            Sd = 0

        # Main Q/K/V
        q = self.to_q(x).view(B, S1, self.num_heads, self.head_dim)         # [B, S1, H, D]
        k = self.to_k(y).view(B, -1, self.num_heads, self.head_dim)         # [B, S2, H, D]
        v = self.to_v(y).view(B, -1, self.num_heads, self.head_dim)         # [B, S2, H, D]

        # Per-head norm on last dim, then permute to SDPA layout
        q = self.q_norm(q).permute(0, 2, 1, 3)                              # [B, H, S1, D]
        k = self.k_norm(k).permute(0, 2, 1, 3)                              # [B, H, S2, D]
        v = v.permute(0, 2, 1, 3)                                           # [B, H, S2, D]

        with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )                                                               # [B, H, S1, D]

        # Optional DCA branch (uses same q)
        if self.with_dca:
            k_dca = k_dca.permute(0, 2, 1, 3)                               # [B, H, Sd, D]
            v_dca = v_dca.permute(0, 2, 1, 3)                               # [B, H, Sd, D]
            with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                context_dca = F.scaled_dot_product_attention(
                    q, k_dca, v_dca
                )                                                           # [B, H, S1, D]
            context = context + self.dca_weight * context_dca

        # Merge heads and project out
        context = context.permute(0, 2, 1, 3).reshape(B, S1, -1)            # [B, S1, H*D]
        out = self.out_proj(context)                                        # [B, S1, Cq]
        return out


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

    
    def forward(self, x):
        B, N, _ = x.shape


        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim)          # [B, N, H, D]
        k = self.to_k(x).view(B, N, self.num_heads, self.head_dim)          # [B, N, H, D]
        v = self.to_v(x).view(B, N, self.num_heads, self.head_dim)          # [B, N, H, D]

        q = self.q_norm(q).permute(0, 2, 1, 3)                              # [B, H, N, D]
        k = self.k_norm(k).permute(0, 2, 1, 3)                              # [B, H, N, D]
        v = v.permute(0, 2, 1, 3)                                           # [B, H, N, D]

        with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )                                                               # [B, H, N, D]

        out = attn.permute(0, 2, 1, 3).reshape(B, N, -1)                    # [B, N, H*D]
        out = self.out_proj(out)                                            # [B, N, C]
        return out