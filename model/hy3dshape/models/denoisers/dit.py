import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
    get_1d_sincos_pos_embed_from_grid, 
    get_2d_sincos_pos_embed,
    TimestepEmbedder, 
    MLP, 
    Attention, 
    CrossAttention
)

from .moe_layers import MoEBlock
from .profilers import BlockTimer
import numpy as np
try:
    import torch.cuda.nvtx as nvtx
    _HAS_NVTX = True
except Exception:
    _HAS_NVTX = False

class ConditionalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        text_states_dim=1024,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        skip_connection: bool = True,
        # The following are parameters from the original for full compatibility
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        qkv_bias=True,
        timested_modulate=False,
        **kwargs,
    ):
        super().__init__()
        
        self.norm1 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn1 = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, 
                               qk_norm=qk_norm, norm_layer=qk_norm_layer)

        self.norm2 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn2 = CrossAttention(qdim=hidden_size, kdim=text_states_dim, num_heads=num_heads, 
                                     qkv_bias=qkv_bias, qk_norm=qk_norm, norm_layer=qk_norm_layer)

        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        if use_moe:
            self.ff = MoEBlock(dim=hidden_size, num_experts=num_experts, moe_top_k=moe_top_k, ff_bias=True, ff_inner_dim=int(hidden_size*4.0))
        else:
            self.ff = MLP(width=hidden_size)

        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.timested_modulate = timested_modulate
        if self.timested_modulate and c_emb_size is not None:
            self.default_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(c_emb_size, hidden_size, bias=True)
            )

    def forward(self, x, c=None, text_states=None, skip_value=None):
        if self.skip_connection:
            x = self.skip_linear(torch.cat([skip_value, x], dim=-1))
            x = self.skip_norm(x)

        if self.timested_modulate:
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa
        x = x + self.attn1(self.norm1(x))

        x = x + self.attn2(self.norm2(x), text_states)

        x = x + self.ff(self.norm3(x))

        return x

class UnconditionalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        skip_connection: bool = True,
        # The following are parameters from the original for full compatibility
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        qkv_bias=True,
        timested_modulate=False,
        **kwargs,
    ):
        super().__init__()

        self.norm1 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn1 = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_norm=qk_norm, norm_layer=qk_norm_layer)

        self.norm2 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn2 = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_norm=qk_norm, norm_layer=qk_norm_layer)

        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        if use_moe:
            self.ff = MoEBlock(dim=hidden_size, num_experts=num_experts, moe_top_k=moe_top_k, ff_bias=True, ff_inner_dim=int(hidden_size*4.0))
        else:
            self.ff = MLP(width=hidden_size)

        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.timested_modulate = timested_modulate
        if self.timested_modulate and c_emb_size is not None:
            self.default_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(c_emb_size, hidden_size, bias=True)
            )

    def forward(self, x, c=None, text_states=None, skip_value=None):
        # The `c` and `text_states` arguments are ignored but kept for signature compatibility

        if self.skip_connection:
            x = self.skip_linear(torch.cat([skip_value, x], dim=-1))
            x = self.skip_norm(x)

        if self.timested_modulate:
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa
        x = x + self.attn1(self.norm1(x))

        x = x + self.attn2(self.norm2(x)) # No text_states needed here

        x = x + self.ff(self.norm3(x))

        return x
    

class AttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(1, 0, 2)
            global_emb = (x * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)
            x = torch.cat([global_emb[None,], x], dim=0)

        else:
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    
    
class FinalLayer(nn.Module):

    def __init__(self, final_hidden_size, out_channels):
        super().__init__()
        self.final_hidden_size = final_hidden_size
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x
    
    
class DiTPlain(nn.Module):
    def __init__(
        self,
        in_channels=32,
        hidden_size=512,
        depth=12,
        num_heads=8,
        # Faithfully ported arguments from original
        context_dim=None,
        integration_mode="cross_attn",
        timested_modulate=False,
        mlp_ratio=4.0,
        norm_type='layer',
        qk_norm_type='rms',
        qk_norm=False,
        text_len=77,
        use_pos_emb=True,
        pos_emb_type='1d',
        use_attention_pooling=False,
        qkv_bias=True,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        max_seq_len: int = 4096,
        grid_size: int = 32,
        profile_blocks: bool = False,
        **kwargs
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        
        self.norm = nn.LayerNorm if norm_type == 'layer' else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == 'rms' else nn.LayerNorm
        
        self.text_len = text_len
        
        self._profile_blocks = profile_blocks
        self._btimer = None
        self._perblock_param = {}
        
        self.dtype = kwargs.get('dtype', torch.bfloat16)

        self.integration_mode = integration_mode
        self.timested_modulate = timested_modulate
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        if self.integration_mode == "cross_attn": # if uncond, the context_dim can be None
            self.t_embedder = TimestepEmbedder(hidden_size)
            self.block_class = ConditionalDiTBlock if context_dim is not None else UnconditionalDiTBlock
            self.block_context_dim = context_dim
        elif self.integration_mode == "concat":
            self.t_embedder = TimestepEmbedder(hidden_size, cond_proj_dim=context_dim) if context_dim is not None else TimestepEmbedder(hidden_size)
            self.block_class = UnconditionalDiTBlock
            self.block_context_dim = None
        else:
            raise ValueError(f"Unknown integration mode: {self.integration_mode}")

        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling
        

        
        if use_pos_emb:
            if pos_emb_type == '1d':
                print("Using 1D Sin/Cos positional embeddings.")
                pos_embed = get_1d_sincos_pos_embed_from_grid(hidden_size, np.arange(max_seq_len))
            elif pos_emb_type == '2d':
                print("Using 2D Sin/Cos positional embeddings.")
                assert max_seq_len == grid_size * grid_size
                pos_embed = get_2d_sincos_pos_embed(hidden_size, grid_size)
            else:
                raise ValueError(f"Unknown pos_emb_type: '{pos_emb_type}'")
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
        else:
            self.pos_embed = None

        if use_attention_pooling:
            self.pooler = AttentionPool(self.text_len, context_dim, num_heads=8, output_dim=1024)
            self.extra_embedder = nn.Sequential(
                nn.Linear(1024, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size, bias=True),
            )
            
        
        self.blocks = nn.ModuleList()
        
        for layer in range(depth):
            use_skip_connection = layer >= depth // 2
            is_moe = (num_moe_layers > 0) and (layer >= depth - num_moe_layers)
            
            self.blocks.append(
                self.block_class(
                    hidden_size=hidden_size,
                    c_emb_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    text_states_dim=self.block_context_dim,
                    qk_norm=qk_norm,
                    norm_layer=self.norm,
                    qk_norm_layer=self.qk_norm,
                    skip_connection=use_skip_connection,
                    qkv_bias=qkv_bias,
                    use_moe=is_moe,
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                    timested_modulate=timested_modulate
                )
            )
        
        self.final_layer = FinalLayer(final_hidden_size=hidden_size, out_channels=self.out_channels)
        
        if self._profile_blocks:
            self._logged_params_once = False
            self._btimer = BlockTimer(use_cuda_events=True, use_nvtx=True)
            def pcount(m): return sum(p.numel() for p in m.parameters())
            for i, blk in enumerate(self.blocks):
                # Attach hooks to the three key submodules
                self._btimer.attach(blk.attn1, f"b{i}.attn1")
                self._btimer.attach(blk.attn2, f"b{i}.attn2")
                self._btimer.attach(blk.ff,    f"b{i}.ff")
                # Stash static param counts for context
                self._perblock_param[f"denoiser/b{i}.attn1_params"] = pcount(blk.attn1)
                self._perblock_param[f"denoiser/b{i}.attn2_params"] = pcount(blk.attn2)
                self._perblock_param[f"denoiser/b{i}.ff_params"]    = pcount(blk.ff)
                if isinstance(blk.ff, MoEBlock):
                    # Attach to MoE sub-parts too!
                    self._btimer.attach(blk.ff.gate,    f"b{i}.ff.gate")
                    self._btimer.attach(blk.ff.shared_experts, f"b{i}.ff.shared")
        
    def collect_block_timings(self, reset: bool = True):
        if not self._profile_blocks or self._btimer is None:
            return {}
        metrics = self._btimer.dump_and_reset(prefix="denoiser") if reset else {}
        if not self._logged_params_once:
            metrics.update(self._perblock_param)
            self._logged_params_once = True
            

        
        return metrics
        
    def forward(self, x, t, contexts):
        seq_len = x.shape[1]
        x = self.x_embedder(x)
        if self.use_pos_emb:
            x = x + self.pos_embed[:, :seq_len, :].to(x.dtype)

        context = contexts.get('main', None)
        if context is not None:
            context = context.to(dtype=x.dtype, device=x.device)
        
        if self.use_attention_pooling:
            if context is None:
                raise ValueError("use_attention_pooling=True but no context provided.")
            if context.dim() != 3:
                raise ValueError(f"Expected context [B, L, C], got {tuple(context.shape)}")
            L, C = context.shape[1], context.shape[2]
            if L != self.text_len:
                raise ValueError(f"context length L={L} must equal text_len={self.text_len} when attention pooling is enabled.")
            if self.block_context_dim is not None and C != self.block_context_dim:
                raise ValueError(f"context dim C={C} must equal context_dim={self.block_context_dim}.")
        else:
            if context is not None and context.dim() == 3 and self.block_context_dim is not None:
                C = context.shape[2]
                if C != self.block_context_dim:
                    raise ValueError(f"context dim C={C} must equal context_dim={self.block_context_dim}.")

        if self.integration_mode == "cross_attn":  # [B, 1, C]
            c_emb = self.t_embedder(t)
        elif self.integration_mode == "concat":
            c_emb = self.t_embedder(t, condition=context)
        else:
            raise ValueError(f"Unknown integration mode: {self.integration_mode}")
        
        c = c_emb
        if self.use_attention_pooling and self.integration_mode == "cross_attn":
            if context is None:
                raise ValueError("Model configured to use attention pooling but no context was provided.")
            pooled_context = self.pooler(context)
            c = c_emb + self.extra_embedder(pooled_context).unsqueeze(dim=1)

        x = torch.cat([c,x], dim=1) # concat c to x. if uncond or cross_attn class, c = t; if cross_attn text, c = t + pooled_text ; if adaln, c = t + cond_proj
        
        skip_value_list = []
        
        
        for layer, block in enumerate(self.blocks):
            skip_value = None
            if layer >= self.depth // 2:
                skip_value = skip_value_list.pop()

            if _HAS_NVTX: nvtx.range_push(f"b{layer}/block")
            x = block(x, c=c.squeeze(1), text_states=context, skip_value=skip_value)
            if _HAS_NVTX: nvtx.range_pop()


            if layer < self.depth // 2:
                skip_value_list.append(x)
        
        x = self.final_layer(x)
        return x