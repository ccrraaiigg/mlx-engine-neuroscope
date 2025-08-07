#!/usr/bin/env python3
"""
Create a gpt_oss model implementation for mlx_lm
"""

import os
import shutil

# Path to mlx_lm models directory
mlx_models_path = "/Users/craig/me/behavior/forks/mlx-engine-neuroscope/.venv_test_gpt/lib/python3.13/site-packages/mlx_lm/models"

# Create a basic gpt_oss.py based on a similar model (like llama)
gpt_oss_content = '''# Copyright © 2023-2024 Apple Inc.
# GPT-OSS model implementation for MLX with MoE support

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import math

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gpt_oss"
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_hidden_layers: int = 24
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-05
    rope_theta: float = 150000
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    sliding_window: int = 128
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    experts_per_token: int = 4  # Alternative name used in config
    router_aux_loss_coef: float = 0.9
    swiglu_limit: float = 7.0
    layer_types: list = None
    rope_scaling: dict = None

    def __post_init__(self):
        # Handle alternative naming
        if hasattr(self, 'experts_per_token') and self.experts_per_token:
            self.num_experts_per_tok = self.experts_per_token
            
        if self.layer_types is None:
            # Default alternating pattern
            self.layer_types = ["sliding_attention", "full_attention"] * (self.num_hidden_layers // 2)
        if self.rope_scaling is None:
            self.rope_scaling = {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "rope_type": "yarn",
                "truncate": False
            }


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return nn.RMSNorm(x, self.weight, self.eps)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding to query and key tensors."""
    if position_ids is None:
        cos = cos[:q.shape[-2], :]
        sin = sin[:q.shape[-2], :]
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


class YarnRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0, 
                 original_max_position_embeddings=4096, beta_fast=32, beta_slow=1):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def __call__(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
            
        t = mx.arange(seq_len, dtype=self.inv_freq.dtype)
        
        # Apply YARN scaling
        if seq_len > self.original_max_position_embeddings:
            scale = seq_len / self.original_max_position_embeddings
            inv_freq_scaled = self.inv_freq / scale
        else:
            inv_freq_scaled = self.inv_freq
            
        freqs = mx.outer(t, inv_freq_scaled)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos, sin


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        
        self.args = args
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.num_key_value_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5
        self.sliding_window = args.sliding_window
        
        # Determine attention type
        if layer_idx < len(args.layer_types):
            self.attention_type = args.layer_types[layer_idx]
        else:
            self.attention_type = "full_attention"
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE
        # Filter rope_scaling parameters to only include supported ones
        rope_params = {}
        if args.rope_scaling:
            supported_params = {'scaling_factor', 'original_max_position_embeddings', 'beta_fast', 'beta_slow'}
            rope_params = {k: v for k, v in args.rope_scaling.items() if k in supported_params}
            # Map 'factor' to 'scaling_factor' if present
            if 'factor' in args.rope_scaling:
                rope_params['scaling_factor'] = args.rope_scaling['factor']
        
        self.rotary_emb = YarnRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            **rope_params
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE
        cos, sin = self.rotary_emb(values, seq_len=L)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)

        # Apply sliding window attention if needed
        if self.attention_type == "sliding_attention" and L > self.sliding_window:
            # Create sliding window mask
            sliding_mask = mx.ones((L, L)) * float('-inf')
            for i in range(L):
                start = max(0, i - self.sliding_window + 1)
                sliding_mask = sliding_mask.at[i, start:i+1].set(0)
            
            if mask is not None:
                mask = mx.minimum(mask, sliding_mask)
            else:
                mask = sliding_mask

        output = scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class Expert(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.experts = [Expert(args) for _ in range(self.num_experts)]
        self.gate = nn.Linear(args.hidden_size, self.num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)
        
        # Router logits
        router_logits = self.gate(x_flat)
        routing_weights = mx.softmax(router_logits, axis=-1)
        
        # Select top-k experts
        topk_weights, topk_indices = mx.topk(routing_weights, self.num_experts_per_tok, axis=-1)
        topk_weights = mx.softmax(topk_weights, axis=-1)
        
        # Initialize output
        final_hidden_states = mx.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            expert_mask = (topk_indices == expert_idx).any(axis=-1)
            if expert_mask.sum() > 0:
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get weights for this expert
                expert_weights = mx.zeros(x_flat.shape[0])
                for i in range(self.num_experts_per_tok):
                    mask = topk_indices[:, i] == expert_idx
                    expert_weights = expert_weights + mask.astype(mx.float32) * topk_weights[:, i]
                
                expert_weights = expert_weights[expert_mask]
                final_hidden_states = final_hidden_states.at[expert_mask].add(
                    expert_output * expert_weights.reshape(-1, 1)
                )
        
        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MoE(args)  # Use MoE instead of regular MLP
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class GptOssModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h, cache[i] = layer(h, mask, cache[i])

        return self.norm(h), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = GptOssModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out, cache = self.model(inputs, cache)
        if hasattr(self, "lm_head"):
            out = self.lm_head(out)
        else:
            out = self.model.embed_tokens.as_linear(out)
        return out, cache

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
'''

# Write the gpt_oss.py file
gpt_oss_path = os.path.join(mlx_models_path, "gpt_oss.py")
print(f"Creating gpt_oss.py at: {gpt_oss_path}")

try:
    with open(gpt_oss_path, 'w') as f:
        f.write(gpt_oss_content)
    print("✓ Successfully created gpt_oss.py")
except Exception as e:
    print(f"✗ Failed to create gpt_oss.py: {e}")