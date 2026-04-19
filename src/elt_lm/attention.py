"""Causal multi-head self-attention with RoPE and optional grouped-query attention.

Uses torch.nn.functional.scaled_dot_product_attention (PyTorch >= 2.7, cu128 build)
which dispatches to FlashAttention-2 / memory-efficient kernels automatically.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from elt_lm.rope import apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, \
            f"n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads          # GQA replication factor
        self.dropout_p = dropout

        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, _ = x.shape

        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)      # (B, H, T, D)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, Hk, T, D)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, Hk, T, D)

        q, k = apply_rope(q, k, cos, sin)

        if self.n_rep != 1:
            # GQA: repeat K and V to match Q's head count.
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        # (B, H, T, D) -> (B, T, H*D) -> (B, T, d_model)
        out = attn.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.w_o(out)
