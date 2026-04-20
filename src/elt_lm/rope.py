"""Rotary Positional Embedding (Su et al., 2021) — HuggingFace-style "half" rotation.

RoPE is applied per-head. For a query/key vector of head_dim D, we split its last
dim into two halves [x1, x2] (each of size D/2), compute cosine/sine tables of
frequencies, and rotate:

    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

This matches the convention used in Llama / Qwen / most modern decoder-only LMs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RoPECache(nn.Module):
    """Precomputes cos/sin tables up to max_seq_len.

    The tables have shape (max_seq_len, head_dim). cos/sin are stored as persistent
    buffers — HF's from_pretrained flow wipes non-persistent buffers before calling
    __init__, so keeping them persistent is the robust way to survive save/load.
    """

    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                   # (T, half)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, D)  — [f0..f_{D/2-1}, f0..f_{D/2-1}]

        self.register_buffer("cos_cached", emb.cos(), persistent=True)
        self.register_buffer("sin_cached", emb.sin(), persistent=True)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        assert seq_len <= self.max_seq_len, \
            f"seq_len={seq_len} exceeds RoPE cache max_seq_len={self.max_seq_len}"
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    """Split x's last dim in half and return [-x2, x1] for the half-rotation trick."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary embedding to q and k.

    q, k: (B, H, T, D)
    cos, sin: (T, D)    — will be broadcast over (B, H)
    """
    # reshape cos/sin to (1, 1, T, D) for broadcast over batch and heads
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_rot = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_rot, k_rot
