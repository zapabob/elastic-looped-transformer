"""Pre-Norm Transformer layer — one f_{theta_i} in the composite block g_Theta."""

from __future__ import annotations

from torch import Tensor, nn

from elt_lm.attention import CausalSelfAttention
from elt_lm.config import ModelConfig
from elt_lm.ffn import SwiGLU
from elt_lm.norm import RMSNorm


class TransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.head_dim is not None and cfg.n_kv_heads is not None  # set in __post_init__
        self.norm_attn = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            dropout=cfg.dropout,
        )
        self.norm_ffn = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.ffn(self.norm_ffn(x))
        return x
