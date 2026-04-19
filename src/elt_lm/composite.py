"""Composite block g_Theta from arXiv:2604.09168 eq. (1):

    g_Theta(x) = f_{theta_N} ∘ f_{theta_{N-1}} ∘ ... ∘ f_{theta_1}(x)

N unique layers stacked once. The ELT model iterates this block L times (eq. 2).
"""

from __future__ import annotations

from torch import Tensor, nn

from elt_lm.config import ModelConfig
from elt_lm.layer import TransformerLayer


class CompositeBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(cfg) for _ in range(cfg.n_unique_layers)]
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x
