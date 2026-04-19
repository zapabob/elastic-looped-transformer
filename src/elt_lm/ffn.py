"""SwiGLU feed-forward network (Shazeer, 2020).

    FFN(x) = W_down( SiLU(W_gate(x)) * W_up(x) )

Three linear layers, no biases — the de-facto modern LM choice.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
