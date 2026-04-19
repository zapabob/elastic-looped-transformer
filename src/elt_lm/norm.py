"""RMSNorm — root-mean-square layer normalization (Zhang & Sennrich, 2019)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Compute in fp32 for numerical stability, cast back to input dtype.
        in_dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = (x32 * rms).to(in_dtype)
        return out * self.weight
