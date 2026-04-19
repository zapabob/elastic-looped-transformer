"""Verify the SDPA causal mask: token t cannot see tokens > t.

We check this by perturbing a future token and asserting that the logits at
earlier positions are unchanged.
"""

from __future__ import annotations

import torch

from elt_lm.config import ModelConfig
from elt_lm.model import ELTLanguageModel


def test_no_future_leakage() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_unique_layers=2, n_heads=4, head_dim=8,
        d_ff=64, max_seq_len=16, L_min=1, L_max=2, dropout=0.0,
        grad_checkpoint=False,
    )
    model = ELTLanguageModel(cfg).eval()

    B, T = 2, 12
    base = torch.randint(0, 64, (B, T))
    mod = base.clone()
    perturb_pos = T - 2
    # flip the token at perturb_pos to a different value
    mod[:, perturb_pos] = (mod[:, perturb_pos] + 1) % 64

    with torch.no_grad():
        out_base = model(base, L=2).logits
        out_mod = model(mod, L=2).logits

    # Positions before perturb_pos should be *identical* (causal mask respected).
    diff = (out_base[:, :perturb_pos, :] - out_mod[:, :perturb_pos, :]).abs().max().item()
    assert diff < 1e-5, f"future leakage detected: max diff = {diff}"

    # Positions from perturb_pos onward MUST differ (else the attention is broken).
    future_diff = (out_base[:, perturb_pos:, :] - out_mod[:, perturb_pos:, :]).abs().max().item()
    assert future_diff > 1e-4, f"perturbation had no effect on future tokens (diff={future_diff})"
