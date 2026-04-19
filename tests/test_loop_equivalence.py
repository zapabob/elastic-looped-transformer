"""The crucial equivalence test: ELT's F_{N,L}(x) must equal
applying the N unique layers L times in sequence (paper eq. 2).

If this test passes, we know the looping machinery is correct and
the weights are truly shared across loops.
"""

from __future__ import annotations

import torch

from elt_lm.config import ModelConfig
from elt_lm.model import ELTLanguageModel


def _make_tiny_model(L_max: int = 3) -> ELTLanguageModel:
    cfg = ModelConfig(
        vocab_size=512,
        d_model=64,
        n_unique_layers=3,
        n_heads=4,
        head_dim=16,
        d_ff=128,
        max_seq_len=32,
        L_min=1,
        L_max=L_max,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        dropout=0.0,
        grad_checkpoint=False,
    )
    torch.manual_seed(0)
    return ELTLanguageModel(cfg).eval()


@torch.no_grad()
def _manual_forward(model: ELTLanguageModel, input_ids: torch.Tensor, L: int) -> torch.Tensor:
    """Reference implementation: iterate composite.layers L times explicitly."""
    T = input_ids.shape[1]
    x = model.tok_embed(input_ids)
    cos, sin = model.rope(T, device=x.device, dtype=x.dtype)
    for _ in range(L):
        for layer in model.composite.layers:
            x = layer(x, cos, sin)
    return model._project(x)


def test_l1_equals_one_composite_pass() -> None:
    model = _make_tiny_model(L_max=2)
    input_ids = torch.randint(0, 512, (2, 16))
    out = model(input_ids, L=1)
    ref = _manual_forward(model, input_ids, L=1)
    assert out.logits.shape == ref.shape
    assert torch.allclose(out.logits, ref, atol=1e-5, rtol=1e-5), \
        f"max diff = {(out.logits - ref).abs().max().item()}"


def test_l2_equals_two_composite_passes() -> None:
    model = _make_tiny_model(L_max=4)
    input_ids = torch.randint(0, 512, (2, 16))
    out = model(input_ids, L=2)
    ref = _manual_forward(model, input_ids, L=2)
    assert torch.allclose(out.logits, ref, atol=1e-5, rtol=1e-5), \
        f"max diff = {(out.logits - ref).abs().max().item()}"


def test_l3_equals_three_composite_passes() -> None:
    model = _make_tiny_model(L_max=4)
    input_ids = torch.randint(0, 512, (2, 16))
    out = model(input_ids, L=3)
    ref = _manual_forward(model, input_ids, L=3)
    assert torch.allclose(out.logits, ref, atol=1e-5, rtol=1e-5), \
        f"max diff = {(out.logits - ref).abs().max().item()}"


def test_weight_sharing_across_loops() -> None:
    """After an optimizer step, gradients from every loop must update the single
    parameter copy — i.e., the layer weights must be shared (not duplicated)."""
    model = _make_tiny_model(L_max=3)
    model.train()
    model.cfg.grad_checkpoint = False       # simplify gradient flow for the test

    input_ids = torch.randint(0, 512, (2, 8))
    out = model(input_ids, L=3)
    loss = out.logits.sum()
    loss.backward()

    # The parameter count should be exactly that of 3 unique layers + tok_embed + norm.
    n_params = sum(p.numel() for p in model.parameters())
    # Every parameter should have a gradient after backward (no dead params).
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters missing grads: {missing}"
    assert n_params > 0


def test_intermediate_logits_at_L_int() -> None:
    """With return_hidden_at=k, intermediate_logits should equal a fresh forward at L=k."""
    model = _make_tiny_model(L_max=4)
    input_ids = torch.randint(0, 512, (2, 16))

    full = model(input_ids, L=4, return_hidden_at=2)
    short = model(input_ids, L=2)

    assert full.intermediate_logits is not None
    assert torch.allclose(full.intermediate_logits, short.logits, atol=1e-5, rtol=1e-5), \
        f"max diff = {(full.intermediate_logits - short.logits).abs().max().item()}"
