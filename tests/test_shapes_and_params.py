"""Basic shape / param-count / cpu-forward smoke tests."""

from __future__ import annotations

import torch

from elt_lm.config import ModelConfig
from elt_lm.model import ELTLanguageModel


def test_forward_shapes() -> None:
    cfg = ModelConfig(
        vocab_size=100, d_model=32, n_unique_layers=2, n_heads=4, head_dim=8,
        d_ff=64, max_seq_len=16, L_min=1, L_max=2, grad_checkpoint=False,
    )
    model = ELTLanguageModel(cfg).eval()
    input_ids = torch.randint(0, 100, (3, 10))
    out = model(input_ids, L=2)
    assert out.logits.shape == (3, 10, 100)
    assert out.intermediate_logits is None
    assert out.intermediate_hidden is None
    assert out.per_loop_hidden is None


def test_forward_with_intermediate() -> None:
    cfg = ModelConfig(
        vocab_size=100, d_model=32, n_unique_layers=2, n_heads=4, head_dim=8,
        d_ff=64, max_seq_len=16, L_min=1, L_max=3, grad_checkpoint=False,
    )
    model = ELTLanguageModel(cfg).eval()
    input_ids = torch.randint(0, 100, (3, 10))
    out = model(input_ids, L=3, return_hidden_at=2)
    assert out.logits.shape == (3, 10, 100)
    assert out.intermediate_logits is not None
    assert out.intermediate_logits.shape == (3, 10, 100)
    assert out.intermediate_hidden is not None
    assert out.intermediate_hidden.shape == (3, 10, 32)


def test_forward_with_per_loop_hidden() -> None:
    cfg = ModelConfig(
        vocab_size=100, d_model=32, n_unique_layers=2, n_heads=4, head_dim=8,
        d_ff=64, max_seq_len=16, L_min=1, L_max=3, grad_checkpoint=False,
    )
    model = ELTLanguageModel(cfg).eval()
    input_ids = torch.randint(0, 100, (2, 10))
    out = model(input_ids, L=3, return_all_loop_hidden=True)
    assert out.per_loop_hidden is not None
    assert len(out.per_loop_hidden) == 3
    assert all(h.shape == (2, 10, 32) for h in out.per_loop_hidden)


def test_param_count_100M_config_fits_budget() -> None:
    cfg = ModelConfig(
        vocab_size=248320, d_model=768, n_unique_layers=12, n_heads=12, head_dim=64,
        d_ff=2048, max_seq_len=2048, L_min=1, L_max=4, tie_word_embeddings=True,
        grad_checkpoint=False,
    )
    model = ELTLanguageModel(cfg)
    n_total = model.num_parameters(non_embedding=False)
    n_non_embed = model.num_parameters(non_embedding=True)
    # Embedding alone is 248320 * 768 ≈ 190.7M (tied, so counted once).
    # Transformer params (N=12 unique layers) should be roughly ~85M.
    assert n_total > 200_000_000
    assert n_total < 400_000_000
    assert 50_000_000 < n_non_embed < 150_000_000


def test_generate_runs() -> None:
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_unique_layers=2, n_heads=4, head_dim=8,
        d_ff=64, max_seq_len=32, L_min=1, L_max=2, grad_checkpoint=False,
    )
    model = ELTLanguageModel(cfg).eval()
    prompt = torch.randint(0, 64, (1, 4))
    out = model.generate(prompt, max_new_tokens=5, L=2, top_k=10)
    assert out.shape == (1, 9)
