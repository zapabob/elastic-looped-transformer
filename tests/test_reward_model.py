from __future__ import annotations

import torch

from elt_lm.config import ModelConfig
from elt_lm.reward_model import ELTRewardModel, bradley_terry_loss


def _tiny_reward_model() -> ELTRewardModel:
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_unique_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=32,
        L_min=1,
        L_max=2,
        grad_checkpoint=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return ELTRewardModel(cfg).to(dtype=torch.float32)


def test_reward_model_forward_shape() -> None:
    model = _tiny_reward_model()
    ids = torch.randint(0, 64, (3, 8), dtype=torch.long)
    rewards = model(ids, L=1)
    assert rewards.shape == (3,)
    assert torch.isfinite(rewards).all()


def test_bradley_terry_loss_backprops() -> None:
    chosen = torch.tensor([1.0, 2.0], requires_grad=True)
    rejected = torch.tensor([0.5, 1.5], requires_grad=True)
    loss = bradley_terry_loss(chosen, rejected, margin=0.1)
    assert loss.item() > 0.0
    loss.backward()
    assert chosen.grad is not None
    assert rejected.grad is not None
