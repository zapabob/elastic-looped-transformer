"""End-to-end smoke test for the GRPO trainer: one rollout + one loss step.

Runs on CPU with a tiny ELT model and a fake tokenizer-free prompt (just tensor
ids). Exercises:
  - rollout_group produces the claimed shapes and mask invariants
  - a full GRPO step updates parameters (grad flows) and does not NaN
"""

from __future__ import annotations

import torch

from elt_lm.config import ModelConfig
from elt_lm.grpo import group_advantage, grpo_loss
from elt_lm.model import ELTLanguageModel
from elt_lm.train_grpo import (
    _adapter_parameter_names,
    _load_adapter_parameters,
    _snapshot_adapter_parameters,
    rollout_group,
)


def _tiny_model() -> ELTLanguageModel:
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_unique_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=64,
        L_min=1,
        L_max=2,
        grad_checkpoint=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return ELTLanguageModel(cfg).to(dtype=torch.float32)


def test_lora_adapter_snapshot_restore_only_lora_params() -> None:
    class ToyLoRAModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor([1.0]))
            self.block = torch.nn.Module()
            self.block.lora_A = torch.nn.Parameter(torch.tensor([2.0]))
            self.block.lora_B = torch.nn.Parameter(torch.tensor([3.0]))

    model = ToyLoRAModel()
    names = _adapter_parameter_names(model)
    assert names == ["block.lora_A", "block.lora_B"]

    snapshot = _snapshot_adapter_parameters(model, names)
    with torch.no_grad():
        model.base.fill_(10.0)
        model.block.lora_A.fill_(20.0)
        model.block.lora_B.fill_(30.0)

    _load_adapter_parameters(model, snapshot)

    assert model.base.item() == 10.0
    assert model.block.lora_A.item() == 2.0
    assert model.block.lora_B.item() == 3.0


def test_rollout_group_shapes_and_mask() -> None:
    m = _tiny_model()
    prompt = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    full, mask, lens = rollout_group(
        m, prompt,
        group_size=3, L=1,
        max_new_tokens=6, temperature=1.0, top_k=8,
        pad_id=0, eos_id=63,
    )
    G, T = full.shape
    assert G == 3
    assert T == prompt.numel() + 6  # prompt + max_new_tokens
    assert mask.shape == full.shape
    # prompt positions must be zero in the response mask
    assert mask[:, : prompt.numel()].sum().item() == 0
    # lens match the 1s in the response slice
    for i in range(G):
        assert mask[i, prompt.numel():].sum().item() == int(lens[i])


def test_grpo_step_updates_params() -> None:
    m = _tiny_model()
    # snapshots at t=0 for old/ref
    import copy
    old = copy.deepcopy(m)
    ref = copy.deepcopy(m)
    for p in old.parameters():
        p.requires_grad_(False)
    for p in ref.parameters():
        p.requires_grad_(False)

    prompt = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    full, mask, _ = rollout_group(
        old, prompt,
        group_size=4, L=1, max_new_tokens=4,
        temperature=1.0, top_k=8, pad_id=0, eos_id=63,
    )

    input_ids = full[:, :-1].contiguous()
    targets = full[:, 1:].contiguous()
    action_mask = mask[:, 1:].contiguous()

    # synthetic rewards: two responses "correct", two "wrong"
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = group_advantage(rewards)

    logits_t = m(input_ids, L=1).logits
    logits_o = old(input_ids, L=1).logits
    logits_r = ref(input_ids, L=1).logits

    out = grpo_loss(
        logits_theta=logits_t,
        logits_old=logits_o,
        logits_ref=logits_r,
        actions=targets,
        response_mask=action_mask,
        advantages=adv,
        clip_eps=0.2,
        kl_beta=0.05,
    )
    assert torch.isfinite(out.loss)
    # backprop produces non-zero grad on trainable params
    out.loss.backward()
    grad_norms = [
        p.grad.norm().item() for p in m.parameters() if p.grad is not None
    ]
    assert len(grad_norms) > 0
    assert any(g > 0.0 for g in grad_norms)


def test_grpo_zero_advantage_when_all_rewards_equal() -> None:
    # When every rollout gets the same reward, advantage is 0 → policy_loss 0,
    # KL term still shows if ref != theta. Here ref == theta so total loss == 0.
    m = _tiny_model()
    import copy
    old = copy.deepcopy(m)
    ref = copy.deepcopy(m)

    prompt = torch.tensor([9, 10, 11], dtype=torch.long)
    full, mask, _ = rollout_group(
        old, prompt,
        group_size=3, L=1, max_new_tokens=3,
        temperature=1.0, top_k=8, pad_id=0, eos_id=63,
    )
    input_ids = full[:, :-1].contiguous()
    targets = full[:, 1:].contiguous()
    action_mask = mask[:, 1:].contiguous()

    rewards = torch.tensor([0.5, 0.5, 0.5])
    adv = group_advantage(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)

    logits_t = m(input_ids, L=1).logits
    logits_o = old(input_ids, L=1).logits
    logits_r = ref(input_ids, L=1).logits
    out = grpo_loss(
        logits_theta=logits_t, logits_old=logits_o, logits_ref=logits_r,
        actions=targets, response_mask=action_mask, advantages=adv,
        clip_eps=0.2, kl_beta=0.05,
    )
    # ref == theta ⇒ KL = 0; adv = 0 ⇒ policy_loss = 0 ⇒ total loss = 0
    assert abs(out.loss.item()) < 1e-5
