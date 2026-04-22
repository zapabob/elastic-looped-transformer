"""ILSD stop-grad correctness and loss-composition tests."""

from __future__ import annotations

import torch

from elt_lm.config import ILSDConfig, ModelConfig
from elt_lm.ilsd import (
    ILSDLossFn,
    _causal_lm_soft_ce,
    _entropy_floor_penalty,
    _hidden_local_consistency,
    compute_lambda,
)
from elt_lm.model import ELTLanguageModel


def _make_small() -> tuple[ELTLanguageModel, ModelConfig, ILSDConfig]:
    mcfg = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_unique_layers=2,
        n_heads=4,
        head_dim=8,
        d_ff=64,
        max_seq_len=16,
        L_min=1,
        L_max=3,
        tie_word_embeddings=True,
        grad_checkpoint=False,
    )
    icfg = ILSDConfig(
        enabled=True,
        lambda_anneal_steps=100,
        warmup_steps=0,
        strict_student_below_teacher=True,
    )
    torch.manual_seed(0)
    model = ELTLanguageModel(mcfg)
    return model, mcfg, icfg


def test_soft_ce_teacher_gradient_is_zero() -> None:
    """Teacher logits must remain stop-grad under soft CE."""
    student = torch.randn(2, 8, 16, requires_grad=True)
    teacher = torch.randn(2, 8, 16, requires_grad=True)

    loss = _causal_lm_soft_ce(student, teacher)
    loss.backward()

    assert student.grad is not None
    assert student.grad.abs().sum().item() > 0.0
    assert teacher.grad is None or teacher.grad.abs().sum().item() == 0.0


def test_soft_ce_mask_ignores_masked_positions() -> None:
    student = torch.randn(1, 5, 7)
    teacher = torch.randn(1, 5, 7)
    mask = torch.tensor([[True, False, False, False]])

    baseline = _causal_lm_soft_ce(student, teacher, valid_mask=mask)
    teacher_perturbed = teacher.clone()
    teacher_perturbed[:, 2:, :] = teacher_perturbed[:, 2:, :] + 50.0
    masked = _causal_lm_soft_ce(student, teacher_perturbed, valid_mask=mask)

    assert torch.allclose(baseline, masked, atol=1e-6, rtol=1e-6)


def test_lambda_schedule_bounds() -> None:
    cfg = ILSDConfig(lambda_init=1.0, lambda_final=0.0, lambda_anneal_steps=1000, warmup_steps=0)
    assert abs(compute_lambda(0, cfg) - 1.0) < 1e-9
    assert abs(compute_lambda(1000, cfg) - 0.0) < 1e-9
    assert abs(compute_lambda(2000, cfg) - 0.0) < 1e-9
    assert abs(compute_lambda(500, cfg) - 0.5) < 1e-9


def test_entropy_floor_penalty_zero_above_floor() -> None:
    logits = torch.zeros(1, 4, 8)
    penalty = _entropy_floor_penalty(logits, floor_value=0.5)
    assert abs(float(penalty.item())) < 1e-8


def test_hidden_local_consistency_zero_for_identical_states() -> None:
    hidden = torch.randn(1, 4, 6)
    mask = torch.tensor([[True, True, True]])
    loss = _hidden_local_consistency((hidden, hidden.clone()), metric="cosine", valid_mask=mask)
    assert loss is not None
    assert abs(float(loss.item())) < 1e-6


def test_ilsd_total_decomposes_correctly() -> None:
    model, mcfg, icfg = _make_small()
    loss_fn = ILSDLossFn(mcfg, icfg, seed=7)

    input_ids = torch.randint(0, mcfg.vocab_size, (2, 8))
    labels = input_ids.clone()

    out = loss_fn(model, input_ids, labels, step=50)
    lam = out.lambda_value
    total_expected = out.l_gt_teacher + lam * out.l_gt_student + (1 - lam) * out.l_dist
    assert torch.allclose(out.total.detach(), total_expected, atol=1e-5, rtol=1e-5)


def test_ilsd_total_decomposes_with_regularizers() -> None:
    model, mcfg, icfg = _make_small()
    icfg.distill_teacher_temp = 1.5
    icfg.distill_uniform_mix = 1e-3
    icfg.entropy_floor_weight = 0.02
    icfg.entropy_floor_start = 0.18
    icfg.entropy_floor_end = 0.08
    icfg.local_consistency_weight = 0.05
    icfg.local_consistency_metric = "cosine"
    loss_fn = ILSDLossFn(mcfg, icfg, seed=7)

    input_ids = torch.randint(0, mcfg.vocab_size, (2, 8))
    labels = input_ids.clone()

    out = loss_fn(model, input_ids, labels, step=50)
    lam = out.lambda_value
    total_expected = (
        out.l_gt_teacher
        + lam * out.l_gt_student
        + (1 - lam) * (
            out.l_dist
            + icfg.entropy_floor_weight * out.l_entropy
            + icfg.local_consistency_weight * out.l_local
        )
    )
    assert torch.allclose(out.total.detach(), total_expected, atol=1e-5, rtol=1e-5)


def test_ilsd_warmup_teacher_only() -> None:
    model, mcfg, icfg = _make_small()
    icfg.warmup_steps = 10
    loss_fn = ILSDLossFn(mcfg, icfg, seed=0)

    input_ids = torch.randint(0, mcfg.vocab_size, (2, 8))
    labels = input_ids.clone()

    out = loss_fn(model, input_ids, labels, step=0)
    assert out.L_int == mcfg.L_max
    assert out.l_dist.item() == 0.0
    assert out.l_gt_student.item() == 0.0
    assert out.l_entropy.item() == 0.0
    assert out.l_local.item() == 0.0
    assert torch.allclose(out.total.detach(), out.l_gt_teacher, atol=1e-6)


def test_ilsd_sampling_in_bounds() -> None:
    model, mcfg, icfg = _make_small()
    loss_fn = ILSDLossFn(mcfg, icfg, seed=123)
    input_ids = torch.randint(0, mcfg.vocab_size, (1, 4))
    labels = input_ids.clone()
    seen: set[int] = set()
    for s in range(20):
        out = loss_fn(model, input_ids, labels, step=s + icfg.warmup_steps + 1)
        seen.add(out.L_int)
        assert mcfg.L_min <= out.L_int < mcfg.L_max
    assert seen.issubset({1, 2})
