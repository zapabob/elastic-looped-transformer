"""ILSD sg(·) correctness and λ-mixed loss sanity tests (paper eq. 3)."""

from __future__ import annotations

import torch

from elt_lm.config import ILSDConfig, ModelConfig
from elt_lm.ilsd import ILSDLossFn, _causal_lm_soft_ce, compute_lambda
from elt_lm.model import ELTLanguageModel


def _make_small() -> tuple[ELTLanguageModel, ModelConfig, ILSDConfig]:
    mcfg = ModelConfig(
        vocab_size=128, d_model=32, n_unique_layers=2, n_heads=4, head_dim=8,
        d_ff=64, max_seq_len=16, L_min=1, L_max=3, tie_word_embeddings=True,
        grad_checkpoint=False,
    )
    icfg = ILSDConfig(enabled=True, lambda_anneal_steps=100, warmup_steps=0,
                      strict_student_below_teacher=True)
    torch.manual_seed(0)
    model = ELTLanguageModel(mcfg)
    return model, mcfg, icfg


def test_soft_ce_teacher_gradient_is_zero() -> None:
    """Soft CE is - Σ softmax(teacher) * log_softmax(student). Teacher must be detached."""
    B, T, V = 2, 8, 16
    student = torch.randn(B, T, V, requires_grad=True)
    teacher = torch.randn(B, T, V, requires_grad=True)

    loss = _causal_lm_soft_ce(student, teacher)
    loss.backward()

    assert student.grad is not None
    assert student.grad.abs().sum().item() > 0.0        # student gets real gradient
    # Teacher should receive ZERO gradient because sg(·) was applied.
    assert teacher.grad is None or teacher.grad.abs().sum().item() == 0.0, \
        "teacher should have no gradient (sg operator)"


def test_lambda_schedule_bounds() -> None:
    cfg = ILSDConfig(lambda_init=1.0, lambda_final=0.0, lambda_anneal_steps=1000, warmup_steps=0)
    assert abs(compute_lambda(0, cfg) - 1.0) < 1e-9
    assert abs(compute_lambda(1000, cfg) - 0.0) < 1e-9
    assert abs(compute_lambda(2000, cfg) - 0.0) < 1e-9          # clamped
    assert abs(compute_lambda(500, cfg) - 0.5) < 1e-9


def test_ilsd_total_decomposes_correctly() -> None:
    """For step > warmup, total = L_GT_teacher + λ·L_GT_student + (1-λ)·L_dist (eq. 3)."""
    model, mcfg, icfg = _make_small()
    loss_fn = ILSDLossFn(mcfg, icfg, seed=7)

    input_ids = torch.randint(0, mcfg.vocab_size, (2, 8))
    labels = input_ids.clone()

    out = loss_fn(model, input_ids, labels, step=50)   # past warmup, mid-anneal
    lam = out.lambda_value
    total_expected = (
        out.l_gt_teacher + lam * out.l_gt_student + (1 - lam) * out.l_dist
    )
    # total is the only non-detached one
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
        assert mcfg.L_min <= out.L_int < mcfg.L_max, f"bad L_int={out.L_int}"
    # With L_max=3 and strict_student_below_teacher, valid student = {1, 2}
    assert seen.issubset({1, 2})
