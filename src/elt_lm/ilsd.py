"""Intra-Loop Self-Distillation (ILSD) loss — arXiv:2604.09168 §4, eq. (3).

Paper form (masked models, eq. 3 + CE specialization):

    L_ILSD = L_GT(F_{N,L_max}(x), y)
           + λ · L_GT(F_{N,L_int}(x), y)
           + (1 - λ) · L_dist(F_{N,L_int}(x), sg(F_{N,L_max}(x)))

    L_int ~ U(L_min, L_max)                                       (S^3 sampling)

Causal-LM specialization:

    L_GT(F) = - Σ_t log P_F(y_t | y_<t)                     (shift-target CE)

    L_dist(F_s, F_t) = - Σ_{t,v} softmax(logits_t)_v · log softmax(logits_s)_v
                      (teacher distribution is detached; equivalent to KL up to
                       an entropy constant — the paper writes it as soft CE.)

λ is linearly annealed from lambda_init to lambda_final over lambda_anneal_steps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from elt_lm.config import ILSDConfig, ModelConfig
from elt_lm.model import ELTLanguageModel


@dataclass
class ILSDLossOutput:
    total: Tensor                  # scalar total loss used for .backward()
    l_gt_teacher: Tensor           # scalar
    l_gt_student: Tensor           # scalar (== l_gt_teacher if L_int == L_max)
    l_dist: Tensor                 # scalar (0 if L_int == L_max)
    lambda_value: float            # current lambda
    L_int: int                     # sampled student loop count this step


def compute_lambda(step: int, cfg: ILSDConfig) -> float:
    """Linear curriculum: lambda_init -> lambda_final over lambda_anneal_steps."""
    if cfg.lambda_anneal_steps <= 0:
        return cfg.lambda_final
    t = min(step, cfg.lambda_anneal_steps) / cfg.lambda_anneal_steps
    return cfg.lambda_init * (1.0 - t) + cfg.lambda_final * t


def sample_L_int(model_cfg: ModelConfig, ilsd_cfg: ILSDConfig, rng: random.Random) -> int:
    """S^3: sample the student loop count uniformly from [L_min, L_max].

    With strict_student_below_teacher=True we clamp to [L_min, L_max - 1] to keep
    the distillation term non-trivial. If L_min == L_max we just return that.
    """
    lo, hi = model_cfg.L_min, model_cfg.L_max
    if hi == lo:
        return lo
    if ilsd_cfg.strict_student_below_teacher:
        hi = hi - 1
        if hi < lo:
            return lo
    return rng.randint(lo, hi)


def _causal_lm_ce(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    """Shift-target cross entropy on next-token prediction.

    logits: (B, T, V)
    labels: (B, T)  — token-level; positions to ignore should be set to ignore_index
    """
    shift_logits = logits[..., :-1, :].contiguous()          # (B, T-1, V)
    shift_labels = labels[..., 1:].contiguous()              # (B, T-1)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def _causal_lm_soft_ce(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """Soft cross-entropy: teacher probs (stop-grad) × student log-probs.

    Both tensors (B, T, V). Returns scalar (mean over (B, T-1) positions).

    The teacher side is detached here — this is eq. (3)'s sg(·) operator.
    """
    s_shift = student_logits[..., :-1, :].contiguous()
    t_shift = teacher_logits[..., :-1, :].contiguous()

    # compute in fp32 for numerical stability (log-softmax under bf16 can underflow)
    teacher_p = torch.softmax(t_shift.float().detach(), dim=-1)
    student_logp = torch.log_softmax(s_shift.float(), dim=-1)

    # - Σ_v p_t * log p_s, averaged over (B, T-1)
    per_token = -(teacher_p * student_logp).sum(dim=-1)      # (B, T-1)
    return per_token.mean()


class ILSDLossFn:
    """Callable wrapper that manages λ-schedule state and rng.

    Usage:
        loss_fn = ILSDLossFn(model_cfg, ilsd_cfg, seed=42)
        out = loss_fn(model, input_ids, labels, step=global_step)
        out.total.backward()
    """

    def __init__(self, model_cfg: ModelConfig, ilsd_cfg: ILSDConfig, seed: int = 0):
        self.model_cfg = model_cfg
        self.ilsd_cfg = ilsd_cfg
        self.rng = random.Random(seed)

    def __call__(
        self,
        model: ELTLanguageModel,
        input_ids: Tensor,
        labels: Tensor,
        step: int,
        ignore_index: int = -100,
    ) -> ILSDLossOutput:
        L_max = self.model_cfg.L_max

        # During warmup, run teacher-only.
        in_warmup = step < self.ilsd_cfg.warmup_steps or not self.ilsd_cfg.enabled
        if in_warmup:
            out = model(input_ids, L=L_max, return_hidden_at=None)
            l_gt_teacher = _causal_lm_ce(out.logits, labels, ignore_index=ignore_index)
            zero = torch.zeros((), device=l_gt_teacher.device, dtype=l_gt_teacher.dtype)
            return ILSDLossOutput(
                total=l_gt_teacher,
                l_gt_teacher=l_gt_teacher,
                l_gt_student=zero,
                l_dist=zero,
                lambda_value=1.0,
                L_int=L_max,
            )

        # Post-warmup: S^3 sample + joint ILSD loss.
        L_int = sample_L_int(self.model_cfg, self.ilsd_cfg, self.rng)
        lam = compute_lambda(step - self.ilsd_cfg.warmup_steps, self.ilsd_cfg)

        out = model(input_ids, L=L_max, return_hidden_at=L_int)
        teacher_logits = out.logits
        student_logits = out.intermediate_logits
        assert student_logits is not None, "return_hidden_at must yield student logits"

        l_gt_teacher = _causal_lm_ce(teacher_logits, labels, ignore_index=ignore_index)
        l_gt_student = _causal_lm_ce(student_logits, labels, ignore_index=ignore_index)

        if L_int == L_max:
            # Student == Teacher path; distillation term is zero by construction.
            l_dist = torch.zeros((), device=l_gt_teacher.device, dtype=l_gt_teacher.dtype)
        else:
            l_dist = _causal_lm_soft_ce(student_logits, teacher_logits)

        total = l_gt_teacher + lam * l_gt_student + (1.0 - lam) * l_dist

        return ILSDLossOutput(
            total=total,
            l_gt_teacher=l_gt_teacher.detach(),
            l_gt_student=l_gt_student.detach(),
            l_dist=l_dist.detach(),
            lambda_value=lam,
            L_int=L_int,
        )
