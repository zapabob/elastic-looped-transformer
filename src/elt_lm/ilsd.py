"""Intra-Loop Self-Distillation (ILSD) loss for the causal-LM ELT port.

Paper form (section 4, eq. 3):

    L_ILSD = L_GT(F_{N,L_max}(x), y)
           + lambda * L_GT(F_{N,L_int}(x), y)
           + (1 - lambda) * L_dist(F_{N,L_int}(x), sg(F_{N,L_max}(x)))

    L_int ~ U(L_min, L_max)

This implementation keeps the paper's core structure and layers in three small
stabilizers that matter for looped decoding:

- teacher-only temperature + tiny uniform smoothing for soft distillation
- a soft entropy floor on student logits
- optional hidden-state consistency across adjacent loops
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from elt_lm.config import ILSDConfig, ModelConfig
from elt_lm.model import ELTLanguageModel


@dataclass
class ILSDLossOutput:
    total: Tensor
    l_gt_teacher: Tensor
    l_gt_student: Tensor
    l_dist: Tensor
    l_entropy: Tensor
    l_local: Tensor
    lambda_value: float
    L_int: int


def compute_lambda(step: int, cfg: ILSDConfig) -> float:
    """Linear curriculum: lambda_init -> lambda_final over lambda_anneal_steps."""
    if cfg.lambda_anneal_steps <= 0:
        return cfg.lambda_final
    t = min(step, cfg.lambda_anneal_steps) / cfg.lambda_anneal_steps
    return cfg.lambda_init * (1.0 - t) + cfg.lambda_final * t


def sample_L_int(model_cfg: ModelConfig, ilsd_cfg: ILSDConfig, rng: random.Random) -> int:
    """Sample the student loop count uniformly from [L_min, L_max]."""
    lo, hi = model_cfg.L_min, model_cfg.L_max
    if hi == lo:
        return lo
    if ilsd_cfg.strict_student_below_teacher:
        hi = hi - 1
        if hi < lo:
            return lo
    return rng.randint(lo, hi)


def _masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return x.mean()
    mask_f = mask.to(dtype=x.dtype)
    return (x * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _causal_lm_ce(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    """Shift-target cross entropy on next-token prediction."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def _causal_lm_soft_ce(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    tau_teacher: float = 1.0,
    uniform_mix: float = 0.0,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Masked soft cross-entropy with teacher-only temperature and smoothing."""
    s_shift = student_logits[..., :-1, :].contiguous()
    t_shift = teacher_logits[..., :-1, :].contiguous()

    teacher_p = torch.softmax(t_shift.float().detach() / max(tau_teacher, 1e-6), dim=-1)
    if uniform_mix > 0.0:
        vocab = teacher_p.size(-1)
        teacher_p = (1.0 - uniform_mix) * teacher_p + uniform_mix / vocab

    student_logp = torch.log_softmax(s_shift.float(), dim=-1)
    per_token = -(teacher_p * student_logp).sum(dim=-1)
    return _masked_mean(per_token, valid_mask)


def _normalized_entropy_from_logits(logits: Tensor) -> Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1)
    return entropy / math.log(logits.size(-1))


def _entropy_floor_value(lambda_value: float, cfg: ILSDConfig) -> float:
    progress = 1.0 - lambda_value
    return cfg.entropy_floor_start * (1.0 - progress) + cfg.entropy_floor_end * progress


def _entropy_floor_penalty(
    logits: Tensor,
    *,
    floor_value: float,
    valid_mask: Tensor | None = None,
) -> Tensor:
    entropy = _normalized_entropy_from_logits(logits[..., :-1, :])
    penalty = F.relu(torch.as_tensor(floor_value, device=logits.device, dtype=entropy.dtype) - entropy)
    return _masked_mean(penalty, valid_mask)


def _hidden_local_consistency(
    hidden_states: tuple[Tensor, ...] | None,
    *,
    metric: str,
    valid_mask: Tensor | None = None,
) -> Tensor | None:
    if hidden_states is None or len(hidden_states) < 2:
        return None

    terms: list[Tensor] = []
    for prev_hidden, next_hidden in zip(hidden_states[:-1], hidden_states[1:]):
        prev_shift = prev_hidden[..., :-1, :].float()
        next_shift = next_hidden[..., :-1, :].float()
        if metric == "cosine":
            per_token = 1.0 - F.cosine_similarity(prev_shift, next_shift, dim=-1, eps=1e-8)
        elif metric == "mse":
            per_token = F.mse_loss(prev_shift, next_shift, reduction="none").mean(dim=-1)
        else:
            raise ValueError(f"unknown local consistency metric: {metric}")
        terms.append(_masked_mean(per_token, valid_mask))

    return torch.stack(terms).mean()


class ILSDLossFn:
    """Callable wrapper that manages lambda-schedule state and rng."""

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
                l_entropy=zero,
                l_local=zero,
                lambda_value=1.0,
                L_int=L_max,
            )

        L_int = sample_L_int(self.model_cfg, self.ilsd_cfg, self.rng)
        lam = compute_lambda(step - self.ilsd_cfg.warmup_steps, self.ilsd_cfg)
        need_all_loop_hidden = self.ilsd_cfg.local_consistency_weight > 0.0

        out = model(
            input_ids,
            L=L_max,
            return_hidden_at=L_int,
            return_all_loop_hidden=need_all_loop_hidden,
        )
        teacher_logits = out.logits
        student_logits = out.intermediate_logits
        assert student_logits is not None, "return_hidden_at must yield student logits"

        valid_mask = (labels[..., 1:] != ignore_index)

        l_gt_teacher = _causal_lm_ce(teacher_logits, labels, ignore_index=ignore_index)
        l_gt_student = _causal_lm_ce(student_logits, labels, ignore_index=ignore_index)

        if L_int == L_max:
            zero = torch.zeros((), device=l_gt_teacher.device, dtype=l_gt_teacher.dtype)
            l_dist = zero
            l_entropy = zero
            l_local = zero
        else:
            l_dist = _causal_lm_soft_ce(
                student_logits,
                teacher_logits,
                tau_teacher=self.ilsd_cfg.distill_teacher_temp,
                uniform_mix=self.ilsd_cfg.distill_uniform_mix,
                valid_mask=valid_mask,
            )
            if self.ilsd_cfg.entropy_floor_weight > 0.0:
                floor_value = _entropy_floor_value(lam, self.ilsd_cfg)
                l_entropy = _entropy_floor_penalty(
                    student_logits,
                    floor_value=floor_value,
                    valid_mask=valid_mask,
                )
            else:
                l_entropy = torch.zeros((), device=l_gt_teacher.device, dtype=l_gt_teacher.dtype)

            local = _hidden_local_consistency(
                out.per_loop_hidden,
                metric=self.ilsd_cfg.local_consistency_metric,
                valid_mask=valid_mask,
            )
            if local is None or self.ilsd_cfg.local_consistency_weight <= 0.0:
                l_local = torch.zeros((), device=l_gt_teacher.device, dtype=l_gt_teacher.dtype)
            else:
                l_local = local.to(device=l_gt_teacher.device, dtype=l_gt_teacher.dtype)

        distill_total = (
            l_dist
            + self.ilsd_cfg.entropy_floor_weight * l_entropy
            + self.ilsd_cfg.local_consistency_weight * l_local
        )
        total = l_gt_teacher + lam * l_gt_student + (1.0 - lam) * distill_total

        return ILSDLossOutput(
            total=total,
            l_gt_teacher=l_gt_teacher.detach(),
            l_gt_student=l_gt_student.detach(),
            l_dist=l_dist.detach(),
            l_entropy=l_entropy.detach(),
            l_local=l_local.detach(),
            lambda_value=lam,
            L_int=L_int,
        )
