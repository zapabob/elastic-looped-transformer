"""GRPO (Group Relative Policy Optimization) loss for post-SFT reasoning RL.

Reference: Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models* (arXiv:2402.03300), §4.1.

## The math (single prompt, G rollouts)

For each prompt `q` we sample a group `{o_1, ..., o_G}` of responses from
`π_θ_old` and compute a scalar reward `R_i` per response.

- Group-relative advantage (value-network-free):

      Â_i = (R_i - mean_j R_j) / (std_j R_j + δ)

  which is **shared across all tokens of o_i** (sequence-level advantage).

- Policy ratio and clipped surrogate (per-token):

      r_t = π_θ(o_{i,t} | q, o_{i,<t})
            / π_θ_old(o_{i,t} | q, o_{i,<t})
      L_clip = -min( r_t Â_i,  clip(r_t, 1-ε, 1+ε) Â_i )

- Per-token KL penalty vs frozen reference π_ref (DeepSeek unbiased form):

      KL_t = exp(lp_ref - lp_θ) - (lp_ref - lp_θ) - 1

  `lp_*` are log-probs of the sampled action. This estimator is ≥ 0, zero
  iff π_θ == π_ref, and unbiased for the true KL.

- Final loss (masked mean over response tokens):

      L = mean_t,i [ L_clip + β · KL_t ]

## Anti-hack guarantees

- `π_ref` is the frozen SFT checkpoint; β stays ≥ 0.05 so the policy cannot
  drift into exploit distributions without paying.
- The advantage uses **shared** reward per response: no per-token reward
  shaping, so the model can't rig a single token to game the objective.
- Rewards themselves (see `verifiers.py`) already gate format credit by
  correctness; this module treats rewards as given scalars.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# advantage
# ---------------------------------------------------------------------------

def group_advantage(
    rewards: Tensor, eps: float = 1e-6
) -> Tensor:
    """Compute Â_i = (R_i - mean(R)) / (std(R) + eps) across a group.

    `rewards`: shape (G,). Returns shape (G,).

    When all rewards are equal the advantage is exactly 0 (no learning signal),
    which is the correct behavior — nothing to reinforce or suppress.
    """
    if rewards.numel() == 0:
        return rewards
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


# ---------------------------------------------------------------------------
# KL estimator
# ---------------------------------------------------------------------------

def kl_unbiased(
    lp_theta: Tensor, lp_ref: Tensor
) -> Tensor:
    """Per-token unbiased KL estimator used in DeepSeekMath GRPO.

        KL_t = exp(lp_ref - lp_theta) - (lp_ref - lp_theta) - 1

    Always ≥ 0; equals 0 iff lp_ref == lp_theta pointwise.

    Inputs: gathered log-probs at the sampled action, shape (B, T) each.
    """
    diff = lp_ref - lp_theta
    return diff.exp() - diff - 1.0


# ---------------------------------------------------------------------------
# per-sequence log-probs helper
# ---------------------------------------------------------------------------

def gather_token_logprobs(
    logits: Tensor, actions: Tensor
) -> Tensor:
    """logits: (B, T, V), actions: (B, T) int64 → (B, T) log π(a_t | s_t)."""
    logits_f = logits.float()
    action_logits = logits_f.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return action_logits - torch.logsumexp(logits_f, dim=-1)


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

@dataclass
class GRPOOutput:
    loss: Tensor
    policy_loss: Tensor
    kl: Tensor
    clip_frac: Tensor       # fraction of tokens where the surrogate was clipped
    adv_abs_mean: Tensor    # diagnostic


def grpo_loss(
    logits_theta: Tensor,      # (B, T, V) current policy
    logits_old: Tensor,        # (B, T, V) behavior policy (no grad)
    logits_ref: Tensor,        # (B, T, V) frozen reference (no grad)
    actions: Tensor,           # (B, T) int64
    response_mask: Tensor,     # (B, T) 1 where it's a response token (skip prompt + pad)
    advantages: Tensor,        # (B,) — one scalar per response (shared across tokens)
    clip_eps: float = 0.2,
    kl_beta: float = 0.05,
) -> GRPOOutput:
    """Compute the full GRPO loss over a batch of (possibly padded) responses.

    The batch can be one group (B = G) for a single prompt, or a flattened
    concatenation of multiple groups — in the latter case the caller is
    responsible for computing `advantages` per-group before flattening.
    """
    lp_theta = gather_token_logprobs(logits_theta, actions)
    with torch.no_grad():
        lp_old = gather_token_logprobs(logits_old, actions)
        lp_ref = gather_token_logprobs(logits_ref, actions)

    return grpo_loss_from_action_logprobs(
        lp_theta=lp_theta,
        lp_old=lp_old,
        lp_ref=lp_ref,
        response_mask=response_mask,
        advantages=advantages,
        clip_eps=clip_eps,
        kl_beta=kl_beta,
    )


def grpo_loss_from_action_logprobs(
    lp_theta: Tensor,
    lp_old: Tensor,
    lp_ref: Tensor,
    response_mask: Tensor,
    advantages: Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.05,
) -> GRPOOutput:
    """Compute GRPO after old/ref have been reduced to action log-probs."""
    ratio = (lp_theta - lp_old).exp()                      # (B, T)
    adv = advantages.detach().unsqueeze(-1)                # (B, 1)
    # clipped surrogate (we MAXIMIZE ratio·A, so loss = -min(...))
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.min(unclipped, clipped)
    policy_per_tok = -surrogate                            # (B, T)

    kl_per_tok = kl_unbiased(lp_theta, lp_ref)             # (B, T)

    # masked mean over response tokens only; normalize per-sample so long
    # responses don't dominate the batch loss
    mask = response_mask.to(lp_theta.dtype)
    denom = mask.sum(dim=-1).clamp_min(1.0)                # (B,)
    pol_seq = (policy_per_tok * mask).sum(dim=-1) / denom   # (B,)
    kl_seq = (kl_per_tok * mask).sum(dim=-1) / denom        # (B,)

    policy_loss = pol_seq.mean()
    kl = kl_seq.mean()
    loss = policy_loss + kl_beta * kl

    with torch.no_grad():
        clipped_flag = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float()
        clip_frac = (clipped_flag * mask).sum() / mask.sum().clamp_min(1.0)
        adv_abs_mean = advantages.abs().mean()

    return GRPOOutput(
        loss=loss,
        policy_loss=policy_loss.detach(),
        kl=kl.detach(),
        clip_frac=clip_frac,
        adv_abs_mean=adv_abs_mean,
    )
