"""GRPO loss + verifier correctness tests.

Covers the anti-hacking invariants we care about:
  - advantage normalization produces zero mean + unit std (on non-degenerate
    inputs), and exactly zero when rewards are constant (no signal).
  - KL unbiased estimator is non-negative and zero when policies match.
  - Clipped surrogate actually clips when ratio drifts.
  - Reward composition: format=1 but correct=0 gives 0 (tag-hack prevention).
"""

from __future__ import annotations

import torch

from elt_lm.grpo import (
    GRPOOutput,
    gather_token_logprobs,
    group_advantage,
    grpo_loss,
    kl_unbiased,
)
from elt_lm.verifiers import (
    CompositeVerifier,
    format_score,
    gsm8k_correctness,
    length_penalty,
    python_exec_correctness,
    repeat_penalty,
)


# ---------------------------------------------------------------------------
# advantage
# ---------------------------------------------------------------------------

def test_group_advantage_zero_mean() -> None:
    r = torch.tensor([1.0, 0.0, 0.5, 0.25, 0.75])
    a = group_advantage(r)
    assert torch.allclose(a.mean(), torch.tensor(0.0), atol=1e-6)
    # unit-ish std (up to our eps)
    assert abs(a.std(unbiased=False).item() - 1.0) < 1e-3


def test_group_advantage_constant_rewards_is_zero() -> None:
    r = torch.tensor([0.5, 0.5, 0.5, 0.5])
    a = group_advantage(r)
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-6)


# ---------------------------------------------------------------------------
# KL
# ---------------------------------------------------------------------------

def test_kl_unbiased_nonneg() -> None:
    torch.manual_seed(0)
    lp_theta = torch.randn(4, 8) - 1.0
    lp_ref = torch.randn(4, 8) - 1.0
    k = kl_unbiased(lp_theta, lp_ref)
    assert (k >= -1e-7).all()


def test_kl_unbiased_zero_when_equal() -> None:
    lp = torch.randn(3, 5) - 2.0
    k = kl_unbiased(lp, lp.clone())
    assert torch.allclose(k, torch.zeros_like(k), atol=1e-6)


# ---------------------------------------------------------------------------
# gather helper
# ---------------------------------------------------------------------------

def test_gather_token_logprobs_matches_softmax() -> None:
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 5)
    actions = torch.tensor([[0, 2, 4], [1, 3, 0]])
    got = gather_token_logprobs(logits, actions)
    expected = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(got, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# grpo_loss end-to-end
# ---------------------------------------------------------------------------

def test_grpo_loss_shapes_and_clip() -> None:
    torch.manual_seed(0)
    B, T, V = 4, 6, 17
    logits_theta = torch.randn(B, T, V, requires_grad=True)
    logits_old = logits_theta.detach().clone()
    logits_ref = logits_theta.detach().clone()
    actions = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

    out = grpo_loss(
        logits_theta=logits_theta,
        logits_old=logits_old,
        logits_ref=logits_ref,
        actions=actions,
        response_mask=mask,
        advantages=advantages,
        clip_eps=0.2,
        kl_beta=0.05,
    )
    assert isinstance(out, GRPOOutput)
    # with theta == old, ratio == 1 everywhere → nothing is clipped
    assert out.clip_frac.item() < 1e-6
    # with theta == ref, KL is zero
    assert out.kl.item() < 1e-6
    # loss is backprop-able
    out.loss.backward()
    assert logits_theta.grad is not None


def test_grpo_loss_clips_when_policy_diverges() -> None:
    torch.manual_seed(0)
    B, T, V = 2, 4, 7
    # make theta very different from old
    logits_old = torch.randn(B, T, V)
    logits_theta = logits_old + 5.0 * torch.randn(B, T, V)
    logits_theta.requires_grad_(True)
    logits_ref = logits_old.clone()
    actions = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T)
    advantages = torch.tensor([1.0, -1.0])
    out = grpo_loss(
        logits_theta=logits_theta,
        logits_old=logits_old,
        logits_ref=logits_ref,
        actions=actions,
        response_mask=mask,
        advantages=advantages,
        clip_eps=0.2,
        kl_beta=0.05,
    )
    assert out.clip_frac.item() > 0.0


# ---------------------------------------------------------------------------
# verifier: format + correctness composition
# ---------------------------------------------------------------------------

def test_format_score_accepts_wellformed() -> None:
    resp = "<think>step by step reasoning here</think><answer>42</answer>"
    fmt, ans = format_score(resp)
    assert fmt == 1.0
    assert ans == "42"


def test_format_score_rejects_missing_tags() -> None:
    assert format_score("just plain text")[0] == 0.0
    assert format_score("<think>hi</think> no answer block")[0] == 0.0
    assert format_score("<answer>42</answer> no think")[0] == 0.0


def test_format_score_rejects_answer_tag_spam() -> None:
    # attacker stuffs the answer tag with garbage to skew scoring
    resp = "<think>a</think><answer>" + ("x" * 500) + "</answer>"
    fmt, _ = format_score(resp)
    assert fmt == 0.0


def test_gsm8k_correctness_last_number_wins() -> None:
    # model dumps scratch numbers then a clean final
    assert gsm8k_correctness("so we get 7 then 12, answer 42", "#### 42") == 1.0
    assert gsm8k_correctness("answer 41", "#### 42") == 0.0


def test_composite_correct_zero_with_format_one_gives_zero_total() -> None:
    v = CompositeVerifier(task="gsm8k")
    resp = "<think>fake reasoning</think><answer>99</answer>"
    r = v.reward(prompt="Q: what is 1+1?", response=resp, reference="#### 2")
    assert r.format == 1.0
    assert r.correct == 0.0
    # gate: total must be ≤ 0 (length/repeat penalties only subtract)
    assert r.total() <= 0.0


def test_composite_format_zero_with_correct_substring_gives_zero_total() -> None:
    # model produced the right number but no format tags → no credit
    v = CompositeVerifier(task="gsm8k")
    resp = "The answer is 42."
    r = v.reward(prompt="Q", response=resp, reference="#### 42")
    assert r.format == 0.0
    assert r.correct == 0.0
    assert r.total() <= 0.0


def test_composite_full_credit_path() -> None:
    v = CompositeVerifier(task="gsm8k")
    resp = "<think>1+1=2</think><answer>2</answer>"
    r = v.reward(prompt="Q", response=resp, reference="#### 2")
    assert r.correct == 1.0
    assert r.format == 1.0
    assert r.total() > 0.9


# ---------------------------------------------------------------------------
# penalties
# ---------------------------------------------------------------------------

def test_length_penalty_zero_below_cap() -> None:
    assert length_penalty("x" * 100, cap=1024) == 0.0


def test_length_penalty_negative_above_cap() -> None:
    p = length_penalty("x" * 2048, cap=1024, coef=0.001)
    assert p < 0


def test_repeat_penalty_triggers_on_loops() -> None:
    # "ok ok ok ..." repeated many times → 5-gram repeats
    resp = ("the answer is " * 20).strip()
    p = repeat_penalty(resp, n=5, max_per_ngram=3, coef=0.01)
    assert p < 0


def test_repeat_penalty_zero_on_normal_text() -> None:
    resp = "the quick brown fox jumps over the lazy dog"
    assert repeat_penalty(resp) == 0.0


# ---------------------------------------------------------------------------
# python_exec_correctness (coding-agent reward)
# ---------------------------------------------------------------------------

def test_python_exec_passes_on_correct_function() -> None:
    ans = "def add(a, b):\n    return a + b\n"
    tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
    assert python_exec_correctness(ans, tests) == 1.0


def test_python_exec_fails_on_wrong_function() -> None:
    ans = "def add(a, b):\n    return a - b\n"
    tests = "assert add(2, 3) == 5\n"
    assert python_exec_correctness(ans, tests) == 0.0


def test_python_exec_strips_code_fence() -> None:
    ans = "```python\ndef sq(x):\n    return x * x\n```"
    tests = "assert sq(4) == 16\n"
    assert python_exec_correctness(ans, tests) == 1.0


def test_python_exec_times_out() -> None:
    ans = "def loop():\n    while True:\n        pass\n"
    tests = "loop()\n"
    # 0.5s budget: the infinite loop must trip the timeout guard
    assert python_exec_correctness(ans, tests, timeout_s=0.5) == 0.0


def test_python_exec_exception_gives_zero() -> None:
    ans = "def boom():\n    raise ValueError('x')\n"
    tests = "boom()\n"
    assert python_exec_correctness(ans, tests) == 0.0


def test_python_exec_empty_answer_gives_zero() -> None:
    assert python_exec_correctness("", "pass\n") == 0.0
