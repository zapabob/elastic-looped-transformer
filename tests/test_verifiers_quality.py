from __future__ import annotations

from elt_lm.verifiers import (
    CompositeVerifier,
    RewardBreakdown,
    VerifierPool,
    exact_math_correctness,
    json_match_correctness,
    mcq_reasoning_correctness,
)


class _StubPool(VerifierPool):
    def __init__(self, mypy: float | None, ruff: float | None, bandit: float | None):
        super().__init__(timeout_s=0.1)
        self._mypy = mypy
        self._ruff = ruff
        self._bandit = bandit

    def mypy(self, answer_text: str) -> float | None:
        del answer_text
        return self._mypy

    def ruff(self, answer_text: str) -> float | None:
        del answer_text
        return self._ruff

    def bandit(self, answer_text: str) -> float | None:
        del answer_text
        return self._bandit


def test_reward_breakdown_combines_verifier_and_reward_model_scores() -> None:
    reward = RewardBreakdown(correct=1.0, format=1.0, length_penalty=-0.1)
    total = reward.total(reward_model_score=0.5, reward_alpha=0.3, verifier_beta=0.7)
    assert abs(total - ((0.7 * 0.9) + (0.3 * 0.5))) < 1e-6


def test_composite_verifier_uses_code_quality_channels() -> None:
    verifier = CompositeVerifier(
        task="python_exec",
        enable_code_quality=True,
        verifier_pool=_StubPool(mypy=1.0, ruff=0.0, bandit=1.0),
    )
    response = (
        "<think>Plan the function, validate the signature, and then emit only the code block.</think>"
        "<answer>```python\ndef add(a, b):\n    return a + b\n```</answer>"
    )
    reference = "assert add(1, 2) == 3\n"
    reward = verifier.reward(prompt="add two numbers", response=response, reference=reference)
    assert reward.python_exec == 1.0
    assert reward.mypy == 1.0
    assert reward.ruff == 0.0
    assert reward.bandit == 1.0
    assert reward.verifier_total() < 1.0


def test_exact_math_and_mcq_verifiers_accept_structured_outputs() -> None:
    assert exact_math_correctness("<think>Subtract 1.</think><answer>2</answer>", "2") == 1.0
    assert mcq_reasoning_correctness("<think>Conservation of energy.</think><answer>B</answer>", "B") == 1.0
    assert exact_math_correctness("2", "2") == 1.0
    assert mcq_reasoning_correctness("B", "B") == 1.0


def test_composite_verifier_scores_structured_math_and_mcq_outputs() -> None:
    math_reward = CompositeVerifier(task="exact_math").reward(
        prompt="",
        response="<think>Compute exactly.</think><answer>3/7</answer>",
        reference="3/7",
    )
    stem_reward = CompositeVerifier(task="mcq_reasoning").reward(
        prompt="",
        response="<think>Compare the options.</think><answer>C</answer>",
        reference="C",
    )

    assert math_reward.verifier_total() == 1.0
    assert stem_reward.verifier_total() == 1.0


def test_json_match_verifier_handles_raw_json() -> None:
    payload = '{"tool_name":"shell_command","arguments":{"command":"pwd"}}'
    assert json_match_correctness(payload, payload) == 1.0


def test_composite_verifier_does_not_reward_blanket_refusal() -> None:
    verifier = CompositeVerifier(task="json_match")
    reward = verifier.reward(
        prompt="Choose the best tool call.",
        response='{"tool_name":"refuse","arguments":{"message":"I can\'t help with that."}}',
        reference='{"tool_name":"shell_command","arguments":{"command":"pwd"}}',
    )
    assert reward.correct == 0.0
