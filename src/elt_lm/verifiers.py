"""Deterministic verifier composition for GRPO and coding-agent evaluation."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from elt_lm.agent.sandbox import run_python_code, run_python_module


_GSM8K_ANS = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_TAG_BLOCK = re.compile(
    r"<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>",
    re.DOTALL,
)
_REFUSAL_RE = re.compile(
    r"\b(?:i\s+(?:can'?t|cannot|won'?t)\s+help|sorry[, ]+i\s+(?:can'?t|cannot)|i(?:'| a)?m sorry)\b",
    re.IGNORECASE,
)


@dataclass
class RewardBreakdown:
    correct: float = 0.0
    format: float = 0.0
    python_exec: float | None = None
    mypy: float | None = None
    ruff: float | None = None
    bandit: float | None = None
    length_penalty: float = 0.0
    repeat_penalty: float = 0.0

    def verifier_total(self) -> float:
        base_correct = self.python_exec if self.python_exec is not None else self.correct
        core = base_correct * self.format
        for maybe_value in (self.mypy, self.ruff, self.bandit):
            if maybe_value is None:
                continue
            core *= 0.5 + 0.5 * maybe_value
        return core + self.length_penalty + self.repeat_penalty

    def total(self, reward_model_score: float = 0.0,
              reward_alpha: float = 0.0,
              verifier_beta: float = 1.0) -> float:
        return verifier_beta * self.verifier_total() + reward_alpha * reward_model_score


def format_score(response: str) -> tuple[float, str]:
    m = _TAG_BLOCK.search(response)
    if not m:
        return 0.0, ""
    think = m.group("think").strip()
    answer = m.group("answer").strip()
    if not think or not answer:
        return 0.0, ""
    if len(answer) > 4 * max(1, len(think)):
        return 0.0, ""
    return 1.0, answer


def _unwrap_code_or_json_block(response: str) -> str:
    m = re.search(
        r"```(?:python|py|json|rust|rs|go|typescript|ts|csharp|c#|cs)?\s*\n?(.*?)```",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return response.strip()


def canonical_task_answer(task: str, response: str) -> tuple[float, str]:
    if task in {"gsm8k", "exact_math", "mcq_reasoning"}:
        fmt, answer = format_score(response)
        return fmt, answer
    if task == "exact_match":
        fmt, answer = format_score(response)
        if fmt > 0:
            return fmt, answer
        stripped = response.strip()
        return (1.0 if stripped else 0.0, stripped)
    if task == "python_exec":
        fmt, answer = format_score(response)
        candidate = answer if fmt > 0 else response
        unwrapped = _unwrap_code_or_json_block(candidate)
        return (1.0 if unwrapped else 0.0, unwrapped)
    if task == "code_static_spec":
        unwrapped = _unwrap_code_or_json_block(response)
        return (1.0 if unwrapped else 0.0, unwrapped)
    if task == "json_match":
        fmt, answer = format_score(response)
        candidate = answer if fmt > 0 else response
        unwrapped = _unwrap_code_or_json_block(candidate)
        return (1.0 if unwrapped else 0.0, unwrapped)
    return format_score(response)


def _normalize_numeric(s: str) -> str | None:
    s = s.strip().replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums:
        return None
    return nums[-1]


def gsm8k_correctness(answer_text: str, reference: str) -> float:
    m_gold = _GSM8K_ANS.search(reference)
    gold = m_gold.group(1) if m_gold else _normalize_numeric(reference)
    cand = _normalize_numeric(answer_text)
    if gold is None or cand is None:
        return 0.0
    try:
        return 1.0 if abs(float(cand) - float(gold)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


def exact_match_correctness(answer_text: str, reference: str) -> float:
    return 1.0 if answer_text.strip().lower() == reference.strip().lower() else 0.0


def exact_math_correctness(answer_text: str, reference: str) -> float:
    _, candidate = canonical_task_answer("exact_math", answer_text)
    gold = reference.strip()
    if not candidate or not gold:
        return 0.0
    if candidate.strip().lower() == gold.strip().lower():
        return 1.0
    try:
        import sympy  # type: ignore[import-not-found]

        lhs: Any = sympy.sympify(candidate)
        rhs: Any = sympy.sympify(gold)
        if sympy.simplify(lhs - rhs) == 0:
            return 1.0
        if float(lhs.evalf()) == float(rhs.evalf()):
            return 1.0
    except Exception:
        pass
    return gsm8k_correctness(candidate, gold)


def mcq_reasoning_correctness(answer_text: str, reference: str) -> float:
    _, candidate = canonical_task_answer("mcq_reasoning", answer_text)
    m = re.findall(r"\b([A-E])\b", candidate, re.IGNORECASE)
    if not m:
        return 0.0
    return 1.0 if m[-1].upper() == reference.strip().upper() else 0.0


def json_match_correctness(answer_text: str, reference: str) -> float:
    _, candidate = canonical_task_answer("json_match", answer_text)
    try:
        pred = json.loads(candidate)
        gold = json.loads(reference)
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if pred == gold else 0.0


def _extract_code_block(response: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n?(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def python_exec_correctness(answer_text: str, reference: str, timeout_s: float = 3.0) -> float:
    code = _extract_code_block(answer_text)
    if not code:
        return 0.0
    result = run_python_code(code + "\n\n" + reference, timeout_s=timeout_s)
    return 1.0 if (not result.timed_out and result.returncode == 0) else 0.0


def code_static_spec_correctness(answer_text: str, reference: str) -> float:
    code = _unwrap_code_or_json_block(answer_text)
    ref = reference.strip().lower()
    if len(code.strip()) < 40 or len(ref) < 30:
        return 0.0
    has_expected = any(token in ref for token in ("assert", "expect", "equals", "should", "test", "cargo test", "go test", "npm test", "dotnet test"))
    has_stub = bool(re.search(r"\b(todo|pass|return\s+none|panic!\s*\(\"todo|throw\s+new\s+notimplemented)", code, re.IGNORECASE))
    return 1.0 if has_expected and not has_stub else 0.0


def _module_tool_score(module: str, module_args: list[str], code: str,
                       timeout_s: float = 5.0) -> float | None:
    if not _extract_code_block(code):
        return 0.0
    try:
        __import__(module)
    except ImportError:
        return None

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "candidate.py"
        script.write_text(_extract_code_block(code), encoding="utf-8")
        result = run_python_module(
            module,
            [*module_args, str(script)],
            timeout_s=timeout_s,
            cwd=td,
        )
    return 1.0 if (not result.timed_out and result.returncode == 0) else 0.0


def mypy_strict_score(answer_text: str, timeout_s: float = 5.0) -> float | None:
    return _module_tool_score("mypy", ["--strict"], answer_text, timeout_s=timeout_s)


def ruff_check_score(answer_text: str, timeout_s: float = 5.0) -> float | None:
    return _module_tool_score("ruff", ["check"], answer_text, timeout_s=timeout_s)


def bandit_score(answer_text: str, timeout_s: float = 5.0) -> float | None:
    return _module_tool_score("bandit", ["-q"], answer_text, timeout_s=timeout_s)


class VerifierPool:
    """Thin wrapper for optional code-quality verifiers."""

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s

    def mypy(self, answer_text: str) -> float | None:
        return mypy_strict_score(answer_text, timeout_s=self.timeout_s)

    def ruff(self, answer_text: str) -> float | None:
        return ruff_check_score(answer_text, timeout_s=self.timeout_s)

    def bandit(self, answer_text: str) -> float | None:
        return bandit_score(answer_text, timeout_s=self.timeout_s)


TASK_VERIFIERS: dict[str, Callable[[str, str], float]] = {
    "gsm8k": gsm8k_correctness,
    "exact_match": exact_match_correctness,
    "exact_math": exact_math_correctness,
    "mcq_reasoning": mcq_reasoning_correctness,
    "python_exec": python_exec_correctness,
    "code_static_spec": code_static_spec_correctness,
    "json_match": json_match_correctness,
}


def length_penalty(response: str, cap: int = 1024, coef: float = 0.001) -> float:
    over = max(0, len(response) - cap)
    return -coef * over


def repeat_penalty(response: str, n: int = 5, max_per_ngram: int = 3,
                   coef: float = 0.01) -> float:
    toks = response.split()
    if len(toks) < n:
        return 0.0
    counts: Counter = Counter()
    for i in range(len(toks) - n + 1):
        counts[tuple(toks[i:i + n])] += 1
    excess = sum(max(0, c - max_per_ngram) for c in counts.values())
    return -coef * excess


class CompositeVerifier:
    def __init__(
        self,
        task: str = "gsm8k",
        length_cap: int = 1024,
        length_coef: float = 0.001,
        repeat_n: int = 5,
        repeat_max: int = 3,
        repeat_coef: float = 0.01,
        enable_code_quality: bool | None = None,
        verifier_pool: VerifierPool | None = None,
    ) -> None:
        if task not in TASK_VERIFIERS:
            raise ValueError(f"unknown task: {task}")
        self.task = task
        self._correct_fn = TASK_VERIFIERS[task]
        self.length_cap = length_cap
        self.length_coef = length_coef
        self.repeat_n = repeat_n
        self.repeat_max = repeat_max
        self.repeat_coef = repeat_coef
        self.enable_code_quality = (
            task == "python_exec" if enable_code_quality is None else enable_code_quality
        )
        self.pool = verifier_pool or VerifierPool()

    def reward(self, prompt: str, response: str, reference: str) -> RewardBreakdown:
        del prompt
        fmt, answer_text = canonical_task_answer(self.task, response)
        correct = self._correct_fn(answer_text, reference) if fmt > 0 else 0.0
        python_exec = None
        mypy = None
        ruff = None
        bandit = None
        if _REFUSAL_RE.search(answer_text):
            correct = 0.0
        if self.task == "python_exec" and fmt > 0:
            python_exec = python_exec_correctness(answer_text, reference)
            if self.enable_code_quality:
                mypy = self.pool.mypy(answer_text)
                ruff = self.pool.ruff(answer_text)
                bandit = self.pool.bandit(answer_text)
        lp = length_penalty(response, self.length_cap, self.length_coef)
        rp = repeat_penalty(response, self.repeat_n, self.repeat_max, self.repeat_coef)
        return RewardBreakdown(
            correct=correct,
            format=fmt,
            python_exec=python_exec,
            mypy=mypy,
            ruff=ruff,
            bandit=bandit,
            length_penalty=lp,
            repeat_penalty=rp,
        )
