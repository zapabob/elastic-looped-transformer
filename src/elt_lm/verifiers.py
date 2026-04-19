"""Verifier + reward composition for GRPO post-training.

Hack-resistant principles implemented here:

  1. Correctness and format are computed independently, then combined
     **multiplicatively** so that a format-pass with wrong content scores 0.
     This blocks "tag-hack" exploits (pad the answer with <think>...</think>
     boilerplate and nothing substantive).
  2. Length and repetition penalties are **additive** and always ≤ 0, so they
     can only subtract from honest credit.
  3. All verifiers are stateless and deterministic — no judge-LM.

Usage:
    v = CompositeVerifier(task="gsm8k")
    r = v.reward(prompt, response, reference)
    # r is a `RewardBreakdown` dataclass with scalar fields and `total()`
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


_GSM8K_ANS = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_TAG_BLOCK = re.compile(
    r"<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>",
    re.DOTALL,
)


@dataclass
class RewardBreakdown:
    """Per-sample reward components. `total()` is what GRPO sees."""
    correct: float = 0.0         # 0 or 1 — exact match w/ reference
    format: float = 0.0          # 0 or 1 — think/answer tag structure present
    length_penalty: float = 0.0  # ≤ 0
    repeat_penalty: float = 0.0  # ≤ 0

    def total(self) -> float:
        # anti-hack gate: no format credit if the answer is wrong.
        core = self.correct * self.format
        return core + self.length_penalty + self.repeat_penalty


# ---------------------------------------------------------------------------
# format verifier
# ---------------------------------------------------------------------------

def format_score(response: str) -> tuple[float, str]:
    """Return (1.0, answer_text) if the response has a well-formed
    <think>...</think><answer>...</answer> block, else (0.0, "")."""
    m = _TAG_BLOCK.search(response)
    if not m:
        return 0.0, ""
    think = m.group("think").strip()
    answer = m.group("answer").strip()
    if not think or not answer:
        return 0.0, ""
    # extra tag-spam resistance: the answer block must be short relative
    # to the think block, so padding answer tags won't help.
    if len(answer) > 4 * max(1, len(think)):
        return 0.0, ""
    return 1.0, answer


# ---------------------------------------------------------------------------
# correctness verifiers (per-task)
# ---------------------------------------------------------------------------

def _normalize_numeric(s: str) -> str | None:
    s = s.strip().replace(",", "").rstrip(".")
    # pick the last number the model emitted; reduces credit for partial matches
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums:
        return None
    return nums[-1]


def gsm8k_correctness(answer_text: str, reference: str) -> float:
    """GSM8K: reference is the gold `####` line. We compare the final number
    in each (model and gold) after normalization."""
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
    """Generic: case-insensitive trimmed exact match."""
    return 1.0 if answer_text.strip().lower() == reference.strip().lower() else 0.0


TASK_VERIFIERS: dict[str, Callable[[str, str], float]] = {
    "gsm8k": gsm8k_correctness,
    "exact_match": exact_match_correctness,
}


# ---------------------------------------------------------------------------
# Python exec verifier (for coding-agent reward)
#
# Safety boundary: we run generated code in a **separate Python subprocess**
# with a hard wall-clock timeout. This is NOT a full sandbox — a determined
# attacker inside the code can still touch the filesystem or network — but
# for GRPO on synthetic function tasks (the common case) it's adequate:
#   - each call is one-shot and blocked by timeout
#   - the working directory is a fresh tempdir, deleted on return
#   - stdout is captured and discarded; we read only the exit code
# The reference contract is: the generated response must define a function
# of a known name; the reference provides assert-based tests that import
# that function and return exit code 0 iff all assertions pass.
# ---------------------------------------------------------------------------


def _extract_code_block(response: str) -> str:
    """Prefer the content of a fenced ```python ...``` block; fall back to
    the raw response. Returns the stripped body."""
    m = re.search(r"```(?:python|py)?\s*\n?(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def python_exec_correctness(
    answer_text: str, reference: str, timeout_s: float = 3.0
) -> float:
    """Run `{answer_code}\\n{reference_tests}` in a subprocess.

    - `answer_text` is what the model emitted inside `<answer>...</answer>`.
      We strip any markdown code fences automatically.
    - `reference` is a trusted test harness string — assert-based checks that
      import from the candidate module via `from __main__ import ...` is
      NOT required; we literally prepend the candidate source, then the tests.

    Returns 1.0 iff the subprocess exits with status 0 within `timeout_s`.
    """
    code = _extract_code_block(answer_text)
    if not code:
        return 0.0
    full = code + "\n\n" + reference

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "cand.py"
        script.write_text(full, encoding="utf-8")
        try:
            r = subprocess.run(
                [sys.executable, str(script)],
                cwd=td,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 0.0
        except OSError:
            return 0.0
    return 1.0 if r.returncode == 0 else 0.0


TASK_VERIFIERS["python_exec"] = python_exec_correctness


# ---------------------------------------------------------------------------
# penalties
# ---------------------------------------------------------------------------

def length_penalty(response: str, cap: int = 1024, coef: float = 0.001) -> float:
    """0 for responses ≤ cap chars, linearly more negative beyond."""
    over = max(0, len(response) - cap)
    return -coef * over


def repeat_penalty(response: str, n: int = 5, max_per_ngram: int = 3,
                   coef: float = 0.01) -> float:
    """Penalize any n-gram that appears more than `max_per_ngram` times."""
    toks = response.split()
    if len(toks) < n:
        return 0.0
    counts: Counter = Counter()
    for i in range(len(toks) - n + 1):
        counts[tuple(toks[i : i + n])] += 1
    excess = sum(max(0, c - max_per_ngram) for c in counts.values())
    return -coef * excess


# ---------------------------------------------------------------------------
# composite
# ---------------------------------------------------------------------------

class CompositeVerifier:
    """Encapsulates the full reward pipeline for one task.

    The caller supplies `task` ∈ TASK_VERIFIERS. `reward(prompt, response,
    reference)` returns a `RewardBreakdown` for logging; `RewardBreakdown.total()`
    is the scalar fed to GRPO.
    """

    def __init__(
        self,
        task: str = "gsm8k",
        length_cap: int = 1024,
        length_coef: float = 0.001,
        repeat_n: int = 5,
        repeat_max: int = 3,
        repeat_coef: float = 0.01,
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

    def reward(self, prompt: str, response: str, reference: str) -> RewardBreakdown:
        del prompt  # kept in signature for future prompt-aware verifiers
        fmt, answer_text = format_score(response)
        correct = self._correct_fn(answer_text, reference) if fmt > 0 else 0.0
        lp = length_penalty(response, self.length_cap, self.length_coef)
        rp = repeat_penalty(
            response, self.repeat_n, self.repeat_max, self.repeat_coef
        )
        return RewardBreakdown(
            correct=correct,
            format=fmt,
            length_penalty=lp,
            repeat_penalty=rp,
        )
