"""Deterministic synthetic preference-pair generator for MILSPEC-style tuning."""

from __future__ import annotations

from dataclasses import dataclass
import random

from elt_lm.posttrain_data import render_chat_text


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    category: str
    standard_refs: list[str]

    def as_record(self) -> dict[str, object]:
        return {
            "bucket": "preference_pairs",
            "mode": "preference",
            "source": "synthetic_milspec",
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_text": render_chat_text(self.prompt, self.chosen),
            "rejected_text": render_chat_text(self.prompt, self.rejected),
            "metadata": {
                "category": self.category,
                "standard_refs": self.standard_refs,
            },
        }


_TASKS = [
    "parse a CSV file of telemetry events into typed Python records",
    "validate and normalize a JSON config for a training job",
    "compute a deterministic summary report from benchmark outputs",
    "filter and sort build artifacts by timestamp and retention policy",
    "load a manifest of datasets and emit normalized examples",
    "wrap a subprocess invocation with timeout and structured errors",
    "summarize an audit log into a concise compliance report",
    "compare two model outputs and compute an exact-match score",
]

_GOOD_TRAITS = [
    "type hints and explicit return types",
    "input validation at public boundaries",
    "structured logging for failure paths",
    "small pure helper functions",
    "clear docstrings describing behavior and errors",
    "deterministic behavior with no hidden randomness",
    "bounded resource usage and timeouts",
    "tests or assertions for invariants",
]

_BAD_TRAITS = [
    "bare except blocks",
    "hidden global state",
    "implicit randomness",
    "missing input validation",
    "shell=True subprocess calls",
    "stringly typed return values",
    "side effects during import time",
    "silent failure handling",
]

_STANDARDS = [
    "MIL-STD-498 traceable software work products",
    "JPL Power of Ten small, simple control flow",
]


def _good_response(task: str, trait_a: str, trait_b: str) -> str:
    return (
        f"Implement a focused Python function to {task}. "
        f"Use {trait_a} and {trait_b}. "
        "Reject malformed input with explicit exceptions, keep behavior deterministic, "
        "and return plain data structures that are easy to test."
    )


def _bad_response(task: str, bad_a: str, bad_b: str) -> str:
    return (
        f"Write quick code to {task}. "
        f"It is acceptable to rely on {bad_a} and {bad_b}. "
        "Skip validation, catch every exception generically, and prioritize brevity over auditability."
    )


def generate_synthetic_preference_pairs(count: int, seed: int = 42) -> list[PreferencePair]:
    rng = random.Random(seed)
    pairs: list[PreferencePair] = []
    for idx in range(count):
        task = rng.choice(_TASKS)
        trait_a, trait_b = rng.sample(_GOOD_TRAITS, k=2)
        bad_a, bad_b = rng.sample(_BAD_TRAITS, k=2)
        prompt = (
            f"[pair {idx:05d}] Produce software-engineering guidance for code that must {task}. "
            "Favor reliable, auditable, and deterministic implementation practices."
        )
        pairs.append(PreferencePair(
            prompt=prompt,
            chosen=_good_response(task, trait_a, trait_b),
            rejected=_bad_response(task, bad_a, bad_b),
            category="milspec_software_practice",
            standard_refs=_STANDARDS,
        ))
    return pairs
