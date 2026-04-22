"""Manifest-driven benchmark helpers for any-time evaluation.

The goal is to keep benchmark execution lightweight and configurable:
`anytime_sweep` can evaluate local JSONL fixtures for tests and real HF
datasets for longer runs, without hard-coding every benchmark schema into the
CLI itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Iterator, Literal

import torch
import yaml

from elt_lm.model import ELTLanguageModel
from elt_lm.verifiers import (
    exact_match_correctness,
    gsm8k_correctness,
    python_exec_correctness,
)


BenchmarkKind = Literal["jsonl", "hf"]
BenchmarkTask = Literal[
    "exact_match",
    "gsm8k",
    "multiple_choice",
    "python_exec",
    "json_match",
]


@dataclass
class BenchmarkSpec:
    name: str
    task: BenchmarkTask
    kind: BenchmarkKind = "jsonl"
    path: str | None = None
    dataset: str | None = None
    config: str | None = None
    configs: list[str] = field(default_factory=list)
    split: str = "test"
    prompt_field: str | None = None
    reference_field: str | None = None
    prompt_template: str | None = None
    reference_template: str | None = None
    limit: int = 0


@dataclass
class BenchmarkCase:
    prompt: str
    reference: str
    task: BenchmarkTask
    benchmark: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    benchmark: str
    task: BenchmarkTask
    L: int
    accuracy: float
    correct: int
    total: int
    latency_ms_per_case: float
    tokens_per_sec: float
    attempts_per_case: float


def load_benchmark_manifest(path: str | Path) -> list[BenchmarkSpec]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    specs = [BenchmarkSpec(**item) for item in raw.get("benchmarks", [])]
    if not specs:
        raise ValueError(f"benchmark manifest produced no benchmarks: {path}")
    return specs


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _iter_hf(spec: BenchmarkSpec) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    if spec.dataset is None:
        raise ValueError(f"{spec.name}: hf benchmark requires `dataset`")
    configs = spec.configs or [spec.config]
    for cfg_name in configs:
        ds = load_dataset(
            spec.dataset,
            cfg_name,
            split=spec.split,
            streaming=True,
        )
        for row in ds:
            if isinstance(row, dict):
                yield row


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _render_template(template: str, row: dict[str, Any]) -> str:
    rendered = template
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        rendered = rendered.replace(
            "{" + field_name + "}",
            _stringify(row.get(field_name, "")),
        )
    return rendered


def _build_case(spec: BenchmarkSpec, row: dict[str, Any]) -> BenchmarkCase | None:
    if spec.prompt_template:
        prompt = _render_template(spec.prompt_template, row).strip()
    elif spec.prompt_field:
        prompt = _stringify(row.get(spec.prompt_field)).strip()
    else:
        prompt = ""

    if spec.reference_template:
        reference = _render_template(spec.reference_template, row).strip()
    elif spec.reference_field:
        reference = _stringify(row.get(spec.reference_field)).strip()
    else:
        reference = ""

    if not prompt or not reference:
        return None
    return BenchmarkCase(
        prompt=prompt,
        reference=reference,
        task=spec.task,
        benchmark=spec.name,
        metadata=row,
    )


def load_benchmark_cases(spec: BenchmarkSpec) -> list[BenchmarkCase]:
    if spec.kind == "jsonl":
        if not spec.path:
            raise ValueError(f"{spec.name}: jsonl benchmark requires `path`")
        rows: Iterable[dict[str, Any]] = _iter_jsonl(Path(spec.path))
    elif spec.kind == "hf":
        if not spec.dataset:
            raise ValueError(f"{spec.name}: hf benchmark requires `dataset`")
        rows = _iter_hf(spec)
    else:
        raise ValueError(f"{spec.name}: unsupported benchmark kind {spec.kind!r}")

    cases: list[BenchmarkCase] = []
    for row in rows:
        case = _build_case(spec, row)
        if case is None:
            continue
        cases.append(case)
        if spec.limit and len(cases) >= spec.limit:
            break
    return cases


_MULTIPLE_CHOICE_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def multiple_choice_correctness(answer_text: str, reference: str) -> float:
    pred = _MULTIPLE_CHOICE_RE.findall(answer_text)
    if not pred:
        return 0.0
    return 1.0 if pred[-1].upper() == reference.strip().upper() else 0.0


def json_match_correctness(answer_text: str, reference: str) -> float:
    try:
        pred = json.loads(answer_text)
        gold = json.loads(reference)
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if pred == gold else 0.0


def score_response(task: BenchmarkTask, response: str, reference: str) -> float:
    if task == "exact_match":
        return exact_match_correctness(response, reference)
    if task == "gsm8k":
        return gsm8k_correctness(response, reference)
    if task == "multiple_choice":
        return multiple_choice_correctness(response, reference)
    if task == "python_exec":
        return python_exec_correctness(response, reference)
    if task == "json_match":
        return json_match_correctness(response, reference)
    raise ValueError(f"unknown benchmark task: {task}")


@torch.no_grad()
def evaluate_benchmark(
    model: ELTLanguageModel,
    tokenizer,
    spec: BenchmarkSpec,
    L: int,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    num_samples: int = 1,
    verifier_retries: int = 0,
) -> BenchmarkResult:
    cases = load_benchmark_cases(spec)
    if not cases:
        raise RuntimeError(f"benchmark {spec.name} has no runnable cases")

    correct = 0
    total_resp_tokens = 0
    total_wall = 0.0
    total_attempts = 0
    max_prompt_len = max(1, model.cfg.max_seq_len - max_new_tokens)

    for case in cases:
        prompt_ids = tokenizer.encode(case.prompt, add_special_tokens=False)
        prompt_ids = prompt_ids[-max_prompt_len:]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attempts_left = max(1, num_samples)
        retries_left = max(0, verifier_retries)
        best_score = -1.0

        while True:
            for _ in range(attempts_left):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    L=L,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=tokenizer.eos_token_id,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                total_wall += time.perf_counter() - t0
                total_attempts += 1

                resp_ids = out_ids[0, input_ids.size(1):].tolist()
                total_resp_tokens += len(resp_ids)
                response = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
                score = score_response(case.task, response, case.reference)
                best_score = max(best_score, score)
                if score >= 1.0:
                    break

            if best_score > 0.0 or retries_left <= 0:
                break
            retries_left -= 1
            attempts_left = 1

        correct += int(best_score > 0.0)

    total = len(cases)
    return BenchmarkResult(
        benchmark=spec.name,
        task=spec.task,
        L=L,
        accuracy=correct / max(1, total),
        correct=correct,
        total=total,
        latency_ms_per_case=(total_wall / max(1, total)) * 1000.0,
        tokens_per_sec=total_resp_tokens / max(1e-9, total_wall),
        attempts_per_case=total_attempts / max(1, total),
    )
