from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from .gguf_distill import (
    DistillQualityError,
    DistillTask,
    build_sft_record,
    evaluate_distill_records,
    validate_distill_record_quality,
)
from .verifiers import TASK_VERIFIERS


LANES: tuple[str, ...] = ("code", "math", "stem_reasoning", "tool_use")
SOURCE_NAME = "synthetic-v2-hard"


@dataclass(frozen=True)
class FailureExample:
    label: str
    response: str
    reason: str


@dataclass(frozen=True)
class HardSyntheticExample:
    task: DistillTask
    example: dict[str, Any]
    failures: tuple[FailureExample, ...] = field(default_factory=tuple)
    difficulty: str = "hard"
    requires_loop_depth: int = 3


def _task(
    lane: str,
    name: str,
    description: str,
    target_kind: str,
    index: int,
) -> DistillTask:
    return DistillTask(
        lane=lane,  # type: ignore[arg-type]
        domain=name,
        description=description,
        target_kind=target_kind,
        tags=[lane, "synthetic_v2_hard", "multi_step", "failure_contrast"],
        target_label="",
        risk_tags=[],
        variant_index=index,
        mode="synthetic",
        variant=f"synthetic_v2_hard_{index}",
    )


def _code_response(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def _math_response(answer: str, reason: str = "This skips one required intermediate step.") -> str:
    return f"<think>{reason}</think><answer>{answer}</answer>"


def _stem_response(choice: str, reason: str = "This selects a plausible distractor before checking the constraint.") -> str:
    return f"<think>{reason}</think><answer>{choice}</answer>"


def _json_response(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _frac(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _code_examples(count: int) -> Iterable[HardSyntheticExample]:
    for i in range(count):
        kind = i % 3
        if kind == 0:
            grace = 1 + (i % 4)
            events = [
                {"service": 2, "start": 1, "end": 3},
                {"service": 1, "start": 0, "end": 4},
                {"service": 1, "start": 4 + grace, "end": 8 + grace},
                {"service": 2, "start": 10, "end": 12},
            ]
            expected = [(1, 0, 8 + grace), (2, 1, 3), (2, 10, 12)]
            code = (
                "def summarize_downtime(events: list[dict[str, int]], grace_minutes: int) -> list[tuple[int, int, int]]:\n"
                "    if grace_minutes < 0:\n"
                "        raise ValueError('grace_minutes must be nonnegative')\n"
                "    by_service: dict[int, list[tuple[int, int]]] = {}\n"
                "    for event in events:\n"
                "        service = event['service']\n"
                "        start = event['start']\n"
                "        end = event['end']\n"
                "        if end < start:\n"
                "            raise ValueError('event end before start')\n"
                "        by_service.setdefault(service, []).append((start, end))\n"
                "    merged: list[tuple[int, int, int]] = []\n"
                "    for service in sorted(by_service):\n"
                "        intervals = sorted(by_service[service])\n"
                "        cur_start, cur_end = intervals[0]\n"
                "        for start, end in intervals[1:]:\n"
                "            if start - cur_end <= grace_minutes:\n"
                "                cur_end = max(cur_end, end)\n"
                "            else:\n"
                "                merged.append((service, cur_start, cur_end))\n"
                "                cur_start, cur_end = start, end\n"
                "        merged.append((service, cur_start, cur_end))\n"
                "    return merged"
            )
            verifier = (
                f"events = {events!r}\n"
                f"assert summarize_downtime(events, {grace}) == {expected!r}\n"
                "try:\n"
                "    summarize_downtime([{'service': 1, 'start': 5, 'end': 4}], 1)\n"
                "except ValueError as exc:\n"
                "    assert str(exc) == 'event end before start'\n"
                "else:\n"
                "    raise AssertionError('expected invalid event error')"
            )
            bad = (
                "def summarize_downtime(events: list[dict[str, int]], grace_minutes: int) -> list[tuple[int, int, int]]:\n"
                "    return [(event['service'], event['start'], event['end']) for event in sorted(events, key=lambda e: (e['service'], e['start']))]"
            )
            yield HardSyntheticExample(
                task=_task("code", "merge_downtime_windows", "Merge per-service outage windows with a grace gap and validation.", "python_exec", i),
                example={
                    "user_request": (
                        "Implement summarize_downtime(events, grace_minutes). Events are dicts with integer "
                        "service, start, and end fields. Validate impossible windows, group by service, sort, "
                        "merge intervals whose gap is at most grace_minutes, and return sorted tuples "
                        f"(service, start, end). Synthetic v2 id {i}."
                    ),
                    "assistant_code": code,
                    "verifier_snippet": verifier,
                    "language": "python",
                },
                failures=(FailureExample("no_interval_merge", _code_response(bad), "Leaves adjacent intervals unmerged and misses invalid-window validation."),),
            )
        elif kind == 1:
            initial = {"alpha": 4 + (i % 3), "beta": 2}
            events = [("gamma", 3), ("alpha", -2), ("beta", 5), ("gamma", -1)]
            expected = {"alpha": initial["alpha"] - 2, "beta": 7, "gamma": 2}
            code = (
                "def reconcile_inventory(initial: dict[str, int], events: list[tuple[str, int]]) -> dict[str, int]:\n"
                "    state = dict(initial)\n"
                "    for sku, delta in events:\n"
                "        next_value = state.get(sku, 0) + delta\n"
                "        if next_value < 0:\n"
                "            raise ValueError(f'negative inventory for {sku}')\n"
                "        state[sku] = next_value\n"
                "    return {sku: state[sku] for sku in sorted(state) if state[sku] != 0}"
            )
            verifier = (
                f"assert reconcile_inventory({initial!r}, {events!r}) == {expected!r}\n"
                "try:\n"
                "    reconcile_inventory({'alpha': 1}, [('alpha', -2)])\n"
                "except ValueError as exc:\n"
                "    assert str(exc) == 'negative inventory for alpha'\n"
                "else:\n"
                "    raise AssertionError('expected negative inventory error')"
            )
            bad = (
                "def reconcile_inventory(initial: dict[str, int], events: list[tuple[str, int]]) -> dict[str, int]:\n"
                "    state = dict(initial)\n"
                "    for sku, delta in events:\n"
                "        state[sku] = state.get(sku, 0) + delta\n"
                "    return state"
            )
            yield HardSyntheticExample(
                task=_task("code", "inventory_delta_reconciliation", "Apply ordered inventory deltas with negative-stock rejection and deterministic output.", "python_exec", i),
                example={
                    "user_request": (
                        "Implement reconcile_inventory(initial, events). Apply ordered (sku, delta) events, "
                        "reject any step that would make stock negative, drop zero-stock SKUs, and return a "
                        f"dictionary ordered by SKU name. Synthetic v2 id {i}."
                    ),
                    "assistant_code": code,
                    "verifier_snippet": verifier,
                    "language": "python",
                },
                failures=(FailureExample("missing_negative_guard", _code_response(bad), "Allows impossible negative inventory and keeps unsorted zero entries."),),
            )
        else:
            tasks = [("extract", 3), ("normalize", 4), ("audit", 2), ("load", 5)]
            capacity = 7 + (i % 2)
            expected = [["extract", "normalize"], ["audit", "load"]] if capacity == 7 else [["extract", "normalize"], ["audit", "load"]]
            code = (
                "def plan_batches(tasks: list[tuple[str, int]], capacity: int) -> list[list[str]]:\n"
                "    if capacity <= 0:\n"
                "        raise ValueError('capacity must be positive')\n"
                "    batches: list[list[str]] = []\n"
                "    current: list[str] = []\n"
                "    used = 0\n"
                "    for name, cost in tasks:\n"
                "        if cost <= 0:\n"
                "            raise ValueError('task cost must be positive')\n"
                "        if cost > capacity:\n"
                "            raise ValueError(f'task too large: {name}')\n"
                "        if current and used + cost > capacity:\n"
                "            batches.append(current)\n"
                "            current = []\n"
                "            used = 0\n"
                "        current.append(name)\n"
                "        used += cost\n"
                "    if current:\n"
                "        batches.append(current)\n"
                "    return batches"
            )
            verifier = (
                f"assert plan_batches({tasks!r}, {capacity}) == {expected!r}\n"
                "try:\n"
                "    plan_batches([('huge', 9)], 4)\n"
                "except ValueError as exc:\n"
                "    assert str(exc) == 'task too large: huge'\n"
                "else:\n"
                "    raise AssertionError('expected oversize task error')"
            )
            bad = (
                "def plan_batches(tasks: list[tuple[str, int]], capacity: int) -> list[list[str]]:\n"
                "    return [[name for name, _cost in tasks]]"
            )
            yield HardSyntheticExample(
                task=_task("code", "capacity_batch_planning", "Pack ordered tasks into capacity-bounded batches while preserving order and rejecting impossible tasks.", "python_exec", i),
                example={
                    "user_request": (
                        "Implement plan_batches(tasks, capacity). Preserve task order, start a new batch "
                        "before capacity is exceeded, reject non-positive capacities or costs, and reject a "
                        f"single task that cannot fit into any batch. Synthetic v2 id {i}."
                    ),
                    "assistant_code": code,
                    "verifier_snippet": verifier,
                    "language": "python",
                },
                failures=(FailureExample("ignores_capacity", _code_response(bad), "Places every task in one batch and never rejects oversize tasks."),),
            )


def _math_examples(count: int) -> Iterable[HardSyntheticExample]:
    for i in range(count):
        kind = i % 3
        if kind == 0:
            prior = Fraction(1 + (i % 3), 10)
            hit = Fraction(3, 4)
            false = Fraction(1, 5)
            posterior = prior * hit / (prior * hit + (1 - prior) * false)
            final = posterior * Fraction(2, 3) + (1 - posterior) * Fraction(1, 6)
            answer = _frac(final)
            shortcut = prior * Fraction(2, 3) + (1 - prior) * Fraction(1, 6)
            wrong = f"{float(shortcut):.6f}"
            question = (
                f"A monitoring event has prior incident probability {_frac(prior)}. A detector fires with "
                f"probability {_frac(hit)} during an incident and {_frac(false)} otherwise. After a detector "
                "fire, a reviewer catches the true incident with probability 2/3 and mistakenly escalates a "
                "non-incident with probability 1/6. Compute the exact probability that the case is escalated."
            )
            reasoning = (
                f"First update the incident probability with Bayes: P(I|fire)=({_frac(prior)}*{_frac(hit)})/"
                f"(({_frac(prior)}*{_frac(hit)})+(1-{_frac(prior)})*{_frac(false)}). "
                "Then condition the escalation probability on incident versus non-incident after the fire. "
                f"The weighted probability is {_frac(posterior)}*2/3 + (1-{_frac(posterior)})*1/6 = {answer}."
            )
        elif kind == 1:
            x0 = 2 + (i % 5)
            a = 2
            b = 3 + (i % 4)
            n = 4
            x = x0
            trace = [x]
            for _ in range(n):
                x = a * x + b
                trace.append(x)
            answer = str(x - (i % 3 + 1))
            wrong = str(trace[-2] - (i % 3 + 1))
            question = (
                f"A loop state starts at x0={x0}. Each loop applies x <- {a}x + {b}. "
                f"After {n} loops, subtract {i % 3 + 1} as a readout correction. What exact integer is read out?"
            )
            reasoning = (
                f"Unroll the recurrence through every loop: {trace}. The final loop state is {trace[-1]}, "
                f"then the readout correction subtracts {i % 3 + 1}, giving {answer}."
            )
        else:
            total = 60 + i
            a_count = 25 + (i % 7)
            b_count = 22 + (i % 5)
            both = 9 + (i % 4)
            c_given_union = Fraction(2 + (i % 3), 5)
            union = a_count + b_count - both
            answer_frac = Fraction(union, total) * c_given_union
            answer = _frac(answer_frac)
            shortcut = Fraction(a_count + b_count, total) * c_given_union
            wrong = f"{float(shortcut):.6f}"
            question = (
                f"In a population of {total}, set A has {a_count}, set B has {b_count}, and the overlap has "
                f"{both}. A second-stage property C occurs with probability {_frac(c_given_union)} among A union B "
                "and never outside A union B. Compute P(C)."
            )
            reasoning = (
                f"Use inclusion-exclusion before the second stage: |A union B|={a_count}+{b_count}-{both}={union}. "
                f"Thus P(A union B)={union}/{total}. Multiplying by P(C | A union B)={_frac(c_given_union)} gives {answer}."
            )
        yield HardSyntheticExample(
            task=_task("math", f"multi_step_math_{kind}", "Exact multi-step arithmetic with an explicit intermediate state.", "exact_math", i),
            example={"question": f"{question} Synthetic v2 id {i}.", "reasoning": reasoning, "final_answer": answer, "reference": answer},
            failures=(FailureExample("skipped_intermediate", _math_response(wrong), "Uses a tempting one-step shortcut and skips a required intermediate update."),),
        )


def _stem_examples(count: int) -> Iterable[HardSyntheticExample]:
    for i in range(count):
        correct = "ABCD"[i % 4]
        scenario = i % 3
        if scenario == 0:
            question = (
                "A heat pump report says the compressor work input stayed fixed while the useful heat delivered "
                "rose after insulation was added. Which interpretation best separates energy conservation from "
                "coefficient-of-performance improvement?"
            )
            correct_text = "Energy is conserved; insulation reduces losses, so more delivered heat per unit work improves COP."
            distractors = [
                "Energy conservation is violated because delivered heat became larger than work input.",
                "The compressor must have secretly consumed less work because COP cannot change otherwise.",
                "The reservoir temperature is irrelevant, so insulation cannot affect delivered heat.",
            ]
        elif scenario == 1:
            question = (
                "A buffer receives log events with bursty arrivals. The mean arrival rate is below service rate, "
                "but bursts overflow the buffer unless batching is enabled. Which systems explanation is strongest?"
            )
            correct_text = "Mean-rate stability is not enough; burst variance and finite capacity require smoothing or backpressure."
            distractors = [
                "If average arrival rate is lower, overflow is mathematically impossible.",
                "Batching helps only by deleting events before they enter the queue.",
                "Finite buffers behave like infinite queues whenever utilization is below one.",
            ]
        else:
            question = (
                "A chemical sensor has high sensitivity but moderate specificity. In a low-prevalence setting, "
                "many positive alerts are false. Which conclusion follows from conditional probability?"
            )
            correct_text = "Low prevalence can make the positive predictive value modest even when sensitivity is high."
            distractors = [
                "High sensitivity alone guarantees that most positive alerts are true positives.",
                "Specificity affects only negative alerts and not the positive predictive value.",
                "Prevalence is irrelevant after a positive test result has already occurred.",
            ]
        choices: list[str] = []
        distractor_iter = iter(distractors)
        for letter in "ABCD":
            if letter == correct:
                choices.append(f"{letter}. {correct_text}")
            else:
                choices.append(f"{letter}. {next(distractor_iter)}")
        reasoning = (
            "Compare the tempting one-factor explanation against the multi-step mechanism. "
            "The correct option preserves the governing law, then accounts for the second-stage constraint "
            "that changes the observed outcome."
        )
        wrong = "ABCD"[((i % 4) + 1) % 4]
        yield HardSyntheticExample(
            task=_task("stem_reasoning", f"two_stage_stem_{scenario}", "STEM multiple choice where the correct answer requires rejecting a plausible single-factor shortcut.", "mcq_reasoning", i),
            example={"question": f"{question} Synthetic v2 id {i}.", "choices": choices, "reasoning": reasoning, "final_choice": correct, "reference": correct},
            failures=(FailureExample("single_factor_distractor", _stem_response(wrong), "Chooses the nearest single-factor explanation instead of the two-stage causal account."),),
        )


def _tool_examples(count: int) -> Iterable[HardSyntheticExample]:
    specs = [
        (
            "mcp.files.search",
            "Search the project for the source of GRPO reward collapse, but do not modify files.",
            {"query": "grpo_step reward_std adv_abs_mean", "root": "H:/elt_data/runs", "limit": 8, "read_only": True},
            {"tool_name": "mcp.files.write", "arguments": {"path": "H:/elt_data/runs/fix.txt", "content": "retry", "dry_run": False}},
        ),
        (
            "agent.plan.execute",
            "Create a dry-run execution plan for rebuilding only hard synthetic v2 manifests after tests pass.",
            {"plan_id": "synthetic-v2-hard-refresh", "max_steps": 5, "dry_run": True, "requires_tests": True},
            {"tool_name": "agent.plan.execute", "arguments": {"plan_id": "synthetic-v2-hard-refresh", "max_steps": 5, "dry_run": False}},
        ),
        (
            "mcp.metrics.query",
            "Read GRPO metrics and return recent reward variance without launching training.",
            {"run_dir": "H:/elt_data/runs/grpo_side_lora_code_synthetic_gb", "metric": "reward_std", "window": 16, "read_only": True},
            {"tool_name": "mcp.metrics.query", "arguments": {"run_dir": "H:/elt_data/runs/grpo_side_lora_code_synthetic_gb", "metric": "loss", "window": 1}},
        ),
    ]
    for i in range(count):
        tool, request, arguments, bad_obj = specs[i % len(specs)]
        arguments = dict(arguments)
        arguments["request_id"] = f"synthetic-v2-{i}"
        response_obj = {"tool_name": tool, "arguments": arguments}
        yield HardSyntheticExample(
            task=_task("tool_use", f"agentic_tool_disambiguation_{i % len(specs)}", "Choose the safe read-only or dry-run tool call after resolving constraints.", "json_match", i),
            example={
                "user_request": f"{request} Synthetic v2 id {i}.",
                "tool_name": tool,
                "arguments": arguments,
                "reference": response_obj,
            },
            failures=(FailureExample("unsafe_or_wrong_tool", _json_response(bad_obj), "Drops a safety constraint or selects the wrong tool/metric."),),
        )


def generate_lane_examples(lane: str, count: int) -> list[HardSyntheticExample]:
    if lane == "code":
        return list(_code_examples(count))
    if lane == "math":
        return list(_math_examples(count))
    if lane == "stem_reasoning":
        return list(_stem_examples(count))
    if lane == "tool_use":
        return list(_tool_examples(count))
    raise ValueError(f"unsupported lane: {lane}")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _failure_score(task_name: str, response: str, reference: str) -> float:
    verifier = TASK_VERIFIERS[task_name]
    return float(verifier(response, reference))


def _build_failure_record(
    *,
    record: dict[str, Any],
    item: HardSyntheticExample,
    failure: FailureExample,
    split: str,
) -> dict[str, Any]:
    score = _failure_score(str(record["task"]), failure.response, str(record["reference"]))
    metadata = dict(record.get("metadata") or {})
    metadata.update({
        "difficulty": item.difficulty,
        "requires_loop_depth": item.requires_loop_depth,
        "failure_label": failure.label,
        "failure_reason": failure.reason,
    })
    return {
        "source": SOURCE_NAME,
        "mode": "failure_contrast",
        "split": split,
        "task": record["task"],
        "prompt": record["prompt"],
        "reference": record["reference"],
        "bad_response": failure.response,
        "failure_label": failure.label,
        "failure_reason": failure.reason,
        "expected_score": 0.0,
        "observed_score": score,
        "metadata": metadata,
    }


def _benchmark_row(record: dict[str, Any], item: HardSyntheticExample) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    metadata.update({
        "difficulty": item.difficulty,
        "requires_loop_depth": item.requires_loop_depth,
        "failure_modes": [failure.label for failure in item.failures],
    })
    return {
        "prompt": record["prompt"],
        "reference": record["reference"],
        "task": record["task"],
        "bucket": f"{record['bucket']}_v2_hard",
        "source": SOURCE_NAME,
        "metadata": metadata,
    }


def build_synthetic_v2_bundle(
    *,
    output_root: Path,
    records_per_lane: int,
    val_ratio: float,
    lanes: Iterable[str] = LANES,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Any] = {
        "source": SOURCE_NAME,
        "output_root": str(output_root),
        "records_per_lane": records_per_lane,
        "lanes": {},
    }
    split_mod = max(2, round(1.0 / max(val_ratio, 1e-6)))
    for lane in lanes:
        train_records: list[dict[str, Any]] = []
        val_records: list[dict[str, Any]] = []
        train_failures: list[dict[str, Any]] = []
        val_failures: list[dict[str, Any]] = []
        benchmark_rows: list[dict[str, Any]] = []
        reject_counts: Counter[str] = Counter()
        failure_scores: list[float] = []
        task_counts: Counter[str] = Counter()
        loop_depths: Counter[int] = Counter()

        for idx, item in enumerate(generate_lane_examples(lane, records_per_lane)):
            split = "val" if idx % split_mod == 0 else "train"
            record = build_sft_record(task=item.task, example=item.example, teacher_name=SOURCE_NAME, split=split)
            metadata = dict(record.get("metadata") or {})
            metadata.update({
                "difficulty": item.difficulty,
                "requires_loop_depth": item.requires_loop_depth,
                "failure_modes": [failure.label for failure in item.failures],
            })
            record["metadata"] = metadata
            try:
                validate_distill_record_quality(record, item.example, item.task, None)
            except DistillQualityError as exc:
                reject_counts[str(exc)] += 1
                continue
            failure_records = [
                _build_failure_record(record=record, item=item, failure=failure, split=split)
                for failure in item.failures
            ]
            failure_scores.extend(float(row["observed_score"]) for row in failure_records)
            if split == "val":
                val_records.append(record)
                val_failures.extend(failure_records)
                benchmark_rows.append(_benchmark_row(record, item))
            else:
                train_records.append(record)
                train_failures.extend(failure_records)
            task_counts[str(record["task"])] += 1
            loop_depths[int(metadata["requires_loop_depth"])] += 1

        lane_dir = output_root / lane
        benchmarks_dir = lane_dir / "benchmarks"
        _write_jsonl(lane_dir / "distill_train.jsonl", train_records)
        _write_jsonl(lane_dir / "distill_val.jsonl", val_records)
        _write_jsonl(lane_dir / "failures_train.jsonl", train_failures)
        _write_jsonl(lane_dir / "failures_val.jsonl", val_failures)
        cases_path = benchmarks_dir / f"synthetic_v2_hard_{lane}_val_cases.jsonl"
        manifest_path = benchmarks_dir / f"synthetic_v2_hard_{lane}_val_manifest.yaml"
        _write_jsonl(cases_path, benchmark_rows)
        manifest = {
            "benchmarks": [{
                "name": f"synthetic_v2_hard_{lane}_val",
                "kind": "jsonl",
                "task": benchmark_rows[0]["task"] if benchmark_rows else "exact_match",
                "path": str(cases_path),
                "prompt_field": "prompt",
                "reference_field": "reference",
            }]
        }
        manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
        all_records = [*train_records, *val_records]
        summary = evaluate_distill_records(all_records, quality_counters=reject_counts, run_verifiers=True)
        summary.update({
            "lane": lane,
            "source": SOURCE_NAME,
            "difficulty": "hard",
            "records": len(all_records),
            "failure_records": len(train_failures) + len(val_failures),
            "failure_expected_zero_rate": (
                sum(1 for score in failure_scores if score == 0.0) / len(failure_scores)
                if failure_scores else 0.0
            ),
            "task_counts": dict(task_counts),
            "loop_depth_counts": {str(key): value for key, value in loop_depths.items()},
            "benchmark_cases_path": str(cases_path),
            "benchmark_manifest_path": str(manifest_path),
            "rejected": dict(reject_counts),
        })
        (lane_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (lane_dir / "README.md").write_text(
            f"# Synthetic v2 hard: {lane}\n\n"
            "Verifier-backed hard synthetic data for ELT loop refinement and GRPO reward variance.\n\n"
            f"- Correct SFT records: {summary['records']}\n"
            f"- Failure contrast records: {summary['failure_records']}\n"
            f"- Verifier pass rate: {summary.get('verifier_pass_rate', 0.0):.3f}\n"
            f"- Failure expected-zero rate: {summary['failure_expected_zero_rate']:.3f}\n"
            f"- Benchmark manifest: `{manifest_path}`\n",
            encoding="utf-8",
        )
        overall["lanes"][lane] = summary
    overall["total_records"] = sum(int(item.get("records", 0)) for item in overall["lanes"].values())
    overall["total_failure_records"] = sum(int(item.get("failure_records", 0)) for item in overall["lanes"].values())
    (output_root / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    return overall


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build hard synthetic v2 ELT/GRPO datasets.")
    parser.add_argument("--output-root", type=Path, default=Path("H:/elt_data/synthetic_v2_hard"))
    parser.add_argument("--records-per-lane", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--lanes", nargs="*", default=list(LANES), choices=list(LANES))
    args = parser.parse_args()
    summary = build_synthetic_v2_bundle(
        output_root=args.output_root,
        records_per_lane=args.records_per_lane,
        val_ratio=args.val_ratio,
        lanes=args.lanes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
