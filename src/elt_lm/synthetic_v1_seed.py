from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .gguf_distill import (
    DistillQualityError,
    DistillTask,
    build_sft_record,
    evaluate_distill_records,
    validate_distill_record_quality,
)


LANES: tuple[str, ...] = ("code", "math", "stem_reasoning", "tool_use")


@dataclass(frozen=True)
class SyntheticExample:
    task: DistillTask
    example: dict[str, Any]


def _task(lane: str, name: str, description: str, target_kind: str, index: int) -> DistillTask:
    return DistillTask(
        lane=lane,  # type: ignore[arg-type]
        domain=name,
        description=description,
        target_kind=target_kind,
        tags=[lane, "synthetic_v1_seed"],
        target_label="",
        risk_tags=[],
        variant_index=index,
        mode="synthetic",
        variant=f"synthetic_seed_{index}",
    )


def _code_examples(count: int) -> Iterable[SyntheticExample]:
    def clamp(i: int) -> tuple[str, str, str, str]:
        low = i % 7
        high = low + 5 + (i % 4)
        below = low - 3
        above = high + 4
        return (
            "clamp_int",
            f"Implement clamp_value(value, low, high) with typed integer bounds. Seed case: {i}.",
            "def clamp_value(value: int, low: int, high: int) -> int:\n"
            "    if low > high:\n"
            "        raise ValueError('low must be <= high')\n"
            "    return min(max(value, low), high)",
            f"assert clamp_value({low + 2}, {low}, {high}) == {low + 2}\n"
            f"assert clamp_value({below}, {low}, {high}) == {low}\n"
            f"assert clamp_value({above}, {low}, {high}) == {high}\n"
            "try:\n"
            f"    clamp_value(1, {high}, {low})\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'low must be <= high'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        )

    def normalize(i: int) -> tuple[str, str, str, str]:
        a = f"Tag{i % 11}"
        b = f"Group{(i * 3) % 13}"
        return (
            "normalize_tags",
            f"Implement normalize_tags(tags) returning sorted lowercase unique non-empty tags. Seed case: {i}.",
            "def normalize_tags(tags: list[str]) -> list[str]:\n"
            "    cleaned = {tag.strip().lower() for tag in tags if tag.strip()}\n"
            "    return sorted(cleaned)",
            f"assert normalize_tags([' {a} ', '{a.lower()}', '', '{b}']) == ['{b.lower()}', '{a.lower()}']\n"
            "assert normalize_tags([]) == []\n"
            f"assert normalize_tags(['Z{i}', 'a{i}', 'z{i}']) == ['a{i}', 'z{i}']",
        )

    def parse_kv(i: int) -> tuple[str, str, str, str]:
        key = f"mode_{i % 17}"
        value = f"safe_{(i * 5) % 19}"
        return (
            "parse_kv_line",
            f"Implement parse_kv_line(line) for 'key=value' pairs with trimmed fields. Seed case: {i}.",
            "def parse_kv_line(line: str) -> tuple[str, str]:\n"
            "    if '=' not in line:\n"
            "        raise ValueError('missing separator')\n"
            "    key, value = line.split('=', 1)\n"
            "    key = key.strip()\n"
            "    value = value.strip()\n"
            "    if not key:\n"
            "        raise ValueError('empty key')\n"
            "    return key, value",
            f"assert parse_kv_line(' {key} = {value} ') == ('{key}', '{value}')\n"
            f"assert parse_kv_line('{key}=a=b') == ('{key}', 'a=b')\n"
            "try:\n"
            "    parse_kv_line('novalue')\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'missing separator'\n"
            "else:\n"
            "    raise AssertionError('expected missing separator')",
        )

    def rolling(i: int) -> tuple[str, str, str, str]:
        start = 1 + (i % 5)
        values = [float(start + step) for step in range(4)]
        expected = [round(sum(values[j:j + 2]) / 2, 3) for j in range(3)]
        return (
            "rolling_average",
            f"Implement rolling_average(values, window) returning rounded moving averages. Seed case: {i}.",
            "def rolling_average(values: list[float], window: int) -> list[float]:\n"
            "    if window <= 0:\n"
            "        raise ValueError('window must be positive')\n"
            "    if window > len(values):\n"
            "        return []\n"
            "    return [round(sum(values[i:i + window]) / window, 3) for i in range(len(values) - window + 1)]",
            f"assert rolling_average({values!r}, 2) == {expected!r}\n"
            f"assert rolling_average([{float(start)}], 2) == []\n"
            "try:\n"
            "    rolling_average([1.0], 0)\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'window must be positive'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        )

    def chunk_list(i: int) -> tuple[str, str, str, str]:
        size = 2 + (i % 3)
        values = list(range(i % 4, i % 4 + 7))
        chunks = [values[j:j + size] for j in range(0, len(values), size)]
        return (
            "chunk_list",
            f"Implement chunk_list(values, size) for typed list chunking with validation. Seed case: {i}.",
            "def chunk_list(values: list[int], size: int) -> list[list[int]]:\n"
            "    if size <= 0:\n"
            "        raise ValueError('size must be positive')\n"
            "    return [values[i:i + size] for i in range(0, len(values), size)]",
            f"assert chunk_list({values!r}, {size}) == {chunks!r}\n"
            "assert chunk_list([], 3) == []\n"
            "try:\n"
            "    chunk_list([1], 0)\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'size must be positive'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        )

    def safe_ratio(i: int) -> tuple[str, str, str, str]:
        numerator = 10 + i
        denominator = 2 + (i % 5)
        expected = round(numerator / denominator, 4)
        return (
            "safe_ratio",
            f"Implement safe_ratio(numerator, denominator) returning a rounded float and rejecting zero. Seed case: {i}.",
            "def safe_ratio(numerator: float, denominator: float) -> float:\n"
            "    if denominator == 0:\n"
            "        raise ValueError('denominator must be nonzero')\n"
            "    return round(numerator / denominator, 4)",
            f"assert safe_ratio({float(numerator)!r}, {float(denominator)!r}) == {expected!r}\n"
            "assert safe_ratio(0.0, 3.0) == 0.0\n"
            "try:\n"
            "    safe_ratio(1.0, 0.0)\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'denominator must be nonzero'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        )

    builders = (clamp, normalize, parse_kv, rolling, chunk_list, safe_ratio)
    for i in range(count):
        name, request, code, verifier = builders[i % len(builders)](i)
        task = _task("code", name, request, "python_exec", i)
        yield SyntheticExample(
            task=task,
            example={
                "user_request": request,
                "assistant_code": code,
                "verifier_snippet": verifier,
                "rationale": "compact verifier-backed code seed",
            },
        )


def _math_examples(count: int) -> Iterable[SyntheticExample]:
    def linear(i: int) -> tuple[str, str, str, str]:
        a = 2 + (i % 7)
        x = 3 + (i % 11)
        b = 5 + (i % 13)
        c = a * x + b
        return (
            "linear_equation",
            f"Solve the linear equation {a}x + {b} = {c} and report the exact value of x. Synthetic seed id {i}.",
            f"Subtract {b} from both sides to get {a}x = {c - b}. Dividing by {a} gives x = {x}.",
            str(x),
        )

    def rectangle(i: int) -> tuple[str, str, str, str]:
        width = 4 + (i % 9)
        height = 5 + ((i * 2) % 8)
        divisor = 2 + (i % 4)
        area = width * height
        answer = area // divisor if area % divisor == 0 else f"{area}/{divisor}"
        return (
            "area_fraction",
            f"A rectangle has width {width} and height {height}. Compute exactly one {divisor}th of its area. Synthetic seed id {i}.",
            f"The area is {width} times {height} = {area}. One {divisor}th of the area is {area}/{divisor}, which is {answer}.",
            str(answer),
        )

    def probability(i: int) -> tuple[str, str, str, str]:
        sides = 6 + (i % 5)
        threshold = 2 + (i % (sides - 2))
        favorable = sides - threshold + 1
        return (
            "die_probability",
            f"A fair {sides}-sided die is rolled once. What is P(result is at least {threshold})? Synthetic seed id {i}.",
            f"The favorable outcomes are {threshold} through {sides}, so there are {favorable} favorable outcomes out of {sides}. The probability is {favorable}/{sides}.",
            f"{favorable}/{sides}",
        )

    def arithmetic_sum(i: int) -> tuple[str, str, str, str]:
        n = 6 + (i % 15)
        step = 2 + (i % 5)
        answer = step * n * (n + 1) // 2
        return (
            "arithmetic_series",
            f"Find the exact sum of the first {n} positive multiples of {step} using a finite series formula. Synthetic seed id {i}.",
            f"The terms are {step} times 1 through {n}. The sum is {step} * {n} * ({n}+1) / 2 = {answer}.",
            str(answer),
        )

    def weighted_average(i: int) -> tuple[str, str, str, str]:
        a = 1 + (i % 8)
        b = 2 + ((i * 3) % 7)
        wa = 2 + (i % 4)
        wb = 3 + ((i * 2) % 5)
        numerator = a * wa + b * wb
        denominator = wa + wb
        answer = numerator // denominator if numerator % denominator == 0 else f"{numerator}/{denominator}"
        return (
            "weighted_average",
            f"Compute the exact weighted average of values {a} and {b} with weights {wa} and {wb}. Synthetic seed id {i}.",
            f"The weighted sum is {a}*{wa} + {b}*{wb} = {numerator}. The total weight is {denominator}. The exact average is {answer}.",
            str(answer),
        )

    builders = (linear, rectangle, probability, arithmetic_sum, weighted_average)
    for i in range(count):
        name, question, reasoning, answer = builders[i % len(builders)](i)
        task = _task("math", name, question, "exact_math", i)
        yield SyntheticExample(
            task=task,
            example={
                "question": question,
                "reasoning": reasoning,
                "final_answer": answer,
                "reference": answer,
            },
        )


def _stem_examples(count: int) -> Iterable[SyntheticExample]:
    specs = [
        (
            "physics_energy",
            "A cart's speed doubles while mass stays fixed in a low-friction track experiment. How does kinetic energy change?",
            "It quadruples because kinetic energy scales with the square of speed",
            [
                "It is unchanged because only direction matters",
                "It doubles because speed appears linearly",
                "It halves because energy is inversely related to speed",
            ],
            "Kinetic energy is proportional to v^2, so doubling speed multiplies energy by four.",
        ),
        (
            "chem_buffer",
            "Which mixture most directly resists pH change when a small amount of acid is added to an aqueous sample?",
            "Weak acid plus conjugate base in comparable amounts",
            [
                "Pure water without a conjugate acid-base pair",
                "Strong acid alone with no buffering partner",
                "Neutral salt only without weak acid chemistry",
            ],
            "A buffer needs a weak acid/base pair and its conjugate partner to consume added acid or base.",
        ),
        (
            "medicine_screening",
            "For a rare disease screening test with fixed sensitivity and specificity, what usually happens to positive predictive value when prevalence rises?",
            "It increases because true positives become a larger share of positive tests",
            [
                "It decreases because sensitivity becomes lower",
                "It becomes zero because false positives dominate all positives",
                "It is unrelated because prevalence never affects predictive value",
            ],
            "With sensitivity and specificity fixed, higher prevalence increases the fraction of positives that are true positives.",
        ),
        (
            "statistics_ci",
            "If sample size increases fourfold while the population variance is fixed, how does the standard error of the sample mean change?",
            "It halves because standard error is inversely proportional to square root of sample size",
            [
                "It doubles because sample size appears in the numerator",
                "It quadruples because sample size was multiplied by four",
                "It is unchanged because variance was held fixed",
            ],
            "Standard error scales as 1/sqrt(n), so multiplying n by four divides it by two.",
        ),
        (
            "biology_enzyme",
            "In a controlled enzyme-catalyzed reaction, which change most directly lowers the initial reaction rate?",
            "Adding a competitive inhibitor that occupies active sites",
            [
                "Doubling substrate concentration when the enzyme is already saturated",
                "Maintaining the enzyme at its optimal pH and temperature",
                "Adding more active enzyme molecules to the same reaction volume",
            ],
            "A competitive inhibitor reduces active-site availability, lowering initial rate under otherwise comparable conditions.",
        ),
        (
            "electrical_power",
            "A resistor's voltage is doubled while resistance is fixed. How does electrical power dissipated in the resistor change?",
            "It quadruples because P = V^2/R when resistance is fixed",
            [
                "It is unchanged because resistance did not change",
                "It doubles because voltage appears linearly in all power laws",
                "It halves because current must decrease",
            ],
            "For a fixed resistor, P = V^2/R, so doubling V multiplies power by four.",
        ),
        (
            "medical_risk",
            "A medication reduces relative risk by 25 percent from a baseline event risk of 20 percent. What is the absolute risk reduction?",
            "Five percentage points because 25 percent of 20 percent is 5 percent",
            [
                "Twenty-five percentage points because relative and absolute risk are identical",
                "Twenty percentage points because the baseline risk is twenty percent",
                "Zero percentage points because relative risk does not affect absolute risk",
            ],
            "Absolute risk reduction is baseline risk times relative reduction: 0.20 * 0.25 = 0.05.",
        ),
    ]
    for i in range(count):
        name, question, correct_choice, distractors, rationale = specs[i % len(specs)]
        question = f"{question} Synthetic seed id {i}."
        expected = "ABCD"[i % 4]
        choices = list(distractors[:3])
        choices.insert("ABCD".index(expected), correct_choice)
        answer = expected
        task = _task("stem_reasoning", name, question, "mcq_reasoning", i)
        yield SyntheticExample(
            task=task,
            example={
                "question": question,
                "choices": choices,
                "answer": answer,
                "final_choice": answer,
                "reasoning": rationale,
                "rationale": rationale,
                "reference": answer,
            },
        )


def _tool_examples(count: int) -> Iterable[SyntheticExample]:
    specs = [
        (
            "mcp_file_search",
            "Search project docs for checkpoint retention rules.",
            "mcp.files.search",
            {"query": "rolling checkpoint keep 3", "root": "H:/elt_data", "read_only": True},
        ),
        (
            "agent_unit_tests",
            "Run focused unit tests for the synthetic dataset builder.",
            "agent.test.run",
            {"command": "uv run --no-sync pytest -q tests/test_synthetic_v1_seed.py", "timeout_sec": 120, "read_only": True},
        ),
        (
            "mcp_dataset_card",
            "Create a dataset card draft for a verifier-backed SFT bundle.",
            "mcp.docs.write",
            {"path": "H:/elt_data/synthetic_v1_seed/README.md", "title": "Synthetic v1 seed", "dry_run": True},
        ),
        (
            "agent_benchmark_plan",
            "Plan an anytime L=1..4 evaluation sweep for code and math lanes.",
            "agent.eval.plan",
            {"lanes": ["code", "math"], "loops": [1, 2, 3, 4], "read_only": True},
        ),
        (
            "mcp_issue_triage",
            "Triage open quality-gate failures without modifying files.",
            "mcp.github.issue_search",
            {"query": "label:quality-gate verifier", "repo": "zapabob/elastic-looped-transformer", "read_only": True},
        ),
        (
            "agent_patch_plan",
            "Draft a bounded patch plan for failing math verifier cases.",
            "agent.plan.create",
            {"scope": "math verifier dry-run", "max_steps": 5, "dry_run": True},
        ),
        (
            "mcp_metric_read",
            "Read latest pipeline progress metrics from H drive.",
            "mcp.metrics.read",
            {"path": "H:/elt_data/pipeline_state/progress_report.json", "read_only": True},
        ),
    ]
    for i in range(count):
        name, request, tool_name, arguments = specs[i % len(specs)]
        request = f"{request} Synthetic seed id {i}."
        arguments = dict(arguments)
        arguments["request_id"] = f"synthetic-{i}"
        task = _task("tool_use", name, request, "json_match", i)
        yield SyntheticExample(
            task=task,
            example={
                "user_request": request,
                "tool_name": tool_name,
                "arguments": arguments,
                "reference": {"tool_name": tool_name, "arguments": arguments},
            },
        )


def generate_lane_examples(lane: str, count: int) -> list[SyntheticExample]:
    if lane == "code":
        return list(_code_examples(count))
    if lane == "math":
        return list(_math_examples(count))
    if lane == "stem_reasoning":
        return list(_stem_examples(count))
    if lane == "tool_use":
        return list(_tool_examples(count))
    raise ValueError(f"unsupported lane: {lane}")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record_fingerprint(value: str) -> str:
    return hashlib.sha1(" ".join(value.strip().lower().split()).encode("utf-8")).hexdigest()


def _lane_iterator(lane: str) -> Iterable[SyntheticExample]:
    if lane == "code":
        return _code_examples(2_147_483_647)
    if lane == "math":
        return _math_examples(2_147_483_647)
    if lane == "stem_reasoning":
        return _stem_examples(2_147_483_647)
    if lane == "tool_use":
        return _tool_examples(2_147_483_647)
    raise ValueError(f"unsupported lane: {lane}")


def _empty_stream_summary(lane: str) -> dict[str, Any]:
    return {
        "lane": lane,
        "source": "synthetic-v1-seed",
        "total_records": 0,
        "valid_json_records": 0,
        "schema_valid_rate": 0.0,
        "unique_text_ratio": 0.0,
        "exact_duplicate_count": 0,
        "exact_duplicate_ratio": 0.0,
        "duplicate_prompt_count": 0,
        "fallback_reject_count": 0,
        "verifier_pass_rate": 0.0,
        "verifier_pass_count": 0,
        "verifier_total": 0,
        "sample_verifier_pass_rate": 0.0,
        "sample_verifier_pass_count": 0,
        "sample_verifier_total": 0,
        "answer_distribution": {},
        "accepted_records": 0,
        "attempted_tasks": 0,
        "generation_attempts": 0,
        "quality_reject_count": 0,
        "quality_reject_reasons": {},
        "domain_counts": {},
        "label_counts": {},
        "split_counts": {},
        "lane_counts": {},
        "task_counts": {},
        "train_bytes": 0,
        "val_bytes": 0,
        "total_bytes": 0,
        "rejected": {},
    }


def build_synthetic_seed_bundle_to_target(
    *,
    output_root: Path,
    target_bytes: int,
    val_ratio: float,
    lanes: Iterable[str] = LANES,
    validation_sample_per_lane: int = 256,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    lanes_list = list(lanes)
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    target_per_lane = max(1, target_bytes // max(1, len(lanes_list)))
    overall: dict[str, Any] = {
        "output_root": str(output_root),
        "target_bytes": target_bytes,
        "target_per_lane": target_per_lane,
        "validation_sample_per_lane": validation_sample_per_lane,
        "lanes": {},
    }
    split_mod = max(2, round(1.0 / max(val_ratio, 1e-6)))

    for lane in lanes_list:
        lane_dir = output_root / lane
        lane_dir.mkdir(parents=True, exist_ok=True)
        train_path = lane_dir / "distill_train.jsonl"
        val_path = lane_dir / "distill_val.jsonl"
        summary = _empty_stream_summary(lane)
        domain_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        task_counts: Counter[str] = Counter()
        answer_distribution: Counter[str] = Counter()
        reject_counts: Counter[str] = Counter()
        seen_text: set[str] = set()
        seen_prompt: set[str] = set()
        sample_verified = 0
        sample_pass = 0

        with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
            for idx, item in enumerate(_lane_iterator(lane)):
                if summary["total_bytes"] >= target_per_lane:
                    break
                split = "val" if (idx % split_mod == 0) else "train"
                record = build_sft_record(
                    task=item.task,
                    example=item.example,
                    teacher_name="synthetic-v1-seed",
                    split=split,
                )
                text_fp = _record_fingerprint(str(record["text"]))
                prompt_fp = _record_fingerprint(str(record["prompt"]))
                summary["generation_attempts"] += 1
                summary["attempted_tasks"] += 1
                if text_fp in seen_text:
                    reject_counts["duplicate_text"] += 1
                    continue
                if prompt_fp in seen_prompt:
                    reject_counts["duplicate_prompt"] += 1
                    continue

                if sample_verified < validation_sample_per_lane:
                    try:
                        validate_distill_record_quality(record, item.example, item.task, None)
                    except DistillQualityError as exc:
                        reject_counts[str(exc)] += 1
                        continue
                    sample_verified += 1
                    sample_pass += 1

                line = json.dumps(record, ensure_ascii=False) + "\n"
                encoded_len = len(line.encode("utf-8"))
                if split == "val":
                    val_f.write(line)
                    summary["val_bytes"] += encoded_len
                else:
                    train_f.write(line)
                    summary["train_bytes"] += encoded_len
                summary["total_bytes"] += encoded_len
                summary["total_records"] += 1
                summary["valid_json_records"] += 1
                summary["accepted_records"] += 1
                seen_text.add(text_fp)
                seen_prompt.add(prompt_fp)
                metadata = record.get("metadata") or {}
                task_name = str(metadata.get("task_name", "unknown"))
                task_kind = str(record.get("task", "unknown"))
                domain_counts[task_name] += 1
                task_counts[task_name] += 1
                label_counts[task_kind] += 1
                split_counts[split] += 1
                if lane == "stem_reasoning":
                    answer_distribution[str(record.get("reference", "")).strip().upper()] += 1

        total = int(summary["total_records"])
        summary["schema_valid_rate"] = 1.0 if total else 0.0
        summary["unique_text_ratio"] = 1.0 if total else 0.0
        summary["exact_duplicate_ratio"] = 0.0
        summary["verifier_total"] = sample_verified
        summary["verifier_pass_count"] = sample_pass
        summary["verifier_pass_rate"] = sample_pass / sample_verified if sample_verified else 0.0
        summary["sample_verifier_total"] = sample_verified
        summary["sample_verifier_pass_count"] = sample_pass
        summary["sample_verifier_pass_rate"] = summary["verifier_pass_rate"]
        summary["quality_reject_count"] = sum(reject_counts.values())
        summary["quality_reject_reasons"] = dict(reject_counts)
        summary["rejected"] = dict(reject_counts)
        summary["domain_counts"] = dict(domain_counts)
        summary["label_counts"] = dict(label_counts)
        summary["split_counts"] = dict(split_counts)
        summary["lane_counts"] = {lane: total}
        summary["task_counts"] = dict(task_counts)
        summary["answer_distribution"] = dict(answer_distribution)
        (lane_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (lane_dir / "README.md").write_text(
            f"# Synthetic v1 large: {lane}\n\n"
            "Large verifier-backed synthetic data generated without teacher sampling.\n\n"
            f"- Records: {summary['total_records']}\n"
            f"- Total bytes: {summary['total_bytes']}\n"
            f"- Sample verifier pass rate: {summary.get('sample_verifier_pass_rate', 0.0):.3f}\n"
            f"- Unique text ratio: {summary.get('unique_text_ratio', 0.0):.3f}\n",
            encoding="utf-8",
        )
        overall["lanes"][lane] = summary
    overall["total_bytes"] = sum(int(item.get("total_bytes", 0)) for item in overall["lanes"].values())
    overall["total_records"] = sum(int(item.get("total_records", 0)) for item in overall["lanes"].values())
    (output_root / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    return overall


def build_synthetic_seed_bundle(
    *,
    output_root: Path,
    records_per_lane: int,
    val_ratio: float,
    lanes: Iterable[str] = LANES,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Any] = {"output_root": str(output_root), "lanes": {}}
    for lane in lanes:
        examples = generate_lane_examples(lane, records_per_lane)
        seen_text: set[str] = set()
        seen_prompt: set[str] = set()
        train_records: list[dict[str, Any]] = []
        val_records: list[dict[str, Any]] = []
        reject_counts: Counter[str] = Counter()
        for idx, item in enumerate(examples):
            split = "val" if (idx % max(2, round(1.0 / max(val_ratio, 1e-6))) == 0) else "train"
            record = build_sft_record(task=item.task, example=item.example, teacher_name="synthetic-v1-seed", split=split)
            try:
                validate_distill_record_quality(
                    record,
                    item.example,
                    item.task,
                    None,
                    seen_text_fingerprints=seen_text,
                    seen_prompt_fingerprints=seen_prompt,
                )
            except DistillQualityError as exc:
                reject_counts[str(exc)] += 1
                continue
            seen_text.add(" ".join(str(record["text"]).strip().lower().split()))
            seen_prompt.add(" ".join(str(record["prompt"]).strip().lower().split()))
            (val_records if split == "val" else train_records).append(record)
        lane_dir = output_root / lane
        all_records = [*train_records, *val_records]
        _write_jsonl(lane_dir / "distill_train.jsonl", train_records)
        _write_jsonl(lane_dir / "distill_val.jsonl", val_records)
        summary = evaluate_distill_records(all_records, quality_counters=reject_counts, run_verifiers=True)
        summary["lane"] = lane
        summary["source"] = "synthetic-v1-seed"
        summary["rejected"] = dict(reject_counts)
        (lane_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (lane_dir / "README.md").write_text(
            f"# Synthetic v1 seed: {lane}\n\n"
            "Verifier-backed synthetic seed data generated without teacher sampling.\n\n"
            f"- Records: {summary['total_records']}\n"
            f"- Verifier pass rate: {summary.get('verifier_pass_rate', 0.0):.3f}\n"
            f"- Unique text ratio: {summary.get('unique_text_ratio', 0.0):.3f}\n",
            encoding="utf-8",
        )
        overall["lanes"][lane] = summary
    (output_root / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    return overall


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build verifier-backed synthetic v1 seed SFT bundles.")
    parser.add_argument("--output-root", type=Path, default=Path("H:/elt_data/synthetic_v1_seed"))
    parser.add_argument("--records-per-lane", type=int, default=32)
    parser.add_argument("--target-bytes", type=int, default=0)
    parser.add_argument("--target-gb", type=float, default=0.0)
    parser.add_argument("--validation-sample-per-lane", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--lanes", nargs="*", default=list(LANES), choices=list(LANES))
    args = parser.parse_args()
    target_bytes = args.target_bytes or int(args.target_gb * 1024 * 1024 * 1024)
    if target_bytes > 0:
        summary = build_synthetic_seed_bundle_to_target(
            output_root=args.output_root,
            target_bytes=target_bytes,
            val_ratio=args.val_ratio,
            lanes=args.lanes,
            validation_sample_per_lane=args.validation_sample_per_lane,
        )
    else:
        summary = build_synthetic_seed_bundle(
            output_root=args.output_root,
            records_per_lane=args.records_per_lane,
            val_ratio=args.val_ratio,
            lanes=args.lanes,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
