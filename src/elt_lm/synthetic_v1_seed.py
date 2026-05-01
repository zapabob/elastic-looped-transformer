from __future__ import annotations

import argparse
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
    specs = [
        (
            "clamp_int",
            "Implement clamp_value(value, low, high) with typed integer bounds.",
            "def clamp_value(value: int, low: int, high: int) -> int:\n"
            "    if low > high:\n"
            "        raise ValueError('low must be <= high')\n"
            "    return min(max(value, low), high)",
            "assert clamp_value(5, 0, 10) == 5\n"
            "assert clamp_value(-3, 0, 10) == 0\n"
            "assert clamp_value(12, 0, 10) == 10\n"
            "try:\n"
            "    clamp_value(1, 5, 2)\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'low must be <= high'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        ),
        (
            "normalize_tags",
            "Implement normalize_tags(tags) returning sorted lowercase unique non-empty tags.",
            "def normalize_tags(tags: list[str]) -> list[str]:\n"
            "    cleaned = {tag.strip().lower() for tag in tags if tag.strip()}\n"
            "    return sorted(cleaned)",
            "assert normalize_tags([' AI ', 'ai', '', 'Math']) == ['ai', 'math']\n"
            "assert normalize_tags([]) == []\n"
            "assert normalize_tags(['Z', 'a', 'z']) == ['a', 'z']",
        ),
        (
            "parse_kv_line",
            "Implement parse_kv_line(line) for 'key=value' pairs with trimmed fields.",
            "def parse_kv_line(line: str) -> tuple[str, str]:\n"
            "    if '=' not in line:\n"
            "        raise ValueError('missing separator')\n"
            "    key, value = line.split('=', 1)\n"
            "    key = key.strip()\n"
            "    value = value.strip()\n"
            "    if not key:\n"
            "        raise ValueError('empty key')\n"
            "    return key, value",
            "assert parse_kv_line(' mode = safe ') == ('mode', 'safe')\n"
            "assert parse_kv_line('a=b=c') == ('a', 'b=c')\n"
            "try:\n"
            "    parse_kv_line('novalue')\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'missing separator'\n"
            "else:\n"
            "    raise AssertionError('expected missing separator')",
        ),
        (
            "rolling_average",
            "Implement rolling_average(values, window) returning rounded moving averages.",
            "def rolling_average(values: list[float], window: int) -> list[float]:\n"
            "    if window <= 0:\n"
            "        raise ValueError('window must be positive')\n"
            "    if window > len(values):\n"
            "        return []\n"
            "    return [round(sum(values[i:i + window]) / window, 3) for i in range(len(values) - window + 1)]",
            "assert rolling_average([1.0, 2.0, 3.0], 2) == [1.5, 2.5]\n"
            "assert rolling_average([2.0], 2) == []\n"
            "try:\n"
            "    rolling_average([1.0], 0)\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'window must be positive'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        ),
    ]
    for i in range(count):
        name, request, code, verifier = specs[i % len(specs)]
        request = f"{request} Seed case: {i}."
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
    specs = [
        ("linear_system", "Solve the linear equation 3x + 7 = 28 and report the exact value of x.", "Subtract 7 from both sides to get 3x = 21, then divide both sides by 3. The exact solution is 7.", "7"),
        ("area_ratio", "A rectangle has width 6 and height 9. Compute exactly one half of its area.", "The area is 6 times 9 = 54 square units. Half of 54 is 27, so the final answer is 27.", "27"),
        ("probability", "A fair six-sided die is rolled once. What is P(result is at least 5)?", "The favorable outcomes are 5 and 6, two outcomes out of six. The probability is 2/6 = 1/3.", "1/3"),
        ("sequence_sum", "Find the exact sum of the first 12 positive even integers using a finite series formula.", "The first 12 positive even integers are 2 times 1 through 12. Their sum is 2*(12*13/2)=156.", "156"),
    ]
    for i in range(count):
        name, question, reasoning, answer = specs[i % len(specs)]
        question = f"{question} Synthetic seed id {i}."
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
            "A cart's speed doubles while mass stays fixed. How does kinetic energy change?",
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
            "Which mixture most directly resists pH change when small acid is added?",
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
            "For a rare disease, what usually happens to positive predictive value when prevalence rises?",
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
            "If sample size increases fourfold with variance fixed, how does standard error change?",
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
            "Which change most directly lowers the initial rate of an enzyme-catalyzed reaction?",
            "Adding a competitive inhibitor that occupies active sites",
            [
                "Doubling substrate concentration when the enzyme is already saturated",
                "Maintaining the enzyme at its optimal pH and temperature",
                "Adding more active enzyme molecules to the same reaction volume",
            ],
            "A competitive inhibitor reduces active-site availability, lowering initial rate under otherwise comparable conditions.",
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
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--lanes", nargs="*", default=list(LANES), choices=list(LANES))
    args = parser.parse_args()
    summary = build_synthetic_seed_bundle(
        output_root=args.output_root,
        records_per_lane=args.records_per_lane,
        val_ratio=args.val_ratio,
        lanes=args.lanes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
