from __future__ import annotations

import json
from pathlib import Path

import yaml

from elt_lm.synthetic_v2_hard import (
    LANES,
    SOURCE_NAME,
    build_synthetic_v2_bundle,
    generate_lane_examples,
)
from elt_lm.synthetic_v2_code_bridge import (
    build_code_bridge_prompts,
    generate_bridge_code_prompts,
    generate_easy_code_bridge_prompts,
)
from elt_lm.synthetic_v2_reasoning_bridge import (
    build_lane_bridge_prompts,
    generate_bridge_reasoning_prompts,
    generate_easy_reasoning_bridge_prompts,
)
from elt_lm.verifiers import CompositeVerifier


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_generate_v2_hard_examples_require_loop_depth_and_failures() -> None:
    for lane in LANES:
        examples = generate_lane_examples(lane, 6)

        assert len(examples) == 6
        assert all(example.failures for example in examples)
        assert all(example.requires_loop_depth >= 3 for example in examples)
        assert all("synthetic_v2_hard" in example.task.tags for example in examples)

    code_examples = generate_lane_examples("code", 6)
    assert {example.task.target_kind for example in code_examples} == {"python_exec"}
    assert len({example.example["user_request"] for example in code_examples}) == 6
    assert len({example.task.domain for example in code_examples}) == 6

    math_examples = generate_lane_examples("math", 12)
    assert len({example.task.domain for example in math_examples}) == 6
    assert {
        failure.label
        for example in math_examples
        for failure in example.failures
    } == {"skipped_intermediate"}


def test_build_synthetic_v2_bundle_writes_hard_lanes(tmp_path: Path) -> None:
    summary = build_synthetic_v2_bundle(
        output_root=tmp_path,
        records_per_lane=6,
        val_ratio=0.25,
        lanes=LANES,
    )

    assert summary["source"] == SOURCE_NAME
    assert summary["total_records"] == 24
    assert summary["total_failure_records"] == 24
    assert set(summary["lanes"]) == set(LANES)

    for lane, lane_summary in summary["lanes"].items():
        lane_dir = tmp_path / lane
        train_path = lane_dir / "distill_train.jsonl"
        val_path = lane_dir / "distill_val.jsonl"
        failure_train_path = lane_dir / "failures_train.jsonl"
        failure_val_path = lane_dir / "failures_val.jsonl"
        cases_path = lane_dir / "benchmarks" / f"synthetic_v2_hard_{lane}_val_cases.jsonl"
        manifest_path = lane_dir / "benchmarks" / f"synthetic_v2_hard_{lane}_val_manifest.yaml"

        assert train_path.exists()
        assert val_path.exists()
        assert failure_train_path.exists()
        assert failure_val_path.exists()
        assert cases_path.exists()
        assert manifest_path.exists()
        assert lane_summary["records"] == 6
        assert lane_summary["verifier_pass_rate"] == 1.0
        assert lane_summary["failure_records"] == 6
        assert lane_summary["failure_expected_zero_rate"] == 1.0
        assert lane_summary["loop_depth_counts"] == {"3": 6}

        records = [*_read_jsonl(train_path), *_read_jsonl(val_path)]
        failures = [*_read_jsonl(failure_train_path), *_read_jsonl(failure_val_path)]
        cases = _read_jsonl(cases_path)
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

        assert len(records) == 6
        assert len(failures) == 6
        assert len(cases) == 2
        assert manifest["benchmarks"][0]["task"] == cases[0]["task"]
        assert all(row["metadata"]["difficulty"] == "hard" for row in records)
        assert all(row["metadata"]["requires_loop_depth"] >= 3 for row in cases)
        assert all(row["expected_score"] == 0.0 for row in failures)
        assert all(row["observed_score"] == 0.0 for row in failures)
        assert all(row["bad_response"] for row in failures)


def test_synthetic_v2_code_bridge_prompts_are_verifier_backed(tmp_path: Path) -> None:
    verifier = CompositeVerifier(task="python_exec", enable_code_quality=False)
    examples = [
        *generate_easy_code_bridge_prompts(6),
        *generate_bridge_code_prompts(6),
    ]

    assert {example.difficulty for example in examples} == {"easy", "bridge"}
    assert len({example.domain for example in examples}) == 12
    for example in examples:
        score = verifier.reward(
            example.prompt,
            example.correct_response,
            example.reference,
        ).verifier_total()
        assert score == 1.0

    hard_cases = tmp_path / "hard.jsonl"
    hard_cases.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt": f"hard prompt {i}",
                    "reference": "assert True",
                    "task": "python_exec",
                    "metadata": {"difficulty": "hard", "task_name": f"hard_{i}"},
                }
            )
            for i in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "bridge.jsonl"
    summary = build_code_bridge_prompts(
        output_path=out,
        hard_cases_path=hard_cases,
        total_cases=12,
        easy_cases=3,
        bridge_cases=6,
    )
    rows = _read_jsonl(out)

    assert summary["total_cases"] == 12
    assert summary["difficulty_counts"] == {"easy": 3, "bridge": 6, "hard": 3}
    assert len(rows) == 12
    assert {row["metadata"]["curriculum"] for row in rows} == {"bridge_easy_hard"}
    assert rows[0]["metadata"]["difficulty"] == "bridge"


def test_synthetic_v2_reasoning_bridge_prompts_are_verifier_backed(tmp_path: Path) -> None:
    for lane, task in (("math", "exact_math"), ("stem_reasoning", "mcq_reasoning")):
        verifier = CompositeVerifier(task=task)
        examples = [
            *generate_easy_reasoning_bridge_prompts(lane, 8),
            *generate_bridge_reasoning_prompts(lane, 8),
        ]

        assert {example.difficulty for example in examples} == {"easy", "bridge"}
        assert len({example.domain for example in examples}) >= 9
        for example in examples:
            score = verifier.reward(
                example.prompt,
                example.correct_response,
                example.reference,
            ).verifier_total()
            assert score == 1.0

        hard_cases = tmp_path / f"{lane}_hard.jsonl"
        hard_cases.write_text(
            "\n".join(
                json.dumps(
                    {
                        "prompt": f"{lane} hard prompt {i}",
                        "reference": "1" if task == "exact_math" else "A",
                        "task": task,
                        "metadata": {"difficulty": "hard", "task_name": f"{lane}_hard_{i}"},
                    }
                )
                for i in range(4)
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / f"{lane}_bridge.jsonl"
        summary = build_lane_bridge_prompts(
            lane=lane,
            output_path=out,
            hard_cases_path=hard_cases,
            total_cases=12,
            easy_cases=3,
            bridge_cases=6,
        )
        rows = _read_jsonl(out)

        assert summary["lane"] == lane
        assert summary["total_cases"] == 12
        assert summary["difficulty_counts"] == {"easy": 3, "bridge": 6, "hard": 3}
        assert len(rows) == 12
        assert {row["metadata"]["curriculum"] for row in rows} == {"bridge_easy_hard"}
        assert rows[0]["metadata"]["difficulty"] == "bridge"
