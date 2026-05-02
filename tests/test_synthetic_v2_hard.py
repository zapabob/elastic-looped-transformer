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
