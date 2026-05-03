from __future__ import annotations

import json
from pathlib import Path

import yaml

from elt_lm.synthetic_v2_agent import (
    AGENT_LANE,
    SOURCE_NAME,
    build_synthetic_v2_agent_bundle,
    generate_agent_examples,
)
from elt_lm.verifiers import CompositeVerifier


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_generate_agent_examples_cover_openclaw_helmes_and_verify() -> None:
    verifier = CompositeVerifier(task="json_match", length_cap=10000)
    examples = generate_agent_examples(12)

    assert len(examples) == 12
    assert {example.task.lane for example in examples} == {"tool_use"}
    assert all(example.failures for example in examples)
    assert all(example.requires_loop_depth >= 3 for example in examples)
    assert {"openclaw", "helmes", "general_agent"}.issubset(set(examples[0].task.tags))
    assert len({example.task.domain for example in examples}) == 12
    assert {"openclaw_repo_ops", "helmes_general_agent", "release_ops"}.issubset(
        {example.agent_focus for example in examples}
    )

    for example in examples:
        response = json.dumps(
            {"tool_name": example.example["tool_name"], "arguments": example.example["arguments"]},
            ensure_ascii=False,
            sort_keys=True,
        )
        reference = json.dumps(example.example["reference"], ensure_ascii=False, sort_keys=True)
        assert verifier.reward("", response, reference).verifier_total() == 1.0
        for failure in example.failures:
            assert verifier.reward("", failure.response, reference).verifier_total() == 0.0


def test_build_synthetic_v2_agent_bundle_writes_quality_gated_data(tmp_path: Path) -> None:
    summary = build_synthetic_v2_agent_bundle(
        output_root=tmp_path,
        records=12,
        val_ratio=0.25,
    )

    assert summary["source"] == SOURCE_NAME
    assert summary["agent_lane"] == AGENT_LANE
    assert summary["records"] == 12
    assert summary["train_records"] == 9
    assert summary["val_records"] == 3
    assert summary["failure_records"] == 12
    assert summary["verifier_pass_rate"] == 1.0
    assert summary["failure_expected_zero_rate"] == 1.0
    assert summary["difficulty_counts"] == {"bridge": 6, "hard": 6}
    assert summary["exact_duplicate_count"] == 0
    assert summary["duplicate_prompt_count"] == 0

    train_path = tmp_path / "distill_train.jsonl"
    val_path = tmp_path / "distill_val.jsonl"
    failure_train_path = tmp_path / "failures_train.jsonl"
    failure_val_path = tmp_path / "failures_val.jsonl"
    cases_path = tmp_path / "benchmarks" / "synthetic_v2_agent_val_cases.jsonl"
    manifest_path = tmp_path / "benchmarks" / "synthetic_v2_agent_val_manifest.yaml"

    for path in (train_path, val_path, failure_train_path, failure_val_path, cases_path, manifest_path):
        assert path.exists()

    records = [*_read_jsonl(train_path), *_read_jsonl(val_path)]
    failures = [*_read_jsonl(failure_train_path), *_read_jsonl(failure_val_path)]
    cases = _read_jsonl(cases_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert len(records) == 12
    assert len(failures) == 12
    assert len(cases) == 3
    assert manifest["benchmarks"][0]["task"] == "json_match"
    assert all(row["task"] == "json_match" for row in records)
    assert all(row["metadata"]["lane"] == "tool_use" for row in records)
    assert all(row["metadata"]["agent_lane"] == AGENT_LANE for row in records)
    assert all("openclaw_helmes_agent" in row["metadata"]["tags"] for row in records)
    assert all(row["expected_score"] == 0.0 and row["observed_score"] == 0.0 for row in failures)
