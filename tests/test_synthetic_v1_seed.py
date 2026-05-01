from __future__ import annotations

import json
from pathlib import Path

from elt_lm.synthetic_v1_seed import (
    build_synthetic_seed_bundle,
    build_synthetic_seed_bundle_to_target,
    generate_lane_examples,
)


def test_generate_lane_examples_are_distinct() -> None:
    for lane in ("code", "math", "stem_reasoning", "tool_use"):
        examples = generate_lane_examples(lane, 4)
        prompts = [example.example.get("user_request") or example.example.get("question") for example in examples]
        assert len(set(prompts)) == 4


def test_code_seed_covers_requested_languages() -> None:
    examples = generate_lane_examples("code", 10)
    languages = {example.example["language"] for example in examples}
    target_kinds = {example.task.target_kind for example in examples}

    assert languages == {"python", "rust2024", "go", "typescript", "csharp"}
    assert target_kinds == {"python_exec", "code_static_spec"}


def test_tool_seed_covers_mcp_and_agent_harnesses() -> None:
    examples = generate_lane_examples("tool_use", 24)
    tool_names = {example.example["tool_name"] for example in examples}

    assert any(name.startswith("mcp.") for name in tool_names)
    assert any(name.startswith("agent.") for name in tool_names)
    assert "mcp.tools.list" in tool_names
    assert "agent.plan.execute" in tool_names
    assert "agent.ci.matrix" in tool_names


def test_build_synthetic_seed_bundle_writes_verifier_backed_lanes(tmp_path: Path) -> None:
    summary = build_synthetic_seed_bundle(
        output_root=tmp_path,
        records_per_lane=10,
        val_ratio=0.25,
        lanes=("code", "math", "stem_reasoning", "tool_use"),
    )

    assert set(summary["lanes"]) == {"code", "math", "stem_reasoning", "tool_use"}
    for lane, lane_summary in summary["lanes"].items():
        lane_dir = tmp_path / lane
        train_path = lane_dir / "distill_train.jsonl"
        val_path = lane_dir / "distill_val.jsonl"
        eval_path = lane_dir / "eval_summary.json"
        assert train_path.exists()
        assert val_path.exists()
        assert eval_path.exists()
        assert lane_summary["total_records"] == 10
        assert lane_summary["unique_text_ratio"] == 1.0
        assert lane_summary["verifier_pass_rate"] == 1.0
        rows = [
            *(json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()),
            *(json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines()),
        ]
        assert all(row["metadata"]["lane"] == lane for row in rows)
        assert all(row["reference"] for row in rows)
        if lane == "code":
            assert {row["metadata"]["language"] for row in rows} == {
                "python",
                "rust2024",
                "go",
                "typescript",
                "csharp",
            }


def test_synthetic_stem_seed_balances_answers(tmp_path: Path) -> None:
    summary = build_synthetic_seed_bundle(
        output_root=tmp_path,
        records_per_lane=8,
        val_ratio=0.25,
        lanes=("stem_reasoning",),
    )

    distribution = summary["lanes"]["stem_reasoning"]["answer_distribution"]
    assert sum(distribution.values()) == 8
    assert set(distribution) == {"A", "B", "C", "D"}
    assert max(distribution.values()) - min(distribution.values()) <= 1


def test_build_synthetic_seed_bundle_to_target_streams_large_shape(tmp_path: Path) -> None:
    summary = build_synthetic_seed_bundle_to_target(
        output_root=tmp_path,
        target_bytes=80_000,
        val_ratio=0.125,
        lanes=("code", "math", "stem_reasoning", "tool_use"),
        validation_sample_per_lane=4,
    )

    assert summary["total_bytes"] >= 80_000
    assert summary["total_records"] > 20
    for lane, lane_summary in summary["lanes"].items():
        assert (tmp_path / lane / "distill_train.jsonl").exists()
        assert (tmp_path / lane / "distill_val.jsonl").exists()
        assert lane_summary["total_bytes"] > 0
        assert lane_summary["sample_verifier_pass_rate"] == 1.0
        assert lane_summary["unique_text_ratio"] == 1.0
        assert lane_summary["exact_duplicate_count"] == 0
