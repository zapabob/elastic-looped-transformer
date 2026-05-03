from __future__ import annotations

import json
from pathlib import Path

import yaml

from elt_lm.bridge_followup import build_followup


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _tool_row(idx: int) -> dict:
    reference = {
        "tool_name": "mcp.files.stat",
        "arguments": {"path": f"file-{idx}", "read_only": True, "request_id": f"r-{idx}"},
    }
    return {
        "prompt": "Select the best tool call.\nReturn strict JSON with keys: tool_name, arguments.\nDo not add prose or markdown fences.",
        "response": json.dumps(reference, sort_keys=True),
        "reference": json.dumps(reference, sort_keys=True),
        "task": "json_match",
        "metadata": {"task_name": "stat"},
    }


def _failure_row(idx: int) -> dict:
    row = _tool_row(idx)
    row.update({
        "bad_response": '{"tool_name":"mcp.files.write","arguments":{"path":"file"}}',
        "failure_label": "unsafe_mutation",
        "failure_reason": "writes during read-only inspection",
    })
    return row


def test_bridge_followup_builds_repair_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "synthetic_v2_hard"
    for lane, task, response, reference in [
        ("code", "python_exec", "```python\ndef f():\n    return 1\n```", "assert f() == 1"),
        ("math", "exact_math", "<think>1+1</think><answer>2</answer>", "2"),
    ]:
        row = {"prompt": "p", "response": response, "reference": reference, "task": task}
        _write_jsonl(root / lane / "distill_train.jsonl", [row])
        _write_jsonl(root / lane / "distill_val.jsonl", [row])

    _write_jsonl(root / "tool_use" / "distill_train.jsonl", [_tool_row(1)])
    _write_jsonl(root / "tool_use" / "distill_val.jsonl", [_tool_row(2)])
    _write_jsonl(root / "tool_use" / "failures_train.jsonl", [_failure_row(1)])
    _write_jsonl(root / "tool_use" / "failures_val.jsonl", [_failure_row(2)])
    _write_jsonl(
        root / "tool_use" / "benchmarks" / "synthetic_v2_bridge_tool_use_val_cases.jsonl",
        [_tool_row(3)],
    )
    _write_jsonl(
        root / "stem_reasoning" / "benchmarks" / "synthetic_v2_bridge_stem_reasoning_val_cases.jsonl",
        [{"prompt": "stem", "reference": "A", "task": "mcq_reasoning"}],
    )

    out = tmp_path / "bridge_followup"
    plan = build_followup(root, out, train_limit=1, val_limit=1, stem_eval_limit=1)

    tool_summary = json.loads((out / "tool_use_repair" / "summary.json").read_text())
    assert tool_summary["task"] == "json_tool_call_match"
    assert tool_summary["chosen_min_score"] == 1.0
    assert tool_summary["rejected_max_score"] < 1.0
    assert plan["tool"]["state"] == "blocked_no_reward_signal"
    manifest = yaml.safe_load((out / "stem_eval" / "stem_bridge_eval_manifest.yaml").read_text())
    assert manifest["benchmarks"][0]["limit"] == 1
