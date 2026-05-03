from __future__ import annotations

import json
from pathlib import Path

from elt_lm.bridge_diagnostics import analyze_bridge_runs, render_markdown, write_report


def _write_metrics(run_root: Path, run_name: str, rows: list[dict]) -> None:
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _lane_rows(
    *,
    correct: list[float],
    fmt: float = 1.0,
    reward_std: list[float] | None = None,
    task: str = "json_match",
) -> list[dict]:
    rows = [{"event": "run_start"}]
    reward_std = reward_std or [0.0] * len(correct)
    for step, (correct_rate, std) in enumerate(zip(correct, reward_std)):
        rows.append({
            "event": "grpo_step",
            "step": step,
            "reward_mean": correct_rate,
            "reward_std": std,
            "adv_abs_mean": 1.0 if std > 0.0 else 0.0,
            "correct_rate": correct_rate,
            "format_rate": fmt,
            "kl": 0.001,
            "clip_frac": 0.0,
            "prompt_task": task,
        })
    rows.extend([
        {"event": "checkpoint", "kind": "final", "step": len(correct)},
        {"event": "run_end"},
    ])
    return rows


def test_bridge_diagnostics_classifies_completed_lanes(tmp_path: Path) -> None:
    lane_runs = {
        "code": "code_run",
        "math": "math_run",
        "stem": "stem_run",
        "tool": "tool_run",
    }
    _write_metrics(
        tmp_path,
        "code_run",
        _lane_rows(correct=[0.0] * 20 + [0.5] * 5, reward_std=[0.0] * 20 + [0.5] * 5),
    )
    _write_metrics(
        tmp_path,
        "math_run",
        _lane_rows(correct=[0.25] * 25, reward_std=[0.5] * 25, task="gsm8k"),
    )
    _write_metrics(
        tmp_path,
        "stem_run",
        _lane_rows(correct=[1.0] * 25, reward_std=[0.0] * 25, task="mcq_reasoning"),
    )
    _write_metrics(
        tmp_path,
        "tool_run",
        _lane_rows(correct=[0.0] * 25, reward_std=[0.0] * 25),
    )

    report = analyze_bridge_runs(tmp_path, lane_runs=lane_runs)

    assert report["decisions"]["tool"]["classification"] == "blocked_no_reward_signal"
    assert report["decisions"]["code"]["classification"] == "unstable_sparse_success"
    assert report["decisions"]["math"]["classification"] == "promising_but_unstable"
    assert report["decisions"]["stem"]["classification"] == "ready_for_export_eval"
    assert report["action_order"][0] == "tool"


def test_bridge_diagnostics_writes_json_and_markdown(tmp_path: Path) -> None:
    lane_runs = {"tool": "tool_run"}
    _write_metrics(tmp_path, "tool_run", _lane_rows(correct=[0.0] * 25))

    report = analyze_bridge_runs(tmp_path, lane_runs=lane_runs)
    json_path, md_path = write_report(report, tmp_path / "out", prefix="diagnostics")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert payload["decisions"]["tool"]["classification"] == "blocked_no_reward_signal"
    assert "| tool | blocked_no_reward_signal |" in markdown
    assert "zero advantage lanes" in render_markdown(report)
