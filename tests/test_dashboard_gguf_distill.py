from __future__ import annotations

import json
from pathlib import Path

from dashboard.panels.gguf_distill import (
    discover_distill_runs,
    classify_heartbeat_health,
)


def test_discover_distill_runs_finds_status_dirs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    (run_a / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
    run_b = tmp_path / "run_b"
    run_b.mkdir()
    (run_b / "metrics.jsonl").write_text('{"event":"x"}\n', encoding="utf-8")
    run_c = tmp_path / "run_c"
    run_c.mkdir()

    runs = discover_distill_runs(tmp_path)
    assert [run.name for run in runs] == ["run_b", "run_a"]


def test_classify_heartbeat_health_detects_stall() -> None:
    heartbeat = {"updated_at": 100.0, "state": "running"}
    assert classify_heartbeat_health(heartbeat, now_ts=1600.0, stall_after_sec=300.0) == "stalled"
    assert classify_heartbeat_health(heartbeat, now_ts=150.0, stall_after_sec=300.0) == "healthy"
    assert classify_heartbeat_health({"state": "complete"}, now_ts=1600.0, stall_after_sec=300.0) == "complete"
