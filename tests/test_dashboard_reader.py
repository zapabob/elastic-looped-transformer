"""Tests for dashboard.utils.metrics_reader."""

from __future__ import annotations

import json
from pathlib import Path

from dashboard.utils.metrics_reader import (
    discover_runs,
    filter_events,
    read_json_file,
    read_jsonl,
    read_log_tail,
)


def test_read_jsonl_skips_partial_lines(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        json.dumps({"event": "a", "v": 1}) + "\n"
        + "{not json\n"
        + json.dumps({"event": "b", "v": 2}) + "\n",
        encoding="utf-8",
    )
    events = read_jsonl(p)
    assert len(events) == 2
    assert events[0]["event"] == "a"
    assert events[1]["event"] == "b"


def test_read_jsonl_last_n_truncates(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(100):
            f.write(json.dumps({"event": "e", "i": i}) + "\n")
    events = read_jsonl(p, last_n=5)
    assert len(events) == 5
    assert [e["i"] for e in events] == [95, 96, 97, 98, 99]


def test_filter_events_by_kind():
    events = [{"event": "a"}, {"event": "b"}, {"event": "a"}]
    assert len(filter_events(events, "a")) == 2
    assert len(filter_events(events, {"a", "b"})) == 3


def test_discover_runs_returns_only_dirs_with_metrics(tmp_path: Path):
    (tmp_path / "run1").mkdir()
    (tmp_path / "run1" / "metrics.jsonl").write_text("{}\n")
    (tmp_path / "run2").mkdir()  # no metrics file
    (tmp_path / "run3").mkdir()
    (tmp_path / "run3" / "metrics.jsonl").write_text("{}\n")

    runs = discover_runs(tmp_path)
    names = {r.name for r in runs}
    assert names == {"run1", "run3"}


def test_read_log_tail(tmp_path: Path):
    p = tmp_path / "log.txt"
    p.write_text("\n".join(f"line {i}" for i in range(10)) + "\n", encoding="utf-8")
    tail = read_log_tail(p, n_lines=3)
    assert tail == ["line 7", "line 8", "line 9"]


def test_read_log_tail_missing(tmp_path: Path):
    assert read_log_tail(tmp_path / "nope.txt") == []


def test_read_json_file(tmp_path: Path):
    p = tmp_path / "status.json"
    p.write_text(json.dumps({"state": "running", "processed_tasks": 4}), encoding="utf-8")
    data = read_json_file(p)
    assert data is not None
    assert data["state"] == "running"


def test_read_json_file_missing(tmp_path: Path):
    assert read_json_file(tmp_path / "missing.json") is None
