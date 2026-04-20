"""Tests for elt_lm.telemetry.TelemetryWriter."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from elt_lm.telemetry import NullTelemetry, TelemetryWriter, make_writer


def read_lines(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_emit_writes_jsonl(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    w = TelemetryWriter(path)
    w.emit("train_step", step=1, loss=3.14)
    w.emit("train_step", step=2, loss=2.71)
    w.close()

    lines = read_lines(path)
    assert lines[0]["event"] == "run_start"
    assert lines[1]["event"] == "train_step"
    assert lines[1]["step"] == 1
    assert lines[1]["loss"] == pytest.approx(3.14)
    assert lines[-1]["event"] == "run_end"


def test_every_line_has_timestamp_and_event(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    with TelemetryWriter(path) as w:
        for i in range(5):
            w.emit("tier_read", tier="NVME", bytes=i * 1024, latency_us=500)

    for line in read_lines(path):
        assert "ts" in line and isinstance(line["ts"], float)
        assert "event" in line


def test_thread_safe_emit(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    w = TelemetryWriter(path)

    def worker(tid: int):
        for i in range(200):
            w.emit("train_step", worker=tid, step=i)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    w.close()

    lines = read_lines(path)
    # run_start + run_end + 4*200 train_step
    train_lines = [l for l in lines if l["event"] == "train_step"]
    assert len(train_lines) == 800
    # No line should have a partial / corrupted record (all should parse).
    for l in lines:
        assert isinstance(l, dict)


def test_handles_numpy_like_scalars(tmp_path: Path):
    """Tensors / numpy scalars should serialize via .item() fallback."""
    import torch
    path = tmp_path / "metrics.jsonl"
    w = TelemetryWriter(path)
    w.emit("train_step", loss=torch.tensor(1.5), step=torch.tensor(7))
    w.close()
    rows = read_lines(path)
    event = next(r for r in rows if r["event"] == "train_step")
    assert event["loss"] == pytest.approx(1.5)
    assert event["step"] == 7


def test_null_telemetry_is_noop(tmp_path: Path):
    n = NullTelemetry()
    n.emit("train_step", step=1)
    n.close()  # should not raise


def test_make_writer_factory(tmp_path: Path):
    w_null = make_writer(None)
    assert isinstance(w_null, NullTelemetry)
    w_null.close()

    w = make_writer(tmp_path)
    assert isinstance(w, TelemetryWriter)
    w.emit("train_step", step=0)
    w.close()
    assert (tmp_path / "metrics.jsonl").exists()
