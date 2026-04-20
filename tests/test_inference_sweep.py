"""Tests for the inference_sweep event schema + dashboard.panels.inference reader."""

from __future__ import annotations

import json
from pathlib import Path

from dashboard.panels.inference import _latest_per_L
from dashboard.utils.metrics_reader import filter_events, read_jsonl
from elt_lm.telemetry import TelemetryWriter


def test_latest_per_L_keeps_most_recent(tmp_path):
    events = [
        {"event": "inference_sweep", "L": 1, "ppl": 99.0},
        {"event": "inference_sweep", "L": 2, "ppl": 50.0},
        {"event": "inference_sweep", "L": 1, "ppl": 10.0},   # newer
        {"event": "inference_sweep", "L": 4, "ppl": 5.0},
    ]
    out = _latest_per_L(events)
    by_L = {int(e["L"]): e for e in out}
    assert by_L[1]["ppl"] == 10.0    # overwrote 99
    assert by_L[2]["ppl"] == 50.0
    assert by_L[4]["ppl"] == 5.0
    # Output is sorted by L
    assert [e["L"] for e in out] == sorted(e["L"] for e in out)


def test_sweep_events_round_trip_via_telemetry(tmp_path: Path):
    """Integration: sweep events written by TelemetryWriter can be read back."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = run_dir / "metrics.jsonl"
    w = TelemetryWriter(metrics)
    for L in (1, 2, 3, 4):
        w.emit(
            "inference_sweep",
            L=L,
            rel_flops=L * 28,
            nll=3.0 / L,
            ppl=20.0 - L * 4,
            tokens_per_sec=1000.0 * L,
            latency_ms=10.0 * L,
            total_tokens=4096,
            batches=8,
            ckpt=str(run_dir / "last.pt"),
        )
    w.close()

    events = read_jsonl(metrics, last_n=1000)
    sweep = _latest_per_L(filter_events(events, "inference_sweep"))
    assert [int(e["L"]) for e in sweep] == [1, 2, 3, 4]
    assert all("ppl" in e for e in sweep)
    assert all("latency_ms" in e for e in sweep)
    assert all("tokens_per_sec" in e for e in sweep)


def test_sweep_event_field_types():
    """Fields required by the dashboard must be present and numeric-castable."""
    raw = json.dumps({
        "event": "inference_sweep", "ts": 1.0,
        "L": 2, "rel_flops": 56, "nll": 1.5, "ppl": 4.5,
        "tokens_per_sec": 2000.0, "latency_ms": 12.5,
        "total_tokens": 4096, "batches": 8, "ckpt": "x",
    })
    e = json.loads(raw)
    for key in ("L", "rel_flops", "nll", "ppl",
                "tokens_per_sec", "latency_ms"):
        float(e[key])   # must not raise
