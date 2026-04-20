"""Telemetry writer — line-buffered JSONL emitter.

Inspired by Hypura's TelemetryEvent broadcast channel, adapted for local
file-based tailing so a Streamlit dashboard can follow training / inference /
pipeline progress without any IPC setup.

One file per run: ``<run_dir>/metrics.jsonl``. Every line is one event.
Events are typed via ``event`` field; remaining fields are event-specific.

Event kinds currently emitted:

    train_step     — per-logstep scalars (loss, lr, grad_norm, tok/sec, ...)
    layer_computed — forward/backward timings for a layer (offload path)
    tier_read      — bytes transferred from a storage tier (GPU/PINNED/RAM/NVME)
    prefetch_status — rolling hit-rate and NVMe throughput
    pipeline_stage — orchestrator stage transitions
    eval_point     — one held-out evaluation scalar
    inference_sweep — per-L point on the any-time quality/latency curve
    checkpoint     — rolling / milestone save events
    hardware       — periodic VRAM / RAM / disk snapshot (dashboard-side allowed too)

The writer is fork-safe (reopens on child) and crash-safe (line-buffered flush).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class TelemetryWriter:
    """Thread-safe line-buffered JSONL writer.

    Call ``emit(event_type, **fields)`` from anywhere; the writer serializes
    access with a lock so multiple threads (train loop + prefetcher + eval)
    can share one instance.
    """

    def __init__(self, path: str | Path, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pid = os.getpid()
        # Line-buffered append mode. On Windows ``buffering=1`` in text mode flushes per '\n'.
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        self.run_id = run_id or self.path.parent.name
        self.emit("run_start", run_id=self.run_id, path=str(self.path))

    def emit(self, event: str, **fields: Any) -> None:
        if os.getpid() != self._pid:
            # Child process after fork — reopen to avoid sharing file handle.
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
            self._pid = os.getpid()
        line = {"ts": time.time(), "event": event, **fields}
        data = json.dumps(line, ensure_ascii=False, default=_json_default)
        with self._lock:
            self._fh.write(data + "\n")

    def close(self) -> None:
        try:
            self.emit("run_end")
        finally:
            with self._lock:
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception:
                    pass

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    """Fallback serializer for numpy / torch scalars + Path."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ----------------------------------------------------------------------
# Null writer — used when telemetry is disabled so call sites stay uniform.

class NullTelemetry:
    def emit(self, event: str, **fields: Any) -> None:  # noqa: D401
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "NullTelemetry":
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def make_writer(run_dir: str | Path | None, enabled: bool = True) -> TelemetryWriter | NullTelemetry:
    """Factory: returns a NullTelemetry if disabled or run_dir is None."""
    if not enabled or run_dir is None:
        return NullTelemetry()
    return TelemetryWriter(Path(run_dir) / "metrics.jsonl")
