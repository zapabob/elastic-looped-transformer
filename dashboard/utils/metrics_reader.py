"""Tail-and-parse a JSONL metrics file produced by elt_lm.telemetry.

Returns a list of dicts (one per event). Cached for `ttl` seconds to keep
the Streamlit app responsive even with large metrics files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path, last_n: int | None = 5000) -> list[dict]:
    """Read up to the last N events from a JSONL file. Tolerant of partial lines."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if last_n is not None and len(out) > last_n:
        out = out[-last_n:]
    return out


def read_json_file(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def filter_events(events: Iterable[dict], kind: str | set[str]) -> list[dict]:
    kinds = {kind} if isinstance(kind, str) else set(kind)
    return [e for e in events if e.get("event") in kinds]


def discover_runs(runs_dir: str | Path) -> list[Path]:
    """Return sorted list of run directories that contain metrics.jsonl."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    out = []
    for child in root.iterdir():
        if child.is_dir() and (child / "metrics.jsonl").exists():
            out.append(child)
    out.sort(key=lambda p: (p / "metrics.jsonl").stat().st_mtime, reverse=True)
    return out


def read_log_tail(path: str | Path, n_lines: int = 200) -> list[str]:
    """Read the last n lines of a plain-text log, tolerating binary junk."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    return [l.rstrip("\n") for l in lines[-n_lines:]]
