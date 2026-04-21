"""Append-only audit logger with a simple hash chain."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from pathlib import Path
from typing import Any


@dataclass
class AuditEvent:
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    prev_hash: str = ""
    hash: str = ""


def _event_hash(event: AuditEvent) -> str:
    data = json.dumps(
        {
            "ts": event.ts,
            "action": event.action,
            "payload": event.payload,
            "prev_hash": event.prev_hash,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class AuditLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = ""
        if self.path.exists():
            *_, last = [
                json.loads(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ] or [None]
            if last:
                self._prev_hash = str(last.get("hash", ""))

    def append(self, action: str, payload: dict[str, Any] | None = None) -> AuditEvent:
        event = AuditEvent(action=action, payload=payload or {}, prev_hash=self._prev_hash)
        event.hash = _event_hash(event)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self._prev_hash = event.hash
        return event
