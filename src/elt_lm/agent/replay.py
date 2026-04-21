"""Replay and validate audit logs written by :mod:`elt_lm.agent.audit`."""

from __future__ import annotations

import json
from pathlib import Path

from elt_lm.agent.audit import AuditEvent, _event_hash


def replay_audit_log(path: str | Path) -> tuple[list[AuditEvent], bool]:
    events: list[AuditEvent] = []
    prev_hash = ""
    valid = True
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event = AuditEvent(
            action=row["action"],
            payload=row.get("payload", {}),
            ts=float(row["ts"]),
            prev_hash=row.get("prev_hash", ""),
            hash=row.get("hash", ""),
        )
        if event.prev_hash != prev_hash:
            valid = False
        if event.hash != _event_hash(event):
            valid = False
        prev_hash = event.hash
        events.append(event)
    return events, valid
