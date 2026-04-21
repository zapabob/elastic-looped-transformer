from __future__ import annotations

import json
from pathlib import Path

from elt_lm.agent.audit import AuditLogger
from elt_lm.agent.replay import replay_audit_log
from elt_lm.agent.runtime import load_agent_runtime_config
from elt_lm.agent.sandbox import run_python_code
from elt_lm.agent.sbom import build_spdx_sbom, write_spdx_sbom


def test_audit_log_replay_validates_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path)
    logger.append("tool_call", {"tool": "python_exec"})
    logger.append("result", {"ok": True})
    events, valid = replay_audit_log(path)
    assert valid is True
    assert [e.action for e in events] == ["tool_call", "result"]


def test_audit_log_replay_detects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path)
    logger.append("tool_call", {"tool": "python_exec"})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["payload"] = {"tool": "tampered"}
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    _, valid = replay_audit_log(path)
    assert valid is False


def test_sandbox_runs_python_and_times_out() -> None:
    ok = run_python_code("print('ok')", timeout_s=1.0)
    assert ok.returncode == 0
    assert "ok" in ok.stdout

    timed = run_python_code("while True:\n    pass\n", timeout_s=0.1)
    assert timed.timed_out is True


def test_sbom_build_and_write(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    target.write_text("hello", encoding="utf-8")
    sbom = build_spdx_sbom([target])
    assert sbom["spdxVersion"] == "SPDX-2.3"
    out = write_spdx_sbom([target], tmp_path / "sbom.json")
    assert out.is_file()


def test_agent_runtime_config_loader(tmp_path: Path) -> None:
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        "L: 3\nsandbox:\n  timeout_s: 4.5\naudit:\n  path: runs/test/audit.jsonl\n",
        encoding="utf-8",
    )
    cfg = load_agent_runtime_config(cfg_path)
    assert cfg.L == 3
    assert abs(cfg.sandbox.timeout_s - 4.5) < 1e-6
    assert cfg.audit.path.endswith("audit.jsonl")
