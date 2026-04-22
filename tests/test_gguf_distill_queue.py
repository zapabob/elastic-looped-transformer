from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from elt_lm.gguf_distill_queue import (
    inspect_stage_runtime_state,
    load_gguf_distill_queue_config,
    run_queue,
)


def _write_distill_config(path: Path, output_root: Path) -> None:
    path.write_text(
        f"""
teacher:
  name: test-teacher
  model_path: C:/models/teacher.gguf
pipeline:
  output_root: {output_root.as_posix()}
domains:
  - name: benign_control
    description: safe control samples
    target_label: allow
""".strip(),
        encoding="utf-8",
    )


def test_load_queue_config(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(
        """
output_root: H:/elt_data/gguf_distill/demo_queue
poll_interval_sec: 15
continue_on_failure: true
wait_for_existing: false
stages:
  - name: s1
    config: configs/one.yaml
  - name: s2
    config: configs/two.yaml
    resume: false
""".strip(),
        encoding="utf-8",
    )

    cfg = load_gguf_distill_queue_config(queue_path)
    assert cfg.output_root == "H:/elt_data/gguf_distill/demo_queue"
    assert cfg.poll_interval_sec == 15
    assert cfg.continue_on_failure is True
    assert cfg.wait_for_existing is False
    assert [stage.name for stage in cfg.stages] == ["s1", "s2"]
    assert Path(cfg.stages[0].config).is_absolute()
    assert cfg.stages[1].resume is False


def test_inspect_stage_runtime_state_detects_running_with_live_pid(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True)
    (output_dir / "status.json").write_text(
        json.dumps({"state": "running", "pid": os.getpid()}),
        encoding="utf-8",
    )

    state, status = inspect_stage_runtime_state(output_dir)
    assert state == "running"
    assert status["state"] == "running"


def test_run_queue_skips_completed_then_runs_next(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    queue_out = tmp_path / "queue"
    stage1_cfg = tmp_path / "stage1.yaml"
    stage2_cfg = tmp_path / "stage2.yaml"
    queue_cfg = tmp_path / "queue.yaml"

    _write_distill_config(stage1_cfg, stage1_dir)
    _write_distill_config(stage2_cfg, stage2_dir)
    stage1_dir.mkdir(parents=True)
    (stage1_dir / "status.json").write_text(json.dumps({"state": "complete", "pid": 0}), encoding="utf-8")
    queue_cfg.write_text(
        f"""
output_root: {queue_out.as_posix()}
stages:
  - name: first
    config: {stage1_cfg.as_posix()}
    resume: true
    skip_completed: true
  - name: second
    config: {stage2_cfg.as_posix()}
    resume: true
    skip_completed: true
""".strip(),
        encoding="utf-8",
    )

    called: list[str] = []

    def _fake_run_pipeline(cfg, **kwargs):  # type: ignore[no-untyped-def]
        called.append(cfg.pipeline.output_root)
        return {"teacher_name": cfg.teacher.name, "total_records": 4}

    monkeypatch.setattr("elt_lm.gguf_distill_queue.run_pipeline", _fake_run_pipeline)

    summary = run_queue(load_gguf_distill_queue_config(queue_cfg))

    assert summary["state"] == "complete"
    assert called == [stage2_dir.as_posix()]
    assert summary["stage_results"][0]["state"] == "skipped_complete"
    assert summary["stage_results"][1]["state"] == "complete"
