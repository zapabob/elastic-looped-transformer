from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from elt_lm.gguf_distill import _pid_is_alive, load_gguf_distill_config, run_pipeline


@dataclass
class GGUFQueueStageConfig:
    config: str
    name: str = ""
    output_dir: str = ""
    max_tasks: int = 0
    skip_upload: bool = False
    skip_student_eval: bool = False
    resume: bool = True
    skip_completed: bool = True


@dataclass
class GGUFQueueConfig:
    output_root: str
    poll_interval_sec: int = 30
    continue_on_failure: bool = False
    wait_for_existing: bool = True
    stages: list[GGUFQueueStageConfig] = field(default_factory=list)


def load_gguf_distill_queue_config(path: str | Path) -> GGUFQueueConfig:
    cfg_path = Path(path)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    stage_items = payload.get("stages") or []
    base_dir = cfg_path.parent
    stages: list[GGUFQueueStageConfig] = []
    for item in stage_items:
        stage = GGUFQueueStageConfig(**item)
        config_path = Path(stage.config)
        if not config_path.is_absolute():
            primary = (base_dir / config_path).resolve()
            fallback = config_path.resolve()
            stage.config = str(primary if primary.exists() or not fallback.exists() else fallback)
        if stage.output_dir:
            output_dir = Path(stage.output_dir)
            if not output_dir.is_absolute():
                primary = (base_dir / output_dir).resolve()
                fallback = output_dir.resolve()
                stage.output_dir = str(primary if primary.exists() or not fallback.exists() else fallback)
        stages.append(stage)
    return GGUFQueueConfig(
        output_root=str(payload.get("output_root") or ""),
        poll_interval_sec=int(payload.get("poll_interval_sec", 30)),
        continue_on_failure=bool(payload.get("continue_on_failure", False)),
        wait_for_existing=bool(payload.get("wait_for_existing", True)),
        stages=stages,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _effective_pid(status: dict[str, Any], output_dir: Path) -> int:
    pid = int(status.get("pid", 0) or 0)
    if pid > 0:
        return pid
    lock_payload = _read_json(output_dir / "run.lock")
    return int(lock_payload.get("pid", 0) or 0)


def inspect_stage_runtime_state(output_dir: Path) -> tuple[str, dict[str, Any]]:
    status = _read_json(output_dir / "status.json")
    if not status:
        status = _read_json(output_dir / "heartbeat.json")
    state = str(status.get("state", "")).strip().lower()
    pid = _effective_pid(status, output_dir)
    live = _pid_is_alive(pid)
    if live and state in {"starting", "running"}:
        return "running", status
    if live and not state:
        return "running", status
    if state == "complete":
        return "complete", status
    if state == "failed":
        return "failed", status
    if live:
        return "running", status
    if status:
        return state or "stale", status
    return "missing", {}


def _stage_name(stage: GGUFQueueStageConfig) -> str:
    if stage.name:
        return stage.name
    return Path(stage.config).stem


def build_queue_status(
    *,
    queue_cfg: GGUFQueueConfig,
    current_stage: str,
    stage_index: int,
    total_stages: int,
    state: str,
    started_at: float,
    stage_results: list[dict[str, Any]],
    last_error: str,
) -> dict[str, Any]:
    updated_at = time.time()
    completed = sum(1 for item in stage_results if item.get("state") in {"complete", "skipped_complete"})
    failed = sum(1 for item in stage_results if item.get("state") == "failed")
    return {
        "current_stage": current_stage,
        "stage_index": stage_index,
        "total_stages": total_stages,
        "completed_stages": completed,
        "failed_stages": failed,
        "state": state,
        "started_at": started_at,
        "updated_at": updated_at,
        "elapsed_sec": round(max(0.0, updated_at - started_at), 3),
        "poll_interval_sec": queue_cfg.poll_interval_sec,
        "continue_on_failure": queue_cfg.continue_on_failure,
        "wait_for_existing": queue_cfg.wait_for_existing,
        "last_error": last_error,
        "stages": stage_results,
        "pid": os.getpid(),
    }


def write_queue_status(output_dir: Path, snapshot: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "status.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    heartbeat = {
        "state": snapshot["state"],
        "current_stage": snapshot["current_stage"],
        "stage_index": snapshot["stage_index"],
        "total_stages": snapshot["total_stages"],
        "completed_stages": snapshot["completed_stages"],
        "failed_stages": snapshot["failed_stages"],
        "updated_at": snapshot["updated_at"],
        "last_error": snapshot["last_error"],
        "pid": snapshot["pid"],
    }
    (output_dir / "heartbeat.json").write_text(
        json.dumps(heartbeat, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _wait_for_terminal_stage(output_dir: Path, poll_interval_sec: int) -> tuple[str, dict[str, Any]]:
    while True:
        state, status = inspect_stage_runtime_state(output_dir)
        if state != "running":
            return state, status
        time.sleep(max(1, poll_interval_sec))


def run_queue(cfg: GGUFQueueConfig) -> dict[str, Any]:
    if not cfg.output_root:
        raise ValueError("queue output_root is required")
    if not cfg.stages:
        raise ValueError("queue requires at least one stage")

    output_dir = Path(cfg.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    stage_results: list[dict[str, Any]] = []
    last_error = ""
    total_stages = len(cfg.stages)

    initial = build_queue_status(
        queue_cfg=cfg,
        current_stage="init",
        stage_index=0,
        total_stages=total_stages,
        state="starting",
        started_at=started_at,
        stage_results=stage_results,
        last_error=last_error,
    )
    write_queue_status(output_dir, initial)
    (output_dir / "queue_plan.json").write_text(
        json.dumps({"queue": asdict(cfg)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for index, stage in enumerate(cfg.stages, start=1):
        stage_name = _stage_name(stage)
        distill_cfg = load_gguf_distill_config(stage.config)
        stage_out_dir = Path(stage.output_dir) if stage.output_dir else Path(distill_cfg.pipeline.output_root)

        snapshot = build_queue_status(
            queue_cfg=cfg,
            current_stage=stage_name,
            stage_index=index,
            total_stages=total_stages,
            state="running",
            started_at=started_at,
            stage_results=stage_results,
            last_error=last_error,
        )
        write_queue_status(output_dir, snapshot)

        stage_state, stage_status = inspect_stage_runtime_state(stage_out_dir)
        if cfg.wait_for_existing and stage_state == "running":
            stage_state, stage_status = _wait_for_terminal_stage(stage_out_dir, cfg.poll_interval_sec)

        if stage_state == "complete" and stage.skip_completed:
            stage_results.append(
                {
                    "name": stage_name,
                    "config": stage.config,
                    "output_dir": str(stage_out_dir),
                    "state": "skipped_complete",
                    "summary": stage_status,
                }
            )
            continue

        if stage_state == "failed" and not stage.resume:
            last_error = f"stage {stage_name} is failed and resume=false"
            stage_results.append(
                {
                    "name": stage_name,
                    "config": stage.config,
                    "output_dir": str(stage_out_dir),
                    "state": "failed",
                    "summary": stage_status,
                    "error": last_error,
                }
            )
            if not cfg.continue_on_failure:
                failure = build_queue_status(
                    queue_cfg=cfg,
                    current_stage=stage_name,
                    stage_index=index,
                    total_stages=total_stages,
                    state="failed",
                    started_at=started_at,
                    stage_results=stage_results,
                    last_error=last_error,
                )
                write_queue_status(output_dir, failure)
                raise RuntimeError(last_error)
            continue

        try:
            summary = run_pipeline(
                distill_cfg,
                output_dir=stage_out_dir if stage.output_dir else None,
                max_tasks=stage.max_tasks,
                skip_upload=stage.skip_upload,
                skip_student_eval=stage.skip_student_eval,
                resume=stage.resume,
            )
        except Exception as exc:
            last_error = str(exc)
            stage_results.append(
                {
                    "name": stage_name,
                    "config": stage.config,
                    "output_dir": str(stage_out_dir),
                    "state": "failed",
                    "error": last_error,
                }
            )
            if not cfg.continue_on_failure:
                failure = build_queue_status(
                    queue_cfg=cfg,
                    current_stage=stage_name,
                    stage_index=index,
                    total_stages=total_stages,
                    state="failed",
                    started_at=started_at,
                    stage_results=stage_results,
                    last_error=last_error,
                )
                write_queue_status(output_dir, failure)
                raise
            continue

        stage_results.append(
            {
                "name": stage_name,
                "config": stage.config,
                "output_dir": str(stage_out_dir),
                "state": "complete",
                "summary": summary,
            }
        )

    final_state = "complete" if not any(item.get("state") == "failed" for item in stage_results) else "failed"
    final_snapshot = build_queue_status(
        queue_cfg=cfg,
        current_stage="complete" if final_state == "complete" else "failed",
        stage_index=total_stages,
        total_stages=total_stages,
        state=final_state,
        started_at=started_at,
        stage_results=stage_results,
        last_error=last_error,
    )
    write_queue_status(output_dir, final_snapshot)
    summary = {
        "state": final_state,
        "queue_output_root": str(output_dir),
        "stage_results": stage_results,
        "started_at": started_at,
        "finished_at": time.time(),
    }
    (output_dir / "queue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config for multi-stage GGUF distillation queue")
    args = p.parse_args()
    cfg = load_gguf_distill_queue_config(args.config)
    summary = run_queue(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
