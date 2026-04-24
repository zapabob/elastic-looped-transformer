"""Resumable ELT long-run pipeline for Windows Task Scheduler.

The pipeline is intentionally idempotent:

* stage completion is tracked under H:/elt_data/pipeline_state/*.done
* training stages resume from run_dir/last.pt when present
* GGUF distill output dirs are never regenerated over a completed or suspicious
  bundle; upload/recover stages stop with a clear error instead
* a process lock prevents 5-minute Task Scheduler ticks from double-starting

Manual checks:

    uv run --no-sync python scripts/pipeline.py --dry-run
    uv run --no-sync python scripts/pipeline.py --only 00_pretrain_clean --dry-run
    uv run --no-sync python scripts/pipeline.py --no-start-long-train
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path("H:/elt_data/pipeline_state")
LOG_DIR = Path("H:/elt_data/pipeline_logs")
TELEMETRY_PATH = STATE_DIR / "pipeline.jsonl"
STATUS_PATH = STATE_DIR / "status.json"
HEARTBEAT_PATH = STATE_DIR / "heartbeat.json"
LOCK_PATH = STATE_DIR / "pipeline.lock"

TOKENIZER = "H:/Qwen3.5-9B-official-hf"
HUIHUI_DETECTION_ROOT = Path("H:/elt_data/gguf_distill/huihui_qwen36_detection")
HUIHUI_DETECTION_PREP_ROOT = Path("H:/elt_data/posttrain/detection/huihui_qwen36")


class PipelineError(RuntimeError):
    pass


class AlreadyRunning(RuntimeError):
    pass


class LongStageDeferred(RuntimeError):
    pass


@dataclass
class PipelineContext:
    dry_run: bool = False
    no_start_long_train: bool = False
    poll_interval_sec: int = 30
    started_at: float = field(default_factory=time.time)


@dataclass
class CommandPlan:
    cmd: list[str]
    run_dir: Path | None = None
    resume_path: Path | None = None
    long_running: bool = False


@dataclass
class Stage:
    name: str
    run: Callable[[PipelineContext], None]
    long_running: bool = False

    def marker(self) -> Path:
        return STATE_DIR / f"{self.name}.done"

    def is_done(self) -> bool:
        return self.marker().exists()

    def mark_done(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.marker().write_text(
            f"completed_at={time.time()}\n",
            encoding="utf-8",
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return False
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_pipeline_lock(lock_path: Path = LOCK_PATH) -> Callable[[], None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        payload = _read_json(lock_path)
        pid = int(payload.get("pid", 0) or 0)
        if _pid_is_alive(pid):
            raise AlreadyRunning(f"pipeline already running with pid={pid}")
        try:
            lock_path.unlink()
        except OSError:
            raise AlreadyRunning(f"stale pipeline lock could not be removed: {lock_path}")
    payload = {"pid": os.getpid(), "started_at": time.time(), "cwd": str(ROOT)}
    with open(lock_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))

    def release() -> None:
        try:
            current = _read_json(lock_path)
            if int(current.get("pid", 0) or 0) == os.getpid():
                lock_path.unlink()
        except OSError:
            pass

    return release


def emit(event: str, **fields: object) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), "event": event, **fields}
    with open(TELEMETRY_PATH, "a", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pipeline_status(
    *,
    state: str,
    current_stage: str,
    stage_index: int,
    total_stages: int,
    started_at: float,
    last_error: str = "",
) -> None:
    now = time.time()
    snapshot = {
        "state": state,
        "current_stage": current_stage,
        "stage_index": stage_index,
        "total_stages": total_stages,
        "started_at": started_at,
        "updated_at": now,
        "elapsed_sec": round(now - started_at, 3),
        "last_error": last_error,
        "pid": os.getpid(),
    }
    _write_json(STATUS_PATH, snapshot)
    _write_json(
        HEARTBEAT_PATH,
        {
            "state": state,
            "current_stage": current_stage,
            "stage_index": stage_index,
            "total_stages": total_stages,
            "updated_at": now,
            "last_error": last_error,
            "pid": os.getpid(),
        },
    )


def run_subprocess(cmd: list[str], *, dry_run: bool = False) -> int:
    print("  $ " + " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise PipelineError(f"command failed with rc={result.returncode}: {' '.join(cmd)}")
    return result.returncode


def load_train_yaml(config_path: str | Path) -> dict[str, Any]:
    path = ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def train_run_dir(config_path: str | Path) -> Path:
    raw = load_train_yaml(config_path)
    run_dir = raw.get("run_dir")
    if not run_dir:
        raise PipelineError(f"config has no run_dir: {config_path}")
    return Path(str(run_dir))


def build_training_command(config_path: str, *, entrypoint: str) -> CommandPlan:
    run_dir = train_run_dir(config_path)
    last = run_dir / "last.pt"
    cmd = ["uv", "run", "--no-sync", entrypoint, "--config", config_path]
    resume = last if last.exists() and last.stat().st_size > 0 else None
    if resume is not None:
        cmd += ["--resume", str(resume)]
    return CommandPlan(cmd=cmd, run_dir=run_dir, resume_path=resume, long_running=True)


def run_training_config(ctx: PipelineContext, config_path: str, *, entrypoint: str) -> None:
    plan = build_training_command(config_path, entrypoint=entrypoint)
    print(f"  run_dir : {plan.run_dir}")
    print(f"  resume  : {plan.resume_path or '<none>'}")
    if ctx.no_start_long_train:
        print("  skip execution: --no-start-long-train")
        raise LongStageDeferred(f"deferred long-running command: {' '.join(plan.cmd)}")
    run_subprocess(plan.cmd, dry_run=ctx.dry_run)


def file_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def inspect_distill_bundle(output_dir: Path) -> dict[str, Any]:
    status = _read_json(output_dir / "status.json")
    heartbeat = _read_json(output_dir / "heartbeat.json")
    summary = _read_json(output_dir / "eval_summary.json")
    train_path = output_dir / "distill_train.jsonl"
    val_path = output_dir / "distill_val.jsonl"
    train_nonempty = file_nonempty(train_path)
    val_nonempty = file_nonempty(val_path)
    total_records = int(summary.get("total_records", 0) or 0)
    state = str(status.get("state") or heartbeat.get("state") or "").lower()
    pid = int(status.get("pid") or heartbeat.get("pid") or 0)
    running = _pid_is_alive(pid) and state in {"starting", "running"}
    return {
        "state": state,
        "pid": pid,
        "running": running,
        "current_stage": status.get("current_stage") or heartbeat.get("current_stage", ""),
        "total_records": total_records,
        "train_nonempty": train_nonempty,
        "val_nonempty": val_nonempty,
        "train_size": train_path.stat().st_size if train_path.exists() else -1,
        "val_size": val_path.stat().st_size if val_path.exists() else -1,
        "summary_exists": bool(summary),
    }


def wait_for_distill_terminal(output_dir: Path, poll_interval_sec: int) -> dict[str, Any]:
    while True:
        info = inspect_distill_bundle(output_dir)
        if not info["running"]:
            return info
        print(f"  waiting for distill process pid={info['pid']} stage={info['current_stage']}")
        time.sleep(max(1, poll_interval_sec))


def hf_upload_commands(output_dir: Path, repo_id: str, *, private: bool = True) -> list[list[str]]:
    create = ["hf", "repos", "create", repo_id, "--type", "dataset", "--exist-ok"]
    if private:
        create.append("--private")
    return [
        create,
        ["hf", "upload-large-folder", repo_id, str(output_dir), "--type", "dataset"],
    ]


def retry_hf_upload(ctx: PipelineContext, output_dir: Path, repo_id: str) -> None:
    for cmd in hf_upload_commands(output_dir, repo_id, private=True):
        run_subprocess(cmd, dry_run=ctx.dry_run)


def stage_pretrain_clean(ctx: PipelineContext) -> None:
    run_training_config(ctx, "configs/base_1B_continue_clean.yaml", entrypoint="elt-train")


def stage_distill_huihui_detection_upload_or_recover(ctx: PipelineContext) -> None:
    info = wait_for_distill_terminal(HUIHUI_DETECTION_ROOT, ctx.poll_interval_sec)
    print("  distill bundle:", json.dumps(info, ensure_ascii=False))
    if info["train_nonempty"] and info["val_nonempty"] and info["total_records"] > 0:
        retry_hf_upload(
            ctx,
            HUIHUI_DETECTION_ROOT,
            "zapabobouj/elt-lm-gguf-distill-huihui-qwen36-detection",
        )
        return
    if info["summary_exists"] and info["total_records"] > 0:
        raise PipelineError(
            "protected Huihui detection bundle has nonzero eval_summary but "
            "distill_train.jsonl/distill_val.jsonl are missing or zero-byte. "
            "Stop before regeneration and restore a completed snapshot."
        )
    raise PipelineError(
        "Huihui detection bundle is not complete enough for upload; refusing to "
        "regenerate into the protected output directory."
    )


def stage_prepare_detection_sft(ctx: PipelineContext) -> None:
    info = inspect_distill_bundle(HUIHUI_DETECTION_ROOT)
    if not (info["train_nonempty"] and info["val_nonempty"]):
        raise PipelineError("cannot prepare detection SFT: distill train/val JSONL are missing or empty")
    cmd = [
        "uv", "run", "--no-sync", "elt-prepare-gguf-detection-sft",
        "--input-root", str(HUIHUI_DETECTION_ROOT),
        "--output-root", str(HUIHUI_DETECTION_PREP_ROOT),
        "--tokenizer", TOKENIZER,
    ]
    run_subprocess(cmd, dry_run=ctx.dry_run)


def stage_detection_sft(ctx: PipelineContext) -> None:
    run_training_config(
        ctx,
        "configs/posttrain_detection_sft_huihui_qwen36.yaml",
        entrypoint="elt-train",
    )


def stage_hauhaucs_multilane_distill(ctx: PipelineContext) -> None:
    cmd = [
        "uv", "run", "--no-sync", "elt-gguf-distill-queue",
        "--config", "configs/gguf_distill_qwen35_hauhaucs_multilane_queue.yaml",
    ]
    run_subprocess(cmd, dry_run=ctx.dry_run)


LANE_PREP: list[tuple[str, Path, Path, str]] = [
    ("code", Path("H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_code"), Path("H:/elt_data/posttrain/code/qwen35_hauhaucs"), "configs/posttrain_code_sft_qwen35_hauhaucs.yaml"),
    ("math", Path("H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_math"), Path("H:/elt_data/posttrain/math/qwen35_hauhaucs"), "configs/posttrain_math_sft_qwen35_hauhaucs.yaml"),
    ("stem_reasoning", Path("H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_stem_reasoning"), Path("H:/elt_data/posttrain/stem_reasoning/qwen35_hauhaucs"), "configs/posttrain_stem_sft_qwen35_hauhaucs.yaml"),
    ("tool_use", Path("H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_tool_use"), Path("H:/elt_data/posttrain/tool_use/qwen35_hauhaucs"), "configs/posttrain_tool_sft_qwen35_hauhaucs.yaml"),
]


def stage_lane_sft(ctx: PipelineContext) -> None:
    for lane, input_root, output_root, config_path in LANE_PREP:
        info = inspect_distill_bundle(input_root)
        if not (info["train_nonempty"] and info["val_nonempty"]):
            raise PipelineError(f"lane {lane} distill bundle is missing train/val JSONL: {input_root}")
        prep_cmd = [
            "uv", "run", "--no-sync", "elt-prepare-gguf-lane-sft",
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--tokenizer", TOKENIZER,
            "--lane", lane,
        ]
        run_subprocess(prep_cmd, dry_run=ctx.dry_run)
        run_training_config(ctx, config_path, entrypoint="elt-train")


def stage_kl_grpo(ctx: PipelineContext) -> None:
    for config_path in [
        "configs/grpo_code_qwen35_hauhaucs.yaml",
        "configs/grpo_math_qwen35_hauhaucs.yaml",
        "configs/grpo_tool_qwen35_hauhaucs.yaml",
    ]:
        raw = load_train_yaml(config_path)
        kl_beta = float((raw.get("grpo") or {}).get("kl_beta", 0.0))
        if kl_beta <= 0:
            raise PipelineError(f"GRPO config must keep kl_beta > 0: {config_path}")
        run_training_config(ctx, config_path, entrypoint="elt-train-grpo")


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if file_nonempty(path):
            return path
    return None


def stage_eval_compare(ctx: PipelineContext) -> None:
    ckpt = first_existing([
        Path("H:/elt_data/runs/grpo_tool_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/grpo_math_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/grpo_code_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/posttrain_detection_sft_huihui_qwen36/last.pt"),
        Path("H:/elt_data/runs/base_1B_clean_continue/last.pt"),
    ])
    if ckpt is None:
        raise PipelineError("no checkpoint available for eval_compare")
    cmd = [
        "uv", "run", "--no-sync", "elt-anytime",
        "--ckpt", str(ckpt),
        "--val-bin", "H:/elt_data/bin_clean_2026-04-24/val.bin",
        "--L-list", "1,2,3,4",
    ]
    run_subprocess(cmd, dry_run=ctx.dry_run)


STAGES: list[Stage] = [
    Stage("00_pretrain_clean", stage_pretrain_clean, long_running=True),
    Stage("01_distill_huihui_detection_upload_or_recover", stage_distill_huihui_detection_upload_or_recover),
    Stage("02_prepare_detection_sft", stage_prepare_detection_sft),
    Stage("03_detection_sft", stage_detection_sft, long_running=True),
    Stage("04_hauhaucs_multilane_distill", stage_hauhaucs_multilane_distill, long_running=True),
    Stage("05_lane_sft", stage_lane_sft, long_running=True),
    Stage("06_kl_grpo", stage_kl_grpo, long_running=True),
    Stage("07_eval_compare", stage_eval_compare),
]


def select_stages(stages: list[Stage], only: str = "", skip: str = "") -> list[Stage]:
    only_tags = [item for item in only.split(",") if item] if only else []
    skip_tags = [item for item in skip.split(",") if item] if skip else []
    selected: list[Stage] = []
    for stage in stages:
        if only_tags and not any(tag in stage.name for tag in only_tags):
            continue
        if skip_tags and any(tag in stage.name for tag in skip_tags):
            continue
        selected.append(stage)
    return selected


def print_plan(stages: list[Stage], ctx: PipelineContext) -> None:
    print(f"  ROOT      = {ROOT}")
    print(f"  STATE_DIR = {STATE_DIR}")
    print(f"  dry_run   = {ctx.dry_run}")
    print(f"  no_start_long_train = {ctx.no_start_long_train}")
    print("  plan:")
    for stage in stages:
        state = "done" if stage.is_done() else "pending"
        print(f"    [{state}] {stage.name}")


def run_pipeline(stages: list[Stage], ctx: PipelineContext) -> int:
    release = acquire_pipeline_lock()
    try:
        emit("pipeline_start", plan=[stage.name for stage in stages])
        total = len(stages)
        for index, stage in enumerate(stages, start=1):
            if stage.is_done():
                print(f"  skip done: {stage.name}")
                emit("pipeline_stage", name=stage.name, status="skipped")
                continue
            print(f"\n========== {stage.name} ==========")
            write_pipeline_status(
                state="running",
                current_stage=stage.name,
                stage_index=index,
                total_stages=total,
                started_at=ctx.started_at,
            )
            emit("pipeline_stage", name=stage.name, status="start")
            t0 = time.time()
            try:
                stage.run(ctx)
            except LongStageDeferred as exc:
                message = str(exc)
                write_pipeline_status(
                    state="deferred",
                    current_stage=stage.name,
                    stage_index=index,
                    total_stages=total,
                    started_at=ctx.started_at,
                    last_error=message,
                )
                emit("pipeline_stage", name=stage.name, status="deferred", reason=message)
                print(f"  stage deferred: {message}")
                return 0
            except Exception as exc:
                message = str(exc)
                write_pipeline_status(
                    state="failed",
                    current_stage=stage.name,
                    stage_index=index,
                    total_stages=total,
                    started_at=ctx.started_at,
                    last_error=message,
                )
                emit("pipeline_stage", name=stage.name, status="failed", error=message)
                print(f"  stage failed: {message}")
                return 1
            stage.mark_done()
            elapsed = time.time() - t0
            emit("pipeline_stage", name=stage.name, status="done", elapsed_sec=elapsed)
            print(f"  {stage.name} done in {elapsed/60:.1f} min")

        write_pipeline_status(
            state="complete",
            current_stage="complete",
            stage_index=total,
            total_stages=total,
            started_at=ctx.started_at,
        )
        emit("pipeline_complete")
        return 0
    finally:
        release()


def reset_markers(stages: list[Stage]) -> None:
    for stage in stages:
        marker = stage.marker()
        if marker.exists():
            marker.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated stage name substrings")
    parser.add_argument("--skip", default="", help="comma-separated stage name substrings")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true", help="delete selected stage markers before running")
    parser.add_argument("--no-start-long-train", action="store_true", help="validate/prepare but skip long train stages")
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(
        dry_run=args.dry_run,
        no_start_long_train=args.no_start_long_train,
    )
    stages = select_stages(STAGES, only=args.only, skip=args.skip)
    if args.reset:
        reset_markers(stages)
    print_plan(stages, ctx)
    if args.dry_run:
        return
    try:
        code = run_pipeline(stages, ctx)
    except AlreadyRunning as exc:
        print(f"  {exc}")
        code = 0
    raise SystemExit(code)


if __name__ == "__main__":
    main()
