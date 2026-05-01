"""Resumable ELT long-run pipeline for Windows Task Scheduler.

The pipeline is intentionally idempotent:

* stage completion is tracked under H:/elt_data/pipeline_state/*.done
* training stages resume from run_dir/last.pt when present
* GGUF distill output dirs are never regenerated over a completed or suspicious
  bundle; upload/recover stages stop with a clear error instead
* a process lock prevents 5-minute Task Scheduler ticks from double-starting

Manual checks:

    uv run --no-sync python scripts/pipeline.py --dry-run
    uv run --no-sync python scripts/pipeline.py --profile side-lora --dry-run
    uv run --no-sync python scripts/pipeline.py --profile posttrain-grpo --dry-run
    uv run --no-sync python scripts/pipeline.py --profile replay-refresh --dry-run
    uv run --no-sync python scripts/pipeline.py --only 00_pretrain_clean --dry-run
    uv run --no-sync python scripts/pipeline.py --no-start-long-train
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path("H:/elt_data/pipeline_state")
LOG_DIR = Path("H:/elt_data/pipeline_logs")
H_CACHE_ROOT = Path("H:/elt_data/cache")
H_TEMP_DIR = H_CACHE_ROOT / "tmp"
TELEMETRY_PATH = STATE_DIR / "pipeline.jsonl"
STATUS_PATH = STATE_DIR / "status.json"
HEARTBEAT_PATH = STATE_DIR / "heartbeat.json"
LOCK_PATH = STATE_DIR / "pipeline.lock"

TOKENIZER = "H:/Qwen3.5-9B-official-hf"
HUIHUI_DETECTION_ROOT = Path("H:/elt_data/gguf_distill/huihui_qwen36_detection")
HUIHUI_DETECTION_PREP_ROOT = Path("H:/elt_data/posttrain/detection/huihui_qwen36")
QWEN35_BOOTSTRAP_CKPT = Path("H:/elt_data/runs/qwen35_4b_elt_bootstrap/last.pt")
# Use the 8.3 short path because Python's Windows subprocess quoting can pass
# quoted .bat paths through to cmd.exe as literal escaped quotes.
VSDEV_CMD = "C:\\PROGRA~1\\MICROS~4\\2022\\COMMUN~1\\Common7\\Tools\\VsDevCmd.bat"

H_DRIVE_ENV = {
    # Keep large transient files and framework caches off C: during long runs.
    "TMP": str(H_TEMP_DIR),
    "TEMP": str(H_TEMP_DIR),
    "TMPDIR": str(H_TEMP_DIR),
    "UV_CACHE_DIR": str(H_CACHE_ROOT / "uv"),
    "UV_PYTHON_INSTALL_DIR": str(H_CACHE_ROOT / "uv" / "python"),
    "UV_TOOL_DIR": str(H_CACHE_ROOT / "uv" / "tools"),
    "PIP_CACHE_DIR": str(H_CACHE_ROOT / "pip"),
    "HF_HOME": str(H_CACHE_ROOT / "hf"),
    "HF_HUB_CACHE": str(H_CACHE_ROOT / "hf" / "hub"),
    "HF_DATASETS_CACHE": str(H_CACHE_ROOT / "hf" / "datasets"),
    "TRANSFORMERS_CACHE": str(H_CACHE_ROOT / "hf" / "transformers"),
    "TORCH_HOME": str(H_CACHE_ROOT / "torch"),
    "XDG_CACHE_HOME": str(H_CACHE_ROOT / "xdg"),
    "TRITON_CACHE_DIR": str(H_CACHE_ROOT / "triton"),
    "TORCHINDUCTOR_CACHE_DIR": str(H_CACHE_ROOT / "torchinductor"),
    "CUDA_CACHE_PATH": str(H_CACHE_ROOT / "cuda"),
    "NUMBA_CACHE_DIR": str(H_CACHE_ROOT / "numba"),
    "MPLCONFIGDIR": str(H_CACHE_ROOT / "matplotlib"),
    "PYTHONPYCACHEPREFIX": str(H_CACHE_ROOT / "pycache"),
}


def ensure_h_drive_runtime_dirs() -> None:
    for value in H_DRIVE_ENV.values():
        Path(value).mkdir(parents=True, exist_ok=True)


def h_drive_subprocess_env() -> dict[str, str]:
    ensure_h_drive_runtime_dirs()
    env = os.environ.copy()
    env.update(H_DRIVE_ENV)
    return env


def _cmd_set_env_prefix() -> str:
    return "".join(f"set {key}={value}&& " for key, value in H_DRIVE_ENV.items())


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


def vsdev_command(inner_cmd: list[str]) -> list[str]:
    quoted_inner = subprocess.list2cmdline(inner_cmd)
    return [
        "cmd.exe",
        "/c",
        (
            'chcp 65001 >NUL && '
            f"call {VSDEV_CMD} -arch=x64 -host_arch=x64 && "
            + _cmd_set_env_prefix()
            +
            "set CC=cl.exe&& "
            "set CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8&& "
            "set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8&& "
            f"{quoted_inner}"
        ),
    ]


def run_subprocess(cmd: list[str], *, dry_run: bool = False) -> int:
    print("  $ " + " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=ROOT, env=h_drive_subprocess_env())
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


def build_training_command(
    config_path: str,
    *,
    entrypoint: str,
    initial_resume: Path | None = None,
    use_vsdev: bool = False,
) -> CommandPlan:
    run_dir = train_run_dir(config_path)
    last = run_dir / "last.pt"
    cmd = ["uv", "run", "--no-sync", entrypoint, "--config", config_path]
    resume = last if last.exists() and last.stat().st_size > 0 else initial_resume
    if resume is not None:
        cmd += ["--resume", str(resume)]
    if use_vsdev:
        cmd = vsdev_command(cmd)
    return CommandPlan(cmd=cmd, run_dir=run_dir, resume_path=resume, long_running=True)


def offload_root_for_config(config_path: str | Path) -> Path | None:
    raw = load_train_yaml(config_path)
    offload = raw.get("offload") or {}
    root = offload.get("root") if isinstance(offload, dict) else None
    if not root:
        return None
    return Path(str(root))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def cleanup_completed_offload(config_path: str | Path, *, dry_run: bool = False) -> None:
    """Delete regenerable NvmeAdamW optimizer state after a completed stage.

    The safety checks are intentionally narrow: only a directory literally named
    offload_nvme under H:/elt_data/runs can be removed. Checkpoints such as
    last.pt and adapter exports are outside this tree and are preserved.
    """

    root = offload_root_for_config(config_path)
    if root is None:
        return
    resolved = root.resolve()
    allowed = Path("H:/elt_data/runs").resolve()
    if resolved.name != "offload_nvme" or not _is_relative_to(resolved, allowed):
        raise PipelineError(f"refusing unsafe offload cleanup path: {root}")
    if not resolved.exists():
        return
    print(f"  cleanup offload: {resolved}")
    if dry_run:
        return
    shutil.rmtree(resolved)


def prune_completed_checkpoints(config_path: str | Path, *, dry_run: bool = False) -> None:
    """Keep last.pt for completed stages and remove bulky duplicate snapshots."""

    if not training_run_complete(config_path):
        return
    raw = load_train_yaml(config_path)
    run_dir_value = raw.get("run_dir")
    if not run_dir_value:
        return
    run_dir = Path(str(run_dir_value)).resolve()
    allowed = Path("H:/elt_data/runs").resolve()
    if not _is_relative_to(run_dir, allowed):
        raise PipelineError(f"refusing unsafe checkpoint prune path: {run_dir_value}")
    for path in sorted(run_dir.glob("rolling_*.pt")) + sorted(run_dir.glob("step_*.pt")):
        if not path.is_file():
            continue
        print(f"  prune checkpoint: {path}")
        if not dry_run:
            path.unlink()


def training_run_complete(config_path: str | Path) -> bool:
    """Return True when a training run already emitted a final checkpoint event."""

    raw = load_train_yaml(config_path)
    run_dir = raw.get("run_dir")
    total_steps = int(raw.get("total_steps", 0) or 0)
    if not run_dir or total_steps <= 0:
        return False
    metrics = Path(str(run_dir)) / "metrics.jsonl"
    last = Path(str(run_dir)) / "last.pt"
    if not file_nonempty(metrics) or not file_nonempty(last):
        return False
    try:
        lines = metrics.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "checkpoint" and row.get("kind") == "final":
            return int(row.get("step", -1) or -1) >= total_steps
        if row.get("event") == "run_end":
            continue
    return False


def run_training_config(
    ctx: PipelineContext,
    config_path: str,
    *,
    entrypoint: str,
    initial_resume: Path | None = None,
    use_vsdev: bool = False,
    cleanup_offload_on_success: bool = False,
) -> None:
    plan = build_training_command(
        config_path,
        entrypoint=entrypoint,
        initial_resume=initial_resume,
        use_vsdev=use_vsdev,
    )
    print(f"  run_dir : {plan.run_dir}")
    print(f"  resume  : {plan.resume_path or '<none>'}")
    if ctx.no_start_long_train:
        print("  skip execution: --no-start-long-train")
        raise LongStageDeferred(f"deferred long-running command: {' '.join(plan.cmd)}")
    run_subprocess(plan.cmd, dry_run=ctx.dry_run)
    if cleanup_offload_on_success:
        prune_completed_checkpoints(config_path, dry_run=ctx.dry_run)
        cleanup_completed_offload(config_path, dry_run=ctx.dry_run)


def file_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _iter_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def inspect_stem_v0_benchmark_quality() -> dict[str, Any]:
    cases_path = STEM_VAL_MANIFEST.parent / "gguf_stem_reasoning_val_cases.jsonl"
    rows = _iter_jsonl_dicts(cases_path)
    total = len(rows)
    prompts = [str(row.get("prompt", "")) for row in rows]
    references = [str(row.get("reference", "")) for row in rows]
    duplicate_prompt_count = total - len(set(prompts))
    placeholder_count = sum(
        1
        for prompt in prompts
        if all(f"{letter}. Option {letter}" in prompt for letter in "ABCD")
    )
    reference_counts: dict[str, int] = {}
    for ref in references:
        reference_counts[ref] = reference_counts.get(ref, 0) + 1
    max_ref_ratio = max(reference_counts.values(), default=0) / max(1, total)
    quality_failed = (
        total > 0
        and (
            placeholder_count > 0
            or duplicate_prompt_count / max(1, total) > 0.25
            or max_ref_ratio > 0.75
        )
    )
    return {
        "cases_path": str(cases_path),
        "total_cases": total,
        "duplicate_prompt_count": duplicate_prompt_count,
        "placeholder_choice_count": placeholder_count,
        "reference_counts": reference_counts,
        "max_reference_ratio": max_ref_ratio,
        "quality_failed": quality_failed,
    }


def inspect_v0_lane_distill_quality(lane: str, input_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in ("distill_train.jsonl", "distill_val.jsonl"):
        rows.extend(_iter_jsonl_dicts(input_root / name))
    total = len(rows)
    prompts = [str(row.get("prompt", "")) for row in rows]
    texts = [str(row.get("text", "")) for row in rows]
    responses = [str(row.get("response", "")) for row in rows]
    references = [str(row.get("reference", "")) for row in rows]
    duplicate_prompt_count = total - len(set(prompts))
    duplicate_text_count = total - len(set(texts))
    fallback_return_none = sum(
        1 for value in [*responses, *references, *texts] if "return None" in value
    )
    answer_zero_count = sum(
        1 for row in rows
        if str(row.get("reference", "")).strip() == "0"
        or "<answer>0</answer>" in str(row.get("response", ""))
    )
    placeholder_choice_count = sum(
        1
        for prompt in prompts
        if all(f"{letter}. Option {letter}" in prompt for letter in "ABCD")
    )
    empty_arguments_count = sum(
        1
        for value in [*responses, *references, *texts]
        if '"arguments": {}' in value
        or '"arguments":{}' in value
        or "'arguments': {}" in value
    )
    reference_counts: dict[str, int] = {}
    for ref in references:
        reference_counts[ref] = reference_counts.get(ref, 0) + 1
    max_reference_ratio = max(reference_counts.values(), default=0) / max(1, total)
    duplicate_prompt_ratio = duplicate_prompt_count / max(1, total)
    duplicate_text_ratio = duplicate_text_count / max(1, total)

    lane_reasons: list[str] = []
    if lane == "code" and fallback_return_none:
        lane_reasons.append("code fallback `def solve(): return None` records detected")
    if lane == "math" and answer_zero_count:
        lane_reasons.append("math fallback `<answer>0</answer>` records detected")
    if lane == "stem_reasoning" and placeholder_choice_count:
        lane_reasons.append("stem placeholder `Option A/B/C/D` choices detected")
    if lane == "tool_use" and empty_arguments_count:
        lane_reasons.append("tool-use empty `arguments: {}` records detected")
    if duplicate_prompt_ratio > 0.50:
        lane_reasons.append("duplicate prompt ratio exceeds 50%")
    if duplicate_text_ratio > 0.50:
        lane_reasons.append("duplicate text ratio exceeds 50%")
    if max_reference_ratio > 0.75 and lane in {"math", "stem_reasoning", "tool_use"}:
        lane_reasons.append("reference label/value skew exceeds 75%")

    return {
        "lane": lane,
        "input_root": str(input_root),
        "total_records": total,
        "unique_prompts": len(set(prompts)),
        "duplicate_prompt_count": duplicate_prompt_count,
        "duplicate_prompt_ratio": duplicate_prompt_ratio,
        "unique_texts": len(set(texts)),
        "duplicate_text_count": duplicate_text_count,
        "duplicate_text_ratio": duplicate_text_ratio,
        "fallback_return_none": fallback_return_none,
        "answer_zero_count": answer_zero_count,
        "placeholder_choice_count": placeholder_choice_count,
        "empty_arguments_count": empty_arguments_count,
        "reference_counts": reference_counts,
        "max_reference_ratio": max_reference_ratio,
        "quality_failed": bool(lane_reasons),
        "reasons": lane_reasons,
    }


def enforce_v0_lane_quality(lane: str, input_root: Path) -> None:
    if os.environ.get("ELT_ALLOW_V0_SMOKE_TRAINING") == "1":
        return
    quality = inspect_v0_lane_distill_quality(lane, input_root)
    if not quality["quality_failed"]:
        return
    out = STATE_DIR / f"v0_{lane}_quality_gate.json"
    _write_json(
        out,
        {
            "state": "failed_quality_gate",
            "reason": "v0 HauhauCS lane data is smoke-only and should not be used for further SFT/GRPO.",
            "quality": quality,
            "next_action": (
                "Regenerate HauhauCS v1 distill with concrete tasks, references, "
                "dedup, and verifier gates before training this lane."
            ),
        },
    )
    raise PipelineError(
        f"refusing v0 {lane} SFT due quality gate failure; see {out}"
    )


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
        cleanup_offload_on_success=True,
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

STEM_SFT_CONFIG = "configs/posttrain_stem_sft_qwen35_hauhaucs.yaml"
STEM_VAL_MANIFEST = Path(
    "H:/elt_data/posttrain/stem_reasoning/qwen35_hauhaucs/benchmarks/"
    "gguf_stem_reasoning_val_manifest.yaml"
)
STEM_VAL_EVAL_DIR = Path("H:/elt_data/runs/posttrain_stem_sft_qwen35_hauhaucs/eval")
STEM_VAL_EVAL_JSON = STEM_VAL_EVAL_DIR / "stem_val_format_verifier_summary.json"
STEM_VAL_EVAL_CSV = STEM_VAL_EVAL_DIR / "stem_val_format_verifier_anytime.csv"

MIXED_LANE_PREP: list[tuple[str, Path, Path, str]] = [
    ("code", Path("H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_code"), Path("H:/elt_data/posttrain_mixed/code/qwen35_hauhaucs_replay"), "configs/posttrain_code_sft_qwen35_hauhaucs_replay.yaml"),
    ("math", Path("H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_math"), Path("H:/elt_data/posttrain_mixed/math/qwen35_hauhaucs_replay"), "configs/posttrain_math_sft_qwen35_hauhaucs_replay.yaml"),
    ("stem_reasoning", Path("H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_stem_reasoning"), Path("H:/elt_data/posttrain_mixed/stem_reasoning/qwen35_hauhaucs_replay"), "configs/posttrain_stem_sft_qwen35_hauhaucs_replay.yaml"),
    ("tool_use", Path("H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_tool_use"), Path("H:/elt_data/posttrain_mixed/tool_use/qwen35_hauhaucs_replay"), "configs/posttrain_tool_sft_qwen35_hauhaucs_replay.yaml"),
]


def stage_stem_sft_val_eval(ctx: PipelineContext) -> None:
    """Evaluate v0 stem SFT on its 13-case validation benchmark after completion."""

    if file_nonempty(STEM_VAL_EVAL_JSON):
        print(f"  skip existing stem val eval: {STEM_VAL_EVAL_JSON}")
        return
    quality = inspect_stem_v0_benchmark_quality()
    if quality["quality_failed"]:
        print("  skip stem v0 val eval: benchmark failed quality guard")
        payload = {
            "state": "skipped_quality_gate",
            "reason": (
                "v0 HauhauCS stem benchmark has placeholder choices, duplicate "
                "prompts, or strong answer-label skew; do not spend GPU time on "
                "this as a reasoning benchmark."
            ),
            "quality": quality,
            "interpretation_note": (
                "Treat v0 stem as smoke/pipeline-proof data only. Rebuild v1 "
                "with concrete choices, balanced labels, dedup, and verifier "
                "gates before using it for reasoning evaluation."
            ),
        }
        _write_json(STEM_VAL_EVAL_JSON, payload)
        STEM_VAL_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
        STEM_VAL_EVAL_CSV.write_text(
            "state,reason,total_cases,duplicate_prompt_count,placeholder_choice_count,max_reference_ratio\n"
            f"skipped_quality_gate,v0_quality_guard,{quality['total_cases']},"
            f"{quality['duplicate_prompt_count']},{quality['placeholder_choice_count']},"
            f"{quality['max_reference_ratio']}\n",
            encoding="utf-8",
        )
        return
    if not training_run_complete(STEM_SFT_CONFIG):
        raise PipelineError(f"stem SFT is not complete yet: {STEM_SFT_CONFIG}")
    run_dir = train_run_dir(STEM_SFT_CONFIG)
    ckpt = run_dir / "last.pt"
    if not file_nonempty(ckpt):
        raise PipelineError(f"stem SFT checkpoint is missing: {ckpt}")
    if not file_nonempty(STEM_VAL_MANIFEST):
        raise PipelineError(f"stem benchmark manifest is missing: {STEM_VAL_MANIFEST}")
    cmd = [
        "uv", "run", "--no-sync", "elt-anytime",
        "--ckpt", str(ckpt),
        "--benchmark-manifest", str(STEM_VAL_MANIFEST),
        "--bench-max-new-tokens", "96",
        "--bench-temperature", "0.0",
        "--bench-top-k", "1",
        "--L-list", "1,2,3,4",
        "--out-csv", str(STEM_VAL_EVAL_CSV),
        "--out-json", str(STEM_VAL_EVAL_JSON),
        "--run-dir", str(STEM_VAL_EVAL_DIR),
    ]
    run_subprocess(cmd, dry_run=ctx.dry_run)


def stage_lane_sft(ctx: PipelineContext) -> None:
    for lane, input_root, output_root, config_path in LANE_PREP:
        info = inspect_distill_bundle(input_root)
        if not (info["train_nonempty"] and info["val_nonempty"]):
            raise PipelineError(f"lane {lane} distill bundle is missing train/val JSONL: {input_root}")
        enforce_v0_lane_quality(lane, input_root)
        prep_cmd = [
            "uv", "run", "--no-sync", "elt-prepare-gguf-lane-sft",
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--tokenizer", TOKENIZER,
            "--lane", lane,
        ]
        run_subprocess(prep_cmd, dry_run=ctx.dry_run)
        run_training_config(
            ctx,
            config_path,
            entrypoint="elt-train",
            cleanup_offload_on_success=True,
        )
        if lane == "stem_reasoning":
            stage_stem_sft_val_eval(ctx)


def stage_prepare_hauhaucs_lanes(ctx: PipelineContext) -> None:
    for lane, input_root, output_root, _config_path in LANE_PREP:
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


def stage_hauhaucs_lane_sft_only(ctx: PipelineContext) -> None:
    for lane, _input_root, _output_root, config_path in LANE_PREP:
        enforce_v0_lane_quality(lane, _input_root)
        if training_run_complete(config_path):
            print(f"  skip completed training config: {config_path}")
            prune_completed_checkpoints(config_path, dry_run=ctx.dry_run)
            cleanup_completed_offload(config_path, dry_run=ctx.dry_run)
            if lane == "stem_reasoning":
                stage_stem_sft_val_eval(ctx)
            continue
        run_training_config(
            ctx,
            config_path,
            entrypoint="elt-train",
            cleanup_offload_on_success=True,
        )
        if lane == "stem_reasoning":
            stage_stem_sft_val_eval(ctx)


def run_grpo_configs(ctx: PipelineContext, config_paths: list[str]) -> None:
    for config_path in config_paths:
        raw = load_train_yaml(config_path)
        kl_beta = float((raw.get("grpo") or {}).get("kl_beta", 0.0))
        if kl_beta <= 0:
            raise PipelineError(f"GRPO config must keep kl_beta > 0: {config_path}")
        if training_run_complete(config_path):
            print(f"  skip completed training config: {config_path}")
            prune_completed_checkpoints(config_path, dry_run=ctx.dry_run)
            cleanup_completed_offload(config_path, dry_run=ctx.dry_run)
            continue
        run_training_config(
            ctx,
            config_path,
            entrypoint="elt-train-grpo",
            cleanup_offload_on_success=True,
        )


def stage_kl_grpo(ctx: PipelineContext) -> None:
    run_grpo_configs(
        ctx,
        [
            "configs/grpo_code_qwen35_hauhaucs.yaml",
            "configs/grpo_math_qwen35_hauhaucs.yaml",
            "configs/grpo_tool_qwen35_hauhaucs.yaml",
        ],
    )


SIDE_LORA_SFT_CONFIGS: list[str] = [
    "configs/qwen35_4b_side_lora_code_sft.yaml",
    "configs/qwen35_4b_side_lora_math_sft.yaml",
    "configs/qwen35_4b_side_lora_stem_sft.yaml",
    "configs/qwen35_4b_side_lora_tool_sft.yaml",
]

SIDE_LORA_MIXED_SFT_CONFIGS: list[str] = [
    "configs/qwen35_4b_side_lora_code_sft_replay.yaml",
    "configs/qwen35_4b_side_lora_math_sft_replay.yaml",
    "configs/qwen35_4b_side_lora_stem_sft_replay.yaml",
    "configs/qwen35_4b_side_lora_tool_sft_replay.yaml",
]


def stage_side_lora_sft(ctx: PipelineContext) -> None:
    run_side_lora_sft_configs(ctx, SIDE_LORA_SFT_CONFIGS)


def run_side_lora_sft_configs(ctx: PipelineContext, config_paths: list[str]) -> None:
    if not file_nonempty(QWEN35_BOOTSTRAP_CKPT):
        raise PipelineError(f"missing Qwen3.5-4B bootstrap checkpoint: {QWEN35_BOOTSTRAP_CKPT}")
    for config_path in config_paths:
        if training_run_complete(config_path):
            print(f"  skip completed training config: {config_path}")
            continue
        run_training_config(
            ctx,
            config_path,
            entrypoint="elt-train",
            initial_resume=QWEN35_BOOTSTRAP_CKPT,
            use_vsdev=True,
        )


def stage_side_lora_ilsd(ctx: PipelineContext) -> None:
    code_sft = Path("H:/elt_data/runs/qwen35_4b_side_lora_code_sft/last.pt")
    initial = code_sft if file_nonempty(code_sft) else QWEN35_BOOTSTRAP_CKPT
    run_training_config(
        ctx,
        "configs/qwen35_4b_side_lora_code_ilsd_l2.yaml",
        entrypoint="elt-train",
        initial_resume=initial,
        use_vsdev=True,
    )


def stage_export_side_lora_adapters(ctx: PipelineContext) -> None:
    exports = [
        ("code", Path("H:/elt_data/runs/qwen35_4b_side_lora_code_sft/last.pt")),
        ("math", Path("H:/elt_data/runs/qwen35_4b_side_lora_math_sft/last.pt")),
        ("stem", Path("H:/elt_data/runs/qwen35_4b_side_lora_stem_sft/last.pt")),
        ("tool", Path("H:/elt_data/runs/qwen35_4b_side_lora_tool_sft/last.pt")),
        ("code_ilsd_l2", Path("H:/elt_data/runs/qwen35_4b_side_lora_code_ilsd_l2/last.pt")),
    ]
    for name, ckpt in exports:
        if not file_nonempty(ckpt):
            raise PipelineError(f"cannot export missing side LoRA checkpoint: {ckpt}")
        cmd = [
            "uv", "run", "--no-sync", "python", "-m", "elt_lm.export_lora_adapter",
            "--ckpt", str(ckpt),
            "--out-dir", f"H:/elt_data/adapters/qwen35_4b_side/{name}",
        ]
        run_subprocess(cmd, dry_run=ctx.dry_run)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if file_nonempty(path):
            return path
    return None


def stage_eval_compare(ctx: PipelineContext) -> None:
    ckpt = first_existing([
        Path("H:/elt_data/runs/grpo_tool_qwen35_hauhaucs_replay/last.pt"),
        Path("H:/elt_data/runs/grpo_math_qwen35_hauhaucs_replay/last.pt"),
        Path("H:/elt_data/runs/grpo_code_qwen35_hauhaucs_replay/last.pt"),
        Path("H:/elt_data/runs/grpo_tool_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/grpo_math_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/grpo_code_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/base_1B_clean_replay_phase2/last.pt"),
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


def stage_native_clean_replay_pretrain(ctx: PipelineContext) -> None:
    initial = first_existing([
        Path("H:/elt_data/runs/grpo_tool_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/grpo_math_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/grpo_code_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/posttrain_tool_sft_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/posttrain_stem_sft_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/posttrain_math_sft_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/posttrain_code_sft_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/posttrain_detection_sft_huihui_qwen36/last.pt"),
        Path("H:/elt_data/runs/base_1B_clean_continue/last.pt"),
    ])
    run_training_config(
        ctx,
        "configs/base_1B_clean_replay_phase2.yaml",
        entrypoint="elt-train",
        initial_resume=initial,
        cleanup_offload_on_success=True,
    )


def stage_prepare_mixed_lane_sft(ctx: PipelineContext) -> None:
    for lane, input_root, output_root, _config_path in MIXED_LANE_PREP:
        info = inspect_distill_bundle(input_root)
        if not (info["train_nonempty"] and info["val_nonempty"]):
            raise PipelineError(f"lane {lane} distill bundle is missing train/val JSONL: {input_root}")
        train_bin = output_root / "bin" / "train.bin"
        val_bin = output_root / "bin" / "val.bin"
        summary = output_root / "prep_summary.json"
        if file_nonempty(train_bin) and file_nonempty(val_bin) and file_nonempty(summary):
            print(f"  skip prepared mixed lane: {lane}")
            continue
        cmd = [
            "uv", "run", "--no-sync", "python", "-m", "elt_lm.prepare_mixed_lane_sft",
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--tokenizer", TOKENIZER,
            "--lane", lane,
        ]
        run_subprocess(cmd, dry_run=ctx.dry_run)


def stage_native_mixed_lane_sft(ctx: PipelineContext) -> None:
    initial = first_existing([
        Path("H:/elt_data/runs/base_1B_clean_replay_phase2/last.pt"),
        Path("H:/elt_data/runs/grpo_tool_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/grpo_math_qwen35_hauhaucs/last.pt"),
        Path("H:/elt_data/runs/grpo_code_qwen35_hauhaucs/last.pt"),
    ])
    for _lane, _input_root, _output_root, config_path in MIXED_LANE_PREP:
        if training_run_complete(config_path):
            print(f"  skip completed training config: {config_path}")
            prune_completed_checkpoints(config_path, dry_run=ctx.dry_run)
            cleanup_completed_offload(config_path, dry_run=ctx.dry_run)
            continue
        run_training_config(
            ctx,
            config_path,
            entrypoint="elt-train",
            initial_resume=initial,
            cleanup_offload_on_success=True,
        )


def stage_native_replay_kl_grpo(ctx: PipelineContext) -> None:
    run_grpo_configs(
        ctx,
        [
            "configs/grpo_code_qwen35_hauhaucs_replay.yaml",
            "configs/grpo_math_qwen35_hauhaucs_replay.yaml",
            "configs/grpo_tool_qwen35_hauhaucs_replay.yaml",
        ],
    )


def stage_side_lora_mixed_sft(ctx: PipelineContext) -> None:
    run_side_lora_sft_configs(ctx, SIDE_LORA_MIXED_SFT_CONFIGS)


FULL_STAGES: list[Stage] = [
    Stage("00_pretrain_clean", stage_pretrain_clean, long_running=True),
    Stage("01_distill_huihui_detection_upload_or_recover", stage_distill_huihui_detection_upload_or_recover),
    Stage("02_prepare_detection_sft", stage_prepare_detection_sft),
    Stage("03_detection_sft", stage_detection_sft, long_running=True),
    Stage("04_hauhaucs_multilane_distill", stage_hauhaucs_multilane_distill, long_running=True),
    Stage("05_lane_sft", stage_lane_sft, long_running=True),
    Stage("05a_stem_sft_val_eval", stage_stem_sft_val_eval),
    Stage("06_kl_grpo", stage_kl_grpo, long_running=True),
    Stage("07_side_lora_sft", stage_side_lora_sft, long_running=True),
    Stage("08_side_lora_ilsd", stage_side_lora_ilsd, long_running=True),
    Stage("09_export_side_lora_adapters", stage_export_side_lora_adapters),
    Stage("10_eval_compare", stage_eval_compare),
]

SIDE_LORA_STAGES: list[Stage] = [
    Stage("00_side_lora_sft", stage_side_lora_sft, long_running=True),
    Stage("01_side_lora_ilsd", stage_side_lora_ilsd, long_running=True),
    Stage("02_export_side_lora_adapters", stage_export_side_lora_adapters),
]

POSTTRAIN_GRPO_STAGES: list[Stage] = [
    Stage("00_prepare_detection_sft", stage_prepare_detection_sft),
    Stage("01_detection_sft", stage_detection_sft, long_running=True),
    Stage("02_prepare_hauhaucs_lanes", stage_prepare_hauhaucs_lanes),
    Stage("03_hauhaucs_lane_sft", stage_hauhaucs_lane_sft_only, long_running=True),
    Stage("03a_stem_sft_val_eval", stage_stem_sft_val_eval),
    Stage("04_kl_grpo", stage_kl_grpo, long_running=True),
    Stage("05_side_lora_ilsd", stage_side_lora_ilsd, long_running=True),
    Stage("06_export_side_lora_adapters", stage_export_side_lora_adapters),
    Stage("07_eval_compare", stage_eval_compare),
]

REPLAY_REFRESH_STAGES: list[Stage] = [
    Stage("00_native_clean_replay_pretrain", stage_native_clean_replay_pretrain, long_running=True),
    Stage("01_prepare_mixed_lane_sft", stage_prepare_mixed_lane_sft),
    Stage("02_native_mixed_lane_sft", stage_native_mixed_lane_sft, long_running=True),
    Stage("03_native_kl_grpo", stage_native_replay_kl_grpo, long_running=True),
    Stage("04_side_lora_mixed_sft", stage_side_lora_mixed_sft, long_running=True),
    Stage("05_eval_compare", stage_eval_compare),
]

STAGE_PROFILES: dict[str, list[Stage]] = {
    "full": FULL_STAGES,
    "posttrain-grpo": POSTTRAIN_GRPO_STAGES,
    "replay-refresh": REPLAY_REFRESH_STAGES,
    "side-lora": SIDE_LORA_STAGES,
}

# Backward-compatible public name used by tests and old scripts.
STAGES = FULL_STAGES


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
    parser.add_argument("--profile", default=os.environ.get("ELT_PIPELINE_PROFILE", "full"),
                        choices=sorted(STAGE_PROFILES),
                        help="stage profile to run")
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
    stages = select_stages(STAGE_PROFILES[args.profile], only=args.only, skip=args.skip)
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
