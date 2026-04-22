"""CLI monitor for GGUF distillation progress."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.panels.gguf_distill import classify_heartbeat_health, discover_distill_runs
from dashboard.utils.metrics_reader import read_json_file, read_log_tail


def build_monitor_report(run_dir: Path) -> dict[str, Any]:
    status = read_json_file(run_dir / "status.json") or {}
    heartbeat = read_json_file(run_dir / "heartbeat.json") or status
    summary = read_json_file(run_dir / "eval_summary.json") or {}
    plan = read_json_file(run_dir / "pipeline_plan.json") or {}
    pipeline_raw = plan.get("pipeline")
    pipeline_cfg = pipeline_raw if isinstance(pipeline_raw, dict) else {}
    stall_after_sec = float(pipeline_cfg.get("stall_after_sec", 1800) or 1800)
    now_ts = time.time()
    health = classify_heartbeat_health(heartbeat, now_ts=now_ts, stall_after_sec=stall_after_sec)
    return {
        "run_dir": str(run_dir),
        "state": status.get("state", "unknown"),
        "current_stage": status.get("current_stage", "-"),
        "health": health,
        "processed_tasks": int(status.get("processed_tasks", 0)),
        "total_tasks": int(status.get("total_tasks", 0)),
        "progress_pct": float(status.get("progress_pct", 0.0)),
        "eta_sec": status.get("eta_sec"),
        "error_count": int(status.get("error_count", 0)),
        "last_error": status.get("last_error", ""),
        "updated_at": status.get("updated_at"),
        "age_sec": max(0.0, now_ts - float(status.get("updated_at", now_ts))),
        "summary_records": int(summary.get("total_records", 0)),
        "schema_valid_rate": float(summary.get("schema_valid_rate", 0.0)),
        "stall_after_sec": stall_after_sec,
        "log_tail": read_log_tail(run_dir / "llama_server.log", n_lines=20),
    }


def _select_run(root: Path, run_name: str) -> Path:
    if run_name:
        run_dir = root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        return run_dir
    runs = discover_distill_runs(root)
    if not runs:
        raise FileNotFoundError(f"no GGUF distill runs under {root}")
    return runs[0]


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="H:/elt_data/gguf_distill")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail-on-stall", action="store_true")
    args = parser.parse_args()

    run_dir = _select_run(Path(args.root), args.run_name)
    report = build_monitor_report(run_dir)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"{report['run_dir']} | state={report['state']} | stage={report['current_stage']} "
            f"| health={report['health']} | progress={report['processed_tasks']}/{report['total_tasks']} "
            f"({report['progress_pct']:.1f}%) | errors={report['error_count']}"
        )
        if report["eta_sec"] is not None:
            print(f"eta_sec={report['eta_sec']}")
        if report["last_error"]:
            print(f"last_error={report['last_error']}")
        if report["log_tail"]:
            print("--- log tail ---")
            print("\n".join(report["log_tail"]))

    if args.fail_on_stall and report["health"] in {"stalled", "failed"}:
        raise SystemExit(2)


if __name__ == "__main__":
    cli()
