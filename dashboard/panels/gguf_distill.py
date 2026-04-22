"""GGUF distillation progress panel."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.utils.metrics_reader import read_json_file, read_jsonl, read_log_tail


def discover_distill_runs(root: str | Path) -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    out: list[Path] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if any((child / name).exists() for name in ("status.json", "metrics.jsonl", "pipeline_plan.json")):
            out.append(child)
    out.sort(key=_run_sort_key, reverse=True)
    return out


def classify_heartbeat_health(
    heartbeat: dict[str, Any] | None,
    *,
    now_ts: float,
    stall_after_sec: float,
) -> str:
    data = heartbeat or {}
    state = str(data.get("state", "unknown")).lower()
    if state in {"complete", "completed", "done"}:
        return "complete"
    if state in {"failed", "error", "aborted", "crashed"}:
        return "failed"
    updated_at = data.get("updated_at")
    if state in {"starting", "running"} and isinstance(updated_at, (int, float)):
        if now_ts - float(updated_at) > stall_after_sec:
            return "stalled"
        return "healthy"
    return state or "unknown"


def _run_mtime(path: Path) -> float:
    candidates = [path / name for name in ("status.json", "metrics.jsonl", "pipeline_plan.json", "eval_summary.json")]
    mtimes = [candidate.stat().st_mtime for candidate in candidates if candidate.exists()]
    return max(mtimes) if mtimes else path.stat().st_mtime


def _run_sort_key(path: Path) -> tuple[float, int, int, str]:
    return (
        _run_mtime(path),
        int((path / "metrics.jsonl").exists()),
        int((path / "status.json").exists()),
        path.name,
    )


def _format_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    total = int(round(float(value)))
    if total < 60:
        return f"{total}s"
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _render_bar_chart(title: str, values: dict[str, int]) -> None:
    st.markdown(f"**{title}**")
    if not values:
        st.caption("no data yet")
        return
    df = pd.DataFrame({"name": list(values.keys()), "count": list(values.values())}).set_index("name")
    st.bar_chart(df, height=220)


def render(distill_root: Path) -> None:
    st.subheader("GGUF Distill")

    runs = discover_distill_runs(distill_root)
    if not runs:
        st.info(f"No GGUF distill runs found under {distill_root}")
        return

    labels = [run.name for run in runs]
    selected_idx = st.selectbox(
        "distill run",
        range(len(runs)),
        format_func=lambda idx: labels[idx],
        key="gguf_distill_run_selector",
    )
    run_dir = runs[selected_idx]

    status = read_json_file(run_dir / "status.json") or {}
    heartbeat = read_json_file(run_dir / "heartbeat.json") or status
    summary = read_json_file(run_dir / "eval_summary.json") or {}
    plan = read_json_file(run_dir / "pipeline_plan.json") or {}
    metrics = read_jsonl(run_dir / "metrics.jsonl", last_n=10_000)
    item_events = [event for event in metrics if event.get("event") == "gguf_distill_item"]
    stage_events = [event for event in metrics if event.get("event") == "gguf_distill_stage"]
    stall_after_sec = float(((plan.get("pipeline") or {}).get("stall_after_sec")) or 1800.0)
    health = classify_heartbeat_health(heartbeat, now_ts=time.time(), stall_after_sec=stall_after_sec)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("state", str(status.get("state", "unknown")))
    c2.metric("stage", str(status.get("current_stage", "-")))
    c3.metric("progress", f"{float(status.get('progress_pct', 0.0)):.1f}%")
    c4.metric("processed", f"{int(status.get('processed_tasks', 0))}/{int(status.get('total_tasks', 0))}")
    c5.metric("heartbeat", health)

    eta_col, latency_col, error_col, updated_col = st.columns(4)
    eta_col.metric("ETA", _format_seconds(status.get("eta_sec")))
    latency_col.metric("last latency", _format_seconds(status.get("last_latency_sec")))
    error_col.metric("errors", int(status.get("error_count", 0)))
    updated_col.metric("updated", _format_seconds(time.time() - float(status.get("updated_at", time.time()))))

    total_tasks = int(status.get("total_tasks", 0))
    progress_pct = float(status.get("progress_pct", 0.0))
    if total_tasks > 0:
        st.progress(min(max(progress_pct / 100.0, 0.0), 1.0), text=f"{progress_pct:.1f}% complete")

    with st.expander("Latest status snapshot", expanded=False):
        st.json(status or {"status": "missing"})

    if item_events:
        df = pd.DataFrame(
            {
                "step": [int(event.get("index", 0)) for event in item_events],
                "latency_sec": [float(event.get("latency_sec", 0.0)) for event in item_events],
                "progress_pct": [float(event.get("progress_pct", 0.0)) for event in item_events],
            }
        ).set_index("step")
        tab_latency, tab_progress, tab_stages = st.tabs(["Latency", "Progress", "Stages"])
        with tab_latency:
            st.line_chart(df[["latency_sec"]], height=260)
        with tab_progress:
            st.line_chart(df[["progress_pct"]], height=260)
        with tab_stages:
            if stage_events:
                st.dataframe(
                    [
                        {
                            "ts": event.get("ts"),
                            "stage": event.get("stage"),
                            "status": event.get("status"),
                            "error": event.get("error", ""),
                        }
                        for event in stage_events[-20:]
                    ],
                    use_container_width=True,
                )
            else:
                st.caption("no stage events yet")
    else:
        st.caption("no gguf_distill_item telemetry yet")

    chart_a, chart_b = st.columns(2)
    with chart_a:
        _render_bar_chart("Domain counts", dict(summary.get("domain_counts", {}) or status.get("domain_counts", {})))
    with chart_b:
        _render_bar_chart("Policy labels", dict(summary.get("label_counts", {}) or status.get("label_counts", {})))

    if summary:
        st.markdown("**Eval summary**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("records", int(summary.get("total_records", 0)))
        s2.metric("schema-valid", f"{float(summary.get('schema_valid_rate', 0.0)):.3f}")
        s3.metric("duplicates", int(summary.get("duplicate_prompt_count", 0)))
        s4.metric("train/val", f"{int(status.get('train_records', 0))}/{int(status.get('val_records', 0))}")
        with st.expander("eval_summary.json", expanded=False):
            st.json(summary)

    with st.expander("llama-server log tail", expanded=False):
        tail = read_log_tail(run_dir / "llama_server.log", n_lines=80)
        st.code("\n".join(tail) or "(empty)", language="text")
