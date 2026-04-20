"""Pipeline panel — renders stage status from pipeline.jsonl + .done markers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.utils.metrics_reader import filter_events, read_jsonl, read_log_tail


STAGE_ORDER = [
    "00_download", "01_ingest", "02_clean", "03_tokenize", "04_smoke",
    "05_pretrain", "06_distill", "07_sft", "08_grpo", "09_eval", "10_export_hf",
]


def render(state_dir: Path, pipeline_log_dir: Path | None = None) -> None:
    st.subheader("Pipeline progress")

    telemetry_path = state_dir / "pipeline.jsonl"
    events = read_jsonl(telemetry_path, last_n=1000)
    stage_events = filter_events(events, "pipeline_stage")

    latest_status: dict[str, dict] = {}
    for e in stage_events:
        latest_status[e["name"]] = e

    # Fallback to .done markers for first-boot / no-telemetry runs
    for stage in STAGE_ORDER:
        if stage not in latest_status and (state_dir / f"{stage}.done").exists():
            latest_status[stage] = {"name": stage, "status": "done"}

    cols = st.columns(min(6, len(STAGE_ORDER)))
    glyph = {"done": "[OK]", "start": "[..]", "skipped": "[--]",
             "aborted": "[X] ", "crashed": "[!!]"}
    for i, stage in enumerate(STAGE_ORDER):
        info = latest_status.get(stage)
        status = info["status"] if info else "pending"
        g = glyph.get(status, "[  ]")
        elapsed = info.get("elapsed_sec") if info else None
        label = f"{g} {stage}"
        col = cols[i % len(cols)]
        with col:
            st.markdown(f"**{label}**")
            if elapsed is not None:
                st.caption(f"{elapsed/60:.1f} min" if elapsed >= 60 else f"{elapsed:.1f} s")
            else:
                st.caption(status)

    with st.expander("Recent stage events", expanded=False):
        if stage_events:
            st.dataframe(
                [{"name": e["name"], "status": e["status"],
                  "elapsed_sec": e.get("elapsed_sec", "")} for e in stage_events[-20:]],
                use_container_width=True,
            )
        else:
            st.caption("no pipeline.jsonl events yet — pipeline has not run since telemetry was added")

    if pipeline_log_dir and pipeline_log_dir.exists():
        logs = sorted(pipeline_log_dir.glob("pipeline-*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            with st.expander(f"Latest log tail ({logs[0].name})", expanded=False):
                tail = read_log_tail(logs[0], n_lines=80)
                st.code("\n".join(tail) or "(empty)", language="text")
