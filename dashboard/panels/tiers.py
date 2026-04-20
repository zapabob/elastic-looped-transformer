"""Storage-tiers panel — reads `tier_read` and `layer_computed` events.

Shown only when the selected run has events matching these types, i.e. when
Phase-C offload was active. Summarizes:

  - per-tier read bytes and MB/s
  - pinned-pool hit rate
  - per-layer compute time distribution (ms) broken out by tier
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.utils.metrics_reader import filter_events, read_jsonl


def render(run_dir: Path) -> None:
    st.subheader("Storage tiers")
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        st.caption("no metrics yet")
        return

    events = read_jsonl(metrics, last_n=20_000)
    tier_reads = filter_events(events, "tier_read")
    layer_events = filter_events(events, "layer_computed")
    status_events = filter_events(events, "prefetch_status")

    if not (tier_reads or layer_events or status_events):
        st.caption("no tier events in this run (offload not active)")
        return

    col_a, col_b, col_c = st.columns(3)
    total_bytes = sum(int(e.get("bytes", 0)) for e in tier_reads)
    total_us = sum(float(e.get("latency_us", 0.0)) for e in tier_reads)
    avg_mbps = (total_bytes / max(1.0, total_us / 1e6)) / 1e6 if total_us else 0.0
    col_a.metric("tier reads", f"{len(tier_reads):,}")
    col_b.metric("bytes read", f"{total_bytes/1e9:.2f} GB")
    col_c.metric("avg NVMe MB/s", f"{avg_mbps:.0f}")

    if status_events:
        latest = status_events[-1]
        st.caption(
            f"pinned hit-rate: {latest.get('pinned_hit_rate', 0.0):.2%} · "
            f"last nvme MB/s: {latest.get('avg_mbps', 0.0):.0f}"
        )

    if layer_events:
        with st.expander("per-layer compute times (last 200)", expanded=False):
            tail = layer_events[-200:]
            st.dataframe(
                [{"layer": e.get("layer_idx"),
                  "tier": e.get("tier"),
                  "duration_ms": float(e.get("duration_us", 0)) / 1000.0}
                 for e in tail],
                use_container_width=True,
            )
