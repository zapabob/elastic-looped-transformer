"""Inference panel: Any-Time Pareto curves and loop refinement metrics.

Reads the latest sweep telemetry from ``metrics.jsonl`` and plots:

- L vs perplexity (quality)
- L vs latency_ms (speed)
- L vs tokens_per_sec (throughput)
- benchmark self-correction / overthinking rates across L
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.utils.metrics_reader import filter_events, read_jsonl


def _latest_per_L(events: list[dict]) -> list[dict]:
    """Keep only the most recent event per L (in file order)."""
    latest: dict[int, dict] = {}
    for e in events:
        L = int(e.get("L", -1))
        if L >= 0:
            latest[L] = e
    return [latest[L] for L in sorted(latest.keys())]


def _latest_benchmark_per_key(events: list[dict]) -> list[dict]:
    """Keep the most recent benchmark event per (benchmark, L)."""
    latest: dict[tuple[str, int], dict] = {}
    for e in events:
        benchmark = str(e.get("benchmark", ""))
        L = int(e.get("L", -1))
        if benchmark and L >= 0:
            latest[(benchmark, L)] = e
    return [latest[k] for k in sorted(latest.keys())]


def render(run_dir: Path) -> None:
    st.subheader("Inference: Any-Time Pareto")
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        st.caption("no metrics yet")
        return

    events = read_jsonl(metrics, last_n=10_000)
    sweep = _latest_per_L(filter_events(events, "inference_sweep"))
    benchmarks = _latest_benchmark_per_key(filter_events(events, "benchmark_eval"))

    if not sweep and not benchmarks:
        st.caption(
            "no inference_sweep or benchmark_eval events yet. Run: "
            "`uv run elt-anytime --ckpt <run>/last.pt --val-bin H:/elt_data/bin/val.bin`"
        )
        return

    import pandas as pd

    if sweep:
        Ls = [int(e["L"]) for e in sweep]
        ppls = [float(e.get("ppl", 0.0)) for e in sweep]
        tps = [float(e.get("tokens_per_sec", 0.0)) for e in sweep]
        latencies = [float(e.get("latency_ms", 0.0)) for e in sweep]
        rel_flops = [float(e.get("rel_flops", 0.0)) for e in sweep]

        col_q, col_t, col_l = st.columns(3)
        col_q.metric("best PPL", f"{min(ppls):.3f}" if ppls else "-")
        col_t.metric("peak tok/s", f"{max(tps):.0f}" if tps else "-")
        col_l.metric("min latency", f"{min(latencies):.1f} ms" if latencies else "-")

        tab_quality, tab_latency, tab_throughput, tab_table = st.tabs(
            ["Quality vs L", "Latency vs L", "Throughput vs L", "Table"]
        )

        df = pd.DataFrame({
            "L": Ls,
            "ppl": ppls,
            "tokens_per_sec": tps,
            "latency_ms": latencies,
            "rel_flops": rel_flops,
        }).set_index("L").sort_index()

        with tab_quality:
            st.line_chart(df[["ppl"]], height=280)
            st.caption("perplexity at L in [L_min, L_max]; monotone drop is ideal Any-Time behavior")

        with tab_latency:
            st.line_chart(df[["latency_ms"]], height=280)
            st.caption("wall-clock per-batch latency; usually grows roughly linearly in L")

        with tab_throughput:
            st.line_chart(df[["tokens_per_sec"]], height=280)
            st.caption("tokens / second (higher is better)")

        with tab_table:
            st.dataframe(df, use_container_width=True)

    if benchmarks:
        st.subheader("Loop refinement benchmarks")
        bench_df = pd.DataFrame([{
            "benchmark": e.get("benchmark", ""),
            "task": e.get("task", ""),
            "L": int(e.get("L", -1)),
            "score": float(e.get("score", 0.0)),
            "loop_gain": e.get("loop_gain"),
            "marginal_gain": e.get("marginal_gain"),
            "self_correction_rate": e.get("self_correction_rate"),
            "overthinking_rate": e.get("overthinking_rate"),
            "latency_ms": float(e.get("latency_ms", 0.0)),
            "tokens_per_sec": float(e.get("tokens_per_sec", 0.0)),
        } for e in benchmarks]).sort_values(["benchmark", "L"])
        st.dataframe(bench_df, use_container_width=True)
        st.caption(
            "self_correction = L=1 wrong -> L=k correct; "
            "overthinking = L=1 correct -> L=k wrong."
        )
