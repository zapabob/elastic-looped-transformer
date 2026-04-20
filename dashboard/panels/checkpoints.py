"""Checkpoints panel — rolling slot age + milestone list for the selected run."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st


def render(run_dir: Path) -> None:
    st.subheader("Checkpoints")
    if not run_dir.exists():
        st.caption("run directory not present yet")
        return

    rolls = sorted(run_dir.glob("rolling_*.pt"))
    milestones = sorted(run_dir.glob("step_*.pt"))
    last = run_dir / "last.pt"

    now = time.time()
    cols = st.columns(max(1, len(rolls)))
    for col, r in zip(cols, rolls):
        age = now - r.stat().st_mtime
        size_mb = r.stat().st_size / 1e6
        col.metric(r.name, f"{size_mb:.0f} MB",
                   f"{age/60:.1f} min ago")

    if last.exists():
        st.caption(f"last.pt → {last.resolve().name}  "
                   f"({(now - last.stat().st_mtime)/60:.1f} min old, "
                   f"{last.stat().st_size/1e6:.0f} MB)")
    else:
        st.caption("no last.pt yet")

    if milestones:
        with st.expander(f"{len(milestones)} milestone saves", expanded=False):
            st.dataframe(
                [{"name": m.name,
                  "size_mb": m.stat().st_size / 1e6,
                  "age_min": (now - m.stat().st_mtime) / 60}
                 for m in milestones[-20:]],
                use_container_width=True,
            )
