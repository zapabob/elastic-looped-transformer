"""ELT-LM dashboard — pipeline, training, hardware, checkpoints.

Run from the project root with:

    uv run streamlit run dashboard/app.py

Panels read local state — no network calls, no IPC:
- Pipeline progress : H:/elt_data/pipeline_state/*.done + pipeline.jsonl
- Training curves   : runs/<exp>/metrics.jsonl (JSONL tail)
- Hardware          : NVML + psutil polled per rerun
- Checkpoints       : runs/<exp>/{rolling_*.pt, step_*.pt, last.pt}

Phases C and D add:
- Storage tiers     : tier_read + layer_computed events
- Inference Pareto  : inference_sweep events
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.panels import checkpoints as p_ckpt  # noqa: E402
from dashboard.panels import hardware as p_hw  # noqa: E402
from dashboard.panels import pipeline as p_pipeline  # noqa: E402
from dashboard.panels import tiers as p_tiers  # noqa: E402
from dashboard.panels import training as p_training  # noqa: E402
from dashboard.utils.metrics_reader import discover_runs  # noqa: E402


STATE_DIR = Path("H:/elt_data/pipeline_state")
PIPELINE_LOG_DIR = Path("H:/elt_data/pipeline_logs")
DEFAULT_DISK_PATHS = [Path("C:/"), Path("H:/")]


def main() -> None:
    st.set_page_config(
        page_title="ELT-LM",
        page_icon="EL",
        layout="wide",
    )
    st.title("ELT-LM — training & pipeline monitor")

    with st.sidebar:
        st.markdown("### Run selector")
        all_runs: list[Path] = []
        for base in (ROOT / "runs", Path("H:/elt_data/runs")):
            if base.exists():
                all_runs.extend(discover_runs(base))

        if not all_runs:
            st.info("No runs with metrics.jsonl found yet.")
            selected_run: Path | None = None
        else:
            labels = [f"{r.parent.name}/{r.name}" for r in all_runs]
            idx = st.selectbox("run", range(len(all_runs)),
                               format_func=lambda i: labels[i])
            selected_run = all_runs[idx]

        st.markdown("### Refresh")
        auto = st.checkbox("Auto-refresh every 5 s", value=True)
        if st.button("Refresh now"):
            st.rerun()

    p_pipeline.render(STATE_DIR, pipeline_log_dir=PIPELINE_LOG_DIR)
    st.divider()

    if selected_run is not None:
        p_training.render(selected_run)
        st.divider()
        p_tiers.render(selected_run)
        st.divider()
        p_ckpt.render(selected_run)
        st.divider()

    p_hw.render(DEFAULT_DISK_PATHS)

    if auto:
        # Minimal sleep loop — Streamlit reruns on any interaction anyway.
        import time
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
