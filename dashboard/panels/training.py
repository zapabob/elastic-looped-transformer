"""Training panel — loss / lr / tok-per-sec / ILSD λ + L_int distribution."""

from __future__ import annotations

from pathlib import Path
from collections import Counter

import streamlit as st

from dashboard.utils.metrics_reader import filter_events, read_jsonl


def render(run_dir: Path) -> None:
    st.subheader(f"Training — {run_dir.name}")

    path = run_dir / "metrics.jsonl"
    events = read_jsonl(path, last_n=5000)
    if not events:
        st.info(f"No metrics.jsonl at {path}")
        return

    cfg = next((e for e in events if e.get("event") == "train_config"), None)
    if cfg:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("non-emb params",
                  f"{cfg.get('model_params_non_embedding', 0)/1e6:.1f} M")
        c2.metric("d_model", cfg.get("d_model", "?"))
        c3.metric("N unique layers", cfg.get("n_unique_layers", "?"))
        c4.metric("L_max", cfg.get("L_max", "?"))

    steps = filter_events(events, "train_step")
    grpo_steps = filter_events(events, "grpo_step")
    steps = steps or grpo_steps
    if not steps:
        st.caption("no train_step events yet")
        return

    xs = [e["step"] for e in steps]
    loss = [e.get("loss", 0.0) for e in steps]
    lr = [e.get("lr", 0.0) for e in steps]
    tok_s = [e.get("tokens_per_sec", 0.0) for e in steps if "tokens_per_sec" in e]

    tabs = st.tabs(["Loss", "Learning rate", "tok/sec", "ILSD λ + L_int", "GRPO"])

    with tabs[0]:
        st.line_chart({"loss": loss}, x_label="step")
        if grpo_steps:
            st.line_chart({"policy_loss": [e["policy_loss"] for e in grpo_steps],
                           "kl": [e["kl"] for e in grpo_steps]}, x_label="step")

    with tabs[1]:
        st.line_chart({"lr": lr}, x_label="step")

    with tabs[2]:
        if tok_s:
            st.line_chart({"tokens_per_sec": tok_s}, x_label="step")
        else:
            st.caption("no tok/sec samples (GRPO run or no samples yet)")

    with tabs[3]:
        ilsd = filter_events(events, "train_step")
        if ilsd:
            lam = [e.get("lambda_value", 0.0) for e in ilsd]
            st.line_chart({"λ": lam}, x_label="step")
            lint = [e.get("L_int", 0) for e in ilsd if "L_int" in e]
            if lint:
                hist = Counter(lint)
                st.bar_chart({str(k): [v] for k, v in sorted(hist.items())})
        else:
            st.caption("no ILSD events in this run")

    with tabs[4]:
        if grpo_steps:
            rew = [e.get("reward_mean", 0.0) for e in grpo_steps]
            corr = [e.get("correct_rate", 0.0) for e in grpo_steps]
            fmt = [e.get("format_rate", 0.0) for e in grpo_steps]
            kl = [e.get("kl", 0.0) for e in grpo_steps]
            clip = [e.get("clip_frac", 0.0) for e in grpo_steps]
            st.line_chart({"reward_mean": rew}, x_label="step")
            st.line_chart({"correct_rate": corr, "format_rate": fmt}, x_label="step")
            st.line_chart({"kl": kl, "clip_frac": clip}, x_label="step")
        else:
            st.caption("no grpo_step events in this run")
