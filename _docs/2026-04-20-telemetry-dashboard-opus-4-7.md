# 2026-04-20 — Phase A: Telemetry + Streamlit dashboard

**Model:** opus-4-7
**Branch of work:** ELT 1B + Hypura-style offload plan, Phase A (of A/B/C/D).

## Goal

Add a JSONL telemetry writer and a Streamlit dashboard so that the pipeline,
training loops (SFT/ILSD and GRPO), and upcoming inference sweeps all stream
structured events to one place that a human can watch live. This is the
observability substrate on top of which Phase B (1B config + PagedAdamW8bit)
and Phase C (Hypura-style 4-tier NVMe offload) will be built.

## Files

### New
- `src/elt_lm/telemetry.py` — `TelemetryWriter` (JSONL, thread-safe, PID-aware
  re-open for fork safety), `NullTelemetry` no-op, `make_writer(run_dir, enabled)`
  factory. Serializes torch/numpy scalars via a `default=_json_default` that
  calls `.item()`.
- `dashboard/__init__.py`, `dashboard/utils/__init__.py`,
  `dashboard/panels/__init__.py` — package markers.
- `dashboard/utils/metrics_reader.py` — `read_jsonl(path, last_n=...)` with
  partial-line skip, `filter_events(events, kind)`, `discover_runs(runs_dir)`,
  `read_log_tail(path, n_lines=...)`.
- `dashboard/panels/pipeline.py` — `STAGE_ORDER = [00_download..10_export_hf]`.
  Reads `H:/elt_data/pipeline_state/*.done` and `pipeline.jsonl`. Stage
  glyphs + elapsed times + rolling log tail.
- `dashboard/panels/training.py` — 5 tabs: Loss, Learning rate, tok/sec,
  ILSD (λ + L_int histogram), GRPO (reward stats, clip frac, KL).
- `dashboard/panels/hardware.py` — NVML (via `pynvml`-aliased `nvidia-ml-py`)
  + `psutil`: VRAM used/total, GPU util %, CPU %, RAM, free space per disk.
- `dashboard/panels/checkpoints.py` — rolling slot age + milestone list
  + `last.pt` info for the selected run.
- `dashboard/app.py` — Streamlit entrypoint. Sidebar run selector pulls from
  `./runs` and `H:/elt_data/runs`. Auto-refresh loop.
- `tests/test_telemetry.py` — 6 tests: emit writes JSONL; every line has
  ts+event; thread-safe with 4 workers × 200 emits; numpy/torch scalars
  serialize; `NullTelemetry` no-op; `make_writer` factory.
- `tests/test_dashboard_reader.py` — 6 tests: partial-line skip, `last_n`
  truncate, filter by kind, `discover_runs` only returns dirs with
  `metrics.jsonl`, `read_log_tail` tail+missing.

### Edited
- `pyproject.toml` — added `dashboard` optional-deps group:
  `streamlit>=1.32.0`, `plotly>=5.18.0`, `nvidia-ml-py>=12.535.0`,
  `psutil>=5.9.0`. (Switched from deprecated `pynvml` after FutureWarning.)
- `src/elt_lm/train.py` — instantiate `TelemetryWriter` after
  `RollingCheckpointer`; emit `train_config`, `train_step` (step/lr/loss/
  L_int/lambda/tokens_per_sec/l_gt_teacher/l_gt_student/l_dist),
  `checkpoint` (milestone + rolling). Close writer at end.
- `src/elt_lm/train_grpo.py` — same shape, emits `grpo_config`,
  `grpo_step` (loss/policy_loss/kl/clip_frac/adv_abs_mean/reward stats/
  correct_rate/format_rate), `checkpoint`.
- `scripts/pipeline.py` — write to `H:/elt_data/pipeline_state/pipeline.jsonl`:
  `pipeline_start`, `pipeline_stage` with
  `status={start,done,skipped,aborted,crashed}` + `elapsed_sec`.

## Key decisions

1. **Per-run `metrics.jsonl`, global `pipeline.jsonl`.** Training events live
   next to each run's checkpoints so `discover_runs` can find them; pipeline
   stage events are global and live alongside the `*.done` sentinels.
2. **Line-buffered append writes, single lock.** Each `emit()` serializes
   under `threading.Lock()` and appends one line. Safe for the train loop
   plus background prefetcher + eval threads that Phase C will add.
3. **PID-aware file handle.** If `emit()` is called after a fork, the writer
   re-opens `self._fh` on the new PID so worker processes don't collide.
4. **Torch/numpy scalar serialization.** `default=_json_default` in
   `json.dumps` tries `.item()` on non-JSON values so train code can pass
   tensors directly without conversion.
5. **`nvidia-ml-py` (not `pynvml`).** `pynvml` is formally deprecated; the
   replacement wheel installs as the same import name so `hardware.py` code
   stays unchanged.
6. **`NullTelemetry` default off switch.** `make_writer(None)` returns a
   no-op, so tests and one-shot CLIs don't need to care about a run dir.

## Tests

All passing (73 tests, 0 failures).

- 6 new telemetry tests, 6 new dashboard-reader tests (12 total new).
- 61 prior tests unchanged.

Run: `uv run -- python -m pytest -q` → `73 passed`.

## Notes for next session (Phase B)

- `configs/base_1B.yaml` needs `d_model=1792, N=28, n_heads=28, head_dim=64,
  d_ff=4864, tie=True`. Expected ~1.54B total, ~1.09B non-emb.
- Add `bitsandbytes>=0.43` to a new `offload_8bit` extras group.
- `TrainConfig` needs a new field `optim: "adamw"|"paged_adamw_8bit"` or an
  `OptimConfig` sub-block; Phase C will add `"nvme_adamw"` as a third option.
- Dashboard already handles 1B runs — no dashboard changes needed in Phase B.

## Pending tasks

- #40 Phase B: `configs/base_1B.yaml` + PagedAdamW8bit wiring.
- #41 Phase C: `src/elt_lm/offload/` package (tiered_store / prefetcher /
  placement / optim_offload / hooks / hardware_profile) + tier_read and
  layer_computed telemetry events + a new dashboard "Storage tiers" panel.
- #42 Phase D: `scripts/anytime_sweep.py` emits `inference_sweep` events +
  a new dashboard "Inference" Pareto panel.
