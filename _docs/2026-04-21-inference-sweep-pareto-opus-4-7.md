# 2026-04-21 — Phase D: Inference Any-Time sweep + Pareto panel

**Model:** opus-4-7
**Branch of work:** ELT 1B + Hypura-style offload plan, Phase D (final phase).

## Goal

Close the plan's observability triangle — pipeline (A), training (A/C),
inference (D) — by making `elt-anytime` emit structured `inference_sweep`
events per loop count L, and rendering the resulting quality × latency ×
throughput Pareto in the Streamlit dashboard. This visualizes ELT's headline
Any-Time capability: quality degrades gracefully as L shrinks.

## Files

### Edited
- `src/elt_lm/eval/anytime_sweep.py`
  - `eval_at_L` now measures wall-clock (with CUDA sync bracketing) and
    returns a dict with `nll`, `ppl`, `tokens_per_sec`, `latency_ms_per_batch`,
    `total_tokens`, `batches` (was just `(nll, ppl)`).
  - `run()` opens a `TelemetryWriter` against `--run-dir` (defaulting to
    the checkpoint's parent directory) and emits one `inference_sweep`
    event per L with every metric above + `rel_flops` (= L × N).
  - CSV output now includes `tokens_per_sec` and `latency_ms` columns.
  - New `--run-dir` CLI flag.
- `dashboard/app.py` — imports and renders the new Inference panel between
  Training and Storage-tiers.

### New
- `dashboard/panels/inference.py` — "Inference — Any-Time Pareto" panel.
  Reads `inference_sweep` events, keeps the most recent one per L
  (`_latest_per_L`), shows headline metrics (best PPL, peak tok/s, min
  latency), and draws three tabs: Quality vs L, Latency vs L, Throughput
  vs L, plus a raw Table view.
- `tests/test_inference_sweep.py` — 3 tests:
  - `_latest_per_L` keeps the most recent event per L and returns sorted output.
  - Integration: `TelemetryWriter.emit(...)` → `read_jsonl` →
    `filter_events` → `_latest_per_L` round-trips cleanly.
  - Sweep events expose all numeric fields the dashboard reads.

## Event schema (canonical for `inference_sweep`)

```json
{
  "ts": 1745197200.0,
  "event": "inference_sweep",
  "L": 2,
  "rel_flops": 56,
  "nll": 1.502,
  "ppl": 4.495,
  "tokens_per_sec": 1942.3,
  "latency_ms": 12.7,
  "total_tokens": 4096,
  "batches": 8,
  "ckpt": "H:/elt_data/runs/base_1B/last.pt"
}
```

Field contract matches `dashboard/panels/inference.py` reader and the test
at `tests/test_inference_sweep.py::test_sweep_event_field_types`.

## Key decisions

1. **Reuse existing `elt-anytime` rather than a new `scripts/anytime_sweep.py`.**
   The plan named the latter but `src/elt_lm/eval/anytime_sweep.py` already
   existed with a working `elt-anytime` entry point. Extending it keeps a
   single code path and honors the existing CLI contract. (The plan's
   `scripts/anytime_sweep.py` is effectively implemented as the `elt-anytime`
   script entry.)
2. **Run-dir defaulting from `--ckpt`.** Users typically pass
   `--ckpt runs/foo/last.pt`; inferring `--run-dir = runs/foo` makes the
   dashboard pick up sweep results without any extra flag.
3. **`_latest_per_L`.** Keep the newest event per L so repeated sweeps on
   the same checkpoint overwrite rather than concatenate in the dashboard.
4. **No new heavy plotting dep.** Used `st.line_chart` over Plotly — keeps
   cold-start fast; Plotly is still available for future richer views.

## Tests

All passing: **99 / 99** (96 prior + 3 new).

## Status of the full plan

| Phase | Scope | Status |
|---|---|---|
| A | Telemetry + Streamlit dashboard | **done** — 73 tests, committed |
| B | base_1B.yaml + PagedAdamW8bit | **done** — 77 tests, committed |
| C | Hypura 4-tier NVMe offload primitives | **done** — 96 tests, committed |
| D | Inference Pareto panel | **done** — 99 tests, this commit |

Remaining (not part of the approved plan's 4 phases, but noted in the
Phase-C follow-ups):
- Hot-path integration of the offload package into `CompositeBlock.forward`
  (promote/demote hooks). The primitives are built and tested; wiring is
  the next concrete coding task.
- 1B smoke run: `elt-train --config configs/base_1B.yaml` with PagedAdamW8bit
  first (Phase B path), then once offload hooks land, NvmeAdamW.

## Pending tasks

- (Phase C follow-up) `offload/hooks.py` + CompositeBlock integration
- 1B end-to-end smoke: PagedAdamW8bit VRAM check → NvmeAdamW VRAM check
