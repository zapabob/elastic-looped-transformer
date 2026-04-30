# 2026-05-01 loop refinement eval + README notes - gpt-5

## Goal

Document the ELT/ILSD stability story in both the GitHub README and the
Hugging Face export model-card template, then make the implementation report
the loop-wise metrics described there.

## Files touched

- `README.md`
- `src/elt_lm/hf/model_card_template.md`
- `src/elt_lm/eval/benchmarks.py`
- `src/elt_lm/eval/anytime_sweep.py`
- `dashboard/panels/inference.py`
- `tests/test_benchmarks.py`

## Key decisions

- The benchmark runner now preserves per-case correctness in `BenchmarkResult`.
- `elt-anytime` compares each benchmark result to `L=L_min` and emits:
  - `loop_gain`
  - `marginal_gain`
  - `self_correction_rate`
  - `overthinking_rate`
- The dashboard keeps the original perplexity Pareto table and adds a benchmark
  refinement table when `benchmark_eval` events are available.
- README/model-card wording now explicitly describes:
  - stop-gradient teacher targets
  - teacher-only temperature
  - entropy floor
  - Delta^2 entropy/logit curvature
  - self-correction versus overthinking as the core test-time scaling question

## Tests

Focused verification passed:

```powershell
uv run --no-sync pytest -q tests/test_benchmarks.py tests/test_inference_sweep.py
# 8 passed

uv run --no-sync pytest -q tests/test_ilsd_gradient.py tests/test_benchmarks.py tests/test_inference_sweep.py
# 22 passed

git diff --check
# no whitespace errors
```

## Commands for future checks

```powershell
uv run --no-sync pytest -q tests/test_benchmarks.py tests/test_inference_sweep.py
git diff --check
```

## Next session notes

If a future run adds entropy trajectory telemetry to validation, wire it into
the same dashboard panel so score gain and entropy collapse can be inspected
side by side.
