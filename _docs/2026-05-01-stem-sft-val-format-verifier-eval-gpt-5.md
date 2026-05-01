# 2026-05-01 stem SFT val format/verifier eval

## Goal

After `posttrain_stem_sft_qwen35_hauhaucs` completes, measure the 13-case v0
stem validation benchmark with two separate metrics:

- strict `<think>...</think><answer>...</answer>` format success rate
- `mcq_reasoning` exact verifier success rate

The v0 HauhauCS stem data is duplicate-heavy and uses placeholder choices, so
high scores must be treated as schema memorization evidence, not broad STEM
reasoning generalization.

## Files touched

- `src/elt_lm/eval/benchmarks.py`
- `src/elt_lm/eval/anytime_sweep.py`
- `scripts/pipeline.py`
- `tests/test_benchmarks.py`
- `tests/test_pipeline_orchestrator.py`

## Key decisions

- Kept the existing `BenchmarkResult.accuracy` semantics as verifier accuracy.
- Added `format_correct` / `format_rate` beside verifier accuracy instead of
  replacing the score.
- Extended `elt-anytime` with `--L-list` because the pipeline already uses that
  interface.
- Added `--out-json` for a durable summary with an interpretation warning for
  v0 stem data.
- Added a pipeline hook and an explicit `03a_stem_sft_val_eval` stage. The hook
  runs after stem SFT completion and writes:
  - `H:/elt_data/runs/posttrain_stem_sft_qwen35_hauhaucs/eval/stem_val_format_verifier_summary.json`
  - `H:/elt_data/runs/posttrain_stem_sft_qwen35_hauhaucs/eval/stem_val_format_verifier_anytime.csv`

## Tests

Passed:

```powershell
uv run --no-sync pytest -q tests/test_benchmarks.py tests/test_pipeline_orchestrator.py
```

Result: `27 passed`

Passed:

```powershell
uv run --no-sync python -m py_compile src/elt_lm/eval/benchmarks.py src/elt_lm/eval/anytime_sweep.py scripts/pipeline.py
```

## Current run status

At implementation time, the active stem SFT was still running at step `32/40`.
The post-stem eval output did not exist yet. The checkpoint was healthy:

- `H:/elt_data/runs/posttrain_stem_sft_qwen35_hauhaucs/last.pt`
- size: `4,854,452,679` bytes

The evaluation should run once stem SFT completes and the patched pipeline is
picked up by a subsequent pipeline invocation. If the already-running pipeline
process was started before this patch, the explicit stage can be run later with:

```powershell
uv run --no-sync python scripts/pipeline.py --profile posttrain-grpo --only stem_sft_val_eval
```
