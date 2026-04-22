## Goal

Improve ELT loop stability with minimal-memory ILSD regularization and add a
verifier-driven best-of-N retry path for benchmark evaluation without changing
the general inference API.

## Background / Requirements

- The existing ILSD implementation used plain detached teacher softmax targets
  with no mask-aware soft CE, no entropy floor, and no loop-consistency signal.
- Full per-loop vocab logits were intentionally avoided because they are too
  expensive for the 12 GB target profile.
- The intended sequencing was:
  1. teacher-only temperature + masked soft CE
  2. entropy floor
  3. hidden-based local consistency
  4. best-of-N + verifier rerank
  5. leave adaptive temperature as a later experiment flag

## Assumptions / Decisions

- Hidden-state consistency is the right low-memory substitute for per-loop logit
  JS in this repo because `L_max <= 4` and hidden tensors are cheap relative to
  `B x T x V` logits.
- Benchmark/eval is the right first landing zone for verifier-triggered retries,
  because it has a reference answer and task-specific scoring already wired in.
- General CLI inference was intentionally left unchanged because there is no
  reference verifier there yet.

## Changed Files

- `src/elt_lm/config.py`
- `src/elt_lm/model.py`
- `src/elt_lm/ilsd.py`
- `src/elt_lm/train.py`
- `src/elt_lm/eval/benchmarks.py`
- `src/elt_lm/eval/anytime_sweep.py`
- `configs/base_1B.yaml`
- `tests/test_ilsd_gradient.py`
- `tests/test_shapes_and_params.py`
- `tests/test_smoke_train.py`
- `tests/test_benchmarks.py`
- `_docs/2026-04-22-ilsd-stability-and-benchmark-rerank-gpt-5.md`

## Implementation Details

### ILSD stabilizers

- Added new `ILSDConfig` fields for:
  - `distill_teacher_temp`
  - `distill_uniform_mix`
  - `entropy_floor_weight`
  - `entropy_floor_start`
  - `entropy_floor_end`
  - `local_consistency_weight`
  - `local_consistency_metric`
- Reworked `_causal_lm_soft_ce` to support:
  - teacher-only temperature
  - tiny uniform smoothing
  - valid-position masking
- Added entropy-floor penalty on student logits using normalized entropy.
- Added hidden-based local consistency across adjacent loop states using either
  cosine distance or mean squared error.
- Extended `ELTOutput` and `ELTLanguageModel.forward()` to optionally expose:
  - `intermediate_hidden`
  - `per_loop_hidden`
- Added telemetry/log output for `l_entropy` and `l_local`.

### Benchmark rerank

- Added `num_samples` and `verifier_retries` support to benchmark evaluation.
- Evaluation now:
  - samples multiple candidates per case,
  - scores them with the existing task verifier,
  - retries with one more sample when all current attempts fail,
  - records `attempts_per_case`.
- Surfaced the new controls and metric in `anytime_sweep`.

### Config defaults

- Enabled small nonzero ILSD stabilizer defaults in `configs/base_1B.yaml`.
- Left small test configs untouched to preserve stable smoke behavior.

## Commands Run

```powershell
uv run --no-sync pytest -q tests/test_ilsd_gradient.py tests/test_shapes_and_params.py tests/test_smoke_train.py tests/test_benchmarks.py tests/test_loop_equivalence.py tests/test_inference_sweep.py
uv run --no-sync pyright src/elt_lm/ilsd.py src/elt_lm/model.py src/elt_lm/eval/benchmarks.py src/elt_lm/eval/anytime_sweep.py src/elt_lm/train.py tests/test_ilsd_gradient.py tests/test_benchmarks.py tests/test_shapes_and_params.py
```

## Test / Verification Results

- Targeted pytest slice: passed
- Pyright on changed files: `0 errors, 0 warnings, 0 informations`

## Residual Risks

- `per_loop_hidden` still increases training memory a bit when local consistency
  is enabled, even though it is much cheaper than full per-loop logits.
- Verifier-triggered retries currently live only in benchmark evaluation, not in
  the general inference CLI or agent runtime.
- Adaptive temperature was intentionally not added yet.

## Recommended Next Actions

1. Run a short `base_1B` smoke/post-warmup training slice and inspect
   `l_entropy` / `l_local` telemetry for saturation or collapse.
2. Add benchmark manifests for STEM-heavy tasks so the new rerank path can be
   exercised on real workloads.
3. If retry behavior helps, lift the same pattern into agent/runtime paths that
   have task-specific verifiers.
