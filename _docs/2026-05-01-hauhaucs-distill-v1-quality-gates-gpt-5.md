# HauhauCS Distill v1 Quality Gates

## Goal

Freeze the existing `qwen35_9b_hauhaucs_*` bundles as v0 smoke corpus and add a separate v1 generation path that rejects fallback-shaped records before they can enter SFT or GRPO.

## Files Touched

- `src/elt_lm/gguf_distill.py`
- `tests/test_gguf_distill.py`
- `configs/gguf_distill_code_qwen35_hauhaucs_v1.yaml`
- `configs/gguf_distill_math_qwen35_hauhaucs_v1.yaml`
- `configs/gguf_distill_stem_qwen35_hauhaucs_v1.yaml`
- `configs/gguf_distill_tool_qwen35_hauhaucs_v1.yaml`

## Key Decisions

- Kept the default `quality_profile: smoke` so existing v0 configs and completed bundles remain compatible.
- Added `quality_profile: v1` plus `reject_fallback_outputs`, uniqueness thresholds, duplicate thresholds, and generation retry count to `GGUFDistillPipelineConfig`.
- Added lane-specific validators for code, math, STEM MCQ, and tool-use records.
- Code v1 requires executable assert-based verifier snippets and rejects placeholder `return None` or callable-only checks.
- Math v1 rejects zero fallback answers and uses the exact math verifier for symbolic or numeric equivalence.
- STEM v1 rejects `Option A/B/C/D` placeholder choices and enforces round-robin expected answer letters.
- Tool-use v1 rejects empty arguments and requires exact JSON match between response and reference.
- Added dataset-level quality summaries: unique text ratio, exact duplicate count/ratio, fallback rejects, verifier pass rate, answer distribution, retry count, accepted records, and attempted tasks.
- If a v1 bundle fails the quality gate, the pipeline writes `state=failed_quality_gate` before raising `QualityGateError`.
- Replaced Windows `os.kill(pid, 0)` lock probing with `OpenProcess`/`WaitForSingleObject` so GGUF distill lock tests and live runs do not trigger `KeyboardInterrupt`.

## Tests

- `H:/elt_data/cache/tmp` was used for temporary files to avoid putting more pressure on `C:`.
- `python -m py_compile src/elt_lm/gguf_distill.py tests/test_gguf_distill.py` passed.
- `.venv/Scripts/python.exe -m pytest -q tests/test_gguf_distill.py` passed: `20 passed`.
- `.venv/Scripts/python.exe -m pytest -q tests/test_gguf_distill.py tests/test_prepare_gguf_lane_sft.py tests/test_pipeline_orchestrator.py` passed: `43 passed`.
- Dry-load check for all four v1 configs passed, each resolving to `128` planned tasks and an `_v1` output root.
- Dry-run plan for code v1 with `max_tasks=2` resolved to `H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_code_v1` without touching the v0 bundle.

## Next Notes

- The current v0 HauhauCS bundles remain useful as pipeline smoke proof only.
- Before running long v1 generation, use `--max-tasks 2 --skip-upload --skip-student-eval` once per lane to inspect `eval_summary.json` quality fields.
- Prepare/SFT should only consume v1 bundles after the quality gate passes.
