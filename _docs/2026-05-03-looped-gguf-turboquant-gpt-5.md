# 2026-05-03 looped GGUF Turboquant

## Goal

Prepare the GGUF and Turboquant release path for ELT exports whose loop range
requires `L >= 2`, using the local `zapabob/llama.cpp` and
`zapabob/Turboquant-CUDA` checkouts without claiming native loop execution for
plain Qwen runtimes.

## Files touched

- `README.md`
- `src/elt_lm/export_merged_qwen35_hf.py`
- `src/elt_lm/release_readiness.py`
- `tests/test_eval_statistics.py`
- `tests/test_hf_qwen35_looped.py`

Related companion changes were made in:

- `C:/Users/downl/Desktop/llama.cpp-zapabob/convert_hf_to_gguf.py`
- `C:/Users/downl/Desktop/Turboquant-CUDA/turboquant/weight_gguf.py`

## Key decisions

- Keep looped Qwen3.5 HF exports on the existing llama.cpp `qwen35`
  architecture and add explicit `elt_config` fields for loop metadata.
- Mark `L_max > 1` artifacts as requiring a loop-aware runtime rather than
  presenting them as plain Qwen-compatible GGUF runtime artifacts.
- Use `ELT/Qwen3.5-looped` as the Turboquant model family for looped exports.
- Make release readiness block publication until both llama.cpp loop runtime
  support and Turboquant `elt.*` metadata preservation are declared.

## Tests

- `uv run --no-sync pytest -q tests/test_hf_qwen35_looped.py::test_export_merged_qwen35_hf_writes_elt_config_metadata tests/test_eval_statistics.py::test_release_readiness_blocks_looped_elt_until_runtime_support_is_declared tests/test_eval_statistics.py::test_release_readiness_reports_turboquant_artifact` (`3 passed`)
- `uv run --no-sync python -m py_compile src/elt_lm/export_merged_qwen35_hf.py src/elt_lm/release_readiness.py`
- `C:/Users/downl/Desktop/llama.cpp-zapabob`: `python -m pytest tests/test_elt_metadata.py -q` (`3 passed`) and `python -m py_compile convert_hf_to_gguf.py tests/test_elt_metadata.py`
- `C:/Users/downl/Desktop/Turboquant-CUDA`: `uv run --no-sync pytest -q tests/test_triality_contract.py tests/test_weight_gguf.py -k "not real_gemma4_e2e"` (`22 passed, 1 deselected`) and `uv run --no-sync python -m py_compile turboquant/triality_contract.py turboquant/weight_gguf.py`

## Next session notes

- This session wires metadata, readiness gates, and Turboquant family handling.
  A future native llama.cpp runtime still needs an execution graph that repeats
  the shared Qwen3.5 text model pass for the requested loop count.
