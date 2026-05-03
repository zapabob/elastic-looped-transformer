# 2026-05-03 - benchmark-release-readiness - gpt-5

## Goal

Prepare the repo for vanilla-vs-finished benchmark comparison with
cross-validation summaries, p-values, README reporting, and HF/GGUF release
handoff.

## Files touched

- `src/elt_lm/eval/statistics.py`
- `src/elt_lm/eval/benchmark_comparison.py`
- `src/elt_lm/export_lora_adapter.py`
- `src/elt_lm/release_readiness.py`
- `pyproject.toml`
- `tests/test_eval_statistics.py`
- `tests/test_hf_qwen35_looped.py`
- `README.md`
- `_docs/assets/2026-05-03-deepresearch-elt-llm-implementation/release_readiness_stem_bridge.json`

## Key decisions

- Added paired permutation tests instead of unpaired tests because benchmark
  cases/folds must remain aligned across vanilla, replay-SFT, GRPO, and final
  model groups.
- Added a Friedman within-block permutation p-value for three-or-more model
  groups without requiring SciPy.
- Kept broad lm-eval scores out of the README until the same paired task set has
  completed for vanilla and final models. The current evidence is only internal
  synthetic-v2 bridge diagnostics.
- Exported side-LoRA adapters as both local `adapter.pt` and portable
  `adapter_model.safetensors`; full GGUF remains gated on a merged/full
  HF-loadable directory that llama.cpp can convert.

## Artifacts

- Stem bridge adapter export:
  `H:/elt_data/adapters/qwen35_4b_side/synthetic_stem_v2_bridge_grpo_candidate/adapter_model.safetensors`
  (64,987,976 bytes)
- Release readiness manifest:
  `_docs/assets/2026-05-03-deepresearch-elt-llm-implementation/release_readiness_stem_bridge.json`

## Tests

- `uv run --no-sync pytest -q tests/test_eval_statistics.py tests/test_hf_qwen35_looped.py::test_export_lora_adapter_writes_small_artifact`
  - 7 passed
- `uv run --no-sync python -m py_compile src/elt_lm/eval/statistics.py src/elt_lm/eval/benchmark_comparison.py src/elt_lm/export_lora_adapter.py src/elt_lm/release_readiness.py`
  - passed

## Next session notes

- Run lm-eval-harness after a HF-loadable vanilla export and a HF-loadable final
  export exist. Use logged samples so paired correctness arrays can feed
  `elt_lm.eval.benchmark_comparison`.
- Do not mark GGUF ready for the side-LoRA bridge until a merged/base+adapter HF
  directory has `config.json`, tokenizer files, and safetensors weights and
  `convert_hf_to_gguf.py` succeeds.
