# 2026-05-03 - merged-qwen35-lora-gguf-turboquant - gpt-5

## Goal

Create a HF-loadable Qwen3.5 side-LoRA bridge artifact by merging the vanilla
base checkpoint and LoRA adapter weights, convert it through llama.cpp GGUF, and
produce a Turboquant TQ4_1S GGUF handoff artifact.

## Files touched

- `src/elt_lm/export_merged_qwen35_hf.py`
- `src/elt_lm/release_readiness.py`
- `tests/test_hf_qwen35_looped.py`
- `tests/test_eval_statistics.py`
- `pyproject.toml`
- `README.md`
- `_docs/assets/2026-05-03-deepresearch-elt-llm-implementation/release_readiness_stem_bridge_merged.json`

## Key decisions

- Added a merged export path for `hf_qwen35_looped` checkpoints that loads the
  adapter checkpoint, resolves its `base_checkpoint`, applies each LoRA delta as
  `lora_B @ lora_A * alpha / rank`, strips the local `qwen.` prefix, and writes
  standard Hugging Face `model.safetensors`.
- Kept ELT metadata in `config.json` under `elt_config` and in
  `elt_export_manifest.json` so downstream release tooling can distinguish this
  from a plain Qwen3.5 dump.
- Preserved the side-LoRA bridge boundary explicitly: this artifact is
  `L_min=L_max=1`; native looped ELT runtime support remains separate.
- Extended release readiness with optional Turboquant source/output fields so
  the manifest can record HF safetensors, llama.cpp GGUF, Q8_0 source GGUF, and
  TQ4_1S output GGUF together.
- No llama.cpp source patch was needed for this run; current
  `zapabob/llama.cpp` converted the merged HF directory as Qwen3.5 GGUF.

## Artifacts

- HF safetensors:
  `H:/elt_data/hf_exports/elt-lm-qwen35-side-stem-v2-bridge-merged/model.safetensors`
  - 9,682,950,504 bytes
  - 427 tensors
  - 4,841,450,496 parameters
  - 248 LoRA modules merged
- BF16 GGUF:
  `H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge.gguf`
  - 9,695,799,200 bytes
- Q8_0 GGUF:
  `H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge-Q8_0.gguf`
  - 5,157,840,800 bytes
- Turboquant TQ4_1S GGUF:
  `H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge-TQ4_1S.gguf`
  - 4,467,421,536 bytes
  - Tensor type counts: F32=177, Q8_0=166, TQ4_1S=84
  - Metadata: `hypura.turboquant.weight.policy=qwen35-config-i`

## Tests and verification

- `uv run --no-sync pytest -q`
  - full repo suite exited 0
- `uv run --no-sync pytest -q tests/test_hf_qwen35_looped.py::test_export_lora_adapter_writes_small_artifact tests/test_hf_qwen35_looped.py::test_export_merged_qwen35_hf_merges_lora_delta tests/test_hf_qwen35_looped.py::test_export_merged_qwen35_hf_writes_elt_config_metadata tests/test_eval_statistics.py`
  - 10 passed
- `uv run --no-sync python -m py_compile src/elt_lm/export_merged_qwen35_hf.py src/elt_lm/release_readiness.py`
  - passed
- `uv run --no-sync python -m elt_lm.export_merged_qwen35_hf ...`
  - wrote merged HF safetensors manifest with `tokenizer_ready=true`
- `python C:/Users/downl/Desktop/llama.cpp-zapabob/convert_hf_to_gguf.py ... --outtype bf16`
  - wrote BF16 GGUF successfully
- `llama-quantize.exe ... Q8_0`
  - wrote Q8_0 GGUF successfully
- `uv run --no-sync python C:/Users/downl/Desktop/Turboquant-CUDA/scripts/convert_weight_turboquant_gguf.py ... --replace-existing-turboquant-metadata --force`
  - wrote TQ4_1S GGUF successfully
- `llama-gguf.exe H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge-TQ4_1S.gguf r n`
  - parsed the TQ4_1S GGUF with 86 KV entries and 427 tensors

## Next session notes

- The release manifest now has no blocking notes and includes replay commands
  for HF upload, GGUF upload, and Turboquant upload.
- The TQ4_1S GGUF was validated at the metadata/tensor-type level. Runtime
  generation smoke should be done with the current local Turboquant-enabled
  llama runtime before publishing performance claims.
- If the target shifts from side-LoRA bridge release to native looped ELT
  runtime, start a separate llama.cpp runtime design/PR instead of overloading
  this artifact path.
