# Unsloth QLoRA roleplay SFT launcher

## Goal

Prepare a reproducible Unsloth QLoRA SFT path for the local
`huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated` cache and the
Aratako formatted Japanese roleplay dataset, without disturbing the existing
ELT `elt-train` side-LoRA pipeline.

## Files touched

- `src/elt_lm/unsloth_qwen35_qlora_sft.py`
- `scripts/unsloth_qwen35_qlora_sft.py`
- `scripts/export_unsloth_adapter.py`
- `scripts/merge_peft_adapter.py`
- `scripts/merge_lora_safetensors_sharded.py`
- `tests/test_unsloth_qwen35_qlora_sft.py`
- `_docs/2026-05-15-unsloth-qwen35-qlora-roleplay-gpt-5.md`

## Key decisions

- Added a standalone script entrypoint instead of replacing existing ELT
  training configs.
- Resolve HF cache roots such as
  `H:/hf_cache/hub/models--.../snapshots/<sha>` automatically.
- Default tokenizer path is `H:/Qwen3.5-9B-official-hf` because the local model
  snapshot has weights/config only and no complete chat template.
- If the tokenizer has no chat template, install a minimal Qwen/ChatML template.
- Keep Unsloth imports inside the real training path so dry-run validation and
  unit tests do not require optional fine-tuning dependencies.
- Require `--accept-dataset-terms` before training from the HF dataset or a
  local parquet export of it. The dataset card lists CC-BY-NC-SA 4.0 and an
  Anthropic terms note, so the launcher intentionally makes that confirmation
  explicit.
- Apply a conservative default `adult-consent` lexical filter before SFT. This
  removes obvious minor/non-consent/high-risk records without printing matched
  text.
- Use local parquet as a supported source because this Python venv currently
  fails TLS verification against Hugging Face, while PowerShell can download the
  Dataset Viewer parquet URL successfully.
- On Windows, disable TorchDynamo/Inductor by default for this launcher. The
  local triton-windows/TorchInductor path failed compiling `cuda_utils.c`
  because `tccdefs.h` was missing, while eager CUDA training completed.
- Ignore `unsloth_compiled_cache/`; it is generated runtime cache.
- For GGUF export, Unsloth `save_pretrained_merged()` and Transformers
  `save_pretrained()` both stalled while writing large safetensors shards on
  this Windows run. The completed path directly reads the base safetensors,
  applies the LoRA delta to the 128 matching text tensors, writes six smaller
  merged safetensors shards, and then runs llama.cpp `convert_hf_to_gguf.py`
  with `--outtype q8_0`.

## Current local evidence

- Model snapshot resolved to:
  `H:/hf_cache/hub/models--huihui-ai--Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated/snapshots/794528f9c51127730c7cf8bcfda63164581ae722`
- GPU check: NVIDIA GeForce RTX 3060, 12 GB VRAM, CUDA available.
- Local parquet downloaded to:
  `H:/elt_data/raw/roleplay/aratako_synthetic_japanese_roleplay_nsfw_claude45_formatted/train-0000.parquet`
- Dry-run at `--max-seq-length 4096`:
  - rows before filter: 3,482
  - rows after filter: 763
  - checked rows: 16
  - token p50: 3,865
  - token p95: 6,482
  - rows over max length: 7 / 16
  - mojibake suspect rows: 0
- Local `C:/Users/downl/Desktop/kimeseku_dataset.jsonl` dry-run:
  - rows before filter: 1,000
  - rows after filter: 1,000
  - checked rows: 32
  - token p50: 402
  - token p95: 497
  - rows over max length: 0 / 32
  - mojibake suspect rows: 0
- Actual Unsloth QLoRA SFT run completed:
  - output dir:
    `H:/elt_data/runs/huihui_qwen35_4b_roleplay_unsloth_qlora`
  - adapter:
    `H:/elt_data/runs/huihui_qwen35_4b_roleplay_unsloth_qlora/adapter/adapter_model.safetensors`
  - dataset rows before filter: 3,482
  - dataset rows after filter: 763
  - train examples after eval split: 740
  - max sequence length: 2,048
  - LoRA rank / alpha: 16 / 32
  - trainable parameters: 21,233,664 / 4,560,499,200 (0.47%)
  - total steps: 80
  - checkpoints saved: `checkpoint-40`, `checkpoint-60`, `checkpoint-80`
    (`checkpoint-20` was pruned by `save_total_limit=3`)
  - train runtime: 1,283 seconds
  - reported train loss: 2.063
  - final logged loss window: 1.979
  - adapter safetensors validation: 256 keys readable.
- Merged HF export for GGUF:
  - output dir:
    `H:/elt_data/hf_exports/huihui_qwen35_4b_roleplay_unsloth_qlora_peft_merged_bf16`
  - shard count: 6 safetensors shards
  - indexed weights: 738
  - indexed tensor bytes: 9,319,737,856
  - LoRA targets merged: 128
  - missing LoRA targets: 0
- Q8_0 GGUF export:
  - output file:
    `H:/elt_data/releases/huihui-qwen35-4b-roleplay-unsloth-qlora-q8_0.gguf`
  - file size: 4,482,402,592 bytes
  - SHA-256:
    `B3691522A59B7D334AB4AC87A50046D09C5C76DABDF4C476793ABC98EBA082E1`
  - GGUF architecture/name: `qwen35` /
    `huihui-qwen35-4b-roleplay-unsloth-qlora`
  - GGUF file type: 7 (`Q8_0`)
  - GGUF tensor count: 426
  - tokenizer pre-tokenizer: `qwen35`

## Tests

- `uv --native-tls run pytest -q tests/test_unsloth_qwen35_qlora_sft.py` -> 6 passed.
- `uv --native-tls run python -m py_compile src/elt_lm/unsloth_qwen35_qlora_sft.py scripts/unsloth_qwen35_qlora_sft.py` -> passed.
- `uv --native-tls run python -m py_compile scripts/merge_peft_adapter.py scripts/merge_lora_safetensors_sharded.py` -> passed.
- llama.cpp dry-run:
  `convert_hf_to_gguf.py ... --outtype q8_0 --dry-run` -> passed,
  reported 426 tensors and about 4.5 GB output.
- `llama-gguf.exe H:/elt_data/releases/huihui-qwen35-4b-roleplay-unsloth-qlora-q8_0.gguf r n` -> passed.
- HF direct dry-run through `datasets.load_dataset()` failed due Python TLS
  certificate verification, so local parquet was used for the successful
  validation path.
- `uv --native-tls run python -c "... safe_open(...)"` read the final adapter
  safetensors file and counted 256 LoRA tensors.

## Next session notes

- This `.venv` now has Unsloth/TRL/PEFT/Accelerate/BitsAndBytes installed.
  `uv --native-tls` was needed to avoid local CA failures.
- `uv pip install unsloth ...` initially installed CPU torch. The CUDA stack was
  restored by `uv --native-tls run ...` from the repo's cu128 torch pin, then
  `torchvision==0.26.0+cu128` was installed from the PyTorch cu128 index to
  match `torch==2.11.0+cu128`.
- The completed launch command was:

```powershell
uv --native-tls run python scripts\unsloth_qwen35_qlora_sft.py `
  --accept-dataset-terms `
  --model-path "H:\hf_cache\hub\models--huihui-ai--Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated" `
  --tokenizer-path "H:\Qwen3.5-9B-official-hf" `
  --parquet "H:\elt_data\raw\roleplay\aratako_synthetic_japanese_roleplay_nsfw_claude45_formatted\train-0000.parquet" `
  --output-dir "H:\elt_data\runs\huihui_qwen35_4b_roleplay_unsloth_qlora" `
  --max-seq-length 2048 `
  --per-device-train-batch-size 1 `
  --gradient-accumulation-steps 1 `
  --max-steps 80 `
  --save-steps 20 `
  --save-total-limit 3 `
  --logging-steps 5
```

- If continuing training, resume from
  `H:/elt_data/runs/huihui_qwen35_4b_roleplay_unsloth_qlora/checkpoint-80`.
  A longer run should consider lower LR and a held-out qualitative/eval pass
  because this first run intentionally covered only 0.108 epoch.
