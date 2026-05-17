# 2026-05-16 - Claude 3.5 15.3k roleplay QLoRA stats + train - GPT-5

## Goal

Download `Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-3.5s-15.3k-formatted`,
measure post-filter row count and token length distribution, then run an
additional Unsloth QLoRA pass from the local Huihui Qwen3.5-4B base.

## Files touched

- `src/elt_lm/unsloth_qwen35_qlora_sft.py`
- `tests/test_unsloth_qwen35_qlora_sft.py`
- `_docs/2026-05-16-claude35-15k-qloRA-stats-train-gpt-5.md`

## Dataset

- Local parquet root:
  `H:/elt_data/raw/roleplay/aratako_synthetic_japanese_roleplay_nsfw_claude35_15_3k_formatted`
- Files:
  - `20240817-0000.parquet` - 19,968,051 bytes
  - `20240907-0000.parquet` - 18,840,141 bytes
- Source rows: 15,264
- Adult-consent filter rows: 9,708
- Mojibake suspect rows after render: 1

Full dry-run command:

```powershell
uv --native-tls run python scripts\unsloth_qwen35_qlora_sft.py `
  --accept-dataset-terms `
  --model-path "H:\hf_cache\hub\models--huihui-ai--Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated" `
  --tokenizer-path "H:\Qwen3.5-9B-official-hf" `
  --parquet "H:\elt_data\raw\roleplay\aratako_synthetic_japanese_roleplay_nsfw_claude35_15_3k_formatted\*.parquet" `
  --dry-run `
  --dry-run-rows 0 `
  --max-seq-length 2048
```

Token length distribution over all 9,708 rendered, filtered rows:

| metric | tokens |
| --- | ---: |
| min | 297 |
| mean | 1,274.22 |
| p50 | 1,059 |
| p90 | 2,130 |
| p95 | 2,332 |
| p99 | 2,749 |
| max | 4,452 |
| rows over 2,048 | 1,216 |

## Key decisions

- Kept the hard `adult-consent` filter enabled.
- Added parquet glob/list support so multiple split parquet files can be used
  without merging them first.
- Added `--dry-run-rows 0` semantics for all-row dataset profiling.
- Tried a quality-oriented `max_seq_length=3072` smoke. It loaded and ran one
  step, but the longer 800-step run stalled around step 28 with very slow
  steps, so it was stopped before producing a checkpoint.
- Switched the production adapter run to `max_seq_length=2048` for stability on
  the RTX 3060 12 GB. This truncates about 12.5% of filtered rows but avoids the
  unstable 3072 path.

## Training result

Primary output dir:

`H:/elt_data/runs/huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_filtered_ms2048_s800`

Completed adapter:

`H:/elt_data/runs/huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_filtered_ms2048_s800/adapter/adapter_model.safetensors`

Run settings:

- Base model:
  `H:/hf_cache/hub/models--huihui-ai--Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated`
- Tokenizer: `H:/Qwen3.5-9B-official-hf`
- Max sequence length: 2,048
- LoRA rank / alpha: 16 / 32
- Learning rate: `8e-5`
- Batch size: 1
- Gradient accumulation: 1
- Train/eval split after filter: 9,416 / 292
- Trainable parameters: 21,233,664 / 4,560,499,200 (0.47%)

The first continuous run reached `checkpoint-100`. Continuing toward 800 became
too slow for an interactive pass, so the job was stopped after checkpointing and
resumed from `checkpoint-100` to `max_steps=110` in order to finish normally and
write the final `adapter/` directory.

Loss log:

| step | loss |
| ---: | ---: |
| 10 | 2.3789 |
| 20 | 1.8747 |
| 30 | 1.8780 |
| 40 | 1.6013 |
| 50 | 1.8329 |
| 60 | 1.6003 |
| 70 | 1.5797 |
| 80 | 1.5334 |
| 90 | 1.5353 |
| 100 | 1.5184 |
| 110 | 1.5046 |

Resume-to-final command:

```powershell
uv --native-tls run python scripts\unsloth_qwen35_qlora_sft.py `
  --accept-dataset-terms `
  --model-path "H:\hf_cache\hub\models--huihui-ai--Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated" `
  --tokenizer-path "H:\Qwen3.5-9B-official-hf" `
  --parquet "H:\elt_data\raw\roleplay\aratako_synthetic_japanese_roleplay_nsfw_claude35_15_3k_formatted\*.parquet" `
  --output-dir "H:\elt_data\runs\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_filtered_ms2048_s800" `
  --max-seq-length 2048 `
  --max-steps 110 `
  --eval-ratio 0.03 `
  --logging-steps 5 `
  --save-steps 10 `
  --save-total-limit 3 `
  --learning-rate 8e-5 `
  --warmup-ratio 0.03 `
  --gradient-accumulation-steps 1 `
  --lora-rank 16 `
  --lora-alpha 32 `
  --seed 3407 `
  --resume-from-checkpoint "H:\elt_data\runs\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_filtered_ms2048_s800\checkpoint-100"
```

## Tests

- `uv run pytest -q tests/test_unsloth_qwen35_qlora_sft.py` -> 7 passed.
- `uv --native-tls run python -m py_compile src/elt_lm/unsloth_qwen35_qlora_sft.py scripts/unsloth_qwen35_qlora_sft.py` -> passed.

## Q8_0 GGUF export

Merged HF export:

`H:/elt_data/hf_exports/huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_ms2048_s110_merged_bf16`

Merge command:

```powershell
uv --native-tls run python scripts\merge_lora_safetensors_sharded.py `
  --base-model "H:\hf_cache\hub\models--huihui-ai--Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated\snapshots\794528f9c51127730c7cf8bcfda63164581ae722" `
  --adapter "H:\elt_data\runs\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_filtered_ms2048_s800\adapter" `
  --out-dir "H:\elt_data\hf_exports\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_ms2048_s110_merged_bf16" `
  --tokenizer "H:\elt_data\runs\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_filtered_ms2048_s800\adapter" `
  --max-shard-size 1536MB `
  --force
```

Merge result:

- sharded safetensors: 6 shards
- indexed weights: 738
- indexed tensor bytes: 9,319,737,856
- merged LoRA targets: 128
- missing LoRA targets: 0

Q8_0 GGUF:

`H:/elt_data/releases/huihui-qwen35-4b-roleplay-unsloth-qlora-claude35-15k-ms2048-s110-q8_0.gguf`

Conversion command:

```powershell
$env:PYTHONPATH = "C:\Users\downl\Desktop\triality-platform\repos\llama.cpp\gguf-py"
uv --native-tls run python "C:\Users\downl\Desktop\triality-platform\repos\llama.cpp\convert_hf_to_gguf.py" `
  "H:\elt_data\hf_exports\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_ms2048_s110_merged_bf16" `
  --outfile "H:\elt_data\releases\huihui-qwen35-4b-roleplay-unsloth-qlora-claude35-15k-ms2048-s110-q8_0.gguf" `
  --outtype q8_0
```

GGUF verification:

- file size: 4,482,410,240 bytes
- SHA-256:
  `ABECC72BA1B6A09853E631F835F17EDA17D124F5FB114CAAA00BF3B9100333C5`
- `llama-gguf.exe ... r n` -> passed, 426 tensors readable.
- Python `GGUFReader` metadata:
  - `general.architecture=qwen35`
  - `general.name=Huihui_Qwen35_4B_Roleplay_Unsloth_Qlora_Claude35_15K_Ms2048_S110_Merged_Bf16`
  - `general.file_type=7`
  - `tokenizer.ggml.pre=qwen35`
  - `n_tensors=426`

## Next session notes

- The adapter is real and saved, but this is still a short run: 110 optimizer
  steps, about 0.0117 epoch over the 9,416 training rows.
- For a longer quality run on this RTX 3060, prefer resuming from
  `checkpoint-110` with shorter wall-clock chunks, or add length bucketing /
  max-token curriculum before attempting hundreds of more steps.
- Q8_0 GGUF conversion is complete. A `llama-cli` generation smoke was attempted
  but interrupted by the interactive run; structural GGUF read and metadata
  checks passed.
