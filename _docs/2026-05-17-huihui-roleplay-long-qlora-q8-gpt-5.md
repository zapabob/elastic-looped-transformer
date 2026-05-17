# Huihui Qwen3.5 Roleplay Long QLoRA Q8 Closeout

## Overview

Continued the Huihui Qwen3.5-4B QLoRA roleplay run with
`max_seq_length=2048`, recovered the post-train adapter from the 800-step
checkpoint, merged the adapter into a BF16 HF export, converted it to Q8_0
GGUF, and verified it through an EasyNovelAssistant KoboldCpp real prompt
smoke test.

This log follows the project implementation-log convention and records the
evidence needed for the training/export closeout. It is not a formal
MILSPECLLMOps compliance claim; it is an engineering trace with commands,
artifacts, tests, and residual risks.

## Goal

- Keep `max_seq_length=2048`.
- Continue QLoRA training on the filtered 15.3k roleplay corpus plus extra
  JSONL data, targeting at least 800 steps.
- Check validation loss.
- Re-export the trained adapter as Q8_0 GGUF.
- Run an EasyNovelAssistant real prompt evaluation before declaring the model
  usable.

## Files Touched

- `src/elt_lm/unsloth_qwen35_qlora_sft.py`
- `scripts/unsloth_qwen35_qlora_sft.py`
- `tests/test_unsloth_qwen35_qlora_sft.py`
- `_docs/2026-05-17-huihui-roleplay-long-qlora-q8-gpt-5.md`

Generated artifacts were written under:

- `H:\elt_data\runs\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_plus_kimeseku_ms2048_len1280to1152_s800`
- `H:\elt_data\hf_exports\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_ms2048_s800_curriculum1280_1152_merged_bf16`
- `H:\elt_data\releases\huihui-qwen35-4b-roleplay-unsloth-qlora-claude35-15k-ms2048-s800-curriculum1280-1152-q8_0.gguf`

## Key Decisions

- Kept `max_seq_length=2048` for tokenizer/model configuration.
- Used a curriculum length cap of `max_train_token_length=1152` to keep the
  RTX 3060 training path viable. The dry run confirmed `over_max_seq_length=0`.
- Preserved the `adult-consent` safety filter. The extra
  `C:\Users\downl\Desktop\kimeseku_dataset.jsonl` source contributed zero
  usable rows after filtering, so it was not trained into the adapter.
- Saved the adapter immediately after `trainer.train()` and changed final eval
  handling to reuse the latest `trainer_state.log_history` eval row before
  running a duplicate `trainer.evaluate()`. This prevents losing the adapter
  if the post-train duplicate eval hits CUDA OOM.
- Evaluated EasyNovelAssistant with a safe neutral Japanese writing prompt.
  The existing local EasyNovelAssistant `config.json` contained unrelated
  user-local prompt text and was not used for model-quality evaluation.

## Dataset Evidence

Dry run:

- Artifact: `H:\elt_data\runs\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_plus_kimeseku_ms2048_len1280to1152_s800\dry_run_full.stdout.json`
- `rows_before`: 16264
- `rows_after_filter`: 9582
- `rows_after_length_filter`: 5909
- `safety_filter`: `adult-consent`
- `max_seq_length`: 2048
- Token length: min 297, mean 918.97, p50 949, p90 1093, p95 1121, p99 1146,
  max 1152
- `over_max_seq_length`: 0
- `mojibake_suspect_rows`: 0

Source split:

- Parquet source: 15264 rows before filter, 9582 after safety filter, 5909
  after length filter.
- Extra JSONL source: 1000 rows before filter, 0 after safety filter.

## Training Evidence

Run directory:

`H:\elt_data\runs\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_plus_kimeseku_ms2048_len1280to1152_s800`

Checkpoint evidence:

- `checkpoint-800\trainer_state.json`: `global_step=800`,
  `max_steps=800`, `epoch=0.1395916942941895`
- `checkpoint-800\adapter_model.safetensors`: 84,972,248 bytes
- Final log row at step 800: training loss `1.278938388824463`

Validation evidence:

- `eval_metrics.json`
- `eval_loss=1.2786273956298828`
- `eval_runtime=701.1729`
- `eval_samples_per_second=0.254`
- `step=800`
- `epoch=0.1395916942941895`

The original post-train process reached the 800-step checkpoint and recorded
eval metrics, then hit CUDA OOM during a duplicate final eval call. The code
was patched so future runs save the adapter first and recover the latest eval
metrics from trainer state when available.

## Export Evidence

BF16 merged HF export:

`H:\elt_data\hf_exports\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_ms2048_s800_curriculum1280_1152_merged_bf16`

Merge summary:

- `merged_lora_tensors=128`
- `missing_lora_targets=0`
- 6 safetensor shards plus tokenizer/config/index files

Q8_0 GGUF:

`H:\elt_data\releases\huihui-qwen35-4b-roleplay-unsloth-qlora-claude35-15k-ms2048-s800-curriculum1280-1152-q8_0.gguf`

Verification:

- File size: 4,482,410,272 bytes
- SHA256: `446723801C57DB417B255C90A068A39604400195DE1B97EE7B151A864434859B`
- GGUF reader metadata:
  - `general.architecture=qwen35`
  - `general.file_type=7`
  - `general.quantization_version=2`
  - `qwen35.block_count=32`
  - `qwen35.context_length=262144`
  - `qwen35.embedding_length=2560`
  - `tensor_count=426`
  - tensor types: `8:249`, `0:177`
- `llama-gguf.exe` structural read completed successfully.

## EasyNovelAssistant Real Prompt Evaluation

Temporary server:

- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\KoboldCpp\koboldcpp.exe`
- Port: `5003`
- Model: the new `s800` Q8_0 GGUF above
- Context: 4096
- GPU layers: 0 for the smoke test, to avoid disturbing the already running
  user-local port 5001 server.
- The temporary port 5003 server was stopped after evaluation. The existing
  port 5001 server was left untouched.

Evaluation artifact:

`H:\elt_data\runs\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_plus_kimeseku_ms2048_len1280to1152_s800\easynovel_s800_safe_prompt_eval.json`

Prompt:

`夜の図書館で、主人公が古い日記を見つける場面の続きを三段落で書いてください。心理描写と情景描写を重視してください。`

Checks:

- `prompt_preserved=true`
- `prompt_has_no_question_marks=true`
- `non_empty=true`
- `contains_japanese=true`
- `no_think_tag=true`
- `no_chatml_marker=true`
- `no_obvious_mojibake=true`
- `not_encoding_refusal=true`

The generated answer was a three-paragraph Japanese continuation with no
visible ChatML markers or `<think>` text.

## Commands Run

Representative verification commands:

```powershell
$py='C:\Users\downl\AppData\Roaming\uv\python\cpython-3.12.9-windows-x86_64-none\python.exe'
$repo='C:\Users\downl\Desktop\新しいフォルダー (7)'
$env:PYTHONPATH="$repo\src;$repo\.venv\Lib\site-packages"
$env:PYTHONNOUSERSITE='1'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
& $py -m pytest -q tests\test_unsloth_qwen35_qlora_sft.py
& $py -m py_compile scripts\unsloth_qwen35_qlora_sft.py src\elt_lm\unsloth_qwen35_qlora_sft.py
```

```powershell
Get-FileHash -LiteralPath 'H:\elt_data\releases\huihui-qwen35-4b-roleplay-unsloth-qlora-claude35-15k-ms2048-s800-curriculum1280-1152-q8_0.gguf' -Algorithm SHA256
```

```powershell
Get-Content -LiteralPath 'H:\elt_data\runs\huihui_qwen35_4b_roleplay_unsloth_qlora_claude35_15k_plus_kimeseku_ms2048_len1280to1152_s800\easynovel_s800_safe_prompt_eval.json' -Encoding UTF8 | ConvertFrom-Json
```

## Test Results

- `python -m pytest -q tests\test_unsloth_qwen35_qlora_sft.py`: 11 passed
- `python -m py_compile scripts\unsloth_qwen35_qlora_sft.py src\elt_lm\unsloth_qwen35_qlora_sft.py`: passed
- GGUF metadata read: passed
- EasyNovelAssistant KoboldCpp real prompt smoke: passed

## Residual Risks

- The training run reached exactly 800 steps, not the higher 1500-step end of
  the target range.
- The run covers about 0.14 epoch due the available sequence/token budget.
- The EasyNovelAssistant smoke test proves load/generate/format cleanliness,
  not broad story-quality generalization.
- The extra JSONL source was excluded by safety filtering; any future use of
  extra data must pass the same guardrails first.
- The Q8_0 artifact was validated structurally and by generation smoke, but no
  full benchmark suite was run.

## Next Actions

- If more quality is needed, continue from `checkpoint-800` with the same
  filter and a conservative curriculum length cap, then repeat val loss,
  GGUF conversion, and EasyNovelAssistant smoke.
- Add a small scripted EasyNovelAssistant smoke harness if this path will be
  repeated often.
