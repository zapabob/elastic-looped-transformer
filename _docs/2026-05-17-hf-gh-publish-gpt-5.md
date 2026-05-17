# HF and GitHub Publish Closeout

## Overview

Published the Huihui Qwen3.5 roleplay QLoRA s800 release to both GitHub and
Hugging Face. The Hugging Face release uses descriptive artifact names for the
Q8_0 GGUF and merged BF16 safetensor shards, includes a ChatML prompt template,
and documents the training dataset/filter accounting.

## Scope

GitHub commit/push:

- committed the Unsloth QLoRA launcher
- committed merge/export helpers
- committed focused tests
- committed implementation logs
- committed the Hugging Face model-card source under `_docs/hf_cards/`

Hugging Face commit:

- repository: `zapabobouj/huihui-qwen35-4b-roleplay-qlora-ms2048-s800`
- commit: `b2a8c663156df5be7b1b35b543ccbf469ccee7f4`
- uploaded files: 24 requested files plus Hub-managed `.gitattributes`
- uploaded bytes reported by the Hub client: 13,927,346,487

## Hugging Face Artifacts

Renamed runtime artifact:

- `huihui-qwen35-4b-roleplay-qlora-ms2048-s800-q8_0.gguf`

Renamed BF16 Transformers shards:

- `huihui-qwen35-4b-roleplay-qlora-ms2048-s800-bf16-00001-of-00006.safetensors`
- `huihui-qwen35-4b-roleplay-qlora-ms2048-s800-bf16-00002-of-00006.safetensors`
- `huihui-qwen35-4b-roleplay-qlora-ms2048-s800-bf16-00003-of-00006.safetensors`
- `huihui-qwen35-4b-roleplay-qlora-ms2048-s800-bf16-00004-of-00006.safetensors`
- `huihui-qwen35-4b-roleplay-qlora-ms2048-s800-bf16-00005-of-00006.safetensors`
- `huihui-qwen35-4b-roleplay-qlora-ms2048-s800-bf16-00006-of-00006.safetensors`

Compatibility index:

- `model.safetensors.index.json` maps Transformers weights to the renamed BF16
  shard filenames.

Adapter:

- `adapter/adapter_model.safetensors` keeps the standard PEFT filename for
  compatibility.

Evidence:

- `training/run_manifest.json`
- `training/eval_metrics.json`
- `training/dry_run_full.stdout.json`
- `training/trainer_state_step800.json`
- `eval/easynovel_s800_safe_prompt_eval.json`

## Dataset and Template Documentation

The Hugging Face README documents:

- base model:
  `huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated`
- dataset:
  `Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted`
- dataset terms:
  CC-BY-NC-SA 4.0 and the dataset-card Anthropic competition restriction
- local extra JSONL:
  1,000 rows before filter, 0 rows after the adult-consent filter
- row counts:
  16,264 before filter, 9,582 after adult-consent filter, 5,909 after length
  filter, 5,731 train rows, 178 eval rows
- template:
  Qwen/ChatML with system, user, and assistant turns
- validation:
  `eval_loss=1.2786273956298828` at step 800
- EasyNovelAssistant smoke:
  Japanese output, no `<think>`, no ChatML leakage, no obvious mojibake

## Verification

GitHub:

- `git push origin main` completed.
- Remote `refs/heads/main` resolved to
  `c1c77de634543e0ae18fe5efa0ee247fcb8350a2` after the first publish commit.

Hugging Face:

- Hugging Face plugin authenticated as `zapabobouj`.
- Hub API model metadata showed the repo updated on 2026-05-17.
- Hub API file listing confirmed:
  - README present
  - renamed Q8_0 GGUF present
  - six renamed BF16 safetensor shards present
  - adapter present
  - eval and training evidence present

Chrome:

- The first lightweight Chrome connection retry succeeded.
- Opening the HF page in Chrome was blocked by an extension UI overlay, so the
  final page-content verification used the authenticated Hugging Face plugin
  and Hub API instead.

## Residual Notes

- The `hf` CLI on this machine still fails TLS certificate verification through
  the Python 3.12 install. The successful upload used the authenticated
  `huggingface_hub` API path.
- The worktree still contains unrelated unstaged/untracked L3/SO8 and repo
  changes that were intentionally left out of this publish commit.
