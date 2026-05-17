---
license: cc-by-nc-sa-4.0
base_model:
  - huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated
datasets:
  - Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted
language:
  - ja
library_name: transformers
pipeline_tag: text-generation
tags:
  - qwen3.5
  - qwen35
  - qlora
  - lora
  - gguf
  - q8_0
  - roleplay
  - japanese
  - not-for-all-audiences
---

# Huihui Qwen3.5-4B Roleplay QLoRA ms2048 s800

This repository publishes a local QLoRA continuation of
[`huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated)
trained on filtered Japanese roleplay data. It includes:

- merged BF16 Transformers safetensors with descriptive shard names
- Q8_0 GGUF for llama.cpp/KoboldCpp-style runtimes
- the PEFT LoRA adapter under `adapter/`
- tokenizer files and the ChatML template used for training/evaluation
- training, validation, and EasyNovelAssistant smoke-test evidence

## Artifact Names

The publish names intentionally avoid generic local export names:

| file | purpose |
|---|---|
| `huihui-qwen35-4b-roleplay-qlora-ms2048-s800-q8_0.gguf` | Q8_0 GGUF runtime artifact |
| `huihui-qwen35-4b-roleplay-qlora-ms2048-s800-bf16-00001-of-00006.safetensors` ... `00006-of-00006.safetensors` | merged BF16 Transformers shards |
| `model.safetensors.index.json` | Transformers index mapping weights to the renamed BF16 shard files |
| `adapter/adapter_model.safetensors` | standard PEFT LoRA adapter filename for compatibility |

## Training Data

Primary dataset:

- [`Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted`](https://huggingface.co/datasets/Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted)

Dataset/license note:

- The dataset card identifies the data as CC-BY-NC-SA 4.0 and states additional
  Anthropic terms restricting use to develop models that compete with
  Anthropic services or models. Treat this release as non-commercial and bound
  by the upstream dataset terms.
- The base model card identifies the base model license as Apache-2.0.

Local data/filter accounting:

| source | rows before | rows after adult-consent filter | rows after length filter |
|---|---:|---:|---:|
| local parquet mirror of the Aratako dataset | 15,264 | 9,582 | 5,909 |
| extra local JSONL source | 1,000 | 0 | 0 |
| total | 16,264 | 9,582 | 5,909 |

The extra local JSONL source contributed zero training rows after the
adult-consent safety filter. It is not included as trained data in this release.

Train/eval split:

- train rows: 5,731
- eval rows: 178

Length/curriculum:

- model `max_seq_length`: 2048
- training curriculum cap: 1152 rendered tokens
- token length mean: 918.97
- token length p50/p90/p95/p99/max: 949 / 1093 / 1121 / 1146 / 1152
- rows over `max_seq_length`: 0
- mojibake-suspect rows: 0

## Training Configuration

| field | value |
|---|---|
| base model | `huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated` |
| tokenizer | Qwen3.5 tokenizer, local mirror of `H:/Qwen3.5-9B-official-hf` |
| method | Unsloth QLoRA SFT |
| LoRA rank / alpha | 16 / 32 |
| learning rate | 8e-5 |
| max steps | 800 |
| epoch reached | 0.1395916942941895 |
| trainable parameters | 21,233,664 of 4,560,499,200, about 0.47% |
| safety filter | `adult-consent` lexical guardrail |

Validation:

- `eval_loss`: 1.2786273956298828
- `eval_runtime`: 701.1729 seconds
- `eval_samples_per_second`: 0.254
- recorded at step 800

## Prompt Template

Training and the EasyNovelAssistant smoke test used ChatML/Qwen-style wrapping.
Use the included `chat_template.jinja` or this equivalent shape:

```text
<|im_start|>system
あなたは日本語で自然な小説文を続けるアシスタントです。本文だけを書き、思考過程、英語、箇条書き、タグ、メタ説明を出さないでください。<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

Recommended stop strings:

```text
<|im_end|>
<|endoftext|>
<|end_of_text|>
<|end|>
<|im_start|>user
<|im_start|>system
```

Smoke-test sampler:

| field | value |
|---|---:|
| temperature | 0.55 |
| top_p | 0.9 |
| top_k | 40 |
| min_p | 0.02 |
| repetition penalty | 1.08 |
| max new tokens | 192 |

## GGUF Verification

Q8_0 GGUF:

- file: `huihui-qwen35-4b-roleplay-qlora-ms2048-s800-q8_0.gguf`
- size: 4,482,410,272 bytes
- SHA256: `446723801C57DB417B255C90A068A39604400195DE1B97EE7B151A864434859B`

GGUF metadata readback:

| key | value |
|---|---|
| `general.architecture` | `qwen35` |
| `general.file_type` | `7` |
| `general.quantization_version` | `2` |
| `qwen35.block_count` | `32` |
| `qwen35.context_length` | `262144` |
| `qwen35.embedding_length` | `2560` |
| tensor count | `426` |
| tensor types | `8:249`, `0:177` |

## EasyNovelAssistant Smoke Test

The Q8_0 GGUF was loaded through the local EasyNovelAssistant
`KoboldCpp.generate` path on a temporary port with a safe Japanese writing
prompt. Checks passed:

- prompt preserved
- non-empty Japanese output
- no `<think>` tag
- no ChatML marker leakage
- no obvious mojibake
- no encoding-refusal response

Evidence file:

- `eval/easynovel_s800_safe_prompt_eval.json`

## Usage

### Transformers BF16

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "zapabobouj/huihui-qwen35-4b-roleplay-qlora-ms2048-s800"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    trust_remote_code=True,
    device_map="auto",
)
```

### PEFT Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated"
repo = "zapabobouj/huihui-qwen35-4b-roleplay-qlora-ms2048-s800"

tokenizer = AutoTokenizer.from_pretrained(f"{repo}/adapter")
model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True)
model = PeftModel.from_pretrained(model, f"{repo}/adapter")
```

### llama.cpp / KoboldCpp GGUF

Use:

```text
huihui-qwen35-4b-roleplay-qlora-ms2048-s800-q8_0.gguf
```

with the ChatML template above.

## Safety and Scope

This is an adult-oriented Japanese roleplay fine-tune. It is not intended for
use with minors, non-consensual sexual content, or illegal activity. The local
training script applied an `adult-consent` lexical filter before SFT and
excluded the extra local JSONL source completely.

The smoke test verifies loadability and basic Japanese generation cleanliness;
it is not a broad benchmark or a guarantee of generalization.
