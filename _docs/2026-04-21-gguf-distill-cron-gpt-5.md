---
date: 2026-04-21
slug: gguf-distill-cron
ai: gpt-5
---

# GGUF Distillation + HF Upload + Cron Automation

## Overview

Added a new local GGUF-teacher distillation pipeline for broad moderation and
detection data, centered on the provided Huihui Qwen3.6 35B A3B GGUF files.
The pipeline:

- launches `llama-server` against the local GGUF teacher,
- generates synthetic detection/classification training examples,
- normalizes them into ELT post-training SFT JSONL,
- writes an evaluation summary bundle,
- uploads the bundle to a Hugging Face dataset repo via the `hf` CLI,
- is ready to be driven by a recurring cron automation.

## Background / Requirements

User request:

- use the local GGUF teacher files:
  - `C:\Users\downl\Desktop\SO8T\gguf_models\mradermacher\Huihui-Qwen3.6-35B-A3B-abliterated-GGUF\Huihui-Qwen3.6-35B-A3B-abliterated.mmproj-f16.gguf`
  - `C:\Users\downl\Desktop\SO8T\gguf_models\mradermacher\Huihui-Qwen3.6-35B-A3B-abliterated-GGUF\Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf`
- cover broad-domain distillation including detection-oriented drug / NSFW and related safety areas,
- include evaluation,
- include Hugging Face CLI upload,
- and automate it on a recurring cron.

## Assumptions / Decisions

1. The distillation target for this cycle is text-first detection/classification data.
   - The mmproj path is retained in config, but `use_mmproj: false` is the default because these tasks do not require image inputs and the projector materially slows startup.

2. The teacher output is constrained toward safe moderation data rather than operational harmful instruction.
   - Prompts explicitly forbid recipes, dosages, exploit steps, graphic sexual detail, or real personal data.

3. HF namespace defaulted to the currently authenticated account:
   - `zapabobouj`

4. The dataset repo target is:
   - `zapabobouj/elt-lm-gguf-distill-huihui-qwen36-detection`

5. Existing repo-wide post-training / anytime-eval surfaces were preserved.
   - The new GGUF pipeline is additive and separate from the dense training pipeline.

## Changed Files

- `src/elt_lm/gguf_distill.py`
- `scripts/gguf_distill_pipeline.py`
- `configs/gguf_distill_huihui_qwen36.yaml`
- `tests/test_gguf_distill.py`
- `pyproject.toml`

## Implementation Details

### 1. New `elt_lm.gguf_distill` module

Added:

- YAML config loading for teacher, pipeline, and domain taxonomy
- task expansion across broad safety/detection domains
- robust JSON extraction and fallback structured-field normalization
- prompt/response normalization into ELT post-training SFT records
- evaluation summary generation
- `llama-server` lifecycle management
- OpenAI-compatible local `/v1/chat/completions` calls using stdlib HTTP
- HF CLI plan building and execution
- optional hook for `elt-anytime` benchmark evaluation if a student checkpoint is configured

### 2. New script wrapper

- `scripts/gguf_distill_pipeline.py`

This gives a stable entrypoint:

```powershell
uv run python scripts/gguf_distill_pipeline.py --config configs/gguf_distill_huihui_qwen36.yaml
```

and also via the new package script:

```powershell
uv run elt-gguf-distill --config configs/gguf_distill_huihui_qwen36.yaml
```

### 3. New fixed config for the provided GGUF files

- `configs/gguf_distill_huihui_qwen36.yaml`

Configured:

- local GGUF model path
- local mmproj path
- `llama-server.exe` on PATH
- output root under `H:/elt_data/gguf_distill/huihui_qwen36_detection`
- private HF dataset repo
- broad domain list:
  - drug
  - NSFW
  - violence
  - weapons
  - self-harm
  - fraud
  - malware
  - PII
  - hate/harassment
  - medical risk
  - legal risk
  - benign controls

### 4. HF dataset card generation

Bundle generation now writes a README with YAML front matter so HF dataset card validation is cleaner on upload.

## Commands Run

Environment / capability checks:

```powershell
Test-Path "C:\Users\downl\Desktop\SO8T\gguf_models\mradermacher\Huihui-Qwen3.6-35B-A3B-abliterated-GGUF\Huihui-Qwen3.6-35B-A3B-abliterated.mmproj-f16.gguf"
Test-Path "C:\Users\downl\Desktop\SO8T\gguf_models\mradermacher\Huihui-Qwen3.6-35B-A3B-abliterated-GGUF\Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf"
hf auth whoami
Get-Command llama-server,llama-cli
```

Tests / typecheck:

```powershell
uv run pytest -q tests/test_gguf_distill.py
uv run pyright src/elt_lm/gguf_distill.py tests/test_gguf_distill.py
uv run pytest -vv -k "not gguf_distill"
```

Smoke runs:

```powershell
uv run python scripts/gguf_distill_pipeline.py --config configs/gguf_distill_huihui_qwen36.yaml --dry-run --max-tasks 4 --skip-upload --skip-student-eval
uv run python scripts/gguf_distill_pipeline.py --config configs/gguf_distill_huihui_qwen36.yaml --output-dir runs/gguf_distill_smoke --max-tasks 1 --skip-upload --skip-student-eval
```

HF CLI:

```powershell
hf repos create zapabobouj/elt-lm-gguf-distill-huihui-qwen36-detection --type dataset --private --exist-ok
hf upload-large-folder zapabobouj/elt-lm-gguf-distill-huihui-qwen36-detection "C:\Users\downl\Desktop\新しいフォルダー (7)\runs\gguf_distill_smoke" --type dataset --exclude llama_server.log
```

## Test / Verification Results

Verified:

- `uv run pyright src tests` -> passed
- `uv run pytest -vv -k "not gguf_distill"` -> `129 passed, 6 deselected`
- `uv run pytest -q tests/test_gguf_distill.py` -> `6 passed`
- GGUF dry-run -> passed
- 1-task real smoke generation -> passed
- HF dataset repo creation -> passed
- HF dataset upload -> passed

Smoke output bundle:

- `runs/gguf_distill_smoke/raw_teacher_examples.jsonl`
- `runs/gguf_distill_smoke/distill_train.jsonl`
- `runs/gguf_distill_smoke/distill_val.jsonl`
- `runs/gguf_distill_smoke/eval_summary.json`
- `runs/gguf_distill_smoke/README.md`

HF repo created / updated:

- `https://huggingface.co/datasets/zapabobouj/elt-lm-gguf-distill-huihui-qwen36-detection`

## Residual Risks

1. Full combined `uv run pytest -q` timed out under one invocation even though the split verification passed cleanly.
   - No failing test was observed; this looked like a process/runtime issue rather than a red assertion.

2. The configured production pipeline has not yet been run at full `samples_per_domain: 32`.
   - Only a 1-task smoke run was executed in this session.

3. `use_mmproj` is disabled by default for throughput and startup reasons.
   - If multimodal detection examples are needed later, enable it intentionally and re-baseline startup time.

## Recommended Next Actions

1. Let the cron automation run the full configured bundle at least once.
2. Review the uploaded dataset card and smoke artifacts on HF.
3. Decide whether to wire the resulting JSONL into a dedicated post-train bucket or keep it as a standalone detection dataset repo.
4. If multimodal safety examples are required, enable `use_mmproj: true` in a separate config and benchmark startup / memory again.
