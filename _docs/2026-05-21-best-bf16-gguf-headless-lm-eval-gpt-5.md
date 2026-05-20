# 2026-05-21 best BF16 GGUF headless lm-eval - GPT-5

## Overview

Identified the best BF16 GGUF serving candidate and ran a headless
`lm-eval-harness` cross-validation comparison over external heldout rows.

## Requirements

- Use the best available version as the BF16 GGUF candidate.
- Run headless `lm-eval-harness` CV with multi-group statistics.
- Upload the BF16 GGUF evidence bundle to Hugging Face.
- Keep claim boundaries explicit for `turbo8`, GSM8K, and loop-aware GGUF.

## Assumptions and decisions

- The BF16 GGUF artifact already exists and is the explicit BF16 release
  candidate: `elt-lm-qwen35-side-stem-aha-ilsd-l3-BF16.gguf`.
- The previous Q8_0 run made `K=bf16,V=turbo3` the best mean policy; this run
  re-evaluated the BF16 GGUF directly rather than only relying on that prior
  Q8_0-model result.
- `K=bf16,V=turbo3` and `K=q8_0,V=turbo3` tied in the BF16 GGUF run. The BF16
  K-cache policy is selected because the requested deliverable is BF16.

## Changed files

- `_docs/assets/2026-05-21-best-bf16-gguf-lm-eval-cv/*`
- `_docs/2026-05-21-best-bf16-gguf-headless-lm-eval-gpt-5.md`
- `README.md`
- `H:/elt_data/hf_publish/qwen3.5-elt-l3/README.md`
- `H:/elt_data/hf_publish/qwen3.5-elt-l3/eval/best_bf16_*`

## Commands run

- `llama-gguf.exe H:\elt_data\hf_publish\qwen3.5-elt-l3\elt-lm-qwen35-side-stem-aha-ilsd-l3-BF16.gguf r n`
- `uv run --with lm-eval --with requests --with datasets --with jsonlines --with matplotlib python scripts\run_lm_eval_gguf_kv_cv.py --sources ... --model H:\elt_data\hf_publish\qwen3.5-elt-l3\elt-lm-qwen35-side-stem-aha-ilsd-l3-BF16.gguf --out-dir _docs\assets\2026-05-21-best-bf16-gguf-lm-eval-cv --max-cases 128 --folds 8 --ctx-size 1024 --ngl 999 --policies bf16:turbo3,q8_0:turbo3,bf16:turbo4,q8_0:turbo4,bf16:turbo8,q8_0:turbo8 --startup-timeout-sec 300 --request-timeout-sec 900`
- `uv run --with lm-eval --with requests --with datasets --with jsonlines --with matplotlib python scripts\run_lm_eval_gguf_kv_cv.py ... --policies q8_0:turbo4 --startup-timeout-sec 900 --port 18081`

## Verification results

- BF16 GGUF metadata read returned GGUF v3 with `86` metadata keys including
  `elt.*` runtime metadata.
- Headless CV completed for four measured policies across `128` cases and
  `8` folds.
- Selected policy: `K=bf16,V=turbo3`, tied overall mean `0.5547 +/- 0.0662`.
- Four-policy Friedman p-value: `0.943806`.
- `turbo8` remains unsupported in the installed llama.cpp runtime.
- HF upload/readback succeeded for README plus `eval/best_bf16_*` files at Hub
  SHA `ba377ee0891e5ecb43d725ad008f466c6da8f5c3`; BF16 GGUF readback size is
  `9695800320` bytes.

## Residual risks

- GSM8K is numeric-MCQ, not standard generation exact-match.
- The stock GGUF runtime is still a serving-surface test and does not prove
  native loop-aware `L>=2` GGUF quality.
- `K=q8_0,V=turbo4` needed a longer single-policy startup retry; the successful
  retry logs are preserved separately from the initial timeout logs.
