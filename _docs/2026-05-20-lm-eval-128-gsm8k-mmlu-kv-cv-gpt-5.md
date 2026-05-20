# 2026-05-20 lm-eval 128-case GSM8K/MMLU K/V CV - GPT-5

## Goal

Extend the GGUF K/V cache `lm-eval-harness` evidence gate from the first
16-case MMLU-STEM slice to at least 128 external heldout cases, including both
MMLU-STEM and GSM8K-derived rows, while preserving the claim boundary around
GGUF serving versus loop-aware ELT runtime quality.

## Files touched

- `scripts/run_lm_eval_gguf_kv_cv.py`
- `tests/test_lm_eval_gguf_kv_cv.py`
- `README.md`
- `H:/elt_data/hf_publish/qwen3.5-elt-l3/README.md`
- `_docs/assets/2026-05-20-lm-eval-gguf-kv-cv-128/*`

## Key decisions

- Kept the existing lm-eval multiple-choice path and generalized it to multiple
  JSONL sources.
- Converted GSM8K test rows into deterministic numeric multiple-choice rows so
  they can be scored through the same one-token A/B/C/D log-probability adapter.
  This is intentionally not a standard GSM8K exact-match leaderboard result.
- Used `8` deterministic folds over `128` cases: `64` native MMLU-STEM MCQ rows
  and `64` GSM8K numeric-MCQ rows.
- Increased request timeout to `900` seconds after the first full run exited
  during `q8_0,V=turbo4`; the failed policy completed when rerun alone with the
  longer timeout, and the final all-policy run completed successfully.

## Results

- Successful policies: `K=q8_0,V=turbo3`, `K=bf16,V=turbo3`,
  `K=q8_0,V=turbo4`, and `K=bf16,V=turbo4`.
- Overall accuracy mean +/- SEM over eight folds:
  - `K=q8_0,V=turbo3`: `0.5547 +/- 0.0630`
  - `K=bf16,V=turbo3`: `0.5625 +/- 0.0765`
  - `K=q8_0,V=turbo4`: `0.5469 +/- 0.0763`
  - `K=bf16,V=turbo4`: `0.5469 +/- 0.0763`
- Pairwise paired permutation p-values are `1.000000` except the two
  `bf16,V=turbo3` versus `V=turbo4` comparisons, which are `0.750973`.
- Friedman within-fold permutation p-value across the four measured policies is
  `0.781022`.
- MMLU-STEM slice: `0.7031` accuracy for all four measured policies.
- GSM8K numeric-MCQ slice: `0.3906` to `0.4219` accuracy depending on K/V policy.
- `K=q8_0,V=turbo8` and `K=bf16,V=turbo8` remain unsupported by the installed
  llama.cpp runtime (`Unsupported cache type: turbo8`).

## Verification

- `uv run --with datasets python scripts\build_external_heldout_cases.py --out-dir _docs\assets\2026-05-20-lm-eval-gguf-kv-cv-128\heldout --gsm8k-limit 64 --mmlu-limit 64`
- `uv run --with lm-eval --with requests --with datasets --with jsonlines --with matplotlib python scripts\run_lm_eval_gguf_kv_cv.py --sources _docs\assets\2026-05-20-lm-eval-gguf-kv-cv-128\heldout\mmlu_stem_external_heldout.jsonl,_docs\assets\2026-05-20-lm-eval-gguf-kv-cv-128\heldout\gsm8k_external_heldout.jsonl --out-dir _docs\assets\2026-05-20-lm-eval-gguf-kv-cv-128 --max-cases 128 --folds 8 --ctx-size 1024 --ngl 999 --startup-timeout-sec 300 --request-timeout-sec 900`
- `.\.venv\Scripts\python.exe -m py_compile scripts\run_lm_eval_gguf_kv_cv.py tests\test_lm_eval_gguf_kv_cv.py`
- `uv run pytest -q tests\test_lm_eval_gguf_kv_cv.py`
- Hugging Face upload/readback verified `zapabobouj/qwen3.5-elt-l3` at
  `3418838249b0601f78d1f74f5300ffa31dc0c11f` with the updated README, 128-case
  `eval/lm_eval_*` files, and `eval/gptimage2_lm_eval_gguf_kv_cv.png`.

## Notes for next session

The 128-case gate now satisfies the requested case scale and includes
GSM8K/MMLU-family external heldouts, but GSM8K is a numeric-MCQ transformation.
Keep this boundary explicit until a true generation/exact-match lm-eval GSM8K
adapter is added.
