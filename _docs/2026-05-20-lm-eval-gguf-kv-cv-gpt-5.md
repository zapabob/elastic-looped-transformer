# 2026-05-20 lm-eval GGUF K/V CV - GPT-5

## Goal

Add a logged `lm-eval-harness` evidence gate for the Qwen3.5 ELT L3 GGUF
serving surface: K cache `q8_0`/`bf16`, V cache `turbo3`/`turbo4`/`turbo8`,
external MMLU-STEM heldout rows, fold-level error bars, paired p-values, and a
gptimage2-style publication figure for README/Hugging Face.

## Files touched

- `scripts/run_lm_eval_gguf_kv_cv.py`
- `tests/test_lm_eval_gguf_kv_cv.py`
- `README.md`
- `H:/elt_data/hf_publish/qwen3.5-elt-l3/README.md`
- `_docs/assets/2026-05-20-lm-eval-gguf-kv-cv/*`

## Key decisions

- Used an isolated `uv run --with lm-eval ...` environment because the repo
  environment had incompatible global `lm_eval`/`transformers` imports.
- Kept lm-eval's evaluator/task pipeline, but added a small local `LM` adapter
  that scores A/B/C/D continuations through llama.cpp `/completion` top-logprob
  responses. The installed OpenAI-compatible endpoint exposes a newer logprob
  schema that stock lm-eval GGUF adapters do not parse.
- Treated `turbo8` as a runtime/parser support probe. The installed
  `llama-server` exits with `Unsupported cache type: turbo8`, so no throughput
  or quality result is claimed for Turbo8.
- Kept the claim narrow: this is a 16-case external MMLU-STEM serving-surface
  CV gate, not a broad lm-eval leaderboard run and not proof of loop-aware
  `L>=2` GGUF execution.

## Results

- Successful policies: `K=q8_0,V=turbo3`, `K=bf16,V=turbo3`,
  `K=q8_0,V=turbo4`, and `K=bf16,V=turbo4`.
- Each successful policy scored `0.6250 +/- 0.1614` accuracy mean +/- SEM over
  four deterministic folds, with 95% CI `[0.3087, 0.9413]`.
- All successful policy predictions matched exactly on this small slice:
  pairwise paired permutation p-values are `1.000000`; Friedman within-fold
  permutation p=`1.000000`.
- `K=q8_0,V=turbo8` and `K=bf16,V=turbo8` are recorded as unsupported with
  stderr detail `Unsupported cache type: turbo8`.

## Verification

- `.\.venv\Scripts\python.exe -m py_compile scripts\run_lm_eval_gguf_kv_cv.py tests\test_lm_eval_gguf_kv_cv.py`
- `uv run pytest -q tests\test_lm_eval_gguf_kv_cv.py tests\test_kv_triality_goal_report.py`
- `uv run --with lm-eval --with requests --with datasets --with jsonlines --with matplotlib python scripts\run_lm_eval_gguf_kv_cv.py --max-cases 16 --folds 4 --ngl 999 --startup-timeout-sec 180 --request-timeout-sec 180`
- Hugging Face upload/readback verified `zapabobouj/qwen3.5-elt-l3` at
  `a7f31985d71d1634673e42a545631d27235d0e61` with the new `eval/lm_eval_*`
  files, redacted local execution paths, and
  `eval/gptimage2_lm_eval_gguf_kv_cv.png`.

## Notes for next session

The next evidence gate should scale beyond this 16-case slice, preferably with
separate GSM8K/MMLU tasks through a stable lm-eval model adapter and a larger
case budget. Do not collapse these serving-surface KV-cache results with the
loop-aware HF runtime L-depth quality results.
