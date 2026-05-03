# 2026-05-03 GGUF quant CV gptimage report - gpt-5

## Goal

Evaluate the release GGUF artifacts (`BF16`, `Q8_0`, `TQ4_1S`) with local
llama.cpp runtime checks, summarize them with SciPy-style paired statistics,
render README-ready error-bar images, and update both GitHub and Hugging Face
README surfaces.

## Files touched

- `src/elt_lm/eval/gguf_quant_report.py`
- `tests/test_gguf_quant_report.py`
- `pyproject.toml`
- `uv.lock`
- `README.md`
- `_docs/assets/2026-05-03-gguf-quant-cv-gptimage/*`
- `H:/elt_data/hf_exports/elt-lm-qwen35-side-stem-v2-bridge-merged/README.md`
- `H:/elt_data/hf_exports/elt-lm-qwen35-side-stem-v2-bridge-merged/gguf_quant_cv/*`

## Key decisions

- Added `elt-gguf-quant-report` as a reproducible CLI that can run
  `llama-bench`, `llama-perplexity`, optional BF16 logits export + KL checks,
  and then rebuild CSV/JSON/Markdown/PNG artifacts from raw logs without
  rerunning the model.
- Kept the README claim boundary explicit: this is a local short-run release
  validation over verifier-backed synthetic-v2 hard text, not a broad external
  lm-eval leaderboard result.
- Stored the large BF16 logits file outside README assets at
  `H:/elt_data/runs/gguf_quant_cv_2026-05-03/bf16_base_logits.bin`.
- Attempted a `q8_0/q8_0` KV-cache bench first; BF16 failed llama.cpp context
  creation on this local runtime, so the paired CV comparison uses the common
  `f16/f16` KV-cache path. The failed attempt remains in
  `_docs/assets/2026-05-03-gguf-quant-cv-gptimage/run.log`.

## Results

- Artifact sizes: BF16 `9.03 GiB`, Q8_0 `4.80 GiB`, TQ4_1S `4.16 GiB`.
- `llama-bench`, `f16/f16` KV, `n=3` repetitions:
  - prompt eval: BF16 `217.15 +/- 13.59`, Q8_0 `248.77 +/- 16.75`,
    TQ4_1S `0.79 +/- 0.02` tok/s (SEM).
  - decode: BF16 `27.83 +/- 0.07`, Q8_0 `40.40 +/- 0.15`,
    TQ4_1S `0.69 +/- 0.03` tok/s (SEM).
  - Friedman p-value: `0.049787` for prompt eval and decode.
- `llama-perplexity` / logits one-chunk release check:
  - PPL: BF16 `11313.67`, Q8_0 `13677.23`, TQ4_1S `21648.79`.
  - KL vs BF16: Q8_0 `1.64035`, TQ4_1S `1.88190`.
  - KV / recurrent state: `1024 / 6432 MiB` for all three formats.

## Tests

- `uv run --extra dev --extra eval pytest -q tests/test_gguf_quant_report.py`
  - `5 passed`
- `uv run --extra dev --extra eval pytest -q tests/test_gguf_quant_report.py tests/test_eval_statistics.py`
  - `12 passed`

## Next session notes

- TQ4_1S loads and evaluates, but this local llama.cpp build is very slow for
  the TQ4_1S prompt/decode path. Treat that as runtime optimization evidence,
  not a model-quality conclusion.
- For publish-grade external claims, run the same paired sample IDs through
  lm-eval-harness or a llama-server-backed harness and feed correctness arrays
  into `elt-compare-benchmarks` or `elt-gguf-quant-report`.
