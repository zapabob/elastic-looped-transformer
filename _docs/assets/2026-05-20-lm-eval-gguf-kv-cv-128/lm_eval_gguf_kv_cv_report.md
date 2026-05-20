# lm-eval-harness GGUF K/V CV report

- Task: `elt_external_heldout_letter_cv`
- Cases: `128`; folds: `8`
- Benchmarks: `gsm8k_numeric_mcq, mmlu_stem`
- Model: `elt-lm-qwen35-side-stem-aha-ilsd-l3-Q8_0.gguf`
- Runtime: llama-server `--ngl 999` with OpenAI-compatible `/completion` logprobs through lm-eval.

| policy | folds | accuracy mean +/- SEM | 95% CI | status |
|---|---:|---:|---:|---|
| `K=q8_0_V=turbo3` | 8 | 0.5547 +/- 0.0630 | [0.4312, 0.6781] | `ok` |
| `K=bf16_V=turbo3` | 8 | 0.5625 +/- 0.0765 | [0.4125, 0.7125] | `ok` |
| `K=q8_0_V=turbo4` | 8 | 0.5469 +/- 0.0763 | [0.3973, 0.6965] | `ok` |
| `K=bf16_V=turbo4` | 8 | 0.5469 +/- 0.0763 | [0.3973, 0.6965] | `ok` |
| `K=q8_0_V=turbo8` | 0 | n/a | n/a | `error while handling argument "--cache-type-v": Unsupported cache type: turbo8` |
| `K=bf16_V=turbo8` | 0 | n/a | n/a | `error while handling argument "--cache-type-v": Unsupported cache type: turbo8` |

| comparison | mean delta | p | method |
|---|---:|---:|---|
| `K=q8_0_V=turbo3` - `K=bf16_V=turbo3` | -0.0078 | 1.000000 | `paired_permutation_10000` |
| `K=q8_0_V=turbo3` - `K=q8_0_V=turbo4` | 0.0078 | 1.000000 | `paired_permutation_10000` |
| `K=q8_0_V=turbo3` - `K=bf16_V=turbo4` | 0.0078 | 1.000000 | `paired_permutation_10000` |
| `K=bf16_V=turbo3` - `K=q8_0_V=turbo4` | 0.0156 | 0.750973 | `paired_permutation_10000` |
| `K=bf16_V=turbo3` - `K=bf16_V=turbo4` | 0.0156 | 0.750973 | `paired_permutation_10000` |
| `K=q8_0_V=turbo4` - `K=bf16_V=turbo4` | 0.0000 | 1.000000 | `paired_permutation_10000` |

Friedman within-fold permutation p: `0.781022` (statistic `0.7125`, n=8).

## Benchmark slice: `gsm8k_numeric_mcq`

| policy | folds | accuracy mean +/- SEM | 95% CI |
|---|---:|---:|---:|
| `K=q8_0_V=turbo3` | 8 | 0.4062 +/- 0.1199 | [0.1713, 0.6412] |
| `K=bf16_V=turbo3` | 8 | 0.4219 +/- 0.1376 | [0.1521, 0.6916] |
| `K=q8_0_V=turbo4` | 8 | 0.3906 +/- 0.1345 | [0.1269, 0.6543] |
| `K=bf16_V=turbo4` | 8 | 0.3906 +/- 0.1345 | [0.1269, 0.6543] |

Friedman p: `0.421058`.

## Benchmark slice: `mmlu_stem`

| policy | folds | accuracy mean +/- SEM | 95% CI |
|---|---:|---:|---:|
| `K=q8_0_V=turbo3` | 8 | 0.7031 +/- 0.0576 | [0.5903, 0.8159] |
| `K=bf16_V=turbo3` | 8 | 0.7031 +/- 0.0525 | [0.6002, 0.8060] |
| `K=q8_0_V=turbo4` | 8 | 0.7031 +/- 0.0576 | [0.5903, 0.8159] |
| `K=bf16_V=turbo4` | 8 | 0.7031 +/- 0.0576 | [0.5903, 0.8159] |

Friedman p: `1.000000`.

Scope: external-heldout letter-choice CV. MMLU-STEM rows are native MCQ; GSM8K rows are numeric-MCQ transformations of GSM8K test examples. This is a logged lm-eval-harness serving-surface gate, not a broad leaderboard result.
