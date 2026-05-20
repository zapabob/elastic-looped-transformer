# lm-eval-harness GGUF K/V CV report

- Task: `elt_external_heldout_letter_cv`
- Cases: `128`; folds: `8`
- Benchmarks: `gsm8k_numeric_mcq, mmlu_stem`
- Model: `elt-lm-qwen35-side-stem-aha-ilsd-l3-BF16.gguf`
- Runtime: llama-server `--ngl 999` with OpenAI-compatible `/completion` logprobs through lm-eval.

| policy | folds | accuracy mean +/- SEM | 95% CI | status |
|---|---:|---:|---:|---|
| `K=bf16_V=turbo3` | 8 | 0.5547 +/- 0.0662 | [0.4249, 0.6845] | `ok` |
| `K=q8_0_V=turbo3` | 8 | 0.5547 +/- 0.0662 | [0.4249, 0.6845] | `ok` |
| `K=bf16_V=turbo4` | 8 | 0.5469 +/- 0.0754 | [0.3991, 0.6947] | `ok` |
| `K=q8_0_V=turbo4` | 8 | 0.5391 +/- 0.0708 | [0.4003, 0.6778] | `ok` |
| `K=bf16_V=turbo8` | 0 | n/a | n/a | `error while handling argument "--cache-type-v": Unsupported cache type: turbo8` |
| `K=q8_0_V=turbo8` | 0 | n/a | n/a | `error while handling argument "--cache-type-v": Unsupported cache type: turbo8` |

| comparison | mean delta | p | method |
|---|---:|---:|---|
| `K=bf16_V=turbo3` - `K=q8_0_V=turbo3` | 0.0000 | 1.000000 | `paired_permutation_10000` |
| `K=bf16_V=turbo3` - `K=bf16_V=turbo4` | 0.0078 | 1.000000 | `paired_permutation_10000` |
| `K=bf16_V=turbo3` - `K=q8_0_V=turbo4` | 0.0156 | 0.750973 | `paired_permutation_10000` |
| `K=q8_0_V=turbo3` - `K=bf16_V=turbo4` | 0.0078 | 1.000000 | `paired_permutation_10000` |
| `K=q8_0_V=turbo3` - `K=q8_0_V=turbo4` | 0.0156 | 0.782101 | `paired_permutation_10000` |
| `K=bf16_V=turbo4` - `K=q8_0_V=turbo4` | 0.0078 | 1.000000 | `paired_permutation_10000` |

Friedman within-fold permutation p: `0.943806` (statistic `0.4875`, n=8).

## Benchmark slice: `gsm8k_numeric_mcq`

| policy | folds | accuracy mean +/- SEM | 95% CI |
|---|---:|---:|---:|
| `K=bf16_V=turbo3` | 8 | 0.4062 +/- 0.1199 | [0.1713, 0.6412] |
| `K=q8_0_V=turbo3` | 8 | 0.3906 +/- 0.1260 | [0.1437, 0.6375] |
| `K=bf16_V=turbo4` | 8 | 0.4062 +/- 0.1432 | [0.1256, 0.6869] |
| `K=q8_0_V=turbo4` | 8 | 0.3906 +/- 0.1345 | [0.1269, 0.6543] |

Friedman p: `0.955004`.

## Benchmark slice: `mmlu_stem`

| policy | folds | accuracy mean +/- SEM | 95% CI |
|---|---:|---:|---:|
| `K=bf16_V=turbo3` | 8 | 0.7031 +/- 0.0525 | [0.6002, 0.8060] |
| `K=q8_0_V=turbo3` | 8 | 0.7188 +/- 0.0566 | [0.6077, 0.8298] |
| `K=bf16_V=turbo4` | 8 | 0.6875 +/- 0.0528 | [0.5840, 0.7910] |
| `K=q8_0_V=turbo4` | 8 | 0.6875 +/- 0.0528 | [0.5840, 0.7910] |

Friedman p: `0.765023`.

Scope: external-heldout letter-choice CV. MMLU-STEM rows are native MCQ; GSM8K rows are numeric-MCQ transformations of GSM8K test examples. This is a logged lm-eval-harness serving-surface gate, not a broad leaderboard result.
