# lm-eval-harness GGUF K/V CV report

- Task: `elt_mmlu_stem_external_letter_cv`
- Cases: `16`; folds: `4`
- Model: `elt-lm-qwen35-side-stem-aha-ilsd-l3-Q8_0.gguf`
- Runtime: llama-server `--ngl 999` with OpenAI-compatible `/completion` logprobs through lm-eval.

| policy | folds | accuracy mean +/- SEM | 95% CI | status |
|---|---:|---:|---:|---|
| `K=q8_0_V=turbo3` | 4 | 0.6250 +/- 0.1614 | [0.3087, 0.9413] | `ok` |
| `K=bf16_V=turbo3` | 4 | 0.6250 +/- 0.1614 | [0.3087, 0.9413] | `ok` |
| `K=q8_0_V=turbo4` | 4 | 0.6250 +/- 0.1614 | [0.3087, 0.9413] | `ok` |
| `K=bf16_V=turbo4` | 4 | 0.6250 +/- 0.1614 | [0.3087, 0.9413] | `ok` |
| `K=q8_0_V=turbo8` | 0 | n/a | n/a | `error while handling argument "--cache-type-v": Unsupported cache type: turbo8` |
| `K=bf16_V=turbo8` | 0 | n/a | n/a | `error while handling argument "--cache-type-v": Unsupported cache type: turbo8` |

| comparison | mean delta | p | method |
|---|---:|---:|---|
| `K=q8_0_V=turbo3` - `K=bf16_V=turbo3` | 0.0000 | 1.000000 | `paired_permutation_10000` |
| `K=q8_0_V=turbo3` - `K=q8_0_V=turbo4` | 0.0000 | 1.000000 | `paired_permutation_10000` |
| `K=q8_0_V=turbo3` - `K=bf16_V=turbo4` | 0.0000 | 1.000000 | `paired_permutation_10000` |
| `K=bf16_V=turbo3` - `K=q8_0_V=turbo4` | 0.0000 | 1.000000 | `paired_permutation_10000` |
| `K=bf16_V=turbo3` - `K=bf16_V=turbo4` | 0.0000 | 1.000000 | `paired_permutation_10000` |
| `K=q8_0_V=turbo4` - `K=bf16_V=turbo4` | 0.0000 | 1.000000 | `paired_permutation_10000` |

Friedman within-fold permutation p: `1.000000` (statistic `0.0000`, n=4).

Scope: small external MMLU-STEM letter-choice CV. This is a logged lm-eval-harness serving-surface gate, not a broad leaderboard result.
