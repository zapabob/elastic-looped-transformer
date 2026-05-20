# 2026-05-21 best BF16 GGUF selection

## Selection

Selected BF16 release candidate: `elt-lm-qwen35-side-stem-aha-ilsd-l3-BF16.gguf`
with cache policy `K=bf16,V=turbo3`.

Rationale: in the headless 128-case / 8-fold `lm-eval-harness` CV, the BF16
GGUF tied for the highest overall mean accuracy at `0.5547 +/- 0.0662` with
`K=bf16,V=turbo3` and `K=q8_0,V=turbo3`. The selected policy keeps the user
requested BF16 K-cache path and uses the best observed V-cache tier. Pairwise
permutation tests did not show a significant separation among measured groups
(`p >= 0.750973`; Friedman p `0.943806`), so this is a best-evidence selection,
not a statistically dominant winner.

## Headless lm-eval result

Task: `elt_external_heldout_letter_cv`; cases: `128`; folds: `8`; benchmarks:
`64` MMLU-STEM native MCQ rows and `64` GSM8K numeric-MCQ rows. Runtime:
`llama-server --ngl 999`; model: `elt-lm-qwen35-side-stem-aha-ilsd-l3-BF16.gguf`.

| policy | folds | accuracy mean +/- SEM | 95% CI | status |
|---|---:|---:|---:|---|
| `K=bf16,V=turbo3` | 8 | 0.5547 +/- 0.0662 | [0.4249, 0.6845] | selected |
| `K=q8_0,V=turbo3` | 8 | 0.5547 +/- 0.0662 | [0.4249, 0.6845] | tied |
| `K=bf16,V=turbo4` | 8 | 0.5469 +/- 0.0754 | [0.3991, 0.6947] | ok |
| `K=q8_0,V=turbo4` | 8 | 0.5391 +/- 0.0708 | [0.4003, 0.6778] | ok |
| `K=bf16,V=turbo8` | 0 | n/a | n/a | unsupported |
| `K=q8_0,V=turbo8` | 0 | n/a | n/a | unsupported |

`K=q8_0,V=turbo4` timed out during the first grouped startup, then completed
successfully in a single-policy headless retry with a longer startup timeout.
The initial timeout log and successful retry log are both preserved under
`raw/`.

## Claim boundaries

- `turbo8` is not a measured quality result because this llama.cpp build rejects
  it as `Unsupported cache type: turbo8`.
- GSM8K is represented as deterministic numeric multiple-choice rows, not
  standard exact-match generation.
- This evaluates the stock GGUF serving surface; native loop-aware `L>=2`
  semantics remain a separate runtime boundary.
