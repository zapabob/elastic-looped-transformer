# 2026-05-21 goal completion audit

This audit maps the requested evidence gate to current repo, GitHub, and
Hugging Face artifacts. It is intentionally evidence-bounded: unsupported
runtime values and non-standard benchmark transforms are reported as such.

## Requirement status

| requirement | status | evidence |
|---|---|---|
| K in `bf16` or `q8_0`, V in `turbo3`, `turbo4`, `turbo8` | complete with unsupported `turbo8` boundary | 128-case lm-eval CV measured `bf16/q8_0` x `turbo3/turbo4`; `turbo8` was probed and rejected by llama.cpp as `Unsupported cache type: turbo8`. |
| Triality rotation orthogonality and determinant stability for 3/4/8-bit targets | complete | SO(8) audit passed `4608` rows with `0` outliers; max orthogonality error `6.269e-03`, max determinant error `7.628e-03`, under the `0.01` gate. |
| Training divergence prevention and self-distillation entropy monitoring | complete as monitoring evidence | ILSD L2/L3 metric scan found no non-finite loss, distance, or entropy values; L3 distance remains a watch surface, not a convergence guarantee. |
| GGUF artifacts from BF16 through compressed release forms | complete with naming boundary | BF16, Q8_0, and TQ4_1S GGUF release artifacts are present and uploaded. `turbo3` is a runtime KV-cache policy in this stack, not a standalone weight GGUF file type. |
| HF and GH upload | complete | GitHub `main` was pushed at `1e08979`; HF repo `zapabobouj/qwen3.5-elt-l3` readback verified after uploading the README and audit bundle. The final Hub SHA is reported by the release command output because it changes whenever the audit files themselves are refreshed. |
| gptimage2-style figure output | complete | Two PNG dashboards exist: KV/Triality/entropy CV and 128-case lm-eval K/V CV. |
| README with summary statistics, error bars, p-values, and lm-eval-harness cross-validation multi-group comparison | complete | README now includes the 128-case lm-eval table, error-bar chart, pairwise p-values, Friedman p-value, benchmark slice notes, and links to raw artifacts. |

## Key numbers

### lm-eval K/V CV, 128 external heldout cases

Task: `elt_external_heldout_letter_cv`; folds: `8`; cases: `128`; heldout:
`64` MMLU-STEM MCQ rows plus `64` GSM8K test rows converted to numeric MCQ.
Runtime: `llama-server --ngl 999` with Q8_0 GGUF.

| policy | folds | accuracy mean +/- SEM | 95% CI | status |
|---|---:|---:|---:|---|
| `K=q8_0_V=turbo3` | 8 | 0.5547 +/- 0.0630 | [0.4312, 0.6781] | ok |
| `K=bf16_V=turbo3` | 8 | 0.5625 +/- 0.0765 | [0.4125, 0.7125] | ok |
| `K=q8_0_V=turbo4` | 8 | 0.5469 +/- 0.0763 | [0.3973, 0.6965] | ok |
| `K=bf16_V=turbo4` | 8 | 0.5469 +/- 0.0763 | [0.3973, 0.6965] | ok |
| `K=q8_0_V=turbo8` | 0 | n/a | n/a | unsupported |
| `K=bf16_V=turbo8` | 0 | n/a | n/a | unsupported |

Pairwise within-fold p-values over measured policies: `1.000000`,
`1.000000`, `1.000000`, `0.750973`, `0.750973`, `1.000000`. Four-policy
Friedman within-fold permutation p-value: `0.781022`.

Benchmark slices: MMLU-STEM accuracy is `0.7031` for all four measured
policies. GSM8K numeric-MCQ accuracy ranges from `0.3906` to `0.4219`. This is
not a standard GSM8K exact-match generation claim.

### Loop-aware L1/L2/L3 CV

| group | n | mean | SEM | 95% CI |
|---|---:|---:|---:|---|
| `L1` | 32 | 0.4375 | 0.0891 | [0.2629, 0.6121] |
| `L2` | 32 | 0.5625 | 0.0891 | [0.3879, 0.7371] |
| `L3` | 32 | 0.6562 | 0.0853 | [0.4891, 0.8234] |

Paired p-values: L1-L2 `0.122488`, L1-L3 `0.016098`, L2-L3 `0.254775`.
Friedman within-block permutation p-value: `0.002500`.

## Artifact index

- `_docs/assets/2026-05-20-kv-triality-goal/kv_triality_goal_report.md`
- `_docs/assets/2026-05-20-kv-triality-goal/kv_triality_goal_report.json`
- `_docs/assets/2026-05-20-kv-triality-goal/gptimage2_kv_triality_goal_dashboard.png`
- `_docs/assets/2026-05-20-lm-eval-gguf-kv-cv-128/lm_eval_gguf_kv_cv_report.md`
- `_docs/assets/2026-05-20-lm-eval-gguf-kv-cv-128/lm_eval_gguf_kv_cv_report.json`
- `_docs/assets/2026-05-20-lm-eval-gguf-kv-cv-128/gptimage2_lm_eval_gguf_kv_cv.png`
- `H:/elt_data/hf_publish/qwen3.5-elt-l3/README.md`
- `H:/elt_data/hf_publish/qwen3.5-elt-l3/eval/lm_eval_gguf_kv_cv_report.md`

## Remaining claim boundaries

- `turbo8` is evaluated only as an unsupported runtime setting because the
  installed llama.cpp build rejects it before serving.
- GSM8K rows are numeric multiple-choice transforms for the current logprob
  serving adapter, not standard generation exact-match.
- Stock GGUF execution still does not prove native loop-aware `L>=2` quality;
  loop-aware quality evidence is from the PyTorch/HF runtime and paired local
  STEM bridge CV.
