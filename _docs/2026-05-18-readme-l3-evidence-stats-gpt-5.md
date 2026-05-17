# 2026-05-18 README L3 Evidence Stats - gpt-5

## Goal

Rewrite the README with the latest L3 Qwen3.5 ELT benchmark evidence: summary
statistics, error-bar graph, and p-values, then commit and push the scoped
documentation update.

## Files touched

- `README.md`
- `scripts/render_l3_readme_stats.py`
- `_docs/assets/2026-05-17-l3-thetom-k-protected/l3_readme_accuracy_errorbars.png`
- `_docs/assets/2026-05-17-l3-thetom-k-protected/l3_readme_stats.json`
- `_docs/assets/2026-05-17-l3-thetom-k-protected/l3_readme_stats.md`
- `_docs/2026-05-18-readme-l3-evidence-stats-gpt-5.md`

## Key decisions

- Kept the README claim boundary narrow: local STEM bridge is strong, MMLU-STEM
  is a small cached heldout slice, loop depth helps in the loop-aware HF
  runtime, and GSM8K is explicitly not solved.
- Used Wilson 95% confidence intervals from the existing evaluator summaries.
- Used two-sided Fisher exact tests for independent local/external
  correct/incorrect comparisons.
- Used exact paired McNemar/binomial tests for the loop-aware L=1, L=2, L=3
  case-id matched rows.
- Committed only README/documentation/statistical-reporting assets needed for
  the public rewrite, leaving unrelated code/runtime changes local.

## Verification

- `.\.venv\Scripts\python3.exe -m py_compile scripts\render_l3_readme_stats.py`
- `.\.venv\Scripts\python3.exe scripts\render_l3_readme_stats.py`
- Visual inspection of `l3_readme_accuracy_errorbars.png`

## Next-session notes

- The p-values are intentionally framed as descriptive evidence because the
  external heldout slices are small. Do not upgrade them to leaderboard claims
  until broader external evals complete on the same artifact.
