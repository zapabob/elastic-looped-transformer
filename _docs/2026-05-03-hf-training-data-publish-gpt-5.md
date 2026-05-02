# 2026-05-03 HF training data publish - GPT-5

## Goal

Publish the current code/config changes to GitHub and Hugging Face, and include
the active synthetic-v2-hard training data on Hugging Face with source
provenance.

## Files touched

- `training_data/synthetic_v2_hard/**`
- `training_data/DATA_SOURCES.md`
- `training_data/source_citations.yaml`
- `training_data/synthetic_v2_hard/README.md`
- `README.md`

## Key decisions

- Commit the small synthetic-v2-hard JSONL snapshot directly. It is about 18 MB
  across 39 files, below normal GitHub and Hugging Face practical limits.
- Do not commit large tokenized bins from `H:/elt_data/posttrain_synthetic/*` or
  `H:/elt_data/bin/*`; those are generated artifacts.
- List public Hugging Face corpus sources in a machine-readable citation YAML so
  the HF model card has traceable training-data provenance.

## Tests

Run before this publish step:

```powershell
uv run --no-sync pytest tests/test_qwen35_side_smoke_configs.py tests/test_synthetic_v2_hard.py tests/test_verifiers_quality.py -q
```

Result: 17 passed.

