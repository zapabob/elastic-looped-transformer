---
date: 2026-04-22
slug: webdataset-ingest-clean-inventory
ai: gpt-5
---

# WebDataset Ingest + Clean + Token Inventory

## Goal

Verify whether `H:/from_D/webdataset` is actually contributing to the current
training corpus, materialize `H:/elt_data/clean`, and replace planning-only
token estimates with measured source-by-source token counts.

## Files Touched

Generated artifacts only:

- `H:/elt_data/raw/webdataset_alpaca_ja.jsonl`
- `H:/elt_data/raw/webdataset_sharegpt_ja.jsonl`
- `H:/elt_data/raw/webdataset_fujiki_ja.jsonl`
- `H:/elt_data/raw/webdataset_wizard_vicuna.jsonl`
- `H:/elt_data/raw/webdataset_coding.jsonl`
- `H:/elt_data/raw/webdataset_coding_train.jsonl`
- `H:/elt_data/raw/webdataset_domain_knowledge.jsonl`
- `H:/elt_data/raw/webdataset_phi35.jsonl`
- `H:/elt_data/clean/*.jsonl`
- `runs/dashboard_runtime/source_token_inventory_2026-04-22.json`
- `runs/dashboard_runtime/source_token_inventory_2026-04-22.csv`

## Key Results

### 1. `H:/from_D/webdataset` is wired into the pipeline, but only partially materialized

- `scripts/pipeline.py` already routes `H:/from_D/webdataset` through
  `scripts/ingest_webdataset.py` before cleaning/tokenization.
- Before this session, `H:/elt_data/raw` only contained:
  - `camel_sci.jsonl`
  - `tulu3.jsonl`
  - `webdataset_integrated.jsonl`
- So the planned WebDataset-derived sources were mostly *not* present on disk.

### 2. Actual ingest counts from `H:/from_D/webdataset`

- `webdataset_alpaca_ja`: `49,963` docs
- `webdataset_sharegpt_ja`: `5,911` docs
- `webdataset_fujiki_ja`: `0` docs
- `webdataset_wizard_vicuna`: `34,598` docs
- `webdataset_coding`: `2,626` docs
- `webdataset_coding_train`: `1,524` docs
- `webdataset_domain_knowledge`: `62` docs
- `webdataset_phi35`: `86,126` docs

Warnings:

- `webdataset_oasst2`: source file was not actually gzipped
- `webdataset_slimpajama`: skipped because `zstandard` is not installed

### 3. Clean corpus was materialized

- `scripts/clean_corpus.py` completed successfully.
- Final deduplicated unique docs: `440,800`
- Free space on `H:` after the run: `52.41 GB`

### 4. Measured token inventory

- Tokenizer: `H:/Qwen3.5-9B-official-hf`
- EOS appended per document
- Total docs: `440,800`
- Total tokens: `411,487,426`

Top contributors by token count:

1. `tulu3.jsonl` -> `131,434,716`
2. `aegis_local.jsonl` -> `114,444,736`
3. `webdataset_integrated.jsonl` -> `79,050,529`
4. `webdataset_wizard_vicuna.jsonl` -> `28,922,446`
5. `webdataset_phi35.jsonl` -> `23,803,305`
6. `camel_sci.jsonl` -> `12,999,970`
7. `webdataset_alpaca_ja.jsonl` -> `8,882,298`
8. `webdataset_sharegpt_ja.jsonl` -> `6,653,426`

## Important Findings

1. The earlier `~5-6B dedup tokens` and `7.9B planned tokens` numbers were not
   describing the currently materialized corpus on disk.
2. The currently measured clean corpus is only `411.5M` tokens.
3. Several manifest entries are still missing in raw form, so the corpus plan
   and the corpus reality remain out of sync.
4. `clean_corpus.py` creates `*.jsonl.jsonl` placeholders when a manifest file
   path is missing, because the missing path is treated as a non-existent file
   rather than an existing file stem. This does not break counting, but it is a
   cleanup signal for future sessions.

## Commands Run

```powershell
uv run python scripts/ingest_webdataset.py --src H:/from_D/webdataset --out H:/elt_data/raw --mode pretrain
uv run python scripts/clean_corpus.py --config scripts/corpus_manifest.yaml --out H:/elt_data/clean
```

Token inventory was generated with an inline tokenizer-count script and written
to:

- `runs/dashboard_runtime/source_token_inventory_2026-04-22.json`
- `runs/dashboard_runtime/source_token_inventory_2026-04-22.csv`

## Next Session Notes

1. Recompute `tok/param` and pass-count planning from `411.5M` real tokens, not
   from the earlier planning estimate.
2. Decide whether to:
   - install `zstandard` and re-run `webdataset_slimpajama`,
   - fix the OASST2 file/path issue,
   - refresh missing HF/raw sources so the manifest matches reality.
3. If using teacher distillation to compensate for the small materialized
   corpus, treat it as a post-train/SFT lane rather than a replacement for
   foundational pretraining coverage.
