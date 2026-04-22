---
date: 2026-04-22
slug: corpus-weight-manifest-dedup
ai: gpt-5
---

# Corpus Manifest Dedup + Deterministic Source Weighting

## Overview

Applied the first two data-quality improvements from the earlier review:

1. remove duplicated entries from `scripts/corpus_manifest_clean.yaml`
2. make `weight` in `scripts/build_train_bin.py` affect corpus contribution deterministically

Semantic dedup was intentionally left for a later stage.

## Background / requirements

- The cleaned corpus manifest had duplicated path blocks, which would cause already-cleaned sources to be reintroduced during token-bin generation.
- `Source.weight` existed in `scripts/build_train_bin.py` but had no effect on emitted documents.
- The repo convention for this task family is to keep changes small, verified, and documented under `_docs/`.

## Assumptions / decisions

- Chose deterministic source weighting instead of RNG-based sampling so repeated corpus builds remain reproducible for the same manifest order.
- Implemented weighting with a Bresenham-style accumulator:
  - `weight=1.0` keeps each source document once
  - `0 < weight < 1` keeps a stable subset
  - `weight > 1` repeats documents in a stable spread
- Left semantic dedup out of scope for this pass to avoid mixing a larger design change into a manifest/packing fix.

## Changed files

- `scripts/corpus_manifest_clean.yaml`
- `scripts/build_train_bin.py`
- `tests/test_build_train_bin.py`

## Implementation details

- Removed the duplicated trailing block from `scripts/corpus_manifest_clean.yaml`.
  - Verified after the change: `dup_paths=0 total_entries=45 unique_entries=45`.
- Added `iter_weighted_texts()` to `scripts/build_train_bin.py`.
  - The helper deterministically scales per-source document emission based on `weight`.
  - `iter_source()` now yields weighted source texts instead of ignoring `Source.weight`.
- Added targeted tests covering:
  - clean manifest uniqueness
  - deterministic downsampling at `weight=0.5`
  - deterministic upsampling at `weight=2.5`

## Commands run

```powershell
uv run pytest -q tests/test_build_train_bin.py
uv run pytest -q tests/test_build_train_bin.py tests/test_posttrain_data.py tests/test_posttrain_configs.py
$lines = Get-Content scripts\corpus_manifest_clean.yaml | Where-Object { $_ -match '^  - path:' } | ForEach-Object { ($_ -replace '^  - path:\s*','').Trim() }; $groups = $lines | Group-Object; $dupGroups = $groups | Where-Object { $_.Count -gt 1 }; "dup_paths=$($dupGroups.Count) total_entries=$($lines.Count) unique_entries=$($groups.Count)"
```

## Test / verification results

- `uv run pytest -q tests/test_build_train_bin.py` -> passed (`3 passed`)
- `uv run pytest -q tests/test_build_train_bin.py tests/test_posttrain_data.py tests/test_posttrain_configs.py` -> passed (`9 passed`)
- clean manifest duplicate check -> `dup_paths=0 total_entries=45 unique_entries=45`

## Residual risks

- Deterministic weighting is source-order-sensitive by design; reordering manifest entries changes which documents are retained for fractional weights.
- No semantic near-duplicate removal was added yet, so paraphrase-level overlap still remains in the cleaned corpus.
- No source weights were introduced into the manifests in this pass; only the implementation path is now active.

## Recommended next actions

1. Decide target `weight` values for major source families (`math`, `code`, `tool`, `agent`, `chat`, local corpora).
2. Add a small manifest-lint test or script that rejects duplicated `path` entries in all corpus manifests.
3. Implement semantic dedup as a separate stage after cleaning and before token packing.
