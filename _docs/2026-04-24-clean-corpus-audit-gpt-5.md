---
date: 2026-04-24
slug: clean-corpus-audit
ai: gpt-5
---

# Clean Corpus Audit

## Goal

Recheck `H:/elt_data/clean` for duplicate and low-quality residue before the next
native ELT continued pretraining and Huihui/Qwen lane-distill stages.

## Summary

- files: 45
- zero-byte files: 31
- bytes: 1,609,002,877
- docs scanned: 440,800
- parse errors: 0
- empty text records: 0
- low-quality records: 0
- exact duplicate records: 0
- prefix duplicate records: 0
- simhash duplicate records: 38,491

## Top Quality Reasons

- none

## Files Touched

- `scripts/audit_clean_corpus.py`
- `tests/test_audit_clean_corpus.py`
- `runs/dashboard_runtime/clean_corpus_audit_2026-04-24.json`
- `_docs/2026-04-24-clean-corpus-audit-gpt-5.md`
- `H:/elt_data/bin_clean_2026-04-24/train.bin`
- `H:/elt_data/bin_clean_2026-04-24/val.bin`

## Tests / Commands

```powershell
uv run --no-sync pytest -q tests/test_audit_clean_corpus.py
uv run --no-sync pyright scripts/audit_clean_corpus.py tests/test_audit_clean_corpus.py
uv run --no-sync python scripts/audit_clean_corpus.py --clean-dir H:/elt_data/clean --report-json runs/dashboard_runtime/clean_corpus_audit_2026-04-24.json --report-md _docs/2026-04-24-clean-corpus-audit-gpt-5.md
uv run --no-sync python scripts/build_train_bin.py --tokenizer H:/Qwen3.5-9B-official-hf --out-dir H:/elt_data/bin_clean_2026-04-24 --config scripts/corpus_manifest_clean.yaml
```

## Zero-Byte Files

- `aegis_offload.jsonl`
- `agent_instruct.jsonl.jsonl`
- `codeact.jsonl.jsonl`
- `codefeedback.jsonl.jsonl`
- `cosmopedia.jsonl.jsonl`
- `finemath.jsonl.jsonl`
- `general_thought.jsonl.jsonl`
- `glaive_tools.jsonl.jsonl`
- `gsm8k.jsonl.jsonl`
- `hermes_tools.jsonl.jsonl`
- `magicoder.jsonl.jsonl`
- `magicoder_evol.jsonl.jsonl`
- `metamath.jsonl.jsonl`
- `opencode_instruct.jsonl.jsonl`
- `opencode_reasoning.jsonl.jsonl`
- `opencoder_sft.jsonl.jsonl`
- `openhermes.jsonl.jsonl`
- `openmath2.jsonl.jsonl`
- `openthoughts.jsonl.jsonl`
- `openwebmath.jsonl.jsonl`
- `orca_agent.jsonl.jsonl`
- `self_oss_instruct.jsonl.jsonl`
- `slim_orca.jsonl.jsonl`
- `swe_gym.jsonl.jsonl`
- `toolace.jsonl.jsonl`
- `webdataset_fujiki_ja.jsonl`
- `webdataset_oasst2.jsonl`
- `webdataset_slimpajama.jsonl`
- `wiki_en.jsonl.jsonl`
- `wiki_ja.jsonl.jsonl`
- `wildchat.jsonl.jsonl`

## Files With Highest Issue Rates

| file | docs | low_quality | prefix_dups | simhash_dups | parse_errors |
|---|---:|---:|---:|---:|---:|
| aegis_offload.jsonl | 0 | 0 | 0 | 0 | 0 |
| agent_instruct.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |
| codeact.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |
| codefeedback.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |
| cosmopedia.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |
| finemath.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |
| general_thought.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |
| glaive_tools.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |
| gsm8k.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |
| hermes_tools.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |
| magicoder.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |
| magicoder_evol.jsonl.jsonl | 0 | 0 | 0 | 0 | 0 |

## Decisions

- This pass is audit-only and does not delete or rewrite corpus files.
- `prefix_dups` uses the same first-512 normalized-character idea as
  `scripts/clean_corpus.py`, so it is compatible with the existing clean stage.
- `simhash_dups` is a high-confidence audit signal only. It should guide a
  later semantic dedup stage, not silently remove records by itself.
- New train/val bins were built into `H:/elt_data/bin_clean_2026-04-24` instead
  of overwriting the existing `H:/elt_data/bin`.

## Next Session Notes

1. Review the 31 zero-byte outputs and decide whether to regenerate their
   sources or remove them from manifests.
2. Treat the 38,491 simhash duplicate hits as semantic-dedup candidates, not as
   automatic removals.
3. Point a native ELT continued-pretraining config at
   `H:/elt_data/bin_clean_2026-04-24/{train,val}.bin`.
