## Goal

Wire a user-provided Qwen3.5-9B GGUF teacher into the existing `elt-gguf-distill`
pipeline without disturbing the existing Huihui teacher run, and verify the path
with a smoke execution.

## Files Touched

- `configs/gguf_distill_qwen35_9b_hauhaucs.yaml`
- `_docs/2026-04-22-qwen35-gguf-distill-wiring-gpt-5.md`

## Key Decisions

- Added a separate GGUF distill config instead of modifying the existing
  Huihui config, so both teacher routes can coexist.
- Chose port `8092` to avoid colliding with the existing teacher launch
  conventions that may already use `8091`.
- Left `pipeline.repo_id` blank to avoid accidental Hub uploads during
  initial smoke testing.
- Kept the existing detection-domain task layout so the new teacher can be
  compared against the previous GGUF distill flow without changing downstream
  schema expectations.

## Verification

- `uv run elt-gguf-distill --config configs/gguf_distill_qwen35_9b_hauhaucs.yaml --dry-run --max-tasks 4 --skip-upload --skip-student-eval`
- `uv run elt-gguf-distill --config configs/gguf_distill_qwen35_9b_hauhaucs.yaml --output-dir runs/gguf_distill_qwen35_smoke --max-tasks 1 --skip-upload --skip-student-eval`

Smoke result:

- `1/1` valid JSON records
- `schema_valid_rate = 1.0`
- artifacts emitted under `runs/gguf_distill_qwen35_smoke/`

## Next Session Notes

- If the user wants this teacher promoted to a full run, reuse
  `configs/gguf_distill_qwen35_9b_hauhaucs.yaml` and choose whether to set a
  non-empty `pipeline.repo_id` for dataset upload.
- `H:/from_D/webdataset` still contains many large directories that are not
  referenced by the current ingest path; deletion should be explicit and
  conservative because some subtrees are still part of the repo's planned data
  flow.
