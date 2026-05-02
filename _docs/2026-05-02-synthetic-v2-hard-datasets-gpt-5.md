# 2026-05-02 synthetic-v2-hard-datasets - gpt-5

## Goal

Create verifier-backed synthetic v2 hard data for ELT/GRPO so reward variance can come from multi-step tasks and intentional failure contrasts instead of only increasing GRPO steps.

## Files Touched

- `src/elt_lm/synthetic_v2_hard.py`
- `configs/grpo_side_lora_code_synthetic_v2_hard.yaml`
- `configs/grpo_side_lora_math_synthetic_v2_hard.yaml`
- `configs/grpo_side_lora_stem_synthetic_v2_hard.yaml`
- `configs/grpo_side_lora_tool_synthetic_v2_hard.yaml`
- `tests/test_synthetic_v2_hard.py`
- `scripts/pipeline.py`
- `tests/test_pipeline_orchestrator.py`
- `pyproject.toml`
- `_docs/2026-05-02-synthetic-v2-hard-datasets-gpt-5.md`

## Key Decisions

- Added four lanes: `code`, `math`, `stem_reasoning`, and `tool_use`.
- Code v2 is `python_exec`-first, with executable verifier snippets and deliberate failure responses that miss validation, merging, batching, or guard logic.
- Math v2 uses exact multi-step arithmetic and recurrence/inclusion-exclusion cases. Failure answers are emitted as decimal shortcuts where needed so the existing exact-math fallback cannot accidentally treat a wrong fraction with the same denominator as correct.
- STEM v2 balances A/B/C/D labels and forces a two-stage interpretation rather than a single-factor distractor.
- Tool-use v2 focuses on safe agentic or MCP calls, read-only/dry-run constraints, and wrong-tool or unsafe-call failures.
- Each lane writes correct SFT records, failure contrast records, and validation benchmark manifests.
- Added a `synthetic-v2-hard` pipeline profile that builds only this dataset and skips only when all lanes meet the record count, verifier pass, and failure-zero gates.
- Added v2-hard side-LoRA GRPO configs with separate run dirs so the current `synthetic_gb` GRPO job can continue undisturbed.
- Added `synthetic-v2-hard-grpo` pipeline profile: build v2 data, run code/math/STEM/tool GRPO, export adapters, then run CV eval on the v2 hard manifests.
- Increased v2-hard GRPO generation budgets relative to the older 128/192-token settings: code 320, math 192, STEM 160, tool 160.

## Artifacts

Generated:

- `H:/elt_data/synthetic_v2_hard/summary.json`
- `H:/elt_data/synthetic_v2_hard/{code,math,stem_reasoning,tool_use}/distill_train.jsonl`
- `H:/elt_data/synthetic_v2_hard/{code,math,stem_reasoning,tool_use}/distill_val.jsonl`
- `H:/elt_data/synthetic_v2_hard/{code,math,stem_reasoning,tool_use}/failures_train.jsonl`
- `H:/elt_data/synthetic_v2_hard/{code,math,stem_reasoning,tool_use}/failures_val.jsonl`
- `H:/elt_data/synthetic_v2_hard/{code,math,stem_reasoning,tool_use}/benchmarks/synthetic_v2_hard_*_val_cases.jsonl`
- `H:/elt_data/synthetic_v2_hard/{code,math,stem_reasoning,tool_use}/benchmarks/synthetic_v2_hard_*_val_manifest.yaml`

Configured next GRPO run dirs:

- `H:/elt_data/runs/grpo_side_lora_code_synthetic_v2_hard`
- `H:/elt_data/runs/grpo_side_lora_math_synthetic_v2_hard`
- `H:/elt_data/runs/grpo_side_lora_stem_synthetic_v2_hard`
- `H:/elt_data/runs/grpo_side_lora_tool_synthetic_v2_hard`

Final generated counts:

- Correct SFT records: 512 total, 128 per lane.
- Failure contrast records: 512 total, 128 per lane.
- Train/val split: 96 train and 32 val per lane.
- Verifier pass rate: 1.0 for all correct records in all lanes.
- Failure expected-zero rate: 1.0 for all lanes.
- Unique text ratio: 1.0 for all lanes.

## Tests

Passing:

- `uv run --no-sync python -m py_compile src\elt_lm\synthetic_v2_hard.py scripts\pipeline.py`
- `uv run --no-sync pytest tests\test_synthetic_v2_hard.py -q` -> 2 passed
- `uv run --no-sync pytest tests\test_pipeline_orchestrator.py tests\test_synthetic_v2_hard.py -q` -> 43 passed
- `uv run --no-sync python scripts\pipeline.py --profile synthetic-v2-hard --dry-run`
- `uv run --no-sync python scripts\pipeline.py --profile synthetic-v2-hard-grpo --dry-run`
- `load_train_config` smoke for all four `configs/grpo_side_lora_*_synthetic_v2_hard.yaml` files, including existing v2 prompt paths.

## Next Session Notes

- Use `uv run --no-sync python -m elt_lm.synthetic_v2_hard --output-root H:/elt_data/synthetic_v2_hard --records-per-lane 128 --val-ratio 0.25` to refresh the current bundle.
- Use `uv run --no-sync python scripts/pipeline.py --profile synthetic-v2-hard` to run the idempotent stage.
- Use `uv run --no-sync python scripts/pipeline.py --profile synthetic-v2-hard-grpo` after the current GPU pipeline stage is finished; it intentionally uses separate v2-hard run dirs.
- If only the v2 data refresh is needed while a GPU job is active, run the build-only profile and skip the GRPO profile until GPU memory is free.
