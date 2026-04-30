# 2026-05-01 LTF replay refresh

## Goal

Use the original LTF/ELT corpus safely after the current distilled
`posttrain-grpo` run finishes. The original corpus is used as clean replay and
continued pretraining, not as direct GRPO prompt data.

## Files touched

- `src/elt_lm/prepare_mixed_lane_sft.py`
- `scripts/prepare_mixed_lane_sft.py`
- `scripts/pipeline.py`
- `configs/*_replay.yaml`
- `tests/test_prepare_mixed_lane_sft.py`
- `tests/test_replay_refresh_configs.py`
- `tests/test_pipeline_orchestrator.py`
- `pyproject.toml`

## Key decisions

- Added `elt-prepare-mixed-lane-sft` / `python -m elt_lm.prepare_mixed_lane_sft`
  to mix lane distill JSONL with a bounded
  amount of original posttrain data and clean-corpus replay.
- Kept the current `posttrain-grpo` profile unchanged so the active run can
  finish without marker or checkpoint churn.
- Added a separate `replay-refresh` profile:
  clean replay pretrain, mixed lane SFT, KL GRPO, side LoRA mixed SFT, eval.
- Kept GRPO verifier-only for code/math/tool; original clean corpus is not
  inserted into open-ended RL prompts.
- Added phase-2 run directories under new `_replay` names to avoid colliding
  with current posttrain outputs.

## Validation

- Focused tests should cover:
  - mixed lane ratio preparation,
  - empty replay source handling,
  - replay profile ordering,
  - replay config loading and KL constraints.

## Next session notes

- Do not reset current `posttrain-grpo` markers for this change.
- After the current profile completes, run:
  `uv run --no-sync python scripts/pipeline.py --profile replay-refresh --dry-run`
- For a tiny data smoke before full prepare, use:
  `uv run --no-sync python -m elt_lm.prepare_mixed_lane_sft --input-root H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_math --output-root H:/elt_data/posttrain_mixed_smoke/math --lane math --max-distill-records 8`
