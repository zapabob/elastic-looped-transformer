# 2026-05-01 posttrain GRPO recovery prune

## Goal

Keep the long-running `posttrain-grpo` pipeline moving after completed lane
stages by skipping already-finalized training configs and pruning bulky
duplicate checkpoint snapshots.

## Files touched

- `scripts/pipeline.py`
- `tests/test_pipeline_orchestrator.py`
- `configs/posttrain_detection_sft_huihui_qwen36.yaml`
- `configs/posttrain_code_sft_qwen35_hauhaucs.yaml`
- `configs/posttrain_math_sft_qwen35_hauhaucs.yaml`
- `configs/posttrain_stem_sft_qwen35_hauhaucs.yaml`
- `configs/posttrain_tool_sft_qwen35_hauhaucs.yaml`
- `configs/grpo_code_qwen35_hauhaucs.yaml`
- `configs/grpo_math_qwen35_hauhaucs.yaml`
- `configs/grpo_tool_qwen35_hauhaucs.yaml`

## Key decisions

- Added `training_run_complete(...)` so aggregate SFT/GRPO stages can skip
  configs that already emitted a final checkpoint at or beyond `total_steps`.
- Added `prune_completed_checkpoints(...)` to keep `last.pt` while removing
  completed `rolling_*.pt` and `step_*.pt` files under `H:/elt_data/runs`.
- Kept the pruning path guard narrow: only run directories below
  `H:/elt_data/runs` are eligible.
- Set native SFT/GRPO `offload.min_free_gb` to `0.0` because the outer
  pipeline now owns the rolling/offload cleanup policy and runs on a tight H:
  disk budget.

## Validation

- `uv run --no-sync pytest -q tests/test_pipeline_orchestrator.py tests/test_qwen35_side_smoke_configs.py`
  - `22 passed`

## Live state

- The active pipeline remains on `03_hauhaucs_lane_sft`.
- Math SFT is running from `configs/posttrain_math_sft_qwen35_hauhaucs.yaml`
  and has resumed from `H:/elt_data/runs/posttrain_math_sft_qwen35_hauhaucs/last.pt`.
- Detection and code SFT completed earlier; their completed checkpoint clutter
  can now be pruned automatically when the aggregate stage revisits them.

## Next session notes

- If a later lane completes and the aggregate marker is still absent, the
  stage should skip the completed config rather than rerunning it.
- If disk pressure returns, first inspect completed run directories for
  retained non-`last.pt` checkpoint snapshots and stale `offload_nvme`.
