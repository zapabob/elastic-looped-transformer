# 2026-04-29 posttrain GRPO pipeline automation

## Goal

Automate the long-run ELT post-training path so it can resume through:

1. prepared Huihui detection SFT,
2. HauhauCS lane SFT bins,
3. native ELT lane SFT,
4. KL-constrained GRPO for code/math/tool,
5. Qwen3.5-4B side LoRA ILSD/export,
6. final anytime evaluation.

The immediate blocker was not distillation. The first native dense SFT stage
stopped on H: free-space checks before optimizer offload could initialize.

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

- Added safe offload cleanup after successful native SFT/GRPO stages.
  Only a directory literally named `offload_nvme` under `H:/elt_data/runs`
  may be removed.
- Removed old dense side smoke run directories:
  `H:/elt_data/runs/qwen35_4b_side_sft_code_smoke_l1` and
  `H:/elt_data/runs/qwen35_4b_side_sft_code_smoke_l2`.
  The Qwen3.5-4B bootstrap checkpoint and LoRA outputs were kept.
- Rebalanced dense native SFT/GRPO configs from open-ended long schedules to
  phase-1 completion schedules. The previous detection config implied roughly
  65.5M training tokens. With the observed `45.8 tok/s`, detection alone was
  around 16 days. The new schedule is intended to complete the full pipeline
  and produce checkpoints/evals without thousands of repeats over tiny
  distilled bundles.
- Kept KL-constrained GRPO enabled with `kl_beta > 0` for code/math/tool.

## Validation

- `uv run --no-sync pytest -q tests/test_pipeline_orchestrator.py tests/test_qwen35_side_smoke_configs.py`
  - `19 passed`
- `uv run --no-sync python scripts/pipeline.py --profile posttrain-grpo --dry-run`
  - `00_prepare_detection_sft` done
  - `01_detection_sft` pending
  - remaining posttrain/GRPO/side/eval stages pending

## Live state at handoff

- Windows Task Scheduler task: `ELT-LM-Pipeline`
- Active profile: `posttrain-grpo`
- Current stage: `01_detection_sft`
- Detection SFT is running with the phase-1 config:
  - `grad_accum_steps=4`
  - `total_steps=24`
  - `log_every=1`
  - `save_every=8`
  - `eval_every=8`
- The first phase-1 training step produced finite loss:
  - `loss=12.828125`
  - `L_int=4`
  - `lambda_value=1.0`
- H: was recovered from about 10 GB free to over 110 GB free after deleting
  old dense side smoke runs and regenerable offload state.

## Next session notes

- If `01_detection_sft` completes, `scripts/pipeline.py` should clean its
  `offload_nvme` automatically before advancing.
- If a later native SFT/GRPO stage fails on disk, check whether a previous
  completed stage left a non-cleaned `offload_nvme` and inspect
  `H:/elt_data/pipeline_logs/pipeline-*.log`.
- Do not delete `H:/elt_data/runs/qwen35_4b_elt_bootstrap/last.pt`; side LoRA
  stages need it as the base checkpoint.
