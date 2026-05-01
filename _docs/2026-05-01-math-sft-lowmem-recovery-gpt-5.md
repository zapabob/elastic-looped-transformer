# 2026-05-01 math SFT low-memory recovery - gpt-5

## Goal

Recover the `posttrain-grpo` pipeline after reboot revealed a repeated CUDA OOM
loop in `03_hauhaucs_lane_sft` while resuming
`configs/posttrain_math_sft_qwen35_hauhaucs.yaml` from step 41.

## Files touched

- `configs/posttrain_math_sft_qwen35_hauhaucs.yaml`
- `tests/test_multilane_configs.py`

## Key decisions

- Temporarily disabled math-lane loop-hidden auxiliary regularizers:
  - `entropy_floor_weight: 0.0`
  - `entropy_curvature_weight: 0.0`
  - `logit_curvature_weight: 0.0`
  - `logit_curvature_max_positions: 0`
  - `local_consistency_weight: 0.0`
- Reduced `data.seq_len` from `1024` to `512`.
- Reduced math SFT teacher loop from `L_max: 4` to `L_max: 3` for the remaining
  recovery steps. The parameter shapes remain unchanged, so later GRPO/eval
  configs can still instantiate the model with `L_max: 4` and load the weights.
- Kept teacher-temperature KD and masked soft CE active so the run still performs
  ILSD-style distillation rather than becoming plain GT-only SFT.

## Evidence

Recent failing logs showed:

```text
torch.AcceleratorError: CUDA error: out of memory
resumed from H:\elt_data\runs\posttrain_math_sft_qwen35_hauhaucs\last.pt (step 41)
```

## Tests

Focused config verification:

```powershell
uv run --no-sync pytest -q tests/test_multilane_configs.py
```

## Next session notes

After math SFT completes, compare the final checkpoint against the step-41
checkpoint before relying on it for GRPO. If quality regresses, rerun math SFT
later with a larger GPU or split config that restores entropy curvature.
