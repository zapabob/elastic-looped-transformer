# 2026-05-03 aha loop self-distillation protocol - GPT-5

## Goal

Turn the ELT loop mechanism into a staged reasoning curriculum instead of
immediately asking GRPO to discover hard answers from sparse reward. The target
is an "aha/grokking" setup: make a higher loop count produce better verified
traces, distill those traces back into the adapter, then run bridge GRPO where
positive samples are reachable.

## Files touched

- `configs/qwen35_4b_side_lora_math_aha_ilsd_l2.yaml`
- `configs/qwen35_4b_side_lora_stem_aha_ilsd_l2.yaml`
- `configs/qwen35_4b_side_lora_math_aha_ilsd_l3.yaml`
- `configs/qwen35_4b_side_lora_stem_aha_ilsd_l3.yaml`
- `configs/aha_loop_self_distill_protocol.yaml`
- `tests/test_qwen35_side_smoke_configs.py`

## Method

Use the existing ILSD implementation as the loop stabilizer:

1. Start from the lane SFT adapter, not from a randomly initialized LoRA.
2. Introduce L=2 first with a strict lower-depth student and L=2 teacher.
3. Keep `lambda` high early and anneal it down so task loss dominates late.
4. Add entropy floor, loop-axis curvature, and local hidden-state consistency.
5. Only stretch to L=3 after L=2 is stable.
6. Hold L=4 until L=3 has a measured held-out gain; do not spend VRAM on it
   before evidence says the loop is useful.
7. Generate high-L traces, keep only verifier-passing outputs, mix them with
   replay anchors, then run a short SFT refresh.
8. Run bridge GRPO after the verifier fix and self-distillation refresh.

This avoids the main observed failure mode from the hard v2 GRPO run: sparse
positive samples make GRPO mostly optimize formatting and KL behavior. The loop
ladder should make correct traces more reachable before policy optimization.

## Guardrails

- Divergence: `grad_clip <= 0.5`, short first runs, no direct jump to L=4.
- Gradient vanishing: L=2 before L=3, strict student-below-teacher sampling,
  annealed ILSD coefficient.
- Local stable solution: hard bridge prompts are used after the adapter has
  learned recoverable intermediate solutions.
- Catastrophic forgetting: LoRA-only updates, replay-mixed self-distillation,
  and later KL-regularized GRPO.

## Run order

Math L=2:

```powershell
uv run --no-sync elt-train --config configs/qwen35_4b_side_lora_math_aha_ilsd_l2.yaml --resume H:/elt_data/runs/qwen35_4b_side_lora_math_sft_synthetic_gb/last.pt
```

STEM L=2:

```powershell
uv run --no-sync elt-train --config configs/qwen35_4b_side_lora_stem_aha_ilsd_l2.yaml --resume H:/elt_data/runs/qwen35_4b_side_lora_stem_sft_synthetic_gb/last.pt
```

Math L=3, only after L=2 is stable:

```powershell
uv run --no-sync elt-train --config configs/qwen35_4b_side_lora_math_aha_ilsd_l3.yaml --resume H:/elt_data/runs/qwen35_4b_side_lora_math_aha_ilsd_l2/last.pt
```

STEM L=3, only after L=2 is stable:

```powershell
uv run --no-sync elt-train --config configs/qwen35_4b_side_lora_stem_aha_ilsd_l3.yaml --resume H:/elt_data/runs/qwen35_4b_side_lora_stem_aha_ilsd_l2/last.pt
```

## Next session notes

The current synthetic-v2-hard pipeline should be allowed to finish before these
configs are launched, because the GPU is already occupied. These configs are
standalone and are not inserted into `scripts/pipeline.py`, so they cannot
disturb the active run.

After each L2/L3 run, compare held-out bridge accuracy at `L=1`, `L=2`, and
`L=3`. The desired grokking signal is not just lower loss: it is a late,
discrete jump where the higher loop count solves bridge examples that L=1 did
not, followed by output self-distillation that narrows the gap.

