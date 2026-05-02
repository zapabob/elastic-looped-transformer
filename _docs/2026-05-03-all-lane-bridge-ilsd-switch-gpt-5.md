# 2026-05-03 all-lane bridge ILSD switch - GPT-5

## Goal

Switch the next synthetic-v2 direction from hard-only GRPO continuation to an
all-lane bridge/fixed-verifier and ILSD ladder.

## Files touched

- `src/elt_lm/synthetic_v2_tool_bridge.py`
- `configs/grpo_side_lora_code_synthetic_v2_bridge.yaml`
- `configs/grpo_side_lora_tool_synthetic_v2_bridge.yaml`
- `configs/qwen35_4b_side_lora_code_aha_ilsd_l2.yaml`
- `configs/qwen35_4b_side_lora_code_aha_ilsd_l3.yaml`
- `configs/qwen35_4b_side_lora_tool_aha_ilsd_l2.yaml`
- `configs/qwen35_4b_side_lora_tool_aha_ilsd_l3.yaml`
- `configs/aha_loop_self_distill_protocol.yaml`
- `pyproject.toml`
- `scripts/pipeline.py`
- `src/elt_lm/train.py`
- `tests/test_synthetic_v2_hard.py`
- `tests/test_qwen35_side_smoke_configs.py`
- `tests/test_pipeline_orchestrator.py`

## Key decisions

- Keep the completed hard GRPO adapters as artifacts, but stop treating the hard
  lane as the next optimization target.
- Add a separate code bridge config instead of reusing the hard run directory.
- Add a tool-use bridge generator and config so all four lanes now have
  bridge/easy-hard GRPO entrypoints.
- Extend the ILSD ladder to code and tool-use so all four lanes can go through
  L=2 stabilization and L=3 stretch before another GRPO attempt.
- Keep the protocol explicit about all four bridge GRPO configs and both L=2/L=3
  ILSD configs so no lane silently stays on hard-only GRPO.
- Add `synthetic-v2-bridge-ilsd` as the resumable pipeline profile:
  hard data check -> bridge prompt build -> all-lane ILSD L2 -> all-lane ILSD L3
  -> all-lane bridge GRPO -> adapter export.
- Point bridge GRPO initial checkpoints at the corresponding L3 ILSD adapters,
  so bridge reward separation happens after the loop/self-distillation ladder.
- Update `H:/elt_data/pipeline_logs/pipeline_launcher.ps1` to use the new
  `synthetic-v2-bridge-ilsd` profile; otherwise Task Scheduler restarts the old
  hard-GRPO CV stage on the next tick.
- Keep ILSD configs on full-layer LoRA (`hf_lora_top_layers: 0`) to match the
  existing synthetic GB SFT adapters; top-8 ILSD rejects the saved lower-layer
  adapter keys and would discard useful lane memory.
- Add `elt-train --init-from` for transfer initialization. `--resume` preserves
  optimizer/step and caused L2 ILSD to immediately finish at source step 240;
  `--init-from` loads weights/adapters but resets the new run to step 0.
- Removed bogus immediate-final L2 run directories for code/math/stem before
  restarting the corrected bridge/ILSD profile.

## Next session notes

If the old CV eval is still running and the user wants immediate switch, stop
the `elt-anytime`/pipeline process tree first, then start:

`uv run --no-sync python scripts/pipeline.py --profile synthetic-v2-bridge-ilsd`

The profile starts with ILSD before bridge GRPO. If VRAM becomes tight, keep
GRPO at `rollout_L: 1` and shorten the L3 stretch before retrying.

## 2026-05-03 L3 divergence guard update

The first live code L3 run reached step 130 with tail loss around 6.1 and
tail `l_dist` around 8.4, while entropy was low. That looked more like
teacher/student divergence amplification than a clean aha signal.

Changed all L3 configs to a conservative ladder:

- `lambda_init: 0.35`
- `lambda_final: 0.03`
- `lambda_anneal_steps: 80`
- `warmup_steps: 8`
- `total_steps: 80`
- `eval_every: 20`
- `save_every: 40`

Also changed code bridge GRPO to initialize from
`H:/elt_data/runs/qwen35_4b_side_lora_code_aha_ilsd_l3/step_0000080.pt`
instead of `last.pt`, so the bridge stage avoids the over-diverged late code
L3 checkpoint.

Math L3 also showed early high disagreement under the conservative schedule
(`l_dist` around 6 by step 25-40). To keep GRPO from consuming late divergent
L3 checkpoints, math/stem/tool bridge GRPO now initialize from their L3
`step_0000040.pt` milestones instead of `last.pt`. Code keeps `step_0000080.pt`
because that is the only available mid-run milestone from the first 160-step
code L3 run.

Verification:

- `uv run --no-sync python scripts/pipeline.py --profile synthetic-v2-bridge-ilsd --dry-run`
  completed and showed the expected remaining stages.
- Confirmed `step_0000080.pt` exists for the code L3 run.
- Added a bounded `val_probe` path to `elt-train`: when `eval_every` is set and
  `data.val_bin` exists, the next training processes log no-grad validation
  probes over `eval_batches` small batches plus a final probe. This gives the
  monitor a cleaner signal for L3 divergence without running a full eval.
- Updated the active heartbeat monitor to know about the L3 divergence guard,
  the code `step_0000080.pt` GRPO init, and the 3-6 step bridge GRPO decision
  gate.
- Focused verification after the guard update:
  `uv run --no-sync python -m py_compile src/elt_lm/train.py src/elt_lm/config.py`
  and
  `uv run --no-sync pytest tests/test_qwen35_side_smoke_configs.py tests/test_pipeline_orchestrator.py tests/test_clean_continue_configs.py tests/test_posttrain_configs.py -q`
  passed.
- Full `uv run --no-sync pytest -q` exposed an older direct-call test failure
  in `src/elt_lm/eval/anytime_sweep.py`: programmatic `argparse.Namespace`
  callers did not include `cv_folds`, even though the CLI parser has a default.
  Added `getattr(args, "cv_folds", 5)` compatibility before commit/push.

Next session should judge bridge GRPO after 3-6 steps using `format_rate`,
`reward_std`, `correct_rate`, `adv_abs_mean`, and `verifier_reward_mean`.
If reward stays sparse, prefer bridge ratio increase or a short lane-specific
SFT over making L3 stronger again.
