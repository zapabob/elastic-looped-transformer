# Bridge follow-up repair pass

## Goal

Act on the bridge diagnostics: export the strong STEM candidate, keep code/math
on replay SFT and prompt repair before more GRPO, and fix the tool lane's zero
reward/advantage by repairing verifier/schema/failure-contrast data first.

## Files touched

- `src/elt_lm/verifiers.py`
- `src/elt_lm/eval/benchmarks.py`
- `src/elt_lm/bridge_followup.py`
- `tests/test_verifiers_quality.py`
- `tests/test_bridge_followup.py`
- `configs/qwen35_4b_side_lora_code_sft_synthetic_v2_bridge_repair.yaml`
- `configs/qwen35_4b_side_lora_math_sft_synthetic_v2_bridge_repair.yaml`
- `configs/qwen35_4b_side_lora_tool_sft_synthetic_v2_bridge_repair.yaml`
- `configs/grpo_side_lora_tool_synthetic_v2_bridge_repair.yaml`
- `training_data/bridge_followup/**`
- `pyproject.toml`

## Key decisions

- Added `json_tool_call_match` as a repair verifier, not a replacement for
  exact `json_match` final evaluation. Exact JSON still receives score `1.0`,
  but partial tool-call structure now produces nonzero repair signal.
- Safety-critical argument drift is capped: missing or wrong `read_only`,
  `dry_run`, `request_id`, or `requires_tests` limits the shaped score to
  `0.49`.
- Built `tool_use_repair` SFT and preference artifacts from correct traces plus
  failure-contrast records. Chosen repair traces score `1.0`; rejected failures
  are capped at `0.49`.
- Built code/math replay subsets with verifier pass rate `1.0`, intended as
  low-risk SFT repair before another sparse-success GRPO continuation.
- Exported the STEM bridge GRPO adapter candidate to
  `H:/elt_data/adapters/qwen35_4b_side/synthetic_stem_v2_bridge_grpo_candidate`.
  Bounded eval was attempted at 128 cases and 8-case smoke; both timed out
  before summary output, so eval is not complete.

## Local/H artifacts

- `training_data/bridge_followup/lane_action_plan.json`
- `training_data/bridge_followup/tool_use_repair/summary.json`
- `H:/elt_data/bridge_followup/{code_replay,math_replay,tool_use_repair}/bin/{train,val}.bin`
- `H:/elt_data/adapters/qwen35_4b_side/synthetic_stem_v2_bridge_grpo_candidate/adapter.pt`

## Tests / checks

- `uv run --no-sync pytest tests/test_verifiers_quality.py tests/test_bridge_followup.py -q` -> 10 passed
- `uv run --no-sync python -m compileall -q src/elt_lm/bridge_followup.py src/elt_lm/verifiers.py src/elt_lm/eval/benchmarks.py`
- `uv run --no-sync elt-build-bridge-followup --train-limit 192 --val-limit 64 --stem-eval-limit 128`
- `uv run --no-sync elt-tokenize ...` for code/math/tool repair train/val bins
- `uv run --no-sync python -m elt_lm.export_lora_adapter --ckpt H:/elt_data/runs/grpo_side_lora_stem_synthetic_v2_bridge/last.pt --out-dir H:/elt_data/adapters/qwen35_4b_side/synthetic_stem_v2_bridge_grpo_candidate`
- Config load smoke for all four repair configs

## Next session notes

- Start tool repair from SFT, then run the shaped-verifier GRPO probe only after
  confirming nonzero reward variance in the repair prompt set.
- Do not count STEM eval as finished; only the adapter export is complete.
- For STEM eval, either use a longer foreground run or add benchmark per-case
  telemetry so timeout windows leave measurable progress.
- Keep the side-LoRA `L=1` repair/eval path separate from native ELT `L=4`.
