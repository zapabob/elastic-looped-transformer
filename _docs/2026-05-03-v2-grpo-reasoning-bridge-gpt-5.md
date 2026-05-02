# 2026-05-03 v2 GRPO reasoning bridge

## Goal

Prepare the next intervention after the synthetic-v2-hard GRPO math and STEM
lanes completed with no useful reward signal.  Keep the completed hard-run
evidence intact, and create separate bridge/easy-hard rerun inputs for math and
STEM reasoning.

## Files touched

- `src/elt_lm/synthetic_v2_reasoning_bridge.py`
- `src/elt_lm/verifiers.py`
- `tests/test_synthetic_v2_hard.py`
- `tests/test_verifiers_quality.py`
- `configs/grpo_side_lora_math_synthetic_v2_bridge.yaml`
- `configs/grpo_side_lora_stem_synthetic_v2_bridge.yaml`
- `pyproject.toml`

## Key decisions

- Added a math/STEM bridge prompt builder instead of mutating the active hard
  lane configs.  The completed zero-signal hard runs stay under their original
  run directories.
- New rerun configs use separate run directories:
  - `H:/elt_data/runs/grpo_side_lora_math_synthetic_v2_bridge`
  - `H:/elt_data/runs/grpo_side_lora_stem_synthetic_v2_bridge`
- Generated bridge prompt files at:
  - `H:/elt_data/synthetic_v2_hard/math/benchmarks/synthetic_v2_bridge_math_val_cases.jsonl`
  - `H:/elt_data/synthetic_v2_hard/stem_reasoning/benchmarks/synthetic_v2_bridge_stem_reasoning_val_cases.jsonl`
- Each lane uses 256 prompts with a 64 easy / 128 bridge / 64 retained-hard
  split.
- Fixed the math/STEM verifier path so `CompositeVerifier.reward()` scores
  structured `<think>...</think><answer>...</answer>` outputs correctly after it
  extracts the answer text.  This makes the prior all-zero math/STEM metrics
  suspect: some true positives may have been suppressed by verifier plumbing,
  not only by data difficulty.

## Verification

- `uv run --no-sync pytest tests/test_synthetic_v2_hard.py tests/test_verifiers_quality.py -q`
  - 10 passed.
- `uv run --no-sync python -m py_compile src/elt_lm/verifiers.py src/elt_lm/synthetic_v2_reasoning_bridge.py`
  - passed.
- `uv run --no-sync python -m elt_lm.synthetic_v2_reasoning_bridge --total-cases 256`
  - generated math and STEM bridge JSONL plus `.summary.json`.
- Config smoke loaded both bridge configs and verified prompt files exist.

## Next session notes

- Do not read the completed math/STEM hard-lane `correct_rate=0` as clean model
  failure without accounting for the verifier fix in this session.
- The safest next run is the bridge configs directly, after the active
  synthetic-v2-hard pipeline is no longer using the GPU:
  - `uv run --no-sync elt-train-grpo --config configs/grpo_side_lora_math_synthetic_v2_bridge.yaml`
  - `uv run --no-sync elt-train-grpo --config configs/grpo_side_lora_stem_synthetic_v2_bridge.yaml`
- If bridge still produces sparse reward after the verifier fix, then add short
  v2 math/STEM SFT before GRPO.
