# v2 GRPO Bridge Prompt Intervention

## Goal

Recover from the `group_size=4` CUDA OOM by returning the code GRPO lane to
`group_size=2` and replacing hard-only prompts with a bridge/easy-hard mixed
prompt curriculum.

## Files touched

- `configs/grpo_side_lora_code_synthetic_v2_hard.yaml`
- `src/elt_lm/synthetic_v2_code_bridge.py`
- `tests/test_synthetic_v2_hard.py`
- `pyproject.toml`
- `_docs/2026-05-02-v2-grpo-bridge-prompts-gpt-5.md`

## Key decisions

- Stopped the scheduler-restarted `group_size=4` retry before it could OOM
  again.
- Kept the main `synthetic-v2-hard-grpo` profile and run directory unchanged so
  export/eval paths stay stable.
- Reverted the code lane GRPO `group_size` from 4 to 2 for RTX 3060 12GB.
- Pointed code GRPO at
  `H:/elt_data/synthetic_v2_hard/code/benchmarks/synthetic_v2_bridge_code_val_cases.jsonl`.
- Added `elt_lm.synthetic_v2_code_bridge` and the
  `elt-build-synthetic-v2-code-bridge` CLI to generate a deterministic 256-case
  curriculum: easy positives, bridge cases from the same code domains, and a
  retained hard slice.
- Chose bridge/easy-hard mixing before short v2 code SFT because it is cheaper
  and directly targets GRPO reward sparsity.

## Evidence

- `group_size=4` failed in `out.loss.backward()` with
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.57 GiB`.
- The only post-g4 completed step was step 14:
  `reward_std=0.002598`, `correct_rate=0.0`, `format_rate=1.0`.
- After failure, GPU memory dropped back to desktop idle range, confirming the
  training process exited.

## Tests / checks

- `uv run --no-sync pytest tests/test_synthetic_v2_hard.py -q` passed.
- `uv run --no-sync python -m py_compile src/elt_lm/synthetic_v2_code_bridge.py` passed.
- Config load smoke confirmed `grpo.group_size == 2` and
  `prompts_file == H:/elt_data/synthetic_v2_hard/code/benchmarks/synthetic_v2_bridge_code_val_cases.jsonl`.
- Generated `H:/elt_data/synthetic_v2_hard/code/benchmarks/synthetic_v2_bridge_code_val_cases.jsonl`
  with 256 cases: 64 easy, 128 bridge, and 64 hard.
- `uv run --no-sync python scripts/pipeline.py --profile synthetic-v2-hard-grpo --no-start-long-train`
  confirmed resume from `H:/elt_data/runs/grpo_side_lora_code_synthetic_v2_hard/last.pt`.
- Restarted `ELT-LM-Pipeline`; metrics appended `grpo_config` with `group_size=2`.
- First post-bridge step completed at step 14:
  `reward_std=0.1045`, `adv_abs_mean=0.99999`, `correct_rate=0.0`, and
  `format_rate=1.0`.
- `git diff --check` passed with only expected CRLF warnings on edited text files.

## Next session notes

- If g2 + bridge still has `correct_rate=0.0` after several post-bridge steps,
  run a short v2 code SFT warm-up before continuing long GRPO.
