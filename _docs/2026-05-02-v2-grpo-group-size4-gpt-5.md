# v2 GRPO Group Size 4 Intervention

## Goal

Switch the synthetic v2 hard code GRPO lane from `group_size: 2` to
`group_size: 4` after the step 10-12 review showed persistent
`correct_rate=0.0` with only sparse reward variance.

## Files touched

- `configs/grpo_side_lora_code_synthetic_v2_hard.yaml`
- `_docs/2026-05-02-v2-grpo-group-size4-gpt-5.md`

## Key decisions

- Stopped the active `synthetic-v2-hard-grpo` run after metrics reached step 17.
- Kept `run_dir: H:/elt_data/runs/grpo_side_lora_code_synthetic_v2_hard` unchanged so the pipeline can resume from `last.pt` and later export/eval stages keep their existing paths.
- Changed only the code lane GRPO `group_size` from 2 to 4. Math, STEM, and tool lanes remain unchanged.
- Stopped a scheduler-restarted `synthetic-gb-side-lora-long` process that had reacquired the global pipeline lock.
- Updated `H:/elt_data/pipeline_logs/pipeline_launcher.ps1` so future `ELT-LM-Pipeline` scheduler ticks target `synthetic-v2-hard-grpo` instead of the saturated v1 synthetic GB profile.
- The last saved checkpoint before the intervention was `last.pt` / `rolling_2.pt` at `2026-05-02T20:49:37+09:00`, corresponding to the rolling checkpoint after step 13. Metrics had observed through step 17, so at most the post-checkpoint steps are expected to be replayed or skipped depending on resume semantics.

## Evidence before change

- Pipeline stage: `01_side_lora_synthetic_v2_hard_grpo`
- Prompt task: `python_exec`
- Step 10-12 all had `correct_rate=0.0` and `format_rate=1.0`.
- Step 12 had `reward_std=0.0435`, `reward_mean=-0.0435`, and `adv_abs_mean=0.99998`.
- Through step 17, `correct_rate` remained `0.0`; reward variance was still sparse and mostly penalty-derived rather than success-derived.

## Tests / checks

- `uv run --no-sync python -c "import yaml, pathlib; ..."` loaded the config and confirmed `grpo.group_size == 4`.
- `uv run --no-sync python scripts/pipeline.py --profile synthetic-v2-hard-grpo --no-start-long-train` confirmed the stage resumes from `H:/elt_data/runs/grpo_side_lora_code_synthetic_v2_hard/last.pt`.
- Restarted `ELT-LM-Pipeline` through Task Scheduler after updating `H:/elt_data/pipeline_logs/pipeline_launcher.ps1`; the active command now runs `--profile synthetic-v2-hard-grpo`.
- Metrics appended a new `grpo_config` event with `group_size=4` at `2026-05-02T21:02:27+09:00`.
- First post-change GRPO step completed at step 13 with `reward_std=0.0`, `correct_rate=0.0`, and `format_rate=1.0`; the run remained alive without immediate OOM.
- `git diff --check` passed with only the expected CRLF warning for `configs/grpo_side_lora_code_synthetic_v2_hard.yaml`.

## Next session notes

- If `correct_rate` is still `0.0` after several group-size-4 steps, prefer adding bridge/easy-hard mixed code prompts or running a short v2 code SFT before continuing long GRPO.
