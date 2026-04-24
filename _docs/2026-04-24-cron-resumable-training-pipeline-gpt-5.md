---
date: 2026-04-24
slug: cron-resumable-training-pipeline
ai: gpt-5
---

# Cron Resumable Training Pipeline

## Goal

Make the ELT long-run pipeline resumable across power loss and reboot:
`pretrain -> SFT/posttrain -> KL-constrained GRPO -> eval/upload`.

## Files Touched

- `scripts/pipeline.py`
- `scripts/pipeline_register.ps1`
- `scripts/pipeline_unregister.ps1`
- `tests/test_pipeline_orchestrator.py`
- `_docs/2026-04-24-cron-resumable-training-pipeline-gpt-5.md`

## Key Decisions

- Replaced the old generic pipeline with the current ELT stage order:
  `00_pretrain_clean`, `01_distill_huihui_detection_upload_or_recover`,
  `02_prepare_detection_sft`, `03_detection_sft`,
  `04_hauhaucs_multilane_distill`, `05_lane_sft`, `06_kl_grpo`,
  `07_eval_compare`.
- Training stages auto-resume from `run_dir/last.pt` when present.
- Stage completion markers live in `H:/elt_data/pipeline_state/*.done`.
- A process lock at `H:/elt_data/pipeline_state/pipeline.lock` prevents
  overlapping 5-minute scheduler ticks.
- The Huihui detection output dir is protected. If `eval_summary.json` says
  records exist but `distill_train.jsonl` / `distill_val.jsonl` are zero-byte,
  the pipeline stops instead of regenerating over that directory.
- KL-constrained GRPO is represented by the existing code/math/tool configs
  with `grpo.kl_beta > 0`.

## Scheduler

- `scripts/pipeline_register.ps1` now writes the launcher to
  `H:/elt_data/pipeline_logs/pipeline_launcher.ps1` to avoid path quoting issues.
- Preferred registration uses `Register-ScheduledTask` with logon + 5-minute
  triggers and battery-safe settings.
- On this run, `Register-ScheduledTask` failed with `Access is denied`, so the
  script fell back to `schtasks.exe`.
- XML fallback with battery/logon settings also required elevated permissions.
- Simple `schtasks.exe` fallback succeeded and registered a 5-minute task:
  `ELT-LM-Pipeline`.
- Because of the non-elevated fallback, `schtasks /Query` reports default power
  settings (`Stop On Battery Mode`, `No Start On Batteries`). Run the register
  script from an elevated PowerShell to get the full battery/logon settings.

## Verification

Commands run:

```powershell
uv run --no-sync pytest -q tests/test_pipeline_orchestrator.py tests/test_clean_continue_configs.py
uv run --no-sync pyright scripts/pipeline.py tests/test_pipeline_orchestrator.py
uv run --no-sync python scripts/pipeline.py --dry-run
uv run --no-sync python scripts/pipeline.py --only 00_pretrain_clean --dry-run
uv run --no-sync python scripts/pipeline.py --only 00_pretrain_clean --no-start-long-train
powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1 -WhatIf
powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1
schtasks.exe /Query /TN ELT-LM-Pipeline /V /FO LIST
```

Results:

- Pipeline/orchestrator tests passed: `8 passed`.
- Pyright passed: `0 errors, 0 warnings`.
- Dry-run prints all eight current stages.
- `--no-start-long-train` now defers the long stage without writing a `.done`
  marker.
- `ELT-LM-Pipeline` is registered and enabled with a 5-minute repeat.

## Next Session Notes

- The simple scheduler fallback may start the full `00_pretrain_clean` stage on
  the next 5-minute tick. Use `scripts/pipeline_unregister.ps1` to stop the
  schedule.
- Before relying on battery-safe behavior, rerun
  `powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1` from
  an elevated PowerShell.
- The current Huihui detection directory appears to have zero-byte train/val
  JSONL despite a prior completed summary, so stage `01` is expected to stop
  safely until a completed snapshot is restored.
