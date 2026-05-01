# Progress report cron and heartbeat automation

## Goal

Provide two layers of progress reporting for the ELT long-run pipeline:

- Windows Task Scheduler cron-style snapshots that keep writing progress artifacts even if the UI heartbeat fails.
- Codex app heartbeat automation for user-facing notifications when intervention is needed.

## Files touched

- `scripts/pipeline_progress_report.ps1`
- `scripts/pipeline_progress_register.ps1`

## Key decisions

- The reporter is read-only: it never starts or stops training.
- Output is written under `H:/elt_data/pipeline_state`:
  - `progress_report.json`
  - `progress_report.md`
  - `progress_reports.jsonl`
- The reporter records pipeline status, latest metrics event, latest `train_step`, checkpoint freshness, GPU snapshot, and `C:` / `H:` free space.
- The app heartbeat uses a new automation id (`elt-progress-heartbeat`) and should notify only for stage completion, failure, stalls, critical disk pressure, or required intervention.

## Tests and checks

- Manual reporter smoke:
  - `powershell -ExecutionPolicy Bypass -File scripts/pipeline_progress_report.ps1`
- Windows scheduled task registration:
  - `powershell -ExecutionPolicy Bypass -File scripts/pipeline_progress_register.ps1 -IntervalMinutes 30`

## Next session notes

- If app heartbeat resume errors reappear, keep the Windows progress reporter enabled and delete the app heartbeat; the durable status files remain available for manual checks.
