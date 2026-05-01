# Cron Heartbeat ETA Progress

## Goal

Make the Windows Cron-style progress reporter act as the durable heartbeat source and include an estimated completion time for the active training run.

## Files Touched

- `scripts/pipeline_progress_report.ps1`

## Key Decisions

- Kept the app Heartbeat disabled because this thread hits a `C:\...` vs `\\?\C:\...` resume-path mismatch.
- Used `H:/elt_data/pipeline_state/progress_report.json` and `progress_report.md` as the human-readable Cron heartbeat source.
- Added `H:/elt_data/pipeline_state/progress_heartbeat.json` as a compact machine-readable heartbeat snapshot.
- ETA is scoped to the current active training run, not the entire multi-stage pipeline, because later stages may skip, resume, or change duration depending on checkpoints.
- ETA is computed from recent `train_step` timestamp deltas in the latest `metrics.jsonl`.
- `metrics_age_sec` now uses the latest JSONL event timestamp when available, rather than only the file modified time.

## Verification

- Re-ran `scripts/pipeline_progress_report.ps1`.
- Confirmed `progress_report.md` contains `eta_current_run`.
- Confirmed `progress_heartbeat.json` contains `eta.estimated_completion_time`.
- Current snapshot showed `step 25 / 40`, `loss 14.3497`, ETA about `3h 23m`, estimated completion around `2026-05-01T21:39:15+09:00`.

## Next Notes

- The scheduled task `ELT-LM-Progress-Report` remains registered at a 5-minute interval.
- If full-pipeline ETA is needed later, add stage-history based estimates after enough completed stage durations accumulate.
