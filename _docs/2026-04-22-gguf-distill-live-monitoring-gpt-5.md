# 2026-04-22 — gguf-distill-live-monitoring — gpt-5

## Goal

Add real-time monitoring for the Huihui GGUF distillation pipeline, surface it in the existing dashboard, add a CLI monitor for automation, and wire Cron monitoring so the long-running production distill can be followed without guessing.

## Files Touched

- `src/elt_lm/gguf_distill.py`
- `dashboard/utils/metrics_reader.py`
- `dashboard/panels/gguf_distill.py`
- `dashboard/app.py`
- `scripts/monitor_gguf_distill.py`
- `configs/gguf_distill_huihui_qwen36.yaml`
- `tests/test_gguf_distill.py`
- `tests/test_dashboard_reader.py`
- `tests/test_dashboard_gguf_distill.py`

## Key Decisions

- Kept the existing dense-first GGUF distill pipeline and extended it with:
  - `status.json`
  - `heartbeat.json`
  - `metrics.jsonl` events for GGUF distill stages/items/uploads/errors
  - `run.lock` PID locking to prevent overlapping runs
- Added failure-safe status emission so exceptions mark the run as `failed` and release the lock.
- Added a dedicated dashboard panel for `H:/elt_data/gguf_distill/*` runs instead of overloading the training panel.
- Added `scripts/monitor_gguf_distill.py` for automation-friendly progress polling from `status.json`, `heartbeat.json`, `eval_summary.json`, and `llama_server.log`.
- Updated config defaults with `heartbeat_interval_sec: 30` and `stall_after_sec: 1800`.
- Switched `llama_server.log` creation to truncate per run. Note: the currently running production job started before that patch, so its current server log still contains prior-session tail content. Future runs will be clean.

## Tests Added / Passing

- Added/extended tests for:
  - GGUF status snapshot generation
  - status/heartbeat artifact writing
  - run-lock rejection
  - JSON file dashboard reader helper
  - GGUF distill run discovery
  - heartbeat health classification
- `uv run pyright src/elt_lm/gguf_distill.py dashboard/utils/metrics_reader.py dashboard/panels/gguf_distill.py scripts/monitor_gguf_distill.py tests/test_gguf_distill.py tests/test_dashboard_reader.py tests/test_dashboard_gguf_distill.py`
  - result: `0 errors, 0 warnings, 0 informations`
- `uv run pytest -q tests/test_gguf_distill.py tests/test_dashboard_reader.py tests/test_dashboard_gguf_distill.py`
  - output showed `19 passed`

## Automation

- Updated existing cron automation:
  - `gguf-distill-daily`
  - now checks `status.json` / `heartbeat.json` / `run.lock` first and skips duplicate launch when the run is already healthy
- Created new cron automation:
  - `gguf-distill-monitor`
  - hourly monitor for progress / stalls / failures

## Live Runtime State

- Production run started in background:
  - output dir: `H:/elt_data/gguf_distill/huihui_qwen36_detection`
  - pipeline PID at launch check: `5460`
- Dashboard is running:
  - `http://localhost:8501`
- At the end of this session the production run was visible through the monitor CLI as:
  - `state=running`
  - `current_stage=teacher_generation`
  - `processed_tasks=0/384`
  - `health=healthy`
- The first teacher sample appears to be slow on this model/quant, so early progress may stay at `0/384` for several minutes before the first increment.

## Next Session Notes

- Use `uv run python scripts/monitor_gguf_distill.py --root H:/elt_data/gguf_distill --run-name huihui_qwen36_detection --json` as the fastest live check.
- The dashboard panel is now the best human-readable view; it auto-refreshes every 5 seconds.
- If the run later stalls, inspect:
  - `H:/elt_data/gguf_distill/huihui_qwen36_detection/status.json`
  - `H:/elt_data/gguf_distill/huihui_qwen36_detection/heartbeat.json`
  - `H:/elt_data/gguf_distill/huihui_qwen36_detection/metrics.jsonl`
  - `H:/elt_data/gguf_distill/huihui_qwen36_detection/pipeline_stdout.log`
  - `H:/elt_data/gguf_distill/huihui_qwen36_detection/pipeline_stderr.log`
- Because the current run started before the log-truncation patch, `llama_server.log` may include stale historical lines until the next run.
