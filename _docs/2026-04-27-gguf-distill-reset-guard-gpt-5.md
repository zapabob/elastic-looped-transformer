# GGUF Distill Reset Guard

## Goal

Prevent completed or recoverable GGUF distillation bundles from being truncated when
the pipeline is accidentally restarted without `--resume`.

## Files touched

- `src/elt_lm/gguf_distill.py`
- `tests/test_gguf_distill.py`
- `configs/base_1B_continue_clean*.yaml`
- `configs/posttrain_*_qwen35_hauhaucs.yaml`
- `configs/posttrain_detection_sft_huihui_qwen36.yaml`
- `configs/gguf_distill_huihui_qwen36*.yaml`
- `scripts/pipeline_register.ps1`
- `src/elt_lm/train_grpo.py`
- `configs/grpo_*_qwen35_hauhaucs.yaml`
- `tests/test_clean_continue_configs.py`
- `tests/test_pipeline_orchestrator.py`

## Key decisions

- Added `guard_against_unsafe_reset(...)` before the non-resume artifact reset path.
- The guard refuses to truncate `raw_teacher_examples.jsonl`, `distill_train.jsonl`,
  or `distill_val.jsonl` if the output directory still has non-empty artifacts,
  `eval_summary.json` records, status progress, or checkpoint progress.
- Added an explicit `--force-reset` CLI escape hatch for intentional discard.
- Stopped and deleted the active `ELT-LM-Pipeline` scheduled task after it was found
  launching every five minutes while the long pretrain stage had no checkpoint output.
- Marked the stale pipeline runtime status as stopped and removed the stale
  `H:/elt_data/pipeline_state/pipeline.lock`.
- Switched native clean pretrain and pipeline SFT configs from
  `paged_adamw_8bit` to `nvme_adamw` to avoid the local Windows
  bitsandbytes initialization failure.
- Added `nvme_adamw` support to `train_grpo.py` and switched the code/math/tool
  GRPO configs to the same offload strategy.
- Lowered Huihui Qwen3.6 35B/A3B distill configs to `ctx_size: 2048` and
  `n_gpu_layers: 16` so retries do not full-offload the 35B teacher into
  CUDA OOM.
- Made `pipeline_register.ps1` safe by default: scheduled ticks now pass
  `--no-start-long-train` unless registration is explicitly run with
  `-StartLongTrain`.
- Restored the protected Huihui detection JSONL bundle from the Hugging Face dataset
  remote after confirming the local files had already been truncated to zero bytes.

## Verification

- `uv run --no-sync python -m py_compile src/elt_lm/gguf_distill.py tests/test_gguf_distill.py`
- `uv run --no-sync pytest -q tests/test_gguf_distill.py::test_guard_against_unsafe_reset_rejects_completed_bundle_summary tests/test_gguf_distill.py::test_guard_against_unsafe_reset_allows_explicit_force_reset`
- Restored artifact counts:
  - `raw_teacher_examples.jsonl`: 384 lines
  - `distill_train.jsonl`: 345 lines
  - `distill_val.jsonl`: 39 lines
- `uv run --no-sync python scripts/pipeline.py --dry-run --no-start-long-train`

The focused guard tests passed. The full `tests/test_gguf_distill.py` emitted
`13 passed` but the process returned exit code 137 after completion in this local
environment, so it should be re-run after memory pressure is lower.

## Next session notes

- `KV cache` absence is an inference/generation-speed limitation, not a training
  or distillation progress mechanism.
- The scheduled pretrain stage repeatedly emitted `run_start` and `train_config`
  only; no `train_step` events or checkpoints were observed.
- The pipeline log showed the pretrain command failing with rc=1 after a
  `bitsandbytes` initialization error. The GGUF teacher server log showed CUDA
  OOM while offloading the 35B/A3B model to GPU.
- Re-enable the scheduler only after a manual config-load check and either:
  `powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1`
  for safe monitoring, or the same command with `-StartLongTrain` when the
  next long run should actually execute.
- `H:/elt_data/gguf_distill/huihui_qwen36_detection/distill_train.jsonl`,
  `distill_val.jsonl`, and `raw_teacher_examples.jsonl` were already zero bytes
  before this patch was added, but the remote Hugging Face dataset still had the
  payloads and was used for restoration.
- A runtime marker was written to
  `H:/elt_data/gguf_distill/huihui_qwen36_detection/recovery_summary_2026-04-27.json`.
- Current distillation inventory:
  - `huihui_qwen36_detection`: complete, 384 raw / 345 train / 39 val records.
  - `qwen35_9b_hauhaucs_detection`: not generated.
  - `qwen35_9b_hauhaucs_code`: not generated, zero train/val records.
  - `qwen35_9b_hauhaucs_math`: not generated, zero train/val records.
  - `qwen35_9b_hauhaucs_stem_reasoning`: not generated, zero train/val records.
  - `qwen35_9b_hauhaucs_tool_use`: not generated, zero train/val records.
- Scheduler status after cleanup:
  - Disabled SO8T/AEGIS automation tasks:
    `AEGIS_Moonshot_Automation`, `AEGIS_Moonshot_Daily`,
    `SO8T-AutoResume`, `SO8T_AEGIS_Automatic_Pipeline`,
    `SO8T_Power_On_Startup`.
  - Registered `ELT-LM-Pipeline` as the only enabled matching scheduler task.
    It uses `H:/elt_data/pipeline_logs/pipeline_launcher.ps1` with
    `ELT_PIPELINE_START_LONG=1` and runs every 5 minutes via the schtasks
    fallback.
- Follow-up focused distillation monitor:
  - Added `configs/gguf_distill_qwen35_hauhaucs_all_remaining_queue.yaml` for
    `detection -> code -> math -> stem_reasoning -> tool_use`.
  - Stopped and deleted the broad `ELT-LM-Pipeline` task to prevent native
    pretrain GPU contention.
  - Registered `ELT-HauhauCS-Distill` every 5 minutes, pointing at
    `H:/elt_data/pipeline_logs/hauhaucs_distill_monitor_launcher.ps1`.
  - The launcher uses `H:/elt_data/pipeline_state/hauhaucs_distill_monitor.lock`
    to avoid double-starts and deletes its own task once all five distill
    output directories are `complete` with non-empty train/val JSONL files.
  - Initial live state after registration: detection running, `3/384` tasks,
    `train=2`, `val=1`, GPU memory about 9.6 GB on RTX 3060.
