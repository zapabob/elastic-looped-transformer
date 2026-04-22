# Goal

Huihui GGUF distillation pipeline on Windows needed restart tolerance after power loss and OOM-like interruption. The main target was resumability from already produced teacher examples without rewriting the run from scratch.

# Files Touched

- `src/elt_lm/gguf_distill.py`
- `configs/gguf_distill_huihui_qwen36_resume_lowmem.yaml`
- `configs/gguf_distill_huihui_qwen36_resume_smoke.yaml`

# Key Decisions

- Added `--resume` handling that restores progress from `raw_teacher_examples.jsonl` first and falls back to rolling checkpoint progress when JSONL is absent.
- Changed record persistence from end-of-stage buffered writes to per-item append with `flush()` and `os.fsync()` so completed examples survive abrupt power loss.
- Preserved rolling checkpoint behavior, but extended payload to include `processed_tasks_total` so restart can advance even when legacy runs do not have raw JSONL yet.
- Kept the low-memory runtime config (`ctx_size=4096`, `n_gpu_layers=48`) and added a separate smoke config with shorter generation limits for restartability validation.

# Verification

- `uv run python -m py_compile src/elt_lm/gguf_distill.py scripts/gguf_distill_pipeline.py`
- Smoke interruption test on `H:/elt_data/gguf_distill/huihui_qwen36_detection_resume_smoke`:
  - generated 1 record
  - forced process kill
  - confirmed persisted artifacts remained:
    - `raw_teacher_examples.jsonl`: 1 line
    - `distill_val.jsonl`: 1 line
    - `status.json`: `processed_tasks=1`
    - `checkpoint_0.json`: `processed_tasks_total=1`
- Resume attempts advanced from persisted state rather than redoing task 0, but the teacher request later failed on runtime timeout before full completion.

# Tests Added / Passing Count

- No automated pytest added in this session.
- Syntax validation passed for 2 Python entry files via `py_compile`.

# What The Next Session Should Know

- Restart persistence is now materially better than the previous implementation because generated rows are written immediately.
- The remaining blocker is teacher runtime stability, not resume bookkeeping:
  - observed failures were `TimeoutError: timed out`
  - earlier main run also showed `ConnectionResetError: [WinError 10054]`
- If the main run is restarted now with `--resume`, it should skip already persisted work or fall back to rolling checkpoint progress from legacy runs.
- Further hardening worth considering if failures continue:
  - retry current task on timeout / connection reset
  - relaunch `llama-server` automatically on request failure
  - reduce generation cost further (`max_new_tokens`, reasoning/thinking behavior, or model/runtime settings)
