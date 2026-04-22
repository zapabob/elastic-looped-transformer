# Goal

Add a safe sequential GGUF distillation queue so the existing Huihui Qwen3.6 teacher can finish before the local Qwen3.5-9B HauhauCS GGUF starts, avoiding concurrent llama-server launches and VRAM/OOM contention.

# Files Touched

- `C:\Users\downl\Desktop\新しいフォルダー (7)\src\elt_lm\gguf_distill_queue.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\scripts\gguf_distill_queue.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\configs\gguf_distill_detection_two_teacher_queue.yaml`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\tests\test_gguf_distill_queue.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\pyproject.toml`

# Key Decisions

- Kept the existing `elt-gguf-distill` single-teacher pipeline untouched and added a thin outer queue.
- Queue behavior is explicit:
  - if a stage is already `running`, wait for its terminal state;
  - if a stage is already `complete`, skip it by default;
  - if a stage is `failed`, rerun with `resume: true` if configured;
  - never launch two teachers at once.
- Added queue-level `status.json` and `heartbeat.json` so the orchestration itself is monitorable.

# Tests

- Added `tests/test_gguf_distill_queue.py`
- Intended verification:
  - `uv run --no-sync pytest -q tests/test_gguf_distill.py tests/test_gguf_distill_queue.py`
  - `uv run --no-sync pyright src/elt_lm/gguf_distill_queue.py tests/test_gguf_distill_queue.py`
- Observed locally:
  - `pyright` passed with `0 errors`
  - `pytest` printed all tests passing, but this environment returned exit code `137` after completion; treat the textual pass output as the primary signal until the runner oddity is investigated.

# Next Session Notes

- Start the queue with:
  - `uv run elt-gguf-distill-queue --config configs/gguf_distill_detection_two_teacher_queue.yaml`
- Queue was launched once during this session and writes orchestration status under `H:/elt_data/gguf_distill/detection_two_teacher_queue/{status,heartbeat}.json`.
- If the live Huihui run is already running, the queue waits; if it is already complete, the queue skips it and moves straight to the Qwen3.5-9B HauhauCS stage.
