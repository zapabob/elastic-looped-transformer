# Goal

Allow student detection SFT and validation work to start before GGUF distillation fully completes by preparing partial `distill_train.jsonl` and `distill_val.jsonl` bundles into token bins plus a JSON-match benchmark manifest.

# Files Touched

- `C:\Users\downl\Desktop\新しいフォルダー (7)\src\elt_lm\prepare_gguf_detection_sft.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\scripts\prepare_gguf_detection_sft.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\configs\posttrain_detection_sft_huihui_qwen36.yaml`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\tests\test_prepare_gguf_detection_sft.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\pyproject.toml`

# Key Decisions

- Added a dedicated CLI to convert a GGUF distill output dir into:
  - `bin/train.bin`
  - `bin/val.bin`
  - a validation benchmark JSONL
  - a benchmark manifest for `json_match`
- Kept the benchmark task aligned with current detection responses, which already emit strict JSON strings.
- Added a Huihui-specific detection SFT config so training can begin from the prepared partial bundle without waiting for the full distill queue to finish.

# Tests

- Intended verification:
  - `uv run --no-sync pyright src/elt_lm/prepare_gguf_detection_sft.py tests/test_prepare_gguf_detection_sft.py`
  - `uv run --no-sync python -m pytest -q tests/test_prepare_gguf_detection_sft.py tests/test_posttrain_configs.py`

# Next Session Notes

- Prepare the live Huihui bundle with:
  - `uv run elt-prepare-gguf-detection-sft --input-root H:/elt_data/gguf_distill/huihui_qwen36_detection --output-root H:/elt_data/posttrain/detection/huihui_qwen36 --tokenizer H:/Qwen3.5-9B-official-hf`
- Then the student can be trained with:
  - `uv run elt-train --config configs/posttrain_detection_sft_huihui_qwen36.yaml --resume H:/elt_data/runs/posttrain_reasoning_sft/last.pt`
