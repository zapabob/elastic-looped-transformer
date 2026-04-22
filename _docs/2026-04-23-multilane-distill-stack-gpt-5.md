# Goal

Implement the approved ELT multi-lane GGUF distill stack so the repo can generate and consume non-detection teacher data for `code`, `math`, `stem_reasoning`, and `tool_use`, while preserving backward compatibility for the existing detection pipeline.

# Files Touched

- `src/elt_lm/gguf_distill.py`
- `src/elt_lm/prepare_gguf_lane_sft.py`
- `src/elt_lm/prepare_gguf_detection_sft.py`
- `src/elt_lm/verifiers.py`
- `src/elt_lm/eval/benchmarks.py`
- `pyproject.toml`
- `scripts/prepare_gguf_lane_sft.py`
- `configs/gguf_distill_*_qwen35_hauhaucs.yaml`
- `configs/gguf_distill_qwen35_hauhaucs_multilane_queue.yaml`
- `configs/posttrain_*_qwen35_hauhaucs.yaml`
- `configs/grpo_*_qwen35_hauhaucs.yaml`
- `tests/test_gguf_distill.py`
- `tests/test_prepare_gguf_detection_sft.py`
- `tests/test_prepare_gguf_lane_sft.py`
- `tests/test_verifiers_quality.py`
- `tests/test_benchmarks.py`
- `tests/test_multilane_configs.py`

# Key Decisions

- Kept ELT as the only student branch. `H:/Qwen3.5-9B-official-hf` remains tokenizer/reference only; no direct Qwen-weight import into ELT was added.
- Generalized GGUF distill around a fixed `lane` enum: `detection`, `code`, `math`, `stem_reasoning`, `tool_use`.
- Preserved detection backward compatibility by continuing to accept legacy `domains` and `samples_per_domain`, mapping them internally to lane-aware `tasks` and `samples_per_task`.
- Standardized lane records so every distill row now carries `task`, `reference`, and richer `metadata` including `lane`, `task_name`, and `teacher`.
- Added lane-specific benchmark preparation through `prepare_gguf_lane_sft.py`, while keeping `prepare_gguf_detection_sft.py` as a thin compatibility wrapper.
- Extended verifier coverage with `exact_math`, `mcq_reasoning`, and `json_match`, and made task-specific answer canonicalization handle raw code blocks and raw JSON without forcing `<think>/<answer>` everywhere.
- Created HauhauCS multi-lane configs plus a sequential queue config in the order `code -> math -> stem_reasoning -> tool_use`.
- Added lane-specific SFT configs for all four lanes and GRPO configs for `code`, `math`, and `tool_use`. `stem_reasoning` stops at SFT in v1.

# Verification

- `uv run --no-sync pyright src/elt_lm/gguf_distill.py src/elt_lm/prepare_gguf_lane_sft.py src/elt_lm/prepare_gguf_detection_sft.py src/elt_lm/verifiers.py src/elt_lm/eval/benchmarks.py tests/test_gguf_distill.py tests/test_prepare_gguf_detection_sft.py tests/test_prepare_gguf_lane_sft.py tests/test_verifiers_quality.py tests/test_benchmarks.py tests/test_multilane_configs.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest -q tests/test_gguf_distill.py tests/test_gguf_distill_queue.py tests/test_multilane_configs.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest -q tests/test_prepare_gguf_detection_sft.py tests/test_prepare_gguf_lane_sft.py tests/test_posttrain_configs.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest -q tests/test_verifiers_quality.py tests/test_benchmarks.py`
- `uv run --no-sync elt-gguf-distill --config configs/gguf_distill_code_qwen35_hauhaucs.yaml --dry-run --max-tasks 2`
- `uv run --no-sync elt-gguf-distill --config configs/gguf_distill_math_qwen35_hauhaucs.yaml --dry-run --max-tasks 2`
- `uv run --no-sync elt-gguf-distill --config configs/gguf_distill_stem_qwen35_hauhaucs.yaml --dry-run --max-tasks 2`
- `uv run --no-sync elt-gguf-distill --config configs/gguf_distill_tool_qwen35_hauhaucs.yaml --dry-run --max-tasks 2`
- `uv run --no-sync python -m elt_lm.prepare_gguf_lane_sft --input-root H:/elt_data/gguf_distill/huihui_qwen36_detection --output-root H:/elt_data/posttrain/detection/huihui_qwen36 --tokenizer H:/Qwen3.5-9B-official-hf --lane detection`

# Notes For Next Session

- The new console entry `elt-prepare-gguf-lane-sft` was added to `pyproject.toml`, but `uv run --no-sync` does not expose freshly-added script shims until the environment is synced again. Module execution works immediately.
- In this environment, one pytest invocation returned exit code `137` even though the textual output showed all tests passed. The smaller split test runs completed cleanly and were used as the authoritative verification set.
- The existing uncommitted detection-prep work from the prior session was intentionally kept and folded into the generic lane-prep implementation rather than discarded.
