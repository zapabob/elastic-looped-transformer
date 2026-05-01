# Goal

Automate the long-running Qwen3.5-4B side-branch path after GB synthetic LoRA
SFT, then evaluate with fold-based statistical validation and an optional
LM-evaluation-harness handoff.

# Files Touched

- `configs/grpo_side_lora_code_synthetic_gb.yaml`
- `configs/grpo_side_lora_math_synthetic_gb.yaml`
- `configs/grpo_side_lora_tool_synthetic_gb.yaml`
- `scripts/pipeline.py`
- `scripts/pipeline_register.ps1`
- `src/elt_lm/train_grpo.py`
- `src/elt_lm/eval/anytime_sweep.py`
- `src/elt_lm/eval/benchmarks.py`
- `src/elt_lm/eval/statistics.py`
- `tests/test_pipeline_orchestrator.py`
- `tests/test_eval_statistics.py`

# Key Decisions

- The long profile is `synthetic-gb-side-lora-long`.
- Long profile order:
  1. prepare synthetic GB lane bins
  2. run code/math/STEM/tool LoRA SFT
  3. export SFT adapters
  4. run KL-constrained GRPO for code/math/tool only
  5. export GRPO adapters
  6. run 5-fold CV benchmark eval for SFT and GRPO adapters
  7. optionally run `lm-eval` if `ELT_LM_EVAL_MODEL_ARGS` is configured
- STEM remains SFT+eval only in this phase because the verifier is weaker than
  code/math/tool.
- GRPO now respects per-row `task` in benchmark prompt JSONL, so mixed code
  prompt pools can score `python_exec` and `code_static_spec` examples through
  the right verifier.
- `elt-anytime` now uses `build_model` and adapter-aware checkpoint loading, so
  side LoRA checkpoints can be evaluated.
- JSONL benchmark rows may override the manifest task field with their own
  `task`, which is required for mixed code-lane validation.
- The optional LM-eval-harness stage is non-blocking unless the environment is
  configured with `ELT_LM_EVAL_MODEL_ARGS`. This keeps the custom ELT runtime
  path reliable while still allowing standard HF/local-server baselines.

# Evaluation Outputs

Cross-validation artifacts are written under:

`H:/elt_data/eval/synthetic_gb_side_lora/<target>/`

Each target writes:

- `cv_results.csv`
- `cv_results.json`
- `metrics.jsonl`

The CSV/JSON include overall score plus `cv_mean`, `cv_std`, `cv_sem`,
`cv_ci95_low`, `cv_ci95_high`, format CV stats, loop gain, self-correction, and
overthinking fields.

# LM-eval-harness Handoff

The optional stage follows the current EleutherAI CLI shape:

```powershell
$env:ELT_LM_EVAL_MODEL = "hf"
$env:ELT_LM_EVAL_MODEL_ARGS = "pretrained=/path/to/hf/model,dtype=bfloat16"
$env:ELT_LM_EVAL_TASKS = "hellaswag,arc_challenge,gsm8k"
uv run --no-sync python scripts/pipeline.py --profile synthetic-gb-side-lora-long --only 06_lm_eval_harness_optional
```

For the custom looped ELT side branch, the primary eval path remains
`elt-anytime` because it can load `hf_qwen35_looped` adapter-only checkpoints
directly.

# Next Session Notes

- Register long automation with:
  `powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1 -Profile synthetic-gb-side-lora-long -StartLongTrain`
- The first long stage currently pending is synthetic GB side LoRA SFT unless a
  newer `last.pt` exists in the lane run directories.
- If side GRPO OOMs, reduce `group_size`, `rollout_max_new_tokens`, or implement
  shared frozen base/reference storage before raising LoRA rank.
