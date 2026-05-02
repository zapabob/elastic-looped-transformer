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
- CV eval is bounded to 500 cases per lane/target by writing a limited manifest
  before each `elt-anytime` invocation. This prevents the GB-scale validation
  manifests from expanding into multi-day full-case generation during the
  automated long profile.

# Evaluation Outputs

Cross-validation artifacts are written under:

`H:/elt_data/eval/synthetic_gb_side_lora/<target>/`

Each target writes:

- `cv_results.csv`
- `cv_results.json`
- `metrics.jsonl`
- `manifests/*_limit500.yaml`

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
- The live H-drive synthetic GB benchmark manifests were also patched with
  `limit: 500` so an already-running scheduler process reads bounded validation
  even if it loaded `scripts/pipeline.py` before this source patch.

# 2026-05-02 UTF-8 Launcher Recovery Addendum

Goal: recover the scheduled `synthetic-gb-side-lora-long` run after GRPO failed
at startup with `UnicodeEncodeError: 'cp932' codec can't encode character`.

Files touched:

- `scripts/pipeline.py`
- `scripts/pipeline_register.ps1`
- `tests/test_pipeline_orchestrator.py`
- `_docs/2026-05-02-side-lora-long-grpo-eval-pipeline-gpt-5.md`

Key decisions:

- `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` are now propagated through both
  the scheduler launcher and the nested `cmd.exe` / `VsDevCmd.bat` training
  command path.
- The generated launcher also sets PowerShell input/output encoding to UTF-8 so
  scheduler logs and child process output use one text encoding boundary.
- The stale 05:28 scheduler process was ended with `schtasks /End`, then the
  updated task was started with `schtasks /Run`.

Tests:

- `uv run pytest tests/test_pipeline_orchestrator.py -q` passed, 34 tests.

Current run notes:

- The launcher was regenerated at `H:/elt_data/pipeline_logs/pipeline_launcher.ps1`.
- The restarted run began at `2026-05-02T17:09:03+09:00`.
- `H:/elt_data/runs/grpo_side_lora_code_synthetic_gb/metrics.jsonl` now has
  `run_start` and `grpo_config`, confirming GRPO got past the previous startup
  Unicode crash.

# 2026-05-02 Low-VRAM GRPO Addendum

Goal: fix the next GRPO failure after UTF-8 recovery: CUDA OOM during
`log_softmax(logits_old)` in the side-LoRA synthetic GB code GRPO stage.

Files touched:

- `configs/grpo_side_lora_code_synthetic_gb.yaml`
- `scripts/pipeline.py`
- `src/elt_lm/grpo.py`
- `src/elt_lm/train_grpo.py`
- `tests/test_grpo.py`
- `tests/test_grpo_train_step.py`
- `tests/test_pipeline_orchestrator.py`
- `_docs/2026-05-02-side-lora-long-grpo-eval-pipeline-gpt-5.md`

Key decisions:

- HF side-LoRA GRPO now keeps only one GPU-resident Qwen backbone and stores
  `ref`/`old` as CPU LoRA adapter snapshots. The failed path previously used
  `copy.deepcopy(model)` twice, tripling the frozen backbone on GPU.
- Old/ref policy passes now compute sampled-action log-probs and discard logits
  before the current-policy backward pass. `gather_token_logprobs` now avoids
  materializing full `(B,T,V)` `log_softmax`.
- The code lane's generation budget was raised from 192 to 256 tokens for both
  GRPO rollout and CV eval. A 50k-record sample of code synthetic train outputs
  showed p99 response length of 212 tokens and 2.3% of responses above 192,
  while math/tool outputs remained below their 128-token budgets.

Tests:

- `uv run pytest tests/test_grpo.py tests/test_grpo_train_step.py tests/test_train_grpo_hybrid_smoke.py tests/test_hf_qwen35_looped.py -q` passed, 42 tests.
- `uv run pytest tests/test_pipeline_orchestrator.py tests/test_grpo.py tests/test_grpo_train_step.py -q` passed, 64 tests.
- `uv run python -m py_compile src/elt_lm/grpo.py src/elt_lm/train_grpo.py` passed.

Runtime smoke:

- `grpo_side_lora_code_synthetic_gb_lowmem_smoke`: 1 step with 8-token rollout
  completed and emitted `grpo_step`.
- `grpo_side_lora_code_synthetic_gb_lowmem_fullrollout_smoke`: 1 step with the
  old 192-token budget completed.
- `grpo_side_lora_code_synthetic_gb_lowmem_256_smoke`: 1 step with the new
  256-token budget completed.

Next session notes:

- The scheduled task was disabled while patching to prevent repeated OOM
  retries. Re-enable and run `ELT-LM-Pipeline` after the source tree is ready.
- The previous failing real run has no useful GRPO checkpoint; `.done` files
  still allow the profile to skip completed prepare/SFT/export stages and resume
  from `03_side_lora_synthetic_gb_grpo`.
