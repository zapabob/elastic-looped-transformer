# Qwen3.5 4B Side SFT Smoke

## Goal

Prepare completed HauhauCS GGUF distill lane bundles for SFT, bootstrap the
HF-backed Huihui/Qwen3.5-4B ELT side branch, and run short SFT smoke checks
without starting long training or GRPO.

## Files Touched

- `configs/qwen35_4b_side_sft_code_smoke_l1.yaml`
- `configs/qwen35_4b_side_sft_code_smoke_l2.yaml`
- `configs/qwen35_4b_side_lora_code_smoke_l1.yaml`
- `configs/qwen35_4b_side_lora_code_smoke_l2.yaml`
- `pyproject.toml`
- `src/elt_lm/bootstrap_qwen35_elt.py`
- `src/elt_lm/config.py`
- `src/elt_lm/export_lora_adapter.py`
- `src/elt_lm/hf_qwen35_looped.py`
- `src/elt_lm/offload/placement.py`
- `src/elt_lm/offload/tiered_store.py`
- `src/elt_lm/train.py`
- `scripts/pipeline.py`
- `scripts/pipeline_register.ps1`
- `tests/test_hf_qwen35_looped.py`
- `tests/test_offload_hooks.py`
- `tests/test_pipeline_orchestrator.py`
- `tests/test_qwen35_side_smoke_configs.py`
- `_docs/2026-04-28-qwen35-side-sft-smoke-gpt-5.md`

## Key Decisions

- Kept existing native ELT HauhauCS posttrain configs unchanged.
- Added separate HF-backed 4B side smoke configs because the existing
  `posttrain_*_qwen35_hauhaucs.yaml` files are native ELT configs.
- Used `nvme_adamw`, short sequence length, and 1-2 total steps for smoke only.
- Left 4B-side GRPO out of this pass because the GRPO trainer is still native
  `ELTLanguageModel`-specific.
- Saved bootstrap checkpoints in `model.parity_dtype` to avoid accidental fp32
  4B checkpoints.
- Extended NVMe optimizer offload to HF-backed side branches that do not expose
  native ELT's `composite` module.
- Added a `side_fastpath` optional dependency set for the Qwen3.5/Huihui side
  branch fast path. On this Windows host, `causal-conv1d` required a local
  CUDA 12.8 + VS2022 build.
- Added an in-repo minimal LoRA path for the HF Qwen3.5 side branch instead of
  pulling PEFT/QLoRA dependencies into the Windows smoke path.
- LoRA mode freezes the bf16 base and trains only adapter matrices on selected
  attention, MLP, and Qwen3.5 linear-attention projections. The L1 smoke uses
  rank 8 across all layers; the L2 loop-introduction smoke uses rank 16 on the
  top 8 layers.
- The side branch needs a VS2022 `CC=cl.exe` environment for Triton kernels on
  this host. The default TCC path misses `tccdefs.h`.
- Adapter-only checkpoints are now supported for LoRA side training. Each
  adapter checkpoint stores `base_checkpoint` plus LoRA tensors and optimizer
  state instead of another full 4B model copy.
- Added `elt_lm.export_lora_adapter` for exporting `adapter.pt` and
  `adapter_config.json` from either adapter-only or full LoRA checkpoints.
- Added pipeline profile `side-lora` so Task Scheduler can run only
  side-branch LoRA SFT, L2 ILSD, and adapter export without waiting for the
  native full pipeline stages.
- Pipeline VS2022 setup uses the 8.3 path
  `C:\PROGRA~1\MICROS~4\2022\COMMUN~1\Common7\Tools\VsDevCmd.bat` because
  Python subprocess quoting can break quoted `.bat` paths on Windows.

## Commands And Results

- `uv run --no-sync pytest -q tests/test_qwen35_side_smoke_configs.py tests/test_prepare_gguf_lane_sft.py tests/test_hf_qwen35_looped.py`
  - `10 passed`
- `elt-prepare-gguf-lane-sft` for HauhauCS `detection/code/math/stem_reasoning/tool_use`
  - detection: `33,751` train tokens / `3,817` val tokens
  - code: `9,907` train tokens / `1,117` val tokens
  - math: `9,299` train tokens / `1,053` val tokens
  - stem_reasoning: `11,749` train tokens / `1,331` val tokens
  - tool_use: `8,697` train tokens / `975` val tokens
- `uv run --no-sync elt-bootstrap-qwen35-elt --hf-model huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated --out H:/elt_data/runs/qwen35_4b_elt_bootstrap/last.pt --tokenizer H:/Qwen3.5-9B-official-hf --template-config configs/qwen35_4b_elt_bootstrap.yaml --loop-bootstrap-L-max 1`
  - checkpoint created at `H:/elt_data/runs/qwen35_4b_elt_bootstrap/last.pt`
  - final bf16 checkpoint size: about `7.834 GB`
- `uv run --no-sync pytest -q tests/test_offload_hooks.py tests/test_offload_config.py tests/test_qwen35_side_smoke_configs.py tests/test_hf_qwen35_looped.py`
  - `20 passed`
- `uv pip install triton-windows flash-linear-attention --no-deps`
  - installed `triton-windows==3.6.0.post26`, `flash-linear-attention==0.5.0`
- `uv pip install fla-core ninja packaging wheel setuptools --no-deps`
  - installed missing build/runtime helpers, including `fla-core==0.5.0`
- `uv pip install causal-conv1d --no-build-isolation` via ASCII junction,
  VS2022 dev prompt, CUDA 12.8, and `TORCH_DONT_CHECK_COMPILER_ABI=1`
  - built and installed `causal-conv1d==1.6.1`
- Import check:
  - `triton`: present
  - `fla`: present
  - `causal_conv1d`: present
  - `causal_conv1d_cuda`: present
- Full side L1 smoke with VS2022 `CC=cl.exe`:
  - `loss 2.7656`, finite
  - saved `H:/elt_data/runs/qwen35_4b_side_sft_code_smoke_l1/last.pt`
  - completed in `28.3 min`
- Full side L2 smoke with VS2022 `CC=cl.exe`:
  - step 0 `loss 4.6875`, step 1 `loss 25.9732`, finite
  - saved `H:/elt_data/runs/qwen35_4b_side_sft_code_smoke_l2/last.pt`
  - completed in `47.7 min`
- `uv run --no-sync pytest -q tests/test_hf_qwen35_looped.py tests/test_qwen35_side_smoke_configs.py tests/test_offload_hooks.py`
  - `18 passed`
- LoRA L1 smoke with VS2022 `CC=cl.exe`:
  - config: `configs/qwen35_4b_side_lora_code_smoke_l1.yaml`
  - rank 8 adapters on attention, MLP, and linear-attention projections
  - 5 steps completed in `7.7 min`
  - loss trace: `3.2188 -> 2.9062 -> 1.7891 -> 1.7656 -> 1.4297`
  - saved `H:/elt_data/runs/qwen35_4b_side_lora_code_smoke_l1/last.pt`
- LoRA L2 ILSD smoke with VS2022 `CC=cl.exe`:
  - config: `configs/qwen35_4b_side_lora_code_smoke_l2.yaml`
  - rank 16 adapters on the top 8 layers
  - 2 steps completed in `8.7 min`
  - loss trace: `5.0625 -> 8.0313`, finite; short loop-intro smoke only
  - saved `H:/elt_data/runs/qwen35_4b_side_lora_code_smoke_l2/last.pt`
- Adapter-only save smoke:
  - config: `configs/qwen35_4b_side_lora_code_smoke_l1.yaml`
  - override: `total_steps=1`, `run_dir=H:/elt_data/runs/qwen35_4b_side_lora_adapter_smoke_l1`
  - `last.pt` size: `97,990,909` bytes, down from about `8.5 GB`
  - exported adapter: `H:/elt_data/adapters/qwen35_4b_side/adapter_smoke_l1/adapter.pt`
  - `adapter.pt` size: `32,625,247` bytes
  - adapter metadata: `496` tensors, `16,232,448` parameters, rank `8`
- `uv run --no-sync pytest -q tests/test_hf_qwen35_looped.py tests/test_qwen35_side_smoke_configs.py tests/test_pipeline_orchestrator.py tests/test_offload_hooks.py tests/test_rolling_ckpt.py`
  - `36 passed`
- `uv run --no-sync python scripts/pipeline.py --profile side-lora --dry-run`
  - planned `00_side_lora_sft -> 01_side_lora_ilsd -> 02_export_side_lora_adapters`
- `powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1 -Profile side-lora -StartLongTrain`
  - advanced Task Scheduler registration required elevation and failed
  - simple 5-minute `schtasks.exe` fallback succeeded for `ELT-LM-Pipeline`
  - launcher uses `--profile side-lora` and `ELT_PIPELINE_START_LONG=1`
- `schtasks.exe /Run /TN ELT-LM-Pipeline`
  - started side-lora pipeline immediately
  - `H:/elt_data/pipeline_state/status.json` showed `state=running`,
    `current_stage=00_side_lora_sft`

## Next Session Notes

- `flash-linear-attention` imports as `fla`; `flash_linear_attention` is not a
  top-level module in this package version.
- The first L=1 smoke was interrupted, and before interruption exposed that
  native-only NVMe offload expected `model.composite`. That path has now been
  extended for HF-backed side branches.
- If `causal-conv1d` must be rebuilt on Windows, use an ASCII path such as
  `C:/elt_lm_ascii`, VS2022 Community dev prompt, CUDA 12.8, and
  `TORCH_DONT_CHECK_COMPILER_ABI=1`.
- LoRA timing smoke is now complete. Next safe step is to promote the L1 LoRA
  config from smoke to a longer code-lane SFT run, then repeat for math/tool.
- The long side-lora pipeline is registered and has been started. Monitor:
  `Get-Content H:/elt_data/pipeline_state/status.json`
  and latest `H:/elt_data/pipeline_logs/pipeline-*.log`.
- If the simple scheduled task must be removed:
  `powershell -ExecutionPolicy Bypass -File scripts/pipeline_unregister.ps1`.
- Non-LoRA full side smoke is functional but too slow for iteration on this
  RTX 3060 setup. Prefer LoRA first, then consider QLoRA only after the
  bitsandbytes/4-bit path is validated separately.
