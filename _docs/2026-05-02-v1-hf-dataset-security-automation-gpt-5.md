# 2026-05-02 v1 HF dataset security automation - GPT-5

## Goal

Before starting the next v1 training pass, fix the two GitHub Dependabot high
alerts, diversify v1 teacher prompts, connect reviewed Hugging Face dataset
sources, and make the resumable Windows cron pipeline run the v1 path with
5-minute rolling checkpoint policy.

## Files touched

- `uv.lock`
- `pyproject.toml`
- `src/elt_lm/gguf_distill.py`
- `src/elt_lm/hf_dataset_mix.py`
- `scripts/pipeline.py`
- `scripts/pipeline_register.ps1`
- `scripts/pipeline_progress_register.ps1`
- `configs/gguf_distill_qwen35_hauhaucs_multilane_v1_queue.yaml`
- `configs/posttrain_*_sft_qwen35_hauhaucs_v1.yaml`
- `configs/grpo_*_qwen35_hauhaucs_v1.yaml`
- `tests/test_hf_dataset_mix.py`
- `tests/test_gguf_distill.py`
- `tests/test_pipeline_orchestrator.py`

## Key decisions

- Dependabot high alerts were both `GitPython` issues in `uv.lock`; upgraded
  `GitPython` from `3.1.46` to `3.1.49`.
- `codex-security` was not available as an installed tool in this session, so
  the security work used local audit commands and GitHub Dependabot alert data.
- v1 uses dedicated output roots and run dirs: `qwen35_9b_hauhaucs_*_v1`,
  `H:/elt_data/posttrain_v1/...`, and `H:/elt_data/runs/*_v1`.
- v1 prompt generation now rotates difficulty, reasoning style, schema style,
  edge cases, and safety boundaries per `variant_index`.
- v1 teacher requests now include a strict JSON system message plus JSON-object
  response formatting. This was added after the first code-v1 attempts produced
  only fallback/refusal-shaped outputs under the quality gate.
- The HauhauCS v1 GGUF configs now launch `llama-server` with
  `--reasoning off --reasoning-budget 0 --reasoning-format none`; Qwen was
  otherwise spending the entire completion budget in `reasoning_content`.
- v1 generation budgets were raised after the first non-thinking code outputs
  still truncated before valid JSON completion: code uses 2048 tokens; math,
  STEM, and tool-use use 1024 tokens.
- `llama-server` launch now forces `--parallel 1` by default for GGUF
  distillation, preventing accidental concurrent requests from exhausting the
  KV cache during long code generations.
- The code prompt now clarifies that MILSPEC-style means reliable contracts and
  tests, not long CRC/crypto/binary-protocol implementations.
- HF dataset fetching is implemented as a reviewed/sampled acquisition stage.
  Sensitive corpora are retained for detection, contrastive, or boundary
  evaluation use, not as operational harm targets.
- `v1-pretrain-posttrain` profile runs:
  `HF dataset sample -> v1 distill -> v1 prepare -> replay pretrain -> v1 SFT -> KL-GRPO -> eval`.
- All new SFT/GRPO configs keep `rolling_ckpt_interval_sec: 300` and
  `rolling_ckpt_keep: 3`.
- The lightweight progress reporter default interval is now 5 minutes.

## Verification

- `uv lock`
- `uv run --no-sync python -m py_compile src/elt_lm/gguf_distill.py src/elt_lm/hf_dataset_mix.py scripts/pipeline.py`
- `uv run --no-sync pytest -q tests/test_hf_dataset_mix.py tests/test_gguf_distill.py tests/test_pipeline_orchestrator.py tests/test_multilane_configs.py`
- `uv run --no-sync python scripts/pipeline.py --profile v1-pretrain-posttrain --dry-run`
- `uv run --no-sync python -m elt_lm.hf_dataset_mix --config configs/hf_dataset_mix_v1.yaml --output-root H:/elt_data/hf_dataset_mix_v1_smoke_fetch --max-rows-per-source 1 --min-sampled-sources 1`
- `powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1 -Profile v1-pretrain-posttrain -StartLongTrain`
- `powershell -ExecutionPolicy Bypass -File scripts/pipeline_progress_register.ps1 -IntervalMinutes 5`
- `Start-ScheduledTask -TaskName ELT-LM-Pipeline`

## HF smoke result

The HF smoke fetch sampled 10 of 16 reviewed sources with one row each. The
production cron stage then sampled 10 of 16 reviewed sources with 32 rows each,
writing 320 rows to `H:/elt_data/hf_dataset_mix_v1`. Expected non-fatal issues
were recorded for gated datasets, missing split/config choices, and legacy
dataset scripts. The stage requires at least one successful sampled source
before it can proceed.

## Automation state

- `ELT-LM-Pipeline` registered through Task Scheduler fallback as an every
  5-minute task with `--profile v1-pretrain-posttrain`.
- `ELT-LM-Progress-Report` registered every 5 minutes.
- After manual start, `H:/elt_data/pipeline_state/status.json` reported
  `01_hauhaucs_v1_multilane_distill`.
- Latest v1 distill status was `code_v1` teacher generation; quality rejections
  are counted in `error_count` and are expected to occur under the v1 gate.
- The initial partial `code_v1` output with only rejected items was reset before
  restart; the existing HF dataset sample stage marker was preserved.
- A later live audit found that code-v1 could accept verifier snippets whose
  `assert` statements lived inside an uncalled `test_*` function. The pipeline
  was stopped, the partial `code_v1` output was treated as disposable, and the
  v1 validator now requires assertions that execute at top level or a test
  function that is called at top level.
- The same audit found a stronger code-verifier failure mode: a verifier could
  redefine the candidate function/classes and then pass against its own
  redefinition. The code lane now rejects verifier snippets that redefine
  top-level candidate APIs, and the prompt tells the teacher that the verifier
  is appended after `assistant_code`.
- After tightening the gate, early code-v1 attempts showed low pass rate because
  the prompt still encouraged large dataclass/API examples. The code prompt and
  code-v1 task variants were narrowed to compact typed utilities, direct
  top-level asserts, no candidate redefinition, and concise verifier snippets.

## Next session notes

- The Task Scheduler profile should be registered with
  `scripts/pipeline_register.ps1 -Profile v1-pretrain-posttrain -StartLongTrain`.
- The progress reporter should be registered with
  `scripts/pipeline_progress_register.ps1 -IntervalMinutes 5`.
- If v1 code distillation is restarted after this fix, reset only
  `H:/elt_data/gguf_distill/qwen35_9b_hauhaucs_code_v1` and the v1 queue state;
  preserve `H:/elt_data/hf_dataset_mix_v1` and its completed stage marker.
- Some HF datasets need follow-up normalization:
  `codeparrot/apps` and `bigbio/pubmed_qa` require script-compatible handling;
  `cais/mmlu` needs an explicit config; `openai/openai_humaneval` and
  `PKU-Alignment/BeaverTails` need non-default splits; `Idavidrein/gpqa` is gated.
