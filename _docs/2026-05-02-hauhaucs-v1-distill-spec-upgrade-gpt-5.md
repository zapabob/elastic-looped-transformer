# 2026-05-02 HauhauCS v1 Distill Spec Upgrade

## Goal

Upgrade the HauhauCS v1 distillation plan so the next generated corpus is not a
format-only smoke set. The new target is verifier-backed data for:

- MCP / AI-agent harness tool use
- production-shaped Python code with type contracts and warning-zero intent
- advanced exact-answer math reasoning for ELT loop refinement
- broad STEM reasoning across physics, chemistry, biology/medicine, CS, and statistics

The existing v0 bundles remain frozen as smoke-only artifacts.

## Background / Requirements

The v0 HauhauCS bundle showed severe quality failures: repeated prompts, code
stubs returning `None`, math answers collapsing to `0`, placeholder STEM
choices, and empty tool arguments. The user requested democratic/free-expression
coverage, including controversial, NSFW, and drug-related topic understanding,
while still avoiding direct operational harm as a supervised target.

## Decisions

- Sensitive material may be used for understanding, classification, contrastive
  routing, and boundary evaluation.
- Directly actionable crime, procurement, weaponization, self-harm facilitation,
  and patient-specific medical instructions remain excluded from SFT/GRPO
  target responses.
- Tool-use v1 now targets MCP or agent-harness schemas with exact JSON calls.
- Code v1 now requires typed public callables and optionally rejects samples
  that fail `ruff check` or `mypy --strict` when those tools are installed.
- Math and STEM v1 prompts now request higher-difficulty, concrete, verifier
  compatible reasoning instead of generic examples.

## Files Touched

- `src/elt_lm/gguf_distill.py`
- `tests/test_gguf_distill.py`
- `configs/gguf_distill_code_qwen35_hauhaucs_v1.yaml`
- `configs/gguf_distill_math_qwen35_hauhaucs_v1.yaml`
- `configs/gguf_distill_stem_qwen35_hauhaucs_v1.yaml`
- `configs/gguf_distill_tool_qwen35_hauhaucs_v1.yaml`
- `configs/hf_dataset_mix_v1.yaml`

## Implementation Details

- Added typed-public-callable validation for code records.
- Added optional `ruff` and `mypy --strict` checks for code v1 if the tools are
  available in the environment.
- Added stricter math and STEM shallow-content rejection.
- Added MCP/agent harness tool-name validation, concrete argument validation,
  and destructive-call guards for tool-use v1.
- Strengthened lane-specific teacher instructions.
- Added a Hugging Face dataset mix manifest for future source-backed replay and
  v1 seed selection.

## Commands Run

```powershell
uv run --no-sync pytest -q tests/test_gguf_distill.py
uv run --no-sync pytest -q tests/test_gguf_distill.py tests/test_pipeline_orchestrator.py tests/test_prepare_mixed_lane_sft.py
uv run --no-sync python -m py_compile src/elt_lm/gguf_distill.py scripts/pipeline.py src/elt_lm/prepare_mixed_lane_sft.py
@'
from elt_lm.gguf_distill import load_gguf_distill_config, build_task_specs
from pathlib import Path
for p in sorted(Path('configs').glob('gguf_distill_*_qwen35_hauhaucs_v1.yaml')):
    cfg = load_gguf_distill_config(p)
    tasks = build_task_specs(cfg)
    print(p.name, cfg.lane, cfg.pipeline.quality_profile, len(tasks), cfg.pipeline.output_root)
'@ | uv run --no-sync python -
```

## Verification Results

- `tests/test_gguf_distill.py`: 23 passed.
- Focused suite: 49 passed.
- `py_compile`: passed for `gguf_distill.py`, `pipeline.py`, and
  `prepare_mixed_lane_sft.py`.
- v1 config load confirmed 128 planned tasks per lane and separate `_v1`
  output roots.

## Residual Risks

- The HF dataset manifest is a reviewed plan, not an ingest implementation.
- `ruff` / `mypy` checks are optional at runtime; if they are unavailable,
  code quality still relies on prompt constraints, execution tests, and static
  typed-callable validation.
- GPQA is gated and should remain benchmark-only unless access and usage terms
  allow training.

## Next Actions

1. Run a tiny v1 generation smoke with `samples_per_task=1` or `2`.
2. Inspect `eval_summary.json` for reject reasons and verifier pass rate.
3. Only after quality gates pass, prepare v1 lane SFT bins.
4. Keep v0 bundles marked as smoke-only and out of the main SFT/GRPO path.
