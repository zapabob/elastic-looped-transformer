---
date: 2026-04-21
slug: dense-first-milspec-posttrain
ai: gpt-5
---

# Dense-First MILSPEC Post-Training Implementation

## Goal

Implement the dense-first MILSPEC post-training plan without introducing MoE:

- add benchmark-capable baseline measurement before post-training,
- add a manifest-driven post-training data pipeline,
- add reward-model and hybrid GRPO support,
- add a lightweight agent runtime scaffold for audit, replay, sandboxing, and SBOM,
- keep the existing dense training path and tokenizer unchanged.

## Files Touched

### Evaluation

- `src/elt_lm/eval/anytime_sweep.py`
- `src/elt_lm/eval/benchmarks.py`
- `tests/test_benchmarks.py`
- `tests/test_inference_sweep.py`

### Post-training data

- `src/elt_lm/posttrain_data.py`
- `src/elt_lm/synthetic_preferences.py`
- `scripts/download_posttrain_data.py`
- `scripts/generate_milspec_preferences.py`
- `scripts/merge_preference_pairs.py`
- `scripts/posttrain_manifest.yaml`
- `scripts/posttrain_security_style_seed.jsonl`
- `scripts/posttrain_code_tokenize_manifest.yaml`
- `scripts/posttrain_tool_tokenize_manifest.yaml`
- `scripts/posttrain_reasoning_tokenize_manifest.yaml`
- `tests/test_posttrain_data.py`

### Reward model and hybrid GRPO

- `src/elt_lm/config.py`
- `src/elt_lm/reward_model.py`
- `src/elt_lm/train_reward_model.py`
- `src/elt_lm/train_grpo.py`
- `src/elt_lm/verifiers.py`
- `configs/reward_model.yaml`
- `configs/grpo_milspec.yaml`
- `configs/posttrain_code_sft.yaml`
- `configs/posttrain_tool_sft.yaml`
- `configs/posttrain_reasoning_sft.yaml`
- `tests/test_reward_model.py`
- `tests/test_train_grpo_hybrid_smoke.py`
- `tests/test_verifiers_quality.py`
- `tests/test_posttrain_configs.py`

### Agent runtime scaffold

- `src/elt_lm/agent/__init__.py`
- `src/elt_lm/agent/audit.py`
- `src/elt_lm/agent/replay.py`
- `src/elt_lm/agent/sandbox.py`
- `src/elt_lm/agent/sbom.py`
- `src/elt_lm/agent/runtime.py`
- `configs/milspec_agent.yaml`
- `tests/test_agent_runtime.py`

### Small type/compatibility fixes found while bringing the new path fully green

- `src/elt_lm/hf/configuration_elt.py`
- `src/elt_lm/offload/optim_offload.py`
- `src/elt_lm/train.py`
- `tests/test_hf_export.py`
- `pyproject.toml`
- `uv.lock`

## Key Decisions

1. Stayed dense-only for this cycle.
   - No MoE layers, no weight-offload rewrite, no tokenizer expansion.
   - The plan remains compatible with the current dense checkpoint and optimizer-state offload path.

2. Split benchmark execution from perplexity-only sweep.
   - `anytime_sweep` now supports benchmark manifests and can emit measured benchmark rows alongside perplexity rows.
   - This makes "replace predictions with measured numbers" possible before post-training starts.

3. Kept post-training data separate from pretrain data plumbing.
   - New manifest-driven loader/normalizer handles SFT rows and preference pairs without overloading the pretrain downloader.

4. Added reward-model support as a first-class training surface.
   - `GRPOConfig` was extended with reward-model and reward-mix fields only.
   - No architecture-level config drift was introduced.

5. Put MILSPEC runtime concerns outside GRPO reward.
   - Determinism/audit/replay/SBOM/subprocess hardening live in `src/elt_lm/agent/`.
   - GRPO reward channels for this pass are limited to `correct`, `format`, `python_exec`, `mypy`, `ruff`, and `bandit`.

## Tests Added / Passing Count

New tests added:

- `tests/test_benchmarks.py`
- `tests/test_posttrain_data.py`
- `tests/test_reward_model.py`
- `tests/test_agent_runtime.py`
- `tests/test_verifiers_quality.py`
- `tests/test_train_grpo_hybrid_smoke.py`
- `tests/test_posttrain_configs.py`

Verification run:

- `uv run pyright src tests` -> passed
- `uv run pytest -q` -> passed

Collected test count at the end of the session:

- `129` tests collected

## Next Session Notes

1. Phase 0 should be exercised on a real checkpoint first.
   - Prepare a benchmark manifest for HumanEval/MBPP, GSM8K, MMLU-STEM, LiveCodeBench-lite, and BFCL-simple.
   - Run `anytime_sweep` against the current dense checkpoint and overwrite prediction-only planning with measured results.

2. The new post-training scripts/configs are scaffolding, not a completed data pull.
   - No long-running dataset download, tokenization, SFT, RM training, or GRPO run was launched in this session.

3. Preference synthesis is ready for the approved 10k-pair MILSPEC/JPL generation pass.
   - Run the generator/merge scripts on the target destination once the exact corpus mix is locked.

4. The subprocess sandbox path intentionally stays lightweight.
   - This matches the "3" decision from planning: no Docker/WSL requirement in this cycle.

5. Dirty-tree note:
   - `AGENTS.md` and `_docs/2026-04-21-positioning-analysis-opus-4-7.md` were already present as untracked files and were left untouched.
