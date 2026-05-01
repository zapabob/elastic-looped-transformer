# Goal

Prepare the KL-GRPO path for the Qwen3.5-4B side branch after LoRA SFT.

# Files Touched

- `src/elt_lm/train_grpo.py`
- `src/elt_lm/verifiers.py`
- `tests/test_hf_qwen35_looped.py`

# Key Decisions

- GRPO policy construction now uses `build_model(cfg.model)` instead of directly
  instantiating `ELTLanguageModel`.
- The GRPO initial policy loader accepts both full checkpoints and
  side-branch LoRA adapter-only checkpoints.
- The loader intentionally does not restore the SFT optimizer/RNG state. GRPO
  owns its optimizer and rollout sampling state separately.
- The frozen `ref` and `old` policies are still cloned from the initialized
  policy. This keeps native behavior unchanged. For real 4B side-branch GRPO,
  memory planning is still required because policy/ref/old are full model
  instances even though only LoRA parameters are trainable.
- While validating the GRPO suite, the GSM8K/exact math/MCQ verifier fallback
  was tightened so plain untagged answers no longer receive format credit.

# Tests

- Added a tiny Qwen3.5 side-branch fixture test proving that an adapter-only SFT
  checkpoint can initialize the GRPO policy loader.
- `tests/test_grpo.py` now preserves the intended invariant that correctness is
  gated by the `<think>...</think><answer>...</answer>` format for reasoning
  tasks.

# Next Session Notes

- LoRA SFT can proceed first for code/math/STEM/tool.
- After adapter export/eval, code/math/tool GRPO configs can point
  `grpo.init_ckpt` at the corresponding LoRA SFT `last.pt`.
- If 4B GRPO OOMs, the next design step is memory-sharing/frozen-reference
  optimization for `ref` and `old`, not full fine-tuning.
