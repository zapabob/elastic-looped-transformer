# Synthetic v2 Hard Training Data

This directory is the redistributable snapshot of the verifier-backed
synthetic-v2-hard data used by the current ELT side-LoRA GRPO/bridge work.

## Contents

- `code/`: Python-exec tasks with correct traces, wrong contrast traces, and
  held-out GRPO prompts.
- `math/`: exact-answer multi-step math tasks with correct traces, wrong
  contrast traces, and bridge prompts.
- `stem_reasoning/`: multiple-choice STEM reasoning tasks with correct traces,
  wrong contrast traces, and bridge prompts.
- `tool_use/`: structured JSON/tool-use tasks with correct traces, wrong
  contrast traces, and held-out GRPO prompts.
- `summary.json`: lane-level generation and verifier statistics.

## Provenance

The records are generated locally by deterministic templates plus verifier
checks in the ELT repository:

- `src/elt_lm/synthetic_v2_hard.py`
- `src/elt_lm/synthetic_v2_code_bridge.py`
- `src/elt_lm/synthetic_v2_reasoning_bridge.py`
- `src/elt_lm/verifiers.py`

The included bridge prompts are the data used by the bridge/easy-hard GRPO
interventions. The train/val JSONL files are suitable for short SFT or
self-distillation refreshes before GRPO.

## Citation

Please cite:

- this repository commit,
- ELT / ILSD: https://arxiv.org/abs/2604.09168,
- GRPO: https://arxiv.org/abs/2402.03300.

