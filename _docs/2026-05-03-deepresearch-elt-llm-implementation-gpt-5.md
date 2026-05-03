# DeepResearch ELT and LLM implementation pass

## Goal

Use the completed bridge progress/data analysis, current LLM post-training
practice, and the ELT paper to turn the next training decision into code,
configuration, and auditable artifacts.

## Sources checked

- ELT paper: https://arxiv.org/abs/2604.09168 and HTML v2:
  https://arxiv.org/html/2604.09168v2
- DeepSeekMath / GRPO paper: https://arxiv.org/abs/2402.03300
- Hugging Face TRL GRPO Trainer docs:
  https://huggingface.co/docs/trl/grpo_trainer
- DAPO paper: https://arxiv.org/abs/2503.14476
- Dr. GRPO / R1-Zero critical perspective:
  https://arxiv.org/abs/2503.20783
- GSPO paper: https://arxiv.org/abs/2507.18071
- LoRA paper: https://arxiv.org/abs/2106.09685
- QLoRA paper: https://arxiv.org/abs/2305.14314
- Hugging Face PEFT LoRA guide:
  https://huggingface.co/docs/peft/main/conceptual_guides/lora

## Key research decisions

- ELT's useful transfer to this repo is not "more loops blindly"; it is
  preserving useful intermediate exits through ILSD-style supervision and
  treating compute depth as an eval/export axis.
- LoRA is still the right local iteration surface for the Qwen3.5 side branch:
  it keeps the base model frozen and makes repeated lane repair cheap.
- GRPO should only be extended when there is reward diversity inside generated
  groups. TRL exposes `reward_std` and zero-std fractions for this reason, and
  this repo already logs `reward_std` plus `adv_abs_mean`.
- DAPO, Dr. GRPO, and GSPO all point at the same practical guardrail for this
  run: watch length/clip/advantage pathologies before scaling RL. The immediate
  blocker here is simpler than algorithm choice: the tool lane has no reward
  signal at all.

## Data-driven diagnosis

The bridge GRPO sweep is complete, but the lanes are not equally usable:

- `stem`: ready for export/eval. Mean correct `0.896`, final correct `1.000`,
  max correct `1.000`. It carries a format warning (`0.948` mean), but it is
  the only lane that looks like a current winner.
- `code`: sparse success. Mean correct `0.031`, max correct `0.500`, final
  correct `0.000`. More GRPO is likely premature without replay SFT and prompt
  repair.
- `math`: sparse/unstable success. Mean correct `0.229`, max correct `1.000`,
  final correct `0.000`. This is close to the promising threshold but still
  below it, so use replay/verifier checks before another RL run.
- `tool`: blocked. Mean/max/final correct are all `0.000`, reward never becomes
  nonzero, and advantage signal steps are `0`. Continuing GRPO here would not
  create useful policy updates.

## Implementation

Added a read-only diagnostic CLI:

```powershell
uv run --no-sync elt-analyze-bridge-diagnostics `
  --config configs/bridge_diagnostics.yaml `
  --out-dir _docs/assets/2026-05-03-deepresearch-elt-llm-implementation `
  --prefix bridge_diagnostics
```

The CLI reads only existing `metrics.jsonl` files and writes:

- `_docs/assets/2026-05-03-deepresearch-elt-llm-implementation/bridge_diagnostics.json`
- `_docs/assets/2026-05-03-deepresearch-elt-llm-implementation/bridge_diagnostics.md`

The policy lives in `configs/bridge_diagnostics.yaml`, so thresholds can be
adjusted without changing code. Current defaults classify lanes as:

- `blocked_no_reward_signal`
- `unstable_sparse_success`
- `promising_but_unstable`
- `ready_for_export_eval`

## Files touched

- `src/elt_lm/bridge_diagnostics.py`
- `configs/bridge_diagnostics.yaml`
- `tests/test_bridge_diagnostics.py`
- `pyproject.toml`
- `_docs/assets/2026-05-03-deepresearch-elt-llm-implementation/bridge_diagnostics.json`
- `_docs/assets/2026-05-03-deepresearch-elt-llm-implementation/bridge_diagnostics.md`

## Tests

- `uv run --no-sync pytest tests/test_bridge_diagnostics.py -q` -> 2 passed
- `uv run --no-sync python -m compileall -q src/elt_lm/bridge_diagnostics.py`
- `uv run elt-analyze-bridge-diagnostics --help`
- `uv run --no-sync elt-analyze-bridge-diagnostics --help`
- `uv run --no-sync elt-analyze-bridge-diagnostics --config configs/bridge_diagnostics.yaml --out-dir _docs/assets/2026-05-03-deepresearch-elt-llm-implementation --prefix bridge_diagnostics`

## Next session notes

- Do not continue the `tool` bridge GRPO lane until verifier/schema replay can
  produce nonzero rewards.
- Treat `stem` as the first export/eval candidate.
- For `code` and `math`, do a short replay SFT or prompt/verifier repair pass
  before spending another GRPO continuation.
- Keep native ELT `L=4` decisions separate from the side-LoRA `L=1` bridge
  path; this diagnostic is for the completed side-LoRA bridge sweep.
