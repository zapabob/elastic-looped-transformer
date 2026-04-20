# elastic-looped-transformer — repo notes for Claude

Causal-LM port of ELT (arXiv:2604.09168) + ILSD + GRPO.

## Implementation log convention

Every meaningful implementation session records a log under `_docs/`:

```
_docs/YYYY-MM-DD-{slug}-{AI}.md
```

- `YYYY-MM-DD` — date the work was done (absolute, not relative).
- `{slug}` — kebab-case summary of what changed (e.g. `hf-export-rolling-ckpt`).
- `{AI}` — the model that did the work (e.g. `opus-4-7`, `sonnet-4-6`).

Each log should include: goal, files touched, key decisions, tests added / passing
count, and anything the next session should know. Append; do not overwrite past logs.

## Key conventions

- Code: `src/elt_lm/...` — paper equations preserved verbatim in comments.
- Configs: `configs/*.yaml` (tiny_10M / base_100M / grpo_gsm8k / sft_cot).
- Scripts: `scripts/*.py` — data DL, cleaning, bin build, HF export, corpus manifests.
- Tests: `tests/test_*.py` — run all with `uv run pytest -q`.
- Data roots: `H:/elt_data/raw`, `H:/elt_data/clean`, `H:/elt_data/bin`,
  `H:/elt_data/runs` (bulk) and `./runs` (local smoke).
- Checkpoints: `rolling_{0..keep-1}.pt` round-robin every `rolling_ckpt_interval_sec`,
  `last.pt` hardlink, milestone saves at `step_*.pt` every `save_every`.
- Resume: `uv run elt-train --config <cfg> --resume <path-to-ckpt>`.

## Paths (Windows)

- Project: `C:\Users\downl\Desktop\新しいフォルダー (7)`
- Tokenizer: `H:\Qwen3.5-9B-official-hf` (Qwen3.5, vocab 248,320)
- From-D legacy corpus: `H:\from_D\dataset\{wikipedia,final}` (used via `scripts/corpus_manifest.yaml`)
- Python: `uv run ...` — env is Py 3.12.9 + CUDA 12.8 + PyTorch 2.x.
