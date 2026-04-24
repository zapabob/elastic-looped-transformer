---
date: 2026-04-24
slug: base-1b-clean-continue-smoke
ai: gpt-5
---

# Base 1B Clean Continue Smoke

## Goal

Wire the regenerated clean-bin corpus at `H:/elt_data/bin_clean_2026-04-24`
into native ELT 1B continued-pretraining configs without overwriting the
existing `H:/elt_data/bin` or `configs/base_1B.yaml`.

## Files Touched

- `configs/base_1B_continue_clean_smoke.yaml`
- `configs/base_1B_continue_clean.yaml`
- `tests/test_clean_continue_configs.py`
- `src/elt_lm/train.py`
- `_docs/2026-04-24-base-1b-clean-continue-smoke-gpt-5.md`

## Key Decisions

- The smoke config is isolated under
  `H:/elt_data/runs/base_1B_clean_smoke_2026-04-24`.
- The long-run config is prepared under
  `H:/elt_data/runs/base_1B_clean_continue`, but should not be auto-started.
- Both configs keep the native ELT 1B model, ILSD, and `paged_adamw_8bit`
  optimizer shape from `configs/base_1B.yaml`.
- No resume checkpoint was found under `H:/elt_data/runs/base_1B`, so this path
  is treated as scratch smoke unless a checkpoint is supplied explicitly later.
- `_update_last_hardlink()` now uses `shutil.copy2()` as the fallback instead of
  `torch.load()` plus `torch.save()`. The previous fallback could fail or spend
  excessive memory when copying a multi-GB 1B checkpoint.

## Verification

Commands run:

```powershell
uv run --no-sync pytest -q tests/test_clean_continue_configs.py tests/test_audit_clean_corpus.py
uv run --no-sync pyright src/elt_lm/train.py tests/test_clean_continue_configs.py
uv run --no-sync python -c "from elt_lm.config import load_train_config; c=load_train_config('configs/base_1B_continue_clean_smoke.yaml'); print(c.data.train_bin, c.total_steps, c.run_dir)"
uv run --no-sync elt-train --config configs/base_1B_continue_clean_smoke.yaml
uv run --no-sync python -c "from pathlib import Path; from elt_lm.train import _update_last_hardlink; d=Path('H:/elt_data/runs/base_1B_clean_smoke_2026-04-24'); _update_last_hardlink(d, d/'step_0000002.pt'); print((d/'last.pt').exists(), (d/'last.pt').stat().st_size)"
```

Results:

- Config tests passed: `5 passed`.
- Pyright passed: `0 errors, 0 warnings`.
- Config load printed:
  `H:/elt_data/bin_clean_2026-04-24/train.bin 2 H:/elt_data/runs/base_1B_clean_smoke_2026-04-24`.
- Smoke telemetry emitted `train_config` for `1,536,921,344` total params and
  `1,091,931,904` non-embedding params.
- Smoke train steps were finite:
  - step 0: `loss=12.6875`, `tokens_per_sec=160.849`
  - step 1: `loss=12.75`, `tokens_per_sec=156.160`
- Final checkpoint exists:
  `H:/elt_data/runs/base_1B_clean_smoke_2026-04-24/step_0000002.pt`
  at `3,978,452,992` bytes.
- `last.pt` exists and points/copies to the same checkpoint size.
- A later rerun hit a transient `bitsandbytes` initialization error before
  training. The earlier successful telemetry and checkpoint are the acceptance
  evidence for this smoke; retry long runs from a fresh process/GPU state.

## Next Session Notes

- The long-run command is prepared but was not started:
  `uv run --no-sync elt-train --config configs/base_1B_continue_clean.yaml`.
- If OOM occurs, first lower `data.seq_len` in the smoke config to 512 while
  keeping the full config unchanged.
