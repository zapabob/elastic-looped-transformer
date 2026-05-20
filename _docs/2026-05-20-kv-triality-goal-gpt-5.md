# 2026-05-20 KV/Triality evidence bundle - gpt-5

## Goal

Build a publication-facing evidence bundle for the current Qwen3.5 ELT handoff:
K protected as BF16/Q8_0, V swept through Turbo3/Turbo4/Turbo8, Triality SO(8)
rotation audit checks, ILSD entropy/divergence monitoring, paired CV statistics,
README updates, and HF/GH-ready artifacts.

## Files touched

- `scripts/thetom_k_protected_kv_sweep.py`
- `scripts/render_kv_triality_goal_report.py`
- `tests/test_kv_triality_goal_report.py`
- `README.md`
- `_docs/assets/2026-05-17-l3-thetom-k-protected/*`
- `_docs/assets/2026-05-20-kv-triality-goal/*`

## Key decisions

- Added `K=q8_0,V=turbo8` and `K=bf16,V=turbo8` to the KV sweep as probe rows.
  The installed llama.cpp runtime rejects `turbo8`, so the rows are published as
  unsupported runtime evidence rather than omitted or treated as failed quality.
- Reused the existing TheTom K-protected KV logs and only executed the missing
  Turbo8 probes with `--reuse-existing`.
- Copied the Turboquant-CUDA Triality SO(8) audit outputs into this repo's
  evidence bundle. The vector-view 3/4/8-bit rows all pass the 0.01
  orthogonality/determinant gate.
- Generated a deterministic `gptimage2`-style dashboard with matplotlib so the
  figure is reproducible from checked-in numeric artifacts.
- Kept lm-eval-harness claims bounded: the repo `uv` environment cannot import
  `lm_eval`; a global CLI is visible, but no broad logged lm-eval run was
  completed in this bundle.

## Verification

- `.\.venv\Scripts\python.exe -m py_compile scripts\thetom_k_protected_kv_sweep.py scripts\render_kv_triality_goal_report.py tests\test_kv_triality_goal_report.py`
- `.\.venv\Scripts\python.exe -m pytest -q tests\test_kv_triality_goal_report.py tests\test_gguf_quant_report.py::test_quantization_lane_boundaries_keep_kv_and_dflash_separate tests\test_so8_adapter.py`
- `.\.venv\Scripts\python.exe scripts\thetom_k_protected_kv_sweep.py --llama-cli %LOCALAPPDATA%\Programs\llama-turboquant\bin\llama-cli.exe --model H:\elt_data\releases\elt-lm-qwen35-side-stem-aha-ilsd-l3-Q8_0.gguf --out-dir _docs\assets\2026-05-17-l3-thetom-k-protected --repetitions 2 --gen-tokens 8 --ctx-size 128 --ngl 0 --timeout-sec 30 --reuse-existing`
- `.\.venv\Scripts\python.exe scripts\render_kv_triality_goal_report.py`

## Notes for next session

- The Turbo8 gate is blocked by the installed llama.cpp argument parser:
  `Unsupported cache type: turbo8`.
- The new paired CV p-values are loop-aware local STEM bridge statistics, not
  lm-eval leaderboard results.
- If a future runtime adds Turbo8, rerun the same sweep without changing the
  README claim boundary until successful rows and logs exist.
