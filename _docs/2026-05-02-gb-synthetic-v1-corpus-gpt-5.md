# Goal

Build a deterministic, verifier-backed synthetic v1 corpus path for GB-scale
training data after the v0 teacher-distilled HauhauCS bundles proved too weak
for capability training.

# Files Touched

- `src/elt_lm/synthetic_v1_seed.py`
- `scripts/pipeline.py`
- `tests/test_pipeline_orchestrator.py`

# Key Decisions

- Keep the v0 distillation bundles as smoke/pipeline-proof artifacts only.
- Generate exact synthetic data directly on `H:/elt_data` instead of relying on
  weak teacher fallback behavior.
- Use `H:/elt_data/synthetic_v1_seed_gb` as the canonical GB-scale output root.
- Treat `summary.total_bytes >= 1 GiB` as the pipeline completion gate. The old
  `512 records per lane` gate was too weak and caused the pipeline to skip the
  actual GB generation stage.
- Expand the deterministic seed space with more code, math, STEM, and MCP/agent
  tool-use templates while keeping verifier/sample checks at `1.0`.
- Code lane now covers `Python`, `Rust 2024`, `Go`, `TypeScript`, and `C#`.
  Python stays `python_exec`; the other languages use `code_static_spec` with
  concrete compiler/test harness references (`cargo test --edition 2024`,
  `go test`, `tsc --strict`/`npm test`, and `dotnet test`).

# Tests

- `uv run --no-sync pytest -q tests/test_synthetic_v1_seed.py tests/test_pipeline_orchestrator.py tests/test_gguf_distill.py`
- `uv run --no-sync python -m py_compile src/elt_lm/synthetic_v1_seed.py scripts/pipeline.py`
- `uv run --no-sync python scripts/pipeline.py --profile synthetic-v1-pretrain-posttrain --dry-run`
- `uv run --no-sync python -m elt_lm.synthetic_v1_seed --output-root H:/elt_data/synthetic_v1_seed_target_smoke2 --target-bytes 10485760 --validation-sample-per-lane 16 --val-ratio 0.125`

# Smoke Result

The 10 MiB target smoke produced `10,489,461` bytes and `6,936` records.
All lanes had `schema_valid_rate=1.0`, `unique_text_ratio=1.0`, and sampled
`verifier_pass_rate=1.0`. STEM answer labels were balanced across A/B/C/D.

# Next Session Notes

- If `H:/elt_data/pipeline_state/01_build_synthetic_v1_seed.done` exists from
  an older small-seed run, remove it before starting the GB pipeline.
- The synthetic-only profile is:
  `uv run --no-sync python scripts/pipeline.py --profile synthetic-v1-pretrain-posttrain`
- The intended first long artifact is
  `H:/elt_data/synthetic_v1_seed_gb/summary.json` with at least `1,073,741,824`
  bytes.
