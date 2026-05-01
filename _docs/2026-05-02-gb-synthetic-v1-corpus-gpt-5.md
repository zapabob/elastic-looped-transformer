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

# GB Corpus Result

The GB target run wrote `H:/elt_data/synthetic_v1_seed_gb` with
`1,073,745,004` bytes and `693,822` accepted records.

- code: `131,422` records, `268,436,591` bytes, `python_exec=26,285`,
  `code_static_spec=105,137`
- math: `231,995` records, `268,435,551` bytes, sampled verifier pass `1.0`
- stem_reasoning: `148,096` records, `268,436,221` bytes, A/B/C/D each `37,024`
- tool_use: `182,309` records, `268,436,641` bytes, sampled verifier pass `1.0`

All lanes reported `schema_valid_rate=1.0`, `unique_text_ratio=1.0`,
`exact_duplicate_count=0`, and sampled verifier pass `1.0`. Math rejected
`17` generated zero-answer cases via the existing `fallback_zero_answer` gate.

# Math/STEM Dedicated GB Results

After the mixed GB run, dedicated higher-volume math and STEM bundles were
generated as separate artifacts:

- `H:/elt_data/synthetic_v1_math_gb`: `1,073,742,145` bytes, `940,182`
  accepted records, sampled verifier pass `1.0`, exact duplicates `0`.
- `H:/elt_data/synthetic_v1_stem_gb`: `1,073,742,371` bytes, `588,938`
  accepted records, sampled verifier pass `1.0`, exact duplicates `0`.

Math now includes algebra, exact arithmetic, probability, Bayes, quadratic
roots, vector dot products, matrix determinants, polynomial derivatives,
definite integrals, conditional probability, geometric recurrences, and
inclusion-exclusion. STEM now includes physics, chemistry, statistics,
medicine, pharmacokinetics, control systems, signal aliasing, genetics, renal
clearance, machine-learning calibration, thermodynamics, and diagnostic
likelihood-ratio reasoning. STEM answer distribution remained balanced:
`A=147,235`, `B=147,235`, `C=147,234`, `D=147,234`.

# Code/Tool Dedicated GB Results

Code and tool/MCP/agent harness data were then split into their own dedicated
GB-scale artifacts so the 4B side branch can learn them through LoRA without
mixing them into the math/STEM adapter schedule.

- `H:/elt_data/synthetic_v1_code_gb`: `1,073,743,448` bytes, `525,172`
  accepted records, sampled verifier pass `1.0`, exact duplicates `0`.
- `H:/elt_data/synthetic_v1_tool_gb`: `1,073,742,896` bytes, `724,384`
  accepted records, sampled verifier pass `1.0`, exact duplicates `0`.

Both dedicated artifacts were uploaded to Hugging Face:

- `https://huggingface.co/datasets/zapabobouj/elt-synthetic-v1-code-gb`
- `https://huggingface.co/datasets/zapabobouj/elt-synthetic-v1-tool-gb`

The dedicated code corpus covers Python executable tasks plus Rust 2024, Go,
TypeScript, and C# static-spec harnesses. The tool corpus covers JSON-match
tool calls, MCP file/search/resource/dataset/status operations, AI-agent dry
run planning, browser inspection, memory search, static/security scan,
checkpoint resume checks, eval reranking, and CI matrix harnesses.

# LoRA Handoff

The dedicated code/math/STEM/tool GB corpora should feed the 4B side branch
through adapter-only LoRA before later GRPO/eval stages. Full 4B fine-tuning is
avoided. New configs isolate those runs:

- `configs/qwen35_4b_side_lora_code_sft_synthetic_gb.yaml`
- `configs/qwen35_4b_side_lora_math_sft_synthetic_gb.yaml`
- `configs/qwen35_4b_side_lora_stem_sft_synthetic_gb.yaml`
- `configs/qwen35_4b_side_lora_tool_sft_synthetic_gb.yaml`

The pipeline profile is `synthetic-gb-side-lora`:

1. prepare `H:/elt_data/synthetic_v1_code_gb/code`,
   `H:/elt_data/synthetic_v1_math_gb/math`,
   `H:/elt_data/synthetic_v1_stem_gb/stem_reasoning`, and
   `H:/elt_data/synthetic_v1_tool_gb/tool_use`
2. run 4B side LoRA SFT for code, math, STEM, then tool-use
3. export adapter-only artifacts
4. run eval compare

Prepare was executed successfully. Token inventory:

- math: `822,659` train records, `117,523` val records,
  `101,160,065` train tokens, `14,366,028` val tokens
- stem_reasoning: `515,320` train records, `73,618` val records,
  `77,833,888` train tokens, `11,014,091` val tokens
- code: `459,525` train records, `65,647` val records,
  `85,610,590` train tokens, `12,128,176` val tokens
- tool_use: `633,836` train records, `90,548` val records,
  `77,545,121` train tokens, `11,035,548` val tokens

The Qwen3.5-4B side bootstrap checkpoint exists at
`H:/elt_data/runs/qwen35_4b_elt_bootstrap/last.pt`.

# Next Session Notes

- If `H:/elt_data/pipeline_state/01_build_synthetic_v1_seed.done` exists from
  an older small-seed run, remove it before starting the GB pipeline.
- The synthetic-only profile is:
  `uv run --no-sync python scripts/pipeline.py --profile synthetic-v1-pretrain-posttrain`
- The intended first long artifact is
  `H:/elt_data/synthetic_v1_seed_gb/summary.json` with at least `1,073,741,824`
  bytes.
