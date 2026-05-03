# Synthetic v2 OpenClaw/Helmes agent dataset

## Goal

Create an additional high-quality synthetic dataset for OpenClaw and Helmes
general-agent use, separate from the existing code/math/stem/tool lanes but
compatible with the current verifier-backed tool-use pipeline.

## Files touched

- `src/elt_lm/synthetic_v2_agent.py`
- `tests/test_synthetic_v2_agent.py`
- `pyproject.toml`
- `training_data/DATA_SOURCES.md`
- `training_data/synthetic_v2_agent/*` after generation

## Key decisions

- Kept `metadata.lane=tool_use` so existing SFT/tokenization/replay tools can
  consume the data without widening `LaneName`.
- Added `metadata.agent_lane=openclaw_helmes_agent` and focused tags for
  OpenClaw, Helmes, release ops, security triage, monitoring, connector safety,
  and LoRA-SFT-to-GRPO bridge work.
- Used strict `json_match` references for deterministic quality gates.
- Added failure-contrast rows for unsafe mutation, missing evidence, wrong tool
  order, secret exposure, ignored stop conditions, and missing replay guards.
- Positioned the dataset as short low-LR lane LoRA SFT footing with replay and
  early stopping before returning to bridge GRPO.

## Tests

Focused tests:

```powershell
uv run --no-sync pytest tests/test_synthetic_v2_agent.py -q
```

Result: `2 passed`.

Synthetic-v2 + pipeline regression slice:

```powershell
uv run --no-sync pytest tests/test_synthetic_v2_agent.py tests/test_synthetic_v2_hard.py tests/test_pipeline_orchestrator.py -q
```

Result: `54 passed`.

Additional generator smoke:

```powershell
uv run --no-sync python -m elt_lm.synthetic_v2_agent --output-root training_data/synthetic_v2_agent --records 1024 --val-ratio 0.25
```

Result: 1024 SFT records, 1024 failure-contrast records, verifier pass rate
1.000, failure expected-zero rate 1.000, duplicate prompt count 0.

Syntax check:

```powershell
uv run --no-sync python -m compileall -q src/elt_lm/synthetic_v2_agent.py
```

Result: passed.

## Next session notes

- The agent dataset can be prepared as a normal `tool_use` lane input while
  filtering by `metadata.agent_lane == "openclaw_helmes_agent"` if needed.
- For the LoRA SFT bridge, mix this data with v1/tool replay and stop early on
  format rate, JSON verifier pass rate, and val loss before bridge GRPO.
