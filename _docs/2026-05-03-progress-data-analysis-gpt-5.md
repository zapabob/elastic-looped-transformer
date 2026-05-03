# Latest progress and data analysis

## Goal

Capture the 2026-05-03 ELT progress snapshot, analyze the newly generated synthetic-v2-agent dataset, and add project-local visual summaries including a GPT Image variant.

## Files touched

- `_docs/2026-05-03-progress-data-analysis-gpt-5.md`
- `_docs/assets/2026-05-03-progress-data-analysis/latest_progress_analysis.json`
- `_docs/assets/2026-05-03-progress-data-analysis/agent_focus_counts.csv`
- `_docs/assets/2026-05-03-progress-data-analysis/elt_progress_data_analysis.png`
- `_docs/assets/2026-05-03-progress-data-analysis/gptimage_elt_progress_infographic.png`
- `_docs/assets/2026-05-03-progress-data-analysis/gptimage_prompt.md`

## Current progress snapshot

- Generated at: `2026-05-03T10:05:44.725681+09:00`
- Pipeline state: `running` at `04_side_lora_synthetic_v2_bridge_grpo` (stage index 5 of 6).
- Active bridge GRPO run: `H:\elt_data\runs\grpo_side_lora_math_synthetic_v2_bridge`; latest parsed GRPO step `7`.
- Active latest format/correct rates: `1.000` / `0.500`; latest loss `-0.000716`; latest KL `0.000275`.
- Active correct-rate max/mean so far: `1.000` / `0.250` across `8` parsed GRPO steps.
- Completed code bridge GRPO: run_end `True`, final step `96`, final checkpoint age `830.2` sec at analysis time.
- Active checkpoint age at analysis time: `197.7` sec; active metrics age `197.7` sec.
- GPU snapshot from progress reporter: `12009/12288 MB`, util `100%`, temp `52 C`.
- Disk snapshot from progress reporter: C `7.68` GB free; H `87.62` GB free.

## Synthetic-v2-agent dataset analysis

- Correct SFT records: `1024`; failure-contrast records: `1024`.
- Split: train `768`, val `256`.
- Verifier pass rate: `1.000`; failure expected-zero rate: `1.000`.
- Duplicate checks: exact duplicate count `0`, duplicate prompt count `0`.
- Difficulty balance: `{'bridge': 512, 'hard': 512}`.
- Safety-risk balance: `{'medium': 512, 'high': 512}`.
- Domain coverage: `12` task domains, balanced `85` to `86` records each.
- Agent focus coverage: `12` buckets, balanced `85` to `86` records each.

## Visualization

![ELT progress data analysis](assets/2026-05-03-progress-data-analysis/elt_progress_data_analysis.png)

![GPT Image ELT progress infographic](assets/2026-05-03-progress-data-analysis/gptimage_elt_progress_infographic.png)

The exact machine-readable snapshot is stored in `latest_progress_analysis.json`. `elt_progress_data_analysis.png` is the deterministic local chart with exact data labels. `gptimage_elt_progress_infographic.png` is the GPT Image variant generated from `gptimage_prompt.md`.

## Key decisions

- Kept the analysis artifacts under `_docs/assets/` so runtime files under `H:/elt_data` remain local-only.
- Treated progress reporter output as a snapshot, not a completion claim; the bridge stage was still marked running while active metrics had moved to math GRPO.
- Used deterministic local plotting for exact numbers and a separate GPT Image asset for presentation-grade rendering.

## Tests

```powershell
uv run --no-sync pytest tests/test_synthetic_v2_agent.py tests/test_synthetic_v2_hard.py tests/test_pipeline_orchestrator.py -q
```

Result: `54 passed`.

```powershell
uv run --no-sync python scripts/pipeline.py --profile synthetic-v2-agent --dry-run
```

Result: passed; dry-run plan contains `00_build_synthetic_v2_agent`.

```powershell
uv run --no-sync pytest -q
```

Result: full suite passed.

## Next session notes

- Active bridge GRPO has moved from code to math; continue watching whether math maintains nonzero reward windows and correct-rate lift.
- C: free space was low in the snapshot; keep large checkpoint/cache writes on H:.
- Agent data is ready for a short low-LR lane LoRA SFT probe with replay and early stopping before bridge GRPO.
