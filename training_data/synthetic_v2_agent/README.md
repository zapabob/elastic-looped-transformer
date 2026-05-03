# Synthetic v2 agent: OpenClaw / Helmes

Verifier-backed synthetic SFT and failure-contrast data for general-agent routing, safe tool sequencing, evidence collection, and handoff behavior. Records keep `metadata.lane=tool_use` for existing ELT tokenization/replay compatibility and add `metadata.agent_lane=openclaw_helmes_agent` for agent-specific filtering.

- Correct SFT records: 1024
- Failure contrast records: 1024
- Verifier pass rate: 1.000
- Failure expected-zero rate: 1.000
- Benchmark manifest: `training_data\synthetic_v2_agent\benchmarks\synthetic_v2_agent_val_manifest.yaml`

Recommended use: short low-LR lane LoRA SFT with replay, early stopping on format rate / verifier accuracy / val loss, then bridge GRPO only after the probe improves.
