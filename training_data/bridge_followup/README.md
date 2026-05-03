# Bridge follow-up action plan

| lane | state | next artifact |
|---|---|---|
| stem | ready_for_export_eval | `stem_eval/stem_bridge_eval_manifest.yaml` |
| code | sparse_success | `code_replay/distill_train.jsonl` |
| math | sparse_success | `math_replay/distill_train.jsonl` |
| tool | blocked_no_reward_signal | `tool_use_repair/distill_train.jsonl` |

Use the tool repair verifier for probe GRPO only; keep exact JSON match for final eval.
