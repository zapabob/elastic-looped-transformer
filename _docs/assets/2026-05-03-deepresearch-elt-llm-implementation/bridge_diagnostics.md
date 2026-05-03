# Bridge GRPO diagnostics

- Run root: `H:\elt_data\runs`
- Lanes analyzed: `4`
- Classification counts: `{"blocked_no_reward_signal": 1, "ready_for_export_eval": 1, "unstable_sparse_success": 2}`

## Decision table

| lane | class | mean correct | max correct | final correct | format | adv steps | action |
|---|---:|---:|---:|---:|---:|---:|---|
| code | unstable_sparse_success | 0.031 | 0.500 | 0.000 | 1.000 | 25 | run replay SFT and prompt repair before more GRPO; sparse successes mean the verifier can score the task but the policy has not stabilized |
| math | unstable_sparse_success | 0.229 | 1.000 | 0.000 | 0.969 | 24 | run replay SFT and prompt repair before more GRPO; sparse successes mean the verifier can score the task but the policy has not stabilized |
| stem | ready_for_export_eval | 0.896 | 1.000 | 1.000 | 0.948 | 20 | export and run bounded held-out eval; keep the adapter as the current bridge winner unless external eval regresses |
| tool | blocked_no_reward_signal | 0.000 | 0.000 | 0.000 | 1.000 | 0 | stop GRPO continuation; inspect verifier, answer schema, and failure-contrast SFT until nonzero rewards appear |

## Lane notes

### code

- Prompt tasks: `python_exec`
- Rationale: mean correct rate 0.031 < 0.250; max correct rate 0.500

### math

- Prompt tasks: `exact_math`
- Rationale: mean correct rate 0.229 < 0.250; max correct rate 1.000

### stem

- Prompt tasks: `mcq_reasoning`
- Rationale: mean correct rate 0.896 >= 0.750; final correct rate 1.000 >= 0.750
- Warnings: mean format rate 0.948 is below 0.950

### tool

- Prompt tasks: `json_match`
- Rationale: max correct rate is 0.000; reward never becomes nonzero
- Warnings: advantage signal steps 0 < 4

## Operational rule

Treat zero reward variance / zero advantage lanes as data or verifier problems before RL problems. Extending GRPO without a nonzero group signal only spends compute without creating a useful policy update.
