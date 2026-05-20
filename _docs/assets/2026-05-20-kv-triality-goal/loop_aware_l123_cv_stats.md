### loop_aware_l123_stem_bridge_mcq_logprob

| group | n | mean | sd | sem | 95% CI |
|---|---:|---:|---:|---:|---:|
| L1 | 32 | 0.4375 | 0.5040 | 0.0891 | [0.2629, 0.6121] |
| L2 | 32 | 0.5625 | 0.5040 | 0.0891 | [0.3879, 0.7371] |
| L3 | 32 | 0.6562 | 0.4826 | 0.0853 | [0.4891, 0.8234] |

| comparison | mean delta | p | method |
|---|---:|---:|---|
| L1 - L2 | -0.1250 | 0.122488 | paired_permutation_10000 |
| L1 - L3 | -0.2188 | 0.016098 | paired_permutation_10000 |
| L2 - L3 | -0.0938 | 0.254775 | paired_permutation_10000 |

| omnibus | statistic | p | method |
|---|---:|---:|---|
| Friedman | 1.7344 | 0.002500 | friedman_within_block_permutation_10000 |
