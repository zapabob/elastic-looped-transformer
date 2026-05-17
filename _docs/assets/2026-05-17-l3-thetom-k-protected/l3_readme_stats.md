# L3 README evidence statistics

## Accuracy gates

| evaluation | n | correct | accuracy | Wilson 95% CI | SEM | prompt tok/s | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local STEM bridge | 128 | 121 | 94.5% | [89.1, 97.3] | 2.01% | 1241.74 | 50.02 |
| MMLU-STEM heldout | 16 | 13 | 81.2% | [57.0, 93.4] | 9.76% | 1268.50 | 54.24 |
| GSM8K heldout | 16 | 0 | 0.0% | [0.0, 19.4] | 0.00% | 1013.86 | 48.64 |

## Pairwise p-values

| comparison | test | p |
|---|---|---:|
| Local STEM bridge vs MMLU-STEM heldout | Fisher exact two-sided | 0.0835 |
| Local STEM bridge vs GSM8K heldout | Fisher exact two-sided | 3.56e-16 |
| MMLU-STEM heldout vs GSM8K heldout | Fisher exact two-sided | 3.22e-06 |

## Loop-aware depth

| L | n | correct | accuracy | Wilson 95% CI | SEM | mean margin | wall sec/case |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 14 | 43.8% | [28.2, 60.7] | 8.77% | 0.0234 | 0.661 |
| 2 | 32 | 18 | 56.2% | [39.3, 71.8] | 8.77% | 0.1445 | 1.225 |
| 3 | 32 | 21 | 65.6% | [48.3, 79.6] | 8.40% | 0.3154 | 1.780 |

| comparison | improved | regressed | discordant | paired exact p |
|---|---:|---:|---:|---:|
| L1_vs_L2 | 4 | 0 | 4 | 0.125 |
| L1_vs_L3 | 7 | 0 | 7 | 0.0156 |
| L2_vs_L3 | 3 | 0 | 3 | 0.25 |
