## GGUF BF16 / Q8_0 / TQ4_1S cross-validation report

- Generated: `2026-05-03`
- Runtime: `llama.cpp CUDA / RTX 3060`
- Corpus: `_docs\assets\2026-05-03-gguf-quant-cv-gptimage\gguf_quant_eval_corpus.txt`
- Scope: short local GGUF validation; not a broad lm-eval leaderboard claim.

### Lane boundaries

- `weight_compression`: TQ4_1S is reported as a local GGUF weight-compression artifact.
- `kv_compression`: TurboQuant-style KV cache compression is tracked as a separate runtime lane; this report does not claim Google TurboQuant KV-cache serving performance.
- `speculative_decoding`: DFlash is tracked as a separate speculative-decoding lane and should be evaluated with target equivalence, acceptance length, accept rate, and tok/s.

### Summary

| metric | group | n | mean | sd | sem | 95% CI | unit |
|---|---|---:|---:|---:|---:|---:|---|
| file_size_gib | BF16 | 1 | 9.0299 | 0.0000 | 0.0000 | [9.0299, 9.0299] | GiB |
| size_ratio_vs_bf16 | BF16 | 1 | 1.0000 | 0.0000 | 0.0000 | [1.0000, 1.0000] | ratio |
| file_size_gib | Q8_0 | 1 | 4.8036 | 0.0000 | 0.0000 | [4.8036, 4.8036] | GiB |
| size_ratio_vs_bf16 | Q8_0 | 1 | 0.5320 | 0.0000 | 0.0000 | [0.5320, 0.5320] | ratio |
| file_size_gib | TQ4_1S | 1 | 4.1606 | 0.0000 | 0.0000 | [4.1606, 4.1606] | GiB |
| size_ratio_vs_bf16 | TQ4_1S | 1 | 0.4608 | 0.0000 | 0.0000 | [0.4608, 0.4608] | ratio |
| prompt_eval_tps | BF16 | 3 | 217.1507 | 23.5344 | 13.5876 | [190.5190, 243.7823] | tok/s |
| decode_tps | BF16 | 3 | 27.8304 | 0.1232 | 0.0711 | [27.6909, 27.9698] | tok/s |
| prompt_eval_tps | Q8_0 | 3 | 248.7653 | 29.0105 | 16.7492 | [215.9369, 281.5938] | tok/s |
| decode_tps | Q8_0 | 3 | 40.4003 | 0.2562 | 0.1479 | [40.1104, 40.6903] | tok/s |
| prompt_eval_tps | TQ4_1S | 3 | 0.7879 | 0.0386 | 0.0223 | [0.7442, 0.8316] | tok/s |
| decode_tps | TQ4_1S | 3 | 0.6878 | 0.0442 | 0.0255 | [0.6377, 0.7378] | tok/s |
| perplexity | BF16 | 1 | 11313.6738 | 0.0000 | 0.0000 | [11313.6738, 11313.6738] | ppl |
| negative_log_likelihood | BF16 | 1 | 9.3338 | 0.0000 | 0.0000 | [9.3338, 9.3338] | nll |
| kv_cache_mib | BF16 | 1 | 1024.0000 | 0.0000 | 0.0000 | [1024.0000, 1024.0000] | MiB |
| recurrent_state_mib | BF16 | 1 | 6432.0000 | 0.0000 | 0.0000 | [6432.0000, 6432.0000] | MiB |
| perplexity | Q8_0 | 1 | 13677.2349 | 0.0000 | 0.0000 | [13677.2349, 13677.2349] | ppl |
| logits_kl_vs_bf16 | Q8_0 | 1 | 1.6403 | 0.0000 | 0.0000 | [1.6403, 1.6403] | KL |
| kv_cache_mib | Q8_0 | 1 | 1024.0000 | 0.0000 | 0.0000 | [1024.0000, 1024.0000] | MiB |
| recurrent_state_mib | Q8_0 | 1 | 6432.0000 | 0.0000 | 0.0000 | [6432.0000, 6432.0000] | MiB |
| perplexity | TQ4_1S | 1 | 21648.7900 | 0.0000 | 0.0000 | [21648.7900, 21648.7900] | ppl |
| logits_kl_vs_bf16 | TQ4_1S | 1 | 1.8819 | 0.0000 | 0.0000 | [1.8819, 1.8819] | KL |
| kv_cache_mib | TQ4_1S | 1 | 1024.0000 | 0.0000 | 0.0000 | [1024.0000, 1024.0000] | MiB |
| recurrent_state_mib | TQ4_1S | 1 | 6432.0000 | 0.0000 | 0.0000 | [6432.0000, 6432.0000] | MiB |

### Multi-group tests

#### decode_tps

- paired blocks: `3`
- omnibus: scipy.stats.friedmanchisquare, statistic=6.0000, p=0.049787
- BF16 vs Q8_0: delta=-12.5700, p=0.250000, scipy.stats.wilcoxon_zsplit_two_sided
- BF16 vs TQ4_1S: delta=27.1426, p=0.250000, scipy.stats.wilcoxon_zsplit_two_sided
- Q8_0 vs TQ4_1S: delta=39.7126, p=0.250000, scipy.stats.wilcoxon_zsplit_two_sided

#### file_size_gib

- paired blocks: `1`
- BF16 vs Q8_0: delta=4.2263, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided
- BF16 vs TQ4_1S: delta=4.8693, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided
- Q8_0 vs TQ4_1S: delta=0.6430, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided

#### kv_cache_mib

- paired blocks: `1`
- BF16 vs Q8_0: delta=0.0000, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided
- BF16 vs TQ4_1S: delta=0.0000, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided
- Q8_0 vs TQ4_1S: delta=0.0000, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided

#### logits_kl_vs_bf16

- paired blocks: `1`
- Q8_0 vs TQ4_1S: delta=-0.2415, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided

#### prompt_eval_tps

- paired blocks: `3`
- omnibus: scipy.stats.friedmanchisquare, statistic=6.0000, p=0.049787
- BF16 vs Q8_0: delta=-31.6147, p=0.250000, scipy.stats.wilcoxon_zsplit_two_sided
- BF16 vs TQ4_1S: delta=216.3628, p=0.250000, scipy.stats.wilcoxon_zsplit_two_sided
- Q8_0 vs TQ4_1S: delta=247.9775, p=0.250000, scipy.stats.wilcoxon_zsplit_two_sided

#### recurrent_state_mib

- paired blocks: `1`
- BF16 vs Q8_0: delta=0.0000, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided
- BF16 vs TQ4_1S: delta=0.0000, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided
- Q8_0 vs TQ4_1S: delta=0.0000, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided

#### size_ratio_vs_bf16

- paired blocks: `1`
- BF16 vs Q8_0: delta=0.4680, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided
- BF16 vs TQ4_1S: delta=0.5392, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided
- Q8_0 vs TQ4_1S: delta=0.0712, p=1.000000, scipy.stats.wilcoxon_zsplit_two_sided

### Interpretation

- `prompt_eval_tps` and `decode_tps` are llama.cpp runtime throughput blocks paired by cache type and phase.
- `perplexity` and `negative_log_likelihood` are local held-out text checks from verifier-backed synthetic-v2 hard validation records.
- `logits_kl_vs_bf16` is included when the optional llama-perplexity logits file run is available.
- `TQ4_1S` currently reports as a mixed GGUF weight artifact with Turboquant metadata; do not read it as a TurboQuant KV-cache result.
