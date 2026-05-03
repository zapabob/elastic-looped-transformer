16:9 Japanese technical infographic/dashboard using exact measured data.
Title: ???????Turboquant TQ4_1S???: ???? / ?????.
Show four chart panels: artifact size BF16 9.03 GiB, Q8_0 4.80 GiB, TQ4_1S 4.16 GiB; prompt eval throughput mean?SEM BF16 217.15?13.59, Q8_0 248.77?16.75, TQ4_1S 0.79?0.02 tok/s; decode throughput BF16 27.83?0.07, Q8_0 40.40?0.15, TQ4_1S 0.69?0.03 tok/s; short PPL BF16 11313.67, Q8_0 13677.23, TQ4_1S 21648.79.
Add left box ????: 53.9% smaller than BF16, 13.4% smaller than Q8_0, loadable GGUF with Turboquant metadata, useful for HF/GH distribution and quantization research.
Add right box ?????: current runtime slow, PPL 1.91x BF16, logits KL TQ4_1S 1.8819 vs Q8_0 1.64035, q8_0/q8_0 KV not common because BF16 context creation failed.
Footer: local short-run release validation, not external lm-eval leaderboard; next step is TQ4_1S runtime/kernel optimization plus broader paired lm-eval.
