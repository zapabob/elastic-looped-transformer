## Turboquant TQ4_1S merits and demerits in this study

Source data: `_docs\assets\2026-05-03-gguf-quant-cv-gptimage\gguf_quant_cv_summary.csv` and paired SciPy reports in the same directory.

### Merits

- Artifact size: `4.16 GiB`, which is `53.9%` smaller than BF16 and `13.4%` smaller than Q8_0.
- GGUF handoff remains loadable with Turboquant metadata, so it is useful for distribution, HF/GH release packaging, and runtime-integration experiments.
- In the short f16/f16 KV run, KV/recurrent-state allocation did not exceed BF16/Q8_0: `1024 / 6432 MiB`.

### Demerits

- Current local runtime throughput is the blocker: prompt eval is `0.79` tok/s (`0.36%` of BF16), and decode is `0.69` tok/s (`2.47%` of BF16).
- Short held-out PPL is worse: TQ4_1S `21648.79` vs BF16 `11313.67` (`1.91x`).
- Logits KL vs BF16 is higher than Q8_0: TQ4_1S `1.88190` vs Q8_0 `1.64035`.
- A q8_0/q8_0 KV-cache bench was not common across all formats because BF16 failed context creation; the paired statistics therefore use the common f16/f16 KV path.

### Interpretation

Turboquant should be presented as a compact release/research handoff format at this stage. The data supports its storage/distribution value, but not a serving-speed claim for the current local runtime. The next proof point is TQ4_1S kernel/runtime optimization plus broader paired lm-eval validation.
