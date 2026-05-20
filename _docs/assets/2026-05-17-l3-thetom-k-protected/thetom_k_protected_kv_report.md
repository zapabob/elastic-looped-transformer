# TheTom TurboQuant K-Protected KV Sweep

K is held at `q8_0` or `bf16`; only V is swept into TheTom `turbo*` cache types.

## Run Status

| policy | total | ok | failed | timeout |
|---|---:|---:|---:|---:|
| `K=f16_V=f16` | 2 | 2 | 0 | 0 |
| `K=q8_0_V=q8_0` | 2 | 2 | 0 | 0 |
| `K=q8_0_V=turbo2` | 2 | 2 | 0 | 0 |
| `K=bf16_V=turbo2` | 2 | 2 | 0 | 0 |
| `K=q8_0_V=turbo3` | 2 | 2 | 0 | 0 |
| `K=bf16_V=turbo3` | 2 | 2 | 0 | 0 |
| `K=q8_0_V=turbo4` | 2 | 2 | 0 | 0 |
| `K=bf16_V=turbo4` | 2 | 1 | 1 | 0 |
| `K=q8_0_V=turbo8` | 2 | 0 | 2 | 0 |
| `K=bf16_V=turbo8` | 2 | 0 | 2 | 0 |

## Summary

| policy | metric | n | mean | SEM | 95% CI |
|---|---:|---:|---:|---:|---:|
| `K=f16_V=f16` | `gen_tok_s` | 2 | 1.215 | 0.275 | 0.676..1.754 |
| `K=f16_V=f16` | `kv_mib` | 2 | 8 | 0 | 8..8 |
| `K=q8_0_V=q8_0` | `gen_tok_s` | 2 | 1.96 | 0.04 | 1.882..2.038 |
| `K=q8_0_V=q8_0` | `kv_mib` | 2 | 4.25 | 0 | 4.25..4.25 |
| `K=q8_0_V=turbo2` | `gen_tok_s` | 2 | 1.84 | 0.07 | 1.703..1.977 |
| `K=q8_0_V=turbo2` | `kv_mib` | 2 | 2.86 | 0 | 2.86..2.86 |
| `K=bf16_V=turbo2` | `gen_tok_s` | 2 | 2.12 | 0.95 | 0.258..3.982 |
| `K=bf16_V=turbo2` | `kv_mib` | 2 | 4.73 | 0 | 4.73..4.73 |
| `K=q8_0_V=turbo3` | `gen_tok_s` | 2 | 3.18 | 0.36 | 2.474..3.886 |
| `K=q8_0_V=turbo3` | `kv_mib` | 2 | 2.91 | 0 | 2.91..2.91 |
| `K=bf16_V=turbo3` | `gen_tok_s` | 2 | 2.535 | 0.635 | 1.29..3.78 |
| `K=bf16_V=turbo3` | `kv_mib` | 2 | 4.78 | 0 | 4.78..4.78 |
| `K=q8_0_V=turbo4` | `gen_tok_s` | 2 | 2.585 | 0.225 | 2.144..3.026 |
| `K=q8_0_V=turbo4` | `kv_mib` | 2 | 3.19 | 0 | 3.19..3.19 |
| `K=bf16_V=turbo4` | `gen_tok_s` | 1 | 4.21 | 0 | 4.21..4.21 |
| `K=bf16_V=turbo4` | `kv_mib` | 1 | 5.06 | 0 | 5.06..5.06 |

## Pairwise vs K=q8_0/V=q8_0

| policy | metric | n | mean delta | p |
|---|---:|---:|---:|---:|
| `K=f16_V=f16` | `gen_tok_s` | 2 | -0.745 | 0.6 |
| `K=f16_V=f16` | `kv_mib` | 2 | 3.75 | 0.6 |
| `K=q8_0_V=turbo2` | `gen_tok_s` | 2 | -0.12 | 0.6 |
| `K=q8_0_V=turbo2` | `kv_mib` | 2 | -1.39 | 0.6 |
| `K=bf16_V=turbo2` | `gen_tok_s` | 2 | 0.16 | 1 |
| `K=bf16_V=turbo2` | `kv_mib` | 2 | 0.48 | 0.6 |
| `K=q8_0_V=turbo3` | `gen_tok_s` | 2 | 1.22 | 0.6 |
| `K=q8_0_V=turbo3` | `kv_mib` | 2 | -1.34 | 0.6 |
| `K=bf16_V=turbo3` | `gen_tok_s` | 2 | 0.575 | 1 |
| `K=bf16_V=turbo3` | `kv_mib` | 2 | 0.53 | 0.6 |
| `K=q8_0_V=turbo4` | `gen_tok_s` | 2 | 0.625 | 0.6 |
| `K=q8_0_V=turbo4` | `kv_mib` | 2 | -1.06 | 0.6 |
| `K=bf16_V=turbo4` | `gen_tok_s` | 1 | 2.29 |  |
| `K=bf16_V=turbo4` | `kv_mib` | 1 | 0.81 |  |
