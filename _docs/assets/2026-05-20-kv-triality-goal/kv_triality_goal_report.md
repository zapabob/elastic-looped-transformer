# 2026-05-20 KV/Triality evidence bundle

## K-protected TurboQuant KV sweep

| policy | ok / total | decode tok/s mean +/- SEM | KV MiB | delta vs K=q8_0/V=q8_0 | p | status note |
|---|---:|---:|---:|---:|---:|---|
| `K=q8_0_V=turbo3` | 2 / 2 | 3.180 +/- 0.360 | 2.910 | 1.220 | 0.6000 | ok |
| `K=bf16_V=turbo3` | 2 / 2 | 2.535 +/- 0.635 | 4.780 | 0.575 | 1.0000 | ok |
| `K=q8_0_V=turbo4` | 2 / 2 | 2.585 +/- 0.225 | 3.190 | 0.625 | 0.6000 | ok |
| `K=bf16_V=turbo4` | 1 / 2 | 4.210 +/- 0.000 | 5.060 | 2.290 | n/a | ok |
| `K=q8_0_V=turbo8` | 0 / 2 | n/a +/- n/a | n/a | n/a | n/a | error while handling argument "--cache-type-v": Unsupported cache type: turbo8 |
| `K=bf16_V=turbo8` | 0 / 2 | n/a +/- n/a | n/a | n/a | n/a | error while handling argument "--cache-type-v": Unsupported cache type: turbo8 |

## Triality SO(8) rotation audit

Audit status: `pass`; rows audited: `4608`; outliers: `0`.

| bits | max orth err | mean det | max det err | status |
|---:|---:|---:|---:|---|
| 3 | 4.870e-03 | 1.000145130 | 7.628e-03 | `pass` |
| 4 | 4.754e-03 | 1.000145894 | 6.427e-03 | `pass` |
| 8 | 6.269e-03 | 1.000028411 | 7.111e-03 | `pass` |

## ILSD entropy and divergence monitor

| lane | L_max | steps | tail loss | max L-dist | last L-entropy | nonfinite |
|---|---:|---:|---:|---:|---:|---|
| `code` | 2 | 48 | 1.386 | 2.363 | 6.790e-04 | `False` |
| `code` | 3 | 32 | 7.018 | 9.128 | 0.000 | `False` |
| `math` | 2 | 48 | 1.659 | 2.246 | 0.000 | `False` |
| `math` | 3 | 16 | 7.216 | 8.832 | 0.001 | `False` |
| `stem` | 2 | 48 | 1.426 | 2.292 | 0.000 | `False` |
| `stem` | 3 | 16 | 7.476 | 9.233 | 0.006 | `False` |
| `tool` | 2 | 48 | 1.413 | 2.092 | 0.000 | `False` |
| `tool` | 3 | 16 | 9.068 | 11.343 | 0.008 | `False` |

## Loop-aware CV multi-group comparison

| group | n | mean | sd | sem | 95% CI |
|---|---:|---:|---:|---:|---:|
| L1 | 32 | 0.4375 | 0.5040 | 0.0891 | [0.2629, 0.6121] |
| L2 | 32 | 0.5625 | 0.5040 | 0.0891 | [0.3879, 0.7371] |
| L3 | 32 | 0.6562 | 0.4826 | 0.0853 | [0.4891, 0.8234] |

| comparison | mean delta | p |
|---|---:|---:|
| L1 - L2 | -0.1250 | 0.122488 |
| L1 - L3 | -0.2188 | 0.016098 |
| L2 - L3 | -0.0938 | 0.254775 |

Friedman within-block permutation p: `0.002500` (statistic `1.7344`, n=32).

## lm-eval-harness status

- Python module available: `False`
- CLI available: `True` (`lm-eval.EXE`)

A global lm-eval CLI is visible, but this repo's Python environment cannot import lm_eval; no broad logged lm-eval run is included in this bundle.

The current numbers are local bridge/external-heldout evidence. Broad lm-eval-harness leaderboard claims remain blocked until the same paired task set is completed under lm-eval with logged samples.
