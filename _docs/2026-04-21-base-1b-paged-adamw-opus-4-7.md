# 2026-04-21 — Phase B: base_1B config + PagedAdamW8bit optimizer

**Model:** opus-4-7
**Branch of work:** ELT 1B + Hypura-style offload plan, Phase B.

## Goal

Make a 1B-class ELT model actually fit and train on RTX 3060 12 GB. 1B fp32
AdamW naively needs ~18.7 GB; the mitigation in this phase is
`bitsandbytes.optim.PagedAdamW8bit` — 8-bit optimizer state paged via CUDA
managed memory. Phase C will replace this with a custom NVMe-backed
`NvmeAdamW`, but PagedAdamW8bit is the short path to "training actually runs."

## Files

### New
- `configs/base_1B.yaml` — d_model=1792, N=28, n_heads=28, head_dim=64,
  d_ff=4864, tie=true. Reports exact param counts match the plan
  (1.537 B total / 1.092 B non-emb / 445 M embeddings).
  Batch shape: `micro_batch=1 × seq_len=1024 × grad_accum=64 = 65,536 tok/step`.
  120k steps × 65.5k tok ≈ 7.9 B tokens — covers the ~5-6 B dedup corpus with
  ~1.3 passes (Chinchilla-adjacent given the L=4 effective-depth multiplier).
- `tests/test_optim_config.py` — 6 tests: default kind is adamw, unknown kind
  raises, nvme_adamw raises NotImplementedError, paged_adamw_8bit returns a
  `bnb.optim.PagedAdamW8bit` when bnb is importable + CUDA is present, helpful
  ImportError when bnb is missing, YAML round-trip populates OptimConfig.

### Edited
- `src/elt_lm/config.py` — new `OptimConfig` dataclass with fields
  `kind ∈ {adamw, paged_adamw_8bit, nvme_adamw}`, `paged_bits ∈ {8, 32}`,
  `paged_percentile_clipping`. Added `TrainConfig.optim` field and YAML
  parsing hook.
- `src/elt_lm/train.py` — `configure_optimizer` now branches on
  `cfg.optim.kind`. `paged_adamw_8bit` branch lazy-imports bitsandbytes and
  raises a helpful install hint when absent. `nvme_adamw` is stubbed with
  NotImplementedError (Phase C).
- `pyproject.toml` — new `offload_8bit` optional-deps group with
  `bitsandbytes>=0.43.0`.

## Key decisions

1. **Paged 8-bit first, NVMe second.** Plan's risk-hedge path: get 1B fitting
   and training with a well-maintained upstream optimizer, then swap in the
   custom NvmeAdamW once the Hypura-style tiered store is in place.
2. **bitsandbytes is optional.** Put behind `offload_8bit` extra so stock
   installs don't pull in the CUDA wheel. The `adamw` code path stays free
   of the dependency.
3. **`percentile_clipping=100`.** Disables bnb's internal quantile clipping
   because we already apply `torch.nn.utils.clip_grad_norm_(..., cfg.grad_clip)`
   in the train loop; double-clipping hurts.
4. **`seq_len=1024` for 1B**, not 2048. Halves activation memory — critical
   with B=1, grad-ckpt, and 12 GB VRAM. Context length is a future upgrade
   (post-pretraining, via long-context fine-tune).
5. **`run_dir: H:/elt_data/runs/base_1B`.** 1B checkpoints are ~3 GB each
   (bf16 weights + fp32-ish opt state); H: has room, C: does not.

## Tests

All passing (77 / 77 = 71 prior + 6 new).

- `test_optim_config.py` covers the full branch table including a fake
  `__import__` monkeypatch to prove the "missing bitsandbytes" error is
  actionable.
- PagedAdamW8bit smoke-test also ran manually on the RTX 3060:
  `Linear(8,8).cuda()` step round-trips cleanly, `bnb.__version__ = 0.49.2`.

## Notes for next session (Phase C)

- `src/elt_lm/offload/` package needs: `hardware_profile.py`,
  `placement.py`, `prefetcher.py`, `tiered_store.py`, `optim_offload.py`,
  `hooks.py`.
- New telemetry events to add: `tier_read` (tier/bytes/latency_us),
  `layer_computed` (layer_idx/tier/duration_us), `prefetch_status`
  (hit_rate/nvme_mbps). Corresponding dashboard panel: "Storage tiers".
- `CompositeBlock.forward` needs promote/demote hooks; ELT's L-fold iteration
  is the payoff — each NVMe→GPU copy amortizes across L forward + L backward
  passes.
- `NvmeAdamW`: memory-mapped fp32 `m`, `v`, `master_weight` per-param shard.
  Step runs in a background ThreadPoolExecutor to overlap with the next
  micro-batch's forward.

## Pending tasks

- #41 Phase C: Hypura 4-tier NVMe offload
- #42 Phase D: Inference sweep + Pareto panel
