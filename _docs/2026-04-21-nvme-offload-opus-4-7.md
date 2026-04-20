# 2026-04-21 — Phase C: Hypura-style 4-tier NVMe offload

**Model:** opus-4-7
**Branch of work:** ELT 1B + Hypura-style offload plan, Phase C.

## Goal

Port the core data-plane of Hypura's Rust runtime into a PyTorch-native,
Windows-compatible offload package so the ELT composite block can run with
bf16 masters in RAM and fp32 optimizer state on NVMe — unlocking 1B training
on RTX 3060 without DeepSpeed (whose libaio-based NVMe path is Linux-only).

This phase delivers the **data structures and unit-tested primitives**. The
`CompositeBlock.forward` promote/demote hook integration + the end-to-end
`elt-train --config base_1B.yaml --optim nvme_adamw` run are the follow-up.

## Files

### New — `src/elt_lm/offload/`
- `__init__.py` — public re-exports.
- `hardware_profile.py` — `HardwareProfile` dataclass + `probe_hardware(nvme_path)`:
  GPU name/VRAM/bandwidth (NVML), RAM bytes (psutil), NVMe free + **measured**
  read bandwidth via a 16 MB probe to the target drive.
- `placement.py` — `StorageTier` enum (GPU / PINNED / RAM / NVME) + greedy
  `plan_placement(model, hw)` assigning embed/norm/head → GPU and every
  composite-layer param → RAM + NVMe shadow. `assert_fits` raises if the
  plan doesn't clear the VRAM/RAM headroom budget.
- `prefetcher.py` — `NvmePrefetcher` class: `ThreadPoolExecutor` with
  blocking `file.readinto(buf)` where `buf` is a reused pinned-RAM tensor
  drawn from a `_PinnedPool`. Exposes `enqueue(key, path, nbytes)` →
  `Future[Tensor]`, `recycle`, `stats()` (bytes_read / reads_completed /
  avg_mbps / pinned_hit_rate). Libaio-free.
- `tiered_store.py` — `TieredParameterStore` owns tier-wise master copies:
  - bf16 CPU mirror of every composite-layer param (RAM master)
  - fp32 numpy memmap on NVMe for master weight + AdamW `m` + `v`
  per param. Shards persist across process restarts (`open_existing` path).
  `_NvmeShard.read_tensor()` / `write_tensor()` are the only IO primitives.
- `optim_offload.py` — `NvmeAdamW`: hybrid optimizer. For parameters whose
  `id()` appears in `store.ram_param_names()`, the update runs on CPU in
  fp32 using master/m/v read from NVMe, with the result written back and
  downcast to bf16 into the live param and the RAM master. Other params
  (embeddings, norms, head) fall through to a stock AdamW code path inside
  the same optimizer.

### New — dashboard
- `dashboard/panels/tiers.py` — "Storage tiers" panel: reads `tier_read`,
  `layer_computed`, `prefetch_status` events. Shows total reads, aggregate
  bytes, average MB/s, latest pinned hit-rate, per-layer compute-time tail.
- `dashboard/app.py` — added the tiers panel to the main layout.

### New — tests (19 passing, no regressions in the 77 prior)
- `tests/test_placement.py` — 6 tests: embedding placed on GPU, composite on
  RAM, final_norm on GPU, byte accounting non-zero, `assert_fits` accepts
  reasonable plan and rejects impossible one.
- `tests/test_prefetcher.py` — 4 tests: round-trip of a 4 MB file, pinned
  reuse shows hit-rate > 0, `bytes_read` accumulates across multiple enqueues,
  optional GPU copy round-trip when CUDA is available.
- `tests/test_tiered_store.py` — 6 tests: RAM masters populated as bf16/CPU,
  NVMe shards have correct shapes, 3× `.f32` files per RAM param hit disk,
  master mmap equals initial weight, reopen preserves written state,
  `promote_to_gpu` works (CUDA-gated).
- `tests/test_nvme_adamw.py` — 3 tests: NvmeAdamW matches `torch.optim.AdamW`
  within `atol=1e-5, rtol=1e-4` over 5 synthetic-gradient steps, m/v
  mmaps contain non-zero data after a step, bf16 live params don't go
  non-finite when optimizer state lives in fp32.

## Key decisions

1. **Greedy placement, no LP.** ELT's three-group structure (embed+head,
   N layers, final norm) doesn't benefit from a solver. Less code; easier
   to reason about; same result.
2. **ThreadPoolExecutor + pinned pool instead of libaio / asyncio.**
   Blocking `readinto(mv)` into a `torch.empty(..., pin_memory=True)` is
   the one primitive that's production-reliable on Windows. No deadlock
   pitfalls from async.
3. **numpy memmap for NVMe state.** `torch.from_file` is an alternative;
   `np.memmap` with `dtype=np.float32` is simpler, gives us `.flush()`,
   and converts to tensors zero-copy via `torch.from_numpy`.
4. **Hybrid optimizer, not segregated.** `NvmeAdamW` handles tiered AND
   non-tiered params transparently via `name_lookup`. Callers don't need to
   build separate optim groups or risk missing a param.
5. **NVMe files persist.** Reopen path means a crashed-during-step training
   run can resume with master/m/v intact — complements the existing rolling
   checkpointer. Filename format `{safe_name}__{master|m|v}.f32` is stable.
6. **bf16 RAM master is mirrored on step.** `_step_tiered` updates BOTH the
   live param and `store._ram_master_bf16[name]`, so subsequent
   `promote_to_gpu` reloads see post-step weights without touching NVMe.

## Tests

All passing: **96 / 96** (77 prior + 19 new). Representative runtime: the
NvmeAdamW equivalence test over 5 steps on a tiny model finishes in ~1 s,
dominated by mmap.flush().

## Notes for next session (Phase D, and Phase C hot-path integration)

The package is usable as a library today but not yet wired into the training
hot path. Two follow-ups remain:

1. **`offload/hooks.py`** — a context manager or monkey-patch for
   `CompositeBlock.forward` that:
   - before layer i: calls `store.promote_to_gpu(name)` on `layer[i].*`
   - after layer i: emits `layer_computed` telemetry and frees the promoted
     tensor
   - concurrently prefetches layer i+1 on a second CUDA stream.
   The autograd-correct way is to swap `layer.W` via a
   `torch.nn.utils.parametrize.register_parametrization` or to override
   `forward` with functional calls using the promoted tensors.
2. **`train.py` wiring** — when `cfg.optim.kind == "nvme_adamw"`, build
   `HardwareProfile`, `PlacementPlan`, `TieredParameterStore`, then
   construct `NvmeAdamW` with `build_name_lookup(model)`. Emit
   `prefetch_status` every N steps.

Phase D (`scripts/anytime_sweep.py` + inference Pareto panel) is still pending
and is independent of the hot-path integration.

## Pending tasks

- #42 Phase D: Inference sweep + Pareto panel
- (follow-up to this phase) CompositeBlock promote/demote hook integration +
  `elt-train --config configs/base_1B.yaml` on `nvme_adamw` backend
