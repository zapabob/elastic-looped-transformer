"""Async NVMe → pinned-RAM prefetcher (Windows-safe, libaio-free).

Hypura's Rust runtime uses libaio + mpsc channels; on Windows we reach the
same shape with:

    - `ThreadPoolExecutor` workers doing blocking `file.readinto(buf)`
    - `torch.empty(..., pin_memory=True)` as pinned staging buffers
      (equivalent to Hypura's AlignedBuffer)
    - A double-buffered slot pool (A/B) so compute on slot A overlaps with
      load into slot B
    - A `queue.Queue` hand-off between enqueue and wait (mpsc-equivalent)

The prefetcher is **read-only**: it exposes `enqueue(key, nbytes, target_ptr_fn)`
and `wait(key)`. Writing goes through `TieredParameterStore` directly (single
writer = the optimizer step).

Stats (`stats()`) are used by the Storage-tiers dashboard panel:
    - bytes_read, reads_completed
    - total_latency_us, avg_mbps
    - pinned_hit / pinned_miss (for hit-rate)
"""

from __future__ import annotations

import concurrent.futures as _cf
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor


@dataclass
class PrefetchStats:
    bytes_read: int = 0
    reads_completed: int = 0
    total_latency_us: float = 0.0
    pinned_hits: int = 0
    pinned_misses: int = 0

    def to_dict(self) -> dict:
        mbps = (self.bytes_read / max(1.0, self.total_latency_us / 1e6) / 1e6
                if self.total_latency_us > 0 else 0.0)
        hit_rate = (self.pinned_hits / max(1, self.pinned_hits + self.pinned_misses)
                    if (self.pinned_hits + self.pinned_misses) else 0.0)
        return {
            "bytes_read": self.bytes_read,
            "reads_completed": self.reads_completed,
            "avg_mbps": mbps,
            "pinned_hit_rate": hit_rate,
        }


class _PinnedPool:
    """Fixed-size pool of pinned-RAM buffers, reused across prefetches.

    Sizing heuristic: keep `max_buffers` slots, each large enough to stage the
    biggest single layer the model has. On miss, allocate a fresh buffer (slow
    path); on hit, reuse an existing one.
    """

    def __init__(self, max_buffers: int, buf_nbytes: int, dtype: torch.dtype = torch.uint8):
        self.max_buffers = max_buffers
        self.buf_nbytes = buf_nbytes
        self.dtype = dtype
        self._free: list[Tensor] = []
        self._lock = threading.Lock()

    def get(self, nbytes: int) -> tuple[Tensor, bool]:
        """Return (buf, was_hit)."""
        with self._lock:
            if self._free and self._free[-1].numel() >= nbytes:
                return self._free.pop(), True
        # miss — allocate (still pinned, but outside the pool).
        buf = torch.empty(max(nbytes, self.buf_nbytes), dtype=self.dtype, pin_memory=True)
        return buf, False

    def put(self, buf: Tensor) -> None:
        with self._lock:
            if len(self._free) < self.max_buffers:
                self._free.append(buf)


class NvmePrefetcher:
    """Async NVMe reader — blocking IO on a thread pool, pinned staging on host.

    Usage:

        pf = NvmePrefetcher(pinned_pool_mb=256, num_workers=2)
        fut = pf.enqueue("layer_7", path, nbytes=77*1024*1024)
        pinned_tensor = pf.wait(fut)           # completes when IO done
        # ...user copies pinned_tensor → GPU on their own CUDA stream...
        pf.recycle(pinned_tensor)              # back into the pool
    """

    def __init__(self, pinned_pool_mb: int = 256, num_workers: int = 2,
                 max_buffers: int = 4):
        self._pool = _PinnedPool(
            max_buffers=max_buffers,
            buf_nbytes=pinned_pool_mb * 1024 * 1024,
        )
        self._exec = _cf.ThreadPoolExecutor(
            max_workers=max(1, num_workers),
            thread_name_prefix="nvme-prefetch",
        )
        self._stats = PrefetchStats()
        self._stats_lock = threading.Lock()

    def enqueue(self, key: str, path: Path, nbytes: int,
                on_done: Callable[[Tensor], None] | None = None) -> "_cf.Future[Tensor]":
        """Schedule a read. Returns a Future that resolves to a pinned CPU tensor
        containing the bytes. The caller must call `recycle` on the returned
        tensor once it has been copied to GPU.
        """
        return self._exec.submit(self._read, key, Path(path), nbytes, on_done)

    def _read(self, key: str, path: Path, nbytes: int,
              on_done: Callable[[Tensor], None] | None) -> Tensor:
        buf, hit = self._pool.get(nbytes)
        t0 = time.perf_counter()
        # buf may be larger than nbytes; read only nbytes into the prefix.
        view = buf[:nbytes]
        with open(path, "rb", buffering=0) as f:
            mv = memoryview(view.numpy())  # pinned CPU tensor supports numpy() view
            got = f.readinto(mv)
        assert got == nbytes, f"short read: expected {nbytes} got {got} for {path}"
        elapsed_us = (time.perf_counter() - t0) * 1e6

        with self._stats_lock:
            self._stats.bytes_read += nbytes
            self._stats.reads_completed += 1
            self._stats.total_latency_us += elapsed_us
            if hit:
                self._stats.pinned_hits += 1
            else:
                self._stats.pinned_misses += 1

        if on_done is not None:
            on_done(view)
        return view

    def recycle(self, pinned_tensor: Tensor) -> None:
        # `pinned_tensor` came from a buf in the pool; the underlying storage
        # is what we want to hand back. Find the full tensor that owns the
        # storage.
        storage = pinned_tensor.untyped_storage()
        full = torch.empty(0, dtype=pinned_tensor.dtype).set_(storage)  # type: ignore[arg-type]
        self._pool.put(full)

    def stats(self) -> dict:
        with self._stats_lock:
            return self._stats.to_dict()

    def close(self) -> None:
        self._exec.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "NvmePrefetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
