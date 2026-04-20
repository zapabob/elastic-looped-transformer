"""Tests for offload.prefetcher.NvmePrefetcher."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from elt_lm.offload.prefetcher import NvmePrefetcher


def test_roundtrip_small_file(tmp_path: Path):
    """Write a fixed pattern to disk, prefetch it, compare bytes."""
    payload = os.urandom(4 * 1024 * 1024)   # 4 MB
    path = tmp_path / "layer.bin"
    path.write_bytes(payload)

    with NvmePrefetcher(pinned_pool_mb=8, num_workers=1, max_buffers=2) as pf:
        fut = pf.enqueue("layer", path, nbytes=len(payload))
        view = fut.result()
        assert view.numel() == len(payload)
        got = bytes(view.numpy().tobytes())
        assert got == payload


def test_pinned_reuse_shows_hit_rate(tmp_path: Path):
    path = tmp_path / "layer.bin"
    path.write_bytes(os.urandom(256 * 1024))

    with NvmePrefetcher(pinned_pool_mb=1, num_workers=1, max_buffers=2) as pf:
        first = pf.enqueue("layer", path, nbytes=256 * 1024).result()
        pf.recycle(first)
        second = pf.enqueue("layer", path, nbytes=256 * 1024).result()
        pf.recycle(second)

    stats = pf.stats()
    assert stats["reads_completed"] == 2
    # at least one of the two should have been a pool hit
    assert stats["pinned_hit_rate"] > 0.0


def test_bytes_read_accumulates(tmp_path: Path):
    path = tmp_path / "layer.bin"
    path.write_bytes(os.urandom(1024 * 1024))

    with NvmePrefetcher(pinned_pool_mb=2, num_workers=2, max_buffers=2) as pf:
        futs = [pf.enqueue(f"k{i}", path, nbytes=1024 * 1024) for i in range(3)]
        for f in futs:
            view = f.result()
            pf.recycle(view)

    stats = pf.stats()
    assert stats["reads_completed"] == 3
    assert stats["bytes_read"] == 3 * 1024 * 1024


def test_gpu_copy_if_cuda(tmp_path: Path):
    """If CUDA is available, copy prefetched bytes to GPU and verify round-trip."""
    if not torch.cuda.is_available():
        return
    payload = torch.arange(1024 * 16, dtype=torch.float32).numpy().tobytes()
    path = tmp_path / "w.bin"
    path.write_bytes(payload)

    with NvmePrefetcher(pinned_pool_mb=1, num_workers=1) as pf:
        view = pf.enqueue("w", path, nbytes=len(payload)).result()
        # Reinterpret the pinned uint8 buffer as float32.
        view_fp32 = torch.frombuffer(view.numpy(), dtype=torch.float32)
        gpu = view_fp32.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        assert gpu.shape == (1024 * 16,)
        assert torch.allclose(gpu.cpu(), torch.arange(1024 * 16, dtype=torch.float32))
        pf.recycle(view)
