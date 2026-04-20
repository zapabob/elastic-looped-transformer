"""Tests for offload.tiered_store.TieredParameterStore."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from elt_lm.config import ModelConfig
from elt_lm.model import ELTLanguageModel
from elt_lm.offload.hardware_profile import HardwareProfile
from elt_lm.offload.placement import StorageTier, plan_placement
from elt_lm.offload.tiered_store import TieredParameterStore


def _hw(nvme_path: Path) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="cpu", gpu_vram_bytes=12 * 1024**3, gpu_bandwidth_gbps=360.0,
        ram_bytes=32 * 1024**3, ram_bandwidth_gbps=50.0,
        nvme_path=nvme_path, nvme_free_bytes=24 * 1024**3,
        nvme_bandwidth_mbps=2000.0,
    )


def _model() -> ELTLanguageModel:
    return ELTLanguageModel(ModelConfig(
        vocab_size=256, d_model=32, n_unique_layers=2,
        n_heads=2, head_dim=16, d_ff=64, max_seq_len=32, tie_word_embeddings=True,
    ))


def test_ram_masters_populated(tmp_path: Path):
    m = _model()
    plan = plan_placement(m, _hw(tmp_path))
    store = TieredParameterStore(m, plan, nvme_root=tmp_path)

    names = store.ram_param_names()
    assert names, "expected at least one RAM-tier composite parameter"
    for n in names:
        master = store.ram_master(n)
        assert master.dtype is torch.bfloat16
        assert master.device.type == "cpu"


def test_nvme_shards_written_with_correct_shape(tmp_path: Path):
    m = _model()
    plan = plan_placement(m, _hw(tmp_path))
    store = TieredParameterStore(m, plan, nvme_root=tmp_path)

    for n in store.ram_param_names():
        master = store.nvme_master(n)
        mg, vg = store.adam_state(n)
        # All three shards must have the same shape as the original parameter.
        p = dict(m.named_parameters())[n]
        assert master.shape == tuple(p.shape)
        assert mg.shape == tuple(p.shape)
        assert vg.shape == tuple(p.shape)


def test_nvme_files_persist_on_disk(tmp_path: Path):
    m = _model()
    plan = plan_placement(m, _hw(tmp_path))
    TieredParameterStore(m, plan, nvme_root=tmp_path)

    # Every RAM-tier param should have three .f32 files on disk.
    files = list(tmp_path.glob("*.f32"))
    expected_per_param = 3
    ram_param_count = sum(1 for t in plan.param_tier.values() if t is StorageTier.RAM)
    assert len(files) == ram_param_count * expected_per_param


def test_master_mmap_matches_initial_weight(tmp_path: Path):
    m = _model()
    plan = plan_placement(m, _hw(tmp_path))
    store = TieredParameterStore(m, plan, nvme_root=tmp_path)

    name = store.ram_param_names()[0]
    master = store.nvme_master(name)
    p_cpu = dict(m.named_parameters())[name].detach().to("cpu", dtype=torch.float32)
    np.testing.assert_allclose(np.asarray(master.mmap).reshape(p_cpu.shape),
                               p_cpu.numpy(), atol=0, rtol=0)


def test_reopen_keeps_existing_state(tmp_path: Path):
    """Creating a store, closing, and recreating it should reuse disk state."""
    m = _model()
    plan = plan_placement(m, _hw(tmp_path))
    store1 = TieredParameterStore(m, plan, nvme_root=tmp_path)
    name = store1.ram_param_names()[0]
    # Stuff the master shard with a marker.
    master = store1.nvme_master(name)
    marker = torch.full(master.shape, 42.0, dtype=torch.float32)
    master.write_tensor(marker)
    del store1

    # Recreate — should open-existing, preserving the marker.
    store2 = TieredParameterStore(m, plan, nvme_root=tmp_path)
    master2 = store2.nvme_master(name)
    np.testing.assert_allclose(np.asarray(master2.mmap).reshape(master2.shape),
                               marker.numpy(), atol=0)


def test_promote_to_gpu_requires_cuda(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("promote_to_gpu test needs CUDA")
    m = _model()
    plan = plan_placement(m, _hw(tmp_path))
    store = TieredParameterStore(m, plan, nvme_root=tmp_path)
    name = store.ram_param_names()[0]
    gpu = store.promote_to_gpu(name, device=torch.device("cuda"))
    torch.cuda.synchronize()
    assert gpu.device.type == "cuda"
    assert gpu.dtype is torch.bfloat16
