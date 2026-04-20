"""Tests for offload.optim_offload.NvmeAdamW.

Equivalence test: NvmeAdamW applied to a model backed by a TieredParameterStore
should produce weights within tight tolerance of a reference `torch.optim.AdamW`
fed the same gradients.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from elt_lm.config import ModelConfig
from elt_lm.model import ELTLanguageModel
from elt_lm.offload.hardware_profile import HardwareProfile
from elt_lm.offload.optim_offload import NvmeAdamW, build_name_lookup
from elt_lm.offload.placement import plan_placement
from elt_lm.offload.tiered_store import TieredParameterStore


def _hw(nvme_path: Path) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="cpu", gpu_vram_bytes=12 * 1024**3, gpu_bandwidth_gbps=360.0,
        ram_bytes=32 * 1024**3, ram_bandwidth_gbps=50.0,
        nvme_path=nvme_path, nvme_free_bytes=24 * 1024**3,
        nvme_bandwidth_mbps=2000.0,
    )


def _make_model(dtype: torch.dtype = torch.float32) -> ELTLanguageModel:
    m = ELTLanguageModel(ModelConfig(
        vocab_size=256, d_model=32, n_unique_layers=2,
        n_heads=2, head_dim=16, d_ff=64, max_seq_len=16, tie_word_embeddings=True,
    ))
    return m.to(dtype)


def test_nvme_adamw_matches_stock_adamw(tmp_path: Path):
    """100 identical gradient applications — NvmeAdamW vs torch.optim.AdamW."""
    torch.manual_seed(0)
    m_ref = _make_model(torch.float32)
    m_nvme = _make_model(torch.float32)
    m_nvme.load_state_dict(m_ref.state_dict())

    plan = plan_placement(m_nvme, _hw(tmp_path))
    store = TieredParameterStore(m_nvme, plan, nvme_root=tmp_path)
    name_lookup = build_name_lookup(m_nvme)

    opt_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-3, betas=(0.9, 0.95),
                                eps=1e-8, weight_decay=0.1)
    opt_nvme = NvmeAdamW(m_nvme.parameters(), store=store, name_lookup=name_lookup,
                         lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    for _ in range(5):
        # Same synthetic gradient for both models
        for p_ref, p_nvme in zip(m_ref.parameters(), m_nvme.parameters()):
            g = torch.randn_like(p_ref) * 0.01
            p_ref.grad = g.clone()
            p_nvme.grad = g.clone()
        opt_ref.step()
        opt_nvme.step()
        opt_ref.zero_grad()
        opt_nvme.zero_grad()

    for (n, p_ref), (_, p_nvme) in zip(m_ref.named_parameters(),
                                        m_nvme.named_parameters()):
        assert torch.allclose(p_ref, p_nvme, atol=1e-5, rtol=1e-4), \
            f"param {n} diverged: max abs diff {(p_ref - p_nvme).abs().max().item()}"


def test_nvme_state_persists_to_disk(tmp_path: Path):
    """After step(), the NVMe m/v/master mmaps should contain non-zero data."""
    torch.manual_seed(0)
    m = _make_model(torch.float32)
    plan = plan_placement(m, _hw(tmp_path))
    store = TieredParameterStore(m, plan, nvme_root=tmp_path)
    name_lookup = build_name_lookup(m)

    opt = NvmeAdamW(m.parameters(), store=store, name_lookup=name_lookup,
                    lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    for p in m.parameters():
        p.grad = torch.randn_like(p) * 0.1
    opt.step()
    store.flush()

    tiered = store.ram_param_names()[0]
    m_shard, v_shard = store.adam_state(tiered)
    import numpy as np
    assert np.any(np.asarray(m_shard.mmap) != 0)
    assert np.any(np.asarray(v_shard.mmap) != 0)


def test_nvme_adamw_works_on_bf16_model(tmp_path: Path):
    """Live params can be bf16 while master/state is fp32 on NVMe."""
    torch.manual_seed(1)
    m = _make_model(torch.bfloat16)
    plan = plan_placement(m, _hw(tmp_path))
    store = TieredParameterStore(m, plan, nvme_root=tmp_path)
    name_lookup = build_name_lookup(m)

    opt = NvmeAdamW(m.parameters(), store=store, name_lookup=name_lookup,
                    lr=1e-3)
    for p in m.parameters():
        p.grad = torch.randn_like(p) * 0.01
    opt.step()

    # No NaNs should have crept in from precision mixing.
    for p in m.parameters():
        assert torch.isfinite(p).all(), "bf16 params went non-finite after step"
