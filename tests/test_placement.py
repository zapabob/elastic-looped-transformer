"""Tests for offload.placement.plan_placement."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from elt_lm.config import ModelConfig
from elt_lm.model import ELTLanguageModel
from elt_lm.offload.hardware_profile import HardwareProfile
from elt_lm.offload.placement import (
    PlacementPlan,
    StorageTier,
    assert_fits,
    plan_placement,
)


def _rtx3060_profile(nvme_path: Path) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="NVIDIA GeForce RTX 3060",
        gpu_vram_bytes=12 * 1024**3,
        gpu_bandwidth_gbps=360.0,
        ram_bytes=32 * 1024**3,
        ram_bandwidth_gbps=50.0,
        nvme_path=nvme_path,
        nvme_free_bytes=24 * 1024**3,
        nvme_bandwidth_mbps=2000.0,
    )


def _small_model() -> ELTLanguageModel:
    # Keep n_unique_layers * d_model small so tests are fast but shape-faithful.
    return ELTLanguageModel(ModelConfig(
        vocab_size=512, d_model=64, n_unique_layers=4,
        n_heads=4, head_dim=16, d_ff=128, max_seq_len=64, tie_word_embeddings=True,
    ))


def test_plan_places_embedding_on_gpu(tmp_path: Path):
    model = _small_model()
    plan = plan_placement(model, _rtx3060_profile(tmp_path))
    assert "tok_embed.weight" in plan.gpu_params
    assert plan.param_tier["tok_embed.weight"] is StorageTier.GPU


def test_plan_places_composite_on_ram(tmp_path: Path):
    model = _small_model()
    plan = plan_placement(model, _rtx3060_profile(tmp_path))
    for name in plan.ram_params:
        assert name.startswith("composite.")
        assert plan.param_tier[name] is StorageTier.RAM


def test_plan_final_norm_on_gpu(tmp_path: Path):
    model = _small_model()
    plan = plan_placement(model, _rtx3060_profile(tmp_path))
    assert any(n.startswith("final_norm") for n in plan.gpu_params)


def test_plan_reports_nonzero_bytes(tmp_path: Path):
    model = _small_model()
    plan = plan_placement(model, _rtx3060_profile(tmp_path))
    assert plan.bytes_gpu > 0
    assert plan.bytes_ram > 0


def test_assert_fits_rejects_impossible_plan():
    plan = PlacementPlan(bytes_gpu=30 * 1024**3, bytes_ram=100 * 1024**3)
    hw = _rtx3060_profile(Path("/tmp"))
    with pytest.raises(RuntimeError, match="on GPU"):
        assert_fits(plan, hw)


def test_assert_fits_accepts_reasonable_plan(tmp_path: Path):
    # Roughly the 1B shapes. Should pass within headroom.
    model = ELTLanguageModel(ModelConfig(
        vocab_size=1024, d_model=128, n_unique_layers=4,
        n_heads=4, head_dim=32, d_ff=256, max_seq_len=128, tie_word_embeddings=True,
    ))
    plan = plan_placement(model, _rtx3060_profile(tmp_path))
    assert_fits(plan, _rtx3060_profile(tmp_path))
