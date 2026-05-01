"""Rolling checkpoint round-robin + resume RNG restore."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from elt_lm.config import ModelConfig, TrainConfig
from elt_lm.model import ELTLanguageModel
from elt_lm.train import RollingCheckpointer, configure_optimizer, load_checkpoint


def _tiny_cfg() -> TrainConfig:
    cfg = TrainConfig()
    cfg.model = ModelConfig(
        vocab_size=128, d_model=32, n_unique_layers=2, n_heads=4, head_dim=8,
        d_ff=64, max_seq_len=32, grad_checkpoint=False, L_min=1, L_max=2,
    )
    return cfg


def _tiny_model_opt() -> tuple[ELTLanguageModel, torch.optim.Optimizer, TrainConfig]:
    cfg = _tiny_cfg()
    model = ELTLanguageModel(cfg.model)
    opt = configure_optimizer(model, cfg)
    return model, opt, cfg


def test_rolling_round_robin_fills_3_slots(tmp_path: Path) -> None:
    model, opt, cfg = _tiny_model_opt()
    rolling = RollingCheckpointer(tmp_path, interval_sec=1, keep=3)

    # First save is immediate via force=True
    assert rolling.maybe_save(model, opt, cfg, step=0, force=True)
    # Second + third — advance clock manually to cross the interval
    rolling.last_save_t = time.time() - 10
    assert rolling.maybe_save(model, opt, cfg, step=1)
    rolling.last_save_t = time.time() - 10
    assert rolling.maybe_save(model, opt, cfg, step=2)

    for i in range(3):
        assert (tmp_path / f"rolling_{i}.pt").exists(), f"rolling_{i}.pt missing"
    assert (tmp_path / "last.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_rolling_slot_wraps_and_no_extra_files(tmp_path: Path) -> None:
    model, opt, cfg = _tiny_model_opt()
    rolling = RollingCheckpointer(tmp_path, interval_sec=1, keep=3)

    for step in range(5):
        rolling.last_save_t = time.time() - 10  # force crossing the interval
        rolling.maybe_save(model, opt, cfg, step=step)

    # Exactly 3 rolling slots exist
    existing = sorted(p.name for p in tmp_path.glob("rolling_*.pt"))
    assert existing == ["rolling_0.pt", "rolling_1.pt", "rolling_2.pt"]
    assert not (tmp_path / "rolling_3.pt").exists()

    # After 5 saves (round-robin 0,1,2,0,1) next slot is 2; slot 1 wrote last
    assert rolling.next_slot == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_rolling_respects_interval(tmp_path: Path) -> None:
    model, opt, cfg = _tiny_model_opt()
    rolling = RollingCheckpointer(tmp_path, interval_sec=999, keep=3)

    # First save (non-forced) should NOT fire because interval hasn't elapsed
    assert not rolling.maybe_save(model, opt, cfg, step=0)
    # Forced save overrides the interval
    assert rolling.maybe_save(model, opt, cfg, step=0, force=True)
    # Subsequent unforced save within interval still blocked
    assert not rolling.maybe_save(model, opt, cfg, step=1)


def test_resume_restores_step_and_rng(tmp_path: Path) -> None:
    model, opt, cfg = _tiny_model_opt()
    rolling = RollingCheckpointer(tmp_path, interval_sec=1, keep=3)

    torch.manual_seed(12345)
    # burn a few RNG draws so the state is non-default
    _ = torch.randn(4)
    rolling.maybe_save(model, opt, cfg, step=42, force=True)
    rng_before = torch.randn(8)

    # Fresh model+opt, load from last.pt, draw from RNG again
    model2, opt2, _ = _tiny_model_opt()
    step = load_checkpoint(tmp_path / "last.pt", model2, opt2)
    assert step == 42
    rng_after = torch.randn(8)
    assert torch.allclose(rng_before, rng_after), "RNG state was not restored on resume"


def test_last_pt_points_at_most_recent(tmp_path: Path) -> None:
    model, opt, cfg = _tiny_model_opt()
    rolling = RollingCheckpointer(tmp_path, interval_sec=1, keep=3)

    rolling.maybe_save(model, opt, cfg, step=10, force=True)
    first_state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert first_state["step"] == 10

    rolling.last_save_t = time.time() - 10
    rolling.maybe_save(model, opt, cfg, step=99)
    second_state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert second_state["step"] == 99
