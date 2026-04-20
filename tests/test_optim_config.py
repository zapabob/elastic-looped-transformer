"""Tests for OptimConfig selection in train.configure_optimizer."""

from __future__ import annotations

import pytest
import torch

from elt_lm.config import ModelConfig, OptimConfig, TrainConfig
from elt_lm.model import ELTLanguageModel
from elt_lm.train import configure_optimizer


def _tiny_cfg() -> TrainConfig:
    return TrainConfig(model=ModelConfig(
        vocab_size=256,
        d_model=32,
        n_unique_layers=2,
        n_heads=2,
        head_dim=16,
        d_ff=64,
        max_seq_len=32,
    ))


def _tiny_model(cfg: TrainConfig | None = None) -> ELTLanguageModel:
    return ELTLanguageModel((cfg or _tiny_cfg()).model)


def test_default_optim_is_adamw():
    cfg = _tiny_cfg()
    assert cfg.optim.kind == "adamw"
    m = _tiny_model(cfg)
    opt = configure_optimizer(m, cfg)
    assert isinstance(opt, torch.optim.AdamW)


def test_unknown_optim_kind_raises():
    cfg = _tiny_cfg()
    cfg.optim = OptimConfig(kind="does_not_exist")  # type: ignore[arg-type]
    m = _tiny_model(cfg)
    with pytest.raises(ValueError, match="unknown optim.kind"):
        configure_optimizer(m, cfg)


def test_nvme_adamw_is_routed_out_of_configure_optimizer():
    """nvme_adamw is built via install_offload_into_training, not
    configure_optimizer. If someone forgets the special case, fail loud."""
    cfg = _tiny_cfg()
    cfg.optim = OptimConfig(kind="nvme_adamw")
    m = _tiny_model(cfg)
    with pytest.raises(RuntimeError, match="install_offload_into_training"):
        configure_optimizer(m, cfg)


def test_paged_adamw_8bit_selected():
    """If bitsandbytes is importable, the optimizer should be a PagedAdamW8bit.

    bnb requires CUDA for its optim classes to construct; skip gracefully when
    CUDA isn't available or the wheel lacks a compatible kernel.
    """
    bnb = pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        pytest.skip("PagedAdamW8bit construction needs CUDA")
    cfg = _tiny_cfg()
    cfg.optim = OptimConfig(kind="paged_adamw_8bit")
    m = _tiny_model(cfg).cuda()
    opt = configure_optimizer(m, cfg)
    assert isinstance(opt, bnb.optim.PagedAdamW8bit)


def test_paged_adamw_helpful_error_when_bnb_missing(monkeypatch):
    """Emulate bitsandbytes absent → should raise ImportError with install hint."""
    import builtins
    import sys

    real_bnb = sys.modules.pop("bitsandbytes", None)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bitsandbytes" or name.startswith("bitsandbytes."):
            raise ImportError("simulated missing bitsandbytes")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        cfg = _tiny_cfg()
        cfg.optim = OptimConfig(kind="paged_adamw_8bit")
        m = _tiny_model(cfg)
        with pytest.raises(ImportError, match="offload_8bit"):
            configure_optimizer(m, cfg)
    finally:
        if real_bnb is not None:
            sys.modules["bitsandbytes"] = real_bnb


def test_optim_config_yaml_roundtrip(tmp_path):
    """`optim:` block in YAML should populate OptimConfig fields."""
    import yaml
    from elt_lm.config import load_train_config

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "optim": {"kind": "paged_adamw_8bit", "paged_bits": 8},
            "lr": 1.0e-4,
        }),
        encoding="utf-8",
    )
    cfg = load_train_config(cfg_path)
    assert cfg.optim.kind == "paged_adamw_8bit"
    assert cfg.optim.paged_bits == 8
    assert cfg.lr == pytest.approx(1.0e-4)
