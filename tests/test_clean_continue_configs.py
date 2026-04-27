from __future__ import annotations

from pathlib import Path

import pytest

from elt_lm.config import load_train_config
from elt_lm.train import _update_last_hardlink


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "base_1B_continue_clean_smoke.yaml"
FULL_CONFIG = ROOT / "configs" / "base_1B_continue_clean.yaml"
CLEAN_BIN = "H:/elt_data/bin_clean_2026-04-24"


def test_base_1b_clean_smoke_config_is_short_and_isolated() -> None:
    cfg = load_train_config(SMOKE_CONFIG)

    assert cfg.data.train_bin == f"{CLEAN_BIN}/train.bin"
    assert cfg.data.val_bin == f"{CLEAN_BIN}/val.bin"
    assert cfg.run_dir == "H:/elt_data/runs/base_1B_clean_smoke_2026-04-24"
    assert cfg.total_steps <= 2
    assert cfg.grad_accum_steps == 1
    assert cfg.log_every == 1
    assert cfg.eval_every == 2
    assert cfg.save_every == 2
    assert cfg.rolling_ckpt_interval_sec == 600
    assert cfg.optim.kind == "nvme_adamw"
    assert cfg.offload.root == "H:/elt_data/runs/base_1B_clean_smoke_2026-04-24/offload_nvme"
    assert cfg.model.backbone_kind == "native_elt"


def test_base_1b_clean_full_config_keeps_long_run_shape() -> None:
    cfg = load_train_config(FULL_CONFIG)

    assert cfg.data.train_bin == f"{CLEAN_BIN}/train.bin"
    assert cfg.data.val_bin == f"{CLEAN_BIN}/val.bin"
    assert cfg.run_dir == "H:/elt_data/runs/base_1B_clean_continue"
    assert cfg.total_steps == 120000
    assert cfg.grad_accum_steps == 64
    assert cfg.eval_every == 1000
    assert cfg.save_every == 2000
    assert cfg.optim.kind == "nvme_adamw"
    assert cfg.offload.root == "H:/elt_data/runs/base_1B_clean_continue/offload_nvme"
    assert cfg.model.backbone_kind == "native_elt"


def test_last_checkpoint_fallback_stream_copies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "step_0000002.pt"
    source.write_bytes(b"checkpoint bytes")

    def fail_link(src: str | Path, dst: str | Path) -> None:
        raise OSError("hardlink unavailable")

    monkeypatch.setattr("elt_lm.train.os.link", fail_link)

    _update_last_hardlink(tmp_path, source)

    assert (tmp_path / "last.pt").read_bytes() == b"checkpoint bytes"
