from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from elt_lm.config import ModelConfig, OffloadConfig, TrainConfig, load_train_config


def test_offload_config_defaults():
    """Test that OffloadConfig has the expected default values."""
    config = OffloadConfig()
    assert config.enabled == False
    assert config.root is None
    assert config.min_free_gb == 20.0


def test_offload_config_roundtrip():
    """Test that OffloadConfig can be serialized to YAML and deserialized."""
    original = OffloadConfig(
        enabled=True,
        root="H:/elt_data/offload_nvme",
        min_free_gb=15.5
    )
    
    # Serialize to a dict (as YAML would)
    data = {
        'enabled': original.enabled,
        'root': original.root,
        'min_free_gb': original.min_free_gb
    }
    
    # Deserialize back to OffloadConfig
    restored = OffloadConfig(**data)
    
    assert restored.enabled == original.enabled
    assert restored.root == original.root
    assert restored.min_free_gb == original.min_free_gb


def test_train_config_has_offload_field():
    """Test that TrainConfig includes an offload field."""
    config = TrainConfig()
    assert hasattr(config, 'offload')
    assert isinstance(config.offload, OffloadConfig)
    # Check that it has the default values
    assert config.offload.enabled == False
    assert config.offload.root is None
    assert config.offload.min_free_gb == 20.0


def test_offload_config_from_yaml():
    """Test creating OffloadConfig from YAML-like dictionary."""
    yaml_data = {
        'enabled': True,
        'root': 'H:/custom/offload',
        'min_free_gb': 25.0
    }
    config = OffloadConfig(**yaml_data)
    assert config.enabled == True
    assert config.root == 'H:/custom/offload'
    assert config.min_free_gb == 25.0


def test_load_train_config_parses_offload_section(tmp_path: Path):
    """load_train_config should pick up `offload:` from YAML.

    Regression: previously the YAML loader ignored the `offload:` block so
    `cfg.offload.root` always came out as None even when set in the config."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "offload": {
                "enabled": True,
                "root": "H:/elt_data/offload_nvme",
                "min_free_gb": 5.0,
            },
            "optim": {"kind": "nvme_adamw"},
        }),
        encoding="utf-8",
    )
    cfg = load_train_config(cfg_path)
    assert cfg.offload.enabled is True
    assert cfg.offload.root == "H:/elt_data/offload_nvme"
    assert cfg.offload.min_free_gb == pytest.approx(5.0)


def test_load_train_config_offload_missing_block(tmp_path: Path):
    """No `offload:` block => sensible defaults, not a crash."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("lr: 0.001\n", encoding="utf-8")
    cfg = load_train_config(cfg_path)
    assert cfg.offload.enabled is False
    assert cfg.offload.root is None


def test_tiered_store_raises_when_disk_full(tmp_path: Path):
    """Mocked shutil.disk_usage with tiny free bytes should trip the pre-flight."""
    import torch  # noqa: F401

    from elt_lm.model import ELTLanguageModel
    from elt_lm.offload.hardware_profile import HardwareProfile
    from elt_lm.offload.placement import plan_placement
    from elt_lm.offload.tiered_store import TieredParameterStore

    cfg = TrainConfig(model=ModelConfig(
        vocab_size=128, d_model=32, n_unique_layers=2, n_heads=2,
        head_dim=16, d_ff=64, max_seq_len=16, tie_word_embeddings=True,
        grad_checkpoint=False,
    ))
    model = ELTLanguageModel(cfg.model)

    hw = HardwareProfile(
        gpu_name="test-gpu",
        gpu_vram_bytes=12 * 1024**3,
        gpu_bandwidth_gbps=360.0,
        ram_bytes=32 * 1024**3,
        ram_bandwidth_gbps=50.0,
        nvme_path=tmp_path,
        nvme_free_bytes=100 * 1024**3,
        nvme_bandwidth_mbps=2000.0,
    )
    plan = plan_placement(model, hw)

    offload = OffloadConfig(root=str(tmp_path), min_free_gb=1000.0)

    class _Usage:
        total = 500 * 1024**3
        used = 499 * 1024**3
        free = 1 * 1024**3   # 1 GB free < 1000 GB margin requirement

    with patch("elt_lm.offload.tiered_store.shutil.disk_usage",
               return_value=_Usage()):
        with pytest.raises(RuntimeError, match="Insufficient free space"):
            TieredParameterStore(
                model, plan, nvme_root=tmp_path, offload_config=offload,
            )


if __name__ == "__main__":
    test_offload_config_defaults()
    test_offload_config_roundtrip()
    test_train_config_has_offload_field()
    test_offload_config_from_yaml()
    print("All tests passed")