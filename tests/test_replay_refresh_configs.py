from __future__ import annotations

from pathlib import Path

import yaml

from elt_lm.config import load_train_config


REPLAY_NATIVE_SFT_CONFIGS = [
    Path("configs/posttrain_code_sft_qwen35_hauhaucs_replay.yaml"),
    Path("configs/posttrain_math_sft_qwen35_hauhaucs_replay.yaml"),
    Path("configs/posttrain_stem_sft_qwen35_hauhaucs_replay.yaml"),
    Path("configs/posttrain_tool_sft_qwen35_hauhaucs_replay.yaml"),
]

REPLAY_GRPO_CONFIGS = [
    Path("configs/grpo_code_qwen35_hauhaucs_replay.yaml"),
    Path("configs/grpo_math_qwen35_hauhaucs_replay.yaml"),
    Path("configs/grpo_tool_qwen35_hauhaucs_replay.yaml"),
]

REPLAY_SIDE_LORA_CONFIGS = [
    Path("configs/qwen35_4b_side_lora_code_sft_replay.yaml"),
    Path("configs/qwen35_4b_side_lora_math_sft_replay.yaml"),
    Path("configs/qwen35_4b_side_lora_stem_sft_replay.yaml"),
    Path("configs/qwen35_4b_side_lora_tool_sft_replay.yaml"),
]


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_replay_pretrain_config_uses_canonical_clean_bin() -> None:
    cfg = load_train_config("configs/base_1B_clean_replay_phase2.yaml")

    assert cfg.data.train_bin == "H:/elt_data/bin_clean_2026-04-24/train.bin"
    assert cfg.data.val_bin == "H:/elt_data/bin_clean_2026-04-24/val.bin"
    assert cfg.run_dir == "H:/elt_data/runs/base_1B_clean_replay_phase2"
    assert cfg.total_steps == 5000
    assert cfg.rolling_ckpt_interval_sec == 300
    assert cfg.rolling_ckpt_keep == 3


def test_replay_native_sft_configs_point_to_posttrain_mixed_bins() -> None:
    for path in REPLAY_NATIVE_SFT_CONFIGS:
        cfg = load_train_config(path)
        assert "/posttrain_mixed/" in cfg.data.train_bin
        assert cfg.data.train_bin.endswith("/bin/train.bin")
        assert cfg.data.val_bin.endswith("/bin/val.bin")
        assert cfg.run_dir.endswith("_replay")
        assert cfg.optim.kind == "nvme_adamw"


def test_replay_grpo_configs_keep_kl_and_mixed_prompts() -> None:
    for path in REPLAY_GRPO_CONFIGS:
        raw = _raw(path)
        cfg = load_train_config(path)
        assert raw["grpo"]["kl_beta"] > 0
        assert "/posttrain_mixed/" in raw["grpo"]["prompts_file"]
        assert cfg.data.train_bin.endswith("/bin/train.bin")
        assert cfg.run_dir.endswith("_replay")


def test_replay_side_lora_configs_are_adapter_only_and_mixed() -> None:
    for path in REPLAY_SIDE_LORA_CONFIGS:
        raw = _raw(path)
        cfg = load_train_config(path)
        assert raw["model"]["backbone_kind"] == "hf_qwen35_looped"
        assert raw["model"]["hf_trainable_mode"] == "lora"
        assert raw["model"]["hf_save_adapter_only"] is True
        assert "/posttrain_mixed/" in cfg.data.train_bin
        assert cfg.data.seq_len == 128
        assert cfg.run_dir.endswith("_replay")
