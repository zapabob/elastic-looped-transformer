from __future__ import annotations

from pathlib import Path

from elt_lm.config import load_train_config


ROOT = Path(__file__).resolve().parents[1]
CODE_BIN = "H:/elt_data/posttrain/code/qwen35_hauhaucs/bin"


def _assert_common_side_smoke(
    path: str,
    *,
    expected_l_max: int,
    expected_steps: int,
    expected_optim: str = "nvme_adamw",
    max_steps: int = 2,
) -> None:
    cfg = load_train_config(ROOT / "configs" / path)

    assert cfg.model.backbone_kind == "hf_qwen35_looped"
    assert cfg.model.hf_model_path == "huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated"
    assert cfg.model.L_min == 1
    assert cfg.model.L_max == expected_l_max
    assert cfg.data.train_bin == f"{CODE_BIN}/train.bin"
    assert cfg.data.val_bin == f"{CODE_BIN}/val.bin"
    assert cfg.data.seq_len <= 256
    assert cfg.optim.kind == expected_optim
    assert cfg.total_steps == expected_steps
    assert cfg.total_steps <= max_steps
    assert cfg.micro_batch_size == 1
    assert cfg.grad_accum_steps == 1
    assert cfg.rolling_ckpt_interval_sec == 300
    assert cfg.rolling_ckpt_keep == 3
    assert cfg.model.hf_lora_rank >= 0


def test_qwen35_4b_side_l1_smoke_config_is_short_and_isolated() -> None:
    _assert_common_side_smoke(
        "qwen35_4b_side_sft_code_smoke_l1.yaml",
        expected_l_max=1,
        expected_steps=1,
    )


def test_qwen35_4b_side_l2_smoke_config_exercises_ilsd() -> None:
    cfg = load_train_config(ROOT / "configs" / "qwen35_4b_side_sft_code_smoke_l2.yaml")

    _assert_common_side_smoke(
        "qwen35_4b_side_sft_code_smoke_l2.yaml",
        expected_l_max=2,
        expected_steps=2,
    )
    assert cfg.ilsd.enabled is True
    assert cfg.ilsd.warmup_steps == 0
    assert cfg.ilsd.strict_student_below_teacher is True


def test_qwen35_4b_side_lora_l1_smoke_config_is_adapter_only() -> None:
    cfg = load_train_config(ROOT / "configs" / "qwen35_4b_side_lora_code_smoke_l1.yaml")

    _assert_common_side_smoke(
        "qwen35_4b_side_lora_code_smoke_l1.yaml",
        expected_l_max=1,
        expected_steps=5,
        expected_optim="adamw",
        max_steps=20,
    )
    assert cfg.model.hf_trainable_mode == "lora"
    assert cfg.model.hf_lora_rank == 8
    assert cfg.model.hf_lora_top_layers == 0
    assert cfg.total_steps <= 20


def test_qwen35_4b_side_lora_l2_smoke_config_limits_loop_adapters() -> None:
    cfg = load_train_config(ROOT / "configs" / "qwen35_4b_side_lora_code_smoke_l2.yaml")

    _assert_common_side_smoke(
        "qwen35_4b_side_lora_code_smoke_l2.yaml",
        expected_l_max=2,
        expected_steps=2,
        expected_optim="adamw",
        max_steps=20,
    )
    assert cfg.model.hf_trainable_mode == "lora"
    assert cfg.model.hf_lora_rank == 16
    assert cfg.model.hf_lora_top_layers == 8
    assert cfg.ilsd.enabled is True
    assert cfg.total_steps <= 20


def test_qwen35_4b_side_lora_long_configs_are_adapter_only() -> None:
    for name in [
        "qwen35_4b_side_lora_code_sft.yaml",
        "qwen35_4b_side_lora_math_sft.yaml",
        "qwen35_4b_side_lora_stem_sft.yaml",
        "qwen35_4b_side_lora_tool_sft.yaml",
        "qwen35_4b_side_lora_code_ilsd_l2.yaml",
    ]:
        cfg = load_train_config(ROOT / "configs" / name)
        assert cfg.model.backbone_kind == "hf_qwen35_looped"
        assert cfg.model.hf_trainable_mode == "lora"
        assert cfg.model.hf_save_adapter_only is True
        assert cfg.model.hf_adapter_base_ckpt == "H:/elt_data/runs/qwen35_4b_elt_bootstrap/last.pt"
        assert cfg.optim.kind == "adamw"
        assert cfg.rolling_ckpt_interval_sec == 300
        assert cfg.rolling_ckpt_keep == 3
        assert cfg.total_steps <= 120
        if name == "qwen35_4b_side_lora_code_ilsd_l2.yaml":
            assert cfg.model.hf_lora_rank == 8
            assert cfg.model.hf_lora_top_layers == 0
            assert cfg.total_steps > 80
