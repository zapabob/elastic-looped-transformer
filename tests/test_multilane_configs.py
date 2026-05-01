from __future__ import annotations

from elt_lm.config import load_train_config
from elt_lm.gguf_distill import load_gguf_distill_config
from elt_lm.gguf_distill_queue import load_gguf_distill_queue_config


def test_load_multilane_distill_configs() -> None:
    for path, lane in [
        ("configs/gguf_distill_code_qwen35_hauhaucs.yaml", "code"),
        ("configs/gguf_distill_math_qwen35_hauhaucs.yaml", "math"),
        ("configs/gguf_distill_stem_qwen35_hauhaucs.yaml", "stem_reasoning"),
        ("configs/gguf_distill_tool_qwen35_hauhaucs.yaml", "tool_use"),
    ]:
        cfg = load_gguf_distill_config(path)
        assert cfg.lane == lane
        assert cfg.pipeline.samples_per_task == 32
        assert len(cfg.tasks) >= 1


def test_load_v1_multilane_distill_configs_disable_teacher_reasoning() -> None:
    for path, lane in [
        ("configs/gguf_distill_code_qwen35_hauhaucs_v1.yaml", "code"),
        ("configs/gguf_distill_math_qwen35_hauhaucs_v1.yaml", "math"),
        ("configs/gguf_distill_stem_qwen35_hauhaucs_v1.yaml", "stem_reasoning"),
        ("configs/gguf_distill_tool_qwen35_hauhaucs_v1.yaml", "tool_use"),
    ]:
        cfg = load_gguf_distill_config(path)
        assert cfg.lane == lane
        assert cfg.pipeline.quality_profile == "v1"
        assert cfg.teacher.reasoning == "off"
        assert cfg.teacher.reasoning_budget == 0
        assert cfg.teacher.reasoning_format == "none"


def test_load_multilane_queue_config() -> None:
    cfg = load_gguf_distill_queue_config("configs/gguf_distill_qwen35_hauhaucs_multilane_queue.yaml")
    assert [stage.name for stage in cfg.stages] == ["code", "math", "stem_reasoning", "tool_use"]


def test_load_all_remaining_hauhaucs_queue_config() -> None:
    cfg = load_gguf_distill_queue_config("configs/gguf_distill_qwen35_hauhaucs_all_remaining_queue.yaml")
    assert [stage.name for stage in cfg.stages] == [
        "detection",
        "code",
        "math",
        "stem_reasoning",
        "tool_use",
    ]
    assert all(stage.resume for stage in cfg.stages)
    assert all(stage.skip_completed for stage in cfg.stages)


def test_load_multilane_sft_and_grpo_configs() -> None:
    for path in [
        "configs/posttrain_code_sft_qwen35_hauhaucs.yaml",
        "configs/posttrain_math_sft_qwen35_hauhaucs.yaml",
        "configs/posttrain_stem_sft_qwen35_hauhaucs.yaml",
        "configs/posttrain_tool_sft_qwen35_hauhaucs.yaml",
        "configs/grpo_code_qwen35_hauhaucs.yaml",
        "configs/grpo_math_qwen35_hauhaucs.yaml",
        "configs/grpo_tool_qwen35_hauhaucs.yaml",
    ]:
        cfg = load_train_config(path)
        assert cfg.data.tokenizer_path.endswith("Qwen3.5-9B-official-hf")


def test_math_sft_config_stays_low_memory_after_oom_recovery() -> None:
    cfg = load_train_config("configs/posttrain_math_sft_qwen35_hauhaucs.yaml")
    assert cfg.data.seq_len <= 512
    assert cfg.model.L_max <= 3
    assert cfg.ilsd.entropy_floor_weight == 0.0
    assert cfg.ilsd.entropy_curvature_weight == 0.0
    assert cfg.ilsd.logit_curvature_weight == 0.0
    assert cfg.ilsd.logit_curvature_max_positions == 0
    assert cfg.ilsd.local_consistency_weight == 0.0
