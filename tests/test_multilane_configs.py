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
