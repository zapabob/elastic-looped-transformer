from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("elt_pipeline", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_stages_supports_only_and_skip() -> None:
    mod = _load_pipeline_module()

    selected = mod.select_stages(mod.STAGES, only="pretrain")
    assert [stage.name for stage in selected] == ["00_pretrain_clean"]

    selected = mod.select_stages(mod.STAGES, skip="grpo,eval")
    assert "06_kl_grpo" not in [stage.name for stage in selected]
    assert "07_eval_compare" not in [stage.name for stage in selected]


def test_build_training_command_adds_resume_when_last_exists(tmp_path: Path) -> None:
    mod = _load_pipeline_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "last.pt").write_bytes(b"checkpoint")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"run_dir: {run_dir.as_posix()}\n", encoding="utf-8")

    plan = mod.build_training_command(str(cfg), entrypoint="elt-train")

    assert plan.resume_path == run_dir / "last.pt"
    assert plan.cmd[-2:] == ["--resume", str(run_dir / "last.pt")]


def test_inspect_distill_bundle_detects_zero_jsonl_with_summary(tmp_path: Path) -> None:
    mod = _load_pipeline_module()
    (tmp_path / "distill_train.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "distill_val.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "eval_summary.json").write_text(
        json.dumps({"total_records": 384}),
        encoding="utf-8",
    )

    info = mod.inspect_distill_bundle(tmp_path)

    assert info["total_records"] == 384
    assert info["train_nonempty"] is False
    assert info["val_nonempty"] is False


def test_protected_detection_stage_refuses_to_regenerate_zero_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_pipeline_module()
    (tmp_path / "distill_train.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "distill_val.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "eval_summary.json").write_text(
        json.dumps({"total_records": 384}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "HUIHUI_DETECTION_ROOT", tmp_path)

    try:
        mod.stage_distill_huihui_detection_upload_or_recover(mod.PipelineContext(dry_run=True))
    except mod.PipelineError as exc:
        assert "protected Huihui detection bundle" in str(exc)
    else:
        raise AssertionError("expected PipelineError")


def test_learning_configs_keep_five_minute_three_slot_rolling() -> None:
    root = Path(__file__).resolve().parents[1]
    configs = [
        "configs/base_1B_continue_clean.yaml",
        "configs/posttrain_detection_sft_huihui_qwen36.yaml",
        "configs/posttrain_code_sft_qwen35_hauhaucs.yaml",
        "configs/posttrain_math_sft_qwen35_hauhaucs.yaml",
        "configs/posttrain_stem_sft_qwen35_hauhaucs.yaml",
        "configs/posttrain_tool_sft_qwen35_hauhaucs.yaml",
        "configs/grpo_code_qwen35_hauhaucs.yaml",
        "configs/grpo_math_qwen35_hauhaucs.yaml",
        "configs/grpo_tool_qwen35_hauhaucs.yaml",
    ]
    for rel in configs:
        payload = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        assert payload["rolling_ckpt_interval_sec"] == 300, rel
        assert payload["rolling_ckpt_keep"] == 3, rel
        if "grpo" in rel:
            assert payload["grpo"]["kl_beta"] > 0, rel
