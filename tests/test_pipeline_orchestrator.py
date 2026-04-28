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


def test_side_lora_profile_runs_only_side_stages() -> None:
    mod = _load_pipeline_module()

    names = [stage.name for stage in mod.STAGE_PROFILES["side-lora"]]

    assert names == [
        "00_side_lora_sft",
        "01_side_lora_ilsd",
        "02_export_side_lora_adapters",
    ]


def test_posttrain_grpo_profile_runs_distilled_sft_then_kl_grpo() -> None:
    mod = _load_pipeline_module()

    names = [stage.name for stage in mod.STAGE_PROFILES["posttrain-grpo"]]

    assert names == [
        "00_prepare_detection_sft",
        "01_detection_sft",
        "02_prepare_hauhaucs_lanes",
        "03_hauhaucs_lane_sft",
        "04_kl_grpo",
        "05_side_lora_ilsd",
        "06_export_side_lora_adapters",
        "07_eval_compare",
    ]


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


def test_build_training_command_uses_initial_resume_and_vsdev(tmp_path: Path) -> None:
    mod = _load_pipeline_module()
    run_dir = tmp_path / "run"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"run_dir: {run_dir.as_posix()}\n", encoding="utf-8")
    base = tmp_path / "base.pt"
    base.write_bytes(b"base")

    plan = mod.build_training_command(
        str(cfg),
        entrypoint="elt-train",
        initial_resume=base,
        use_vsdev=True,
    )

    assert plan.resume_path == base
    assert plan.cmd[:2] == ["cmd.exe", "/c"]
    assert "VsDevCmd.bat" in plan.cmd[2]
    assert "CC=cl.exe" in plan.cmd[2]
    assert str(base) in plan.cmd[2]


def test_cleanup_completed_offload_only_allows_run_offload_dirs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_pipeline_module()
    allowed = tmp_path / "elt_data" / "runs"
    offload = allowed / "some_run" / "offload_nvme"
    offload.mkdir(parents=True)
    (offload / "state.f32").write_bytes(b"state")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"offload:\n  root: {offload.as_posix()}\n", encoding="utf-8")

    def fake_allowed_path(value: str) -> Path:
        if value == "H:/elt_data/runs":
            return allowed
        return Path(value)

    monkeypatch.setattr(mod, "Path", fake_allowed_path)

    mod.cleanup_completed_offload(str(cfg), dry_run=False)

    assert not offload.exists()


def test_cleanup_completed_offload_refuses_non_offload_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_pipeline_module()
    allowed = tmp_path / "elt_data" / "runs"
    unsafe = allowed / "some_run"
    unsafe.mkdir(parents=True)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"offload:\n  root: {unsafe.as_posix()}\n", encoding="utf-8")

    def fake_allowed_path(value: str) -> Path:
        if value == "H:/elt_data/runs":
            return allowed
        return Path(value)

    monkeypatch.setattr(mod, "Path", fake_allowed_path)

    try:
        mod.cleanup_completed_offload(str(cfg), dry_run=False)
    except mod.PipelineError as exc:
        assert "refusing unsafe offload cleanup path" in str(exc)
    else:
        raise AssertionError("expected PipelineError")


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
        "configs/qwen35_4b_side_lora_code_sft.yaml",
        "configs/qwen35_4b_side_lora_math_sft.yaml",
        "configs/qwen35_4b_side_lora_stem_sft.yaml",
        "configs/qwen35_4b_side_lora_tool_sft.yaml",
        "configs/qwen35_4b_side_lora_code_ilsd_l2.yaml",
    ]
    for rel in configs:
        payload = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        assert payload["rolling_ckpt_interval_sec"] == 300, rel
        assert payload["rolling_ckpt_keep"] == 3, rel
        if "grpo" in rel:
            assert payload["grpo"]["kl_beta"] > 0, rel
        if "side_lora" in rel:
            assert payload["model"]["hf_save_adapter_only"] is True, rel


def test_pipeline_learning_configs_avoid_bitsandbytes_on_windows() -> None:
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
        assert payload["optim"]["kind"] == "nvme_adamw", rel
        assert payload["offload"]["root"].endswith("/offload_nvme"), rel


def test_posttrain_grpo_profile_uses_bounded_phase1_steps() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_max_steps = {
        "configs/posttrain_detection_sft_huihui_qwen36.yaml": 24,
        "configs/posttrain_code_sft_qwen35_hauhaucs.yaml": 48,
        "configs/posttrain_math_sft_qwen35_hauhaucs.yaml": 48,
        "configs/posttrain_stem_sft_qwen35_hauhaucs.yaml": 40,
        "configs/posttrain_tool_sft_qwen35_hauhaucs.yaml": 40,
        "configs/grpo_code_qwen35_hauhaucs.yaml": 64,
        "configs/grpo_math_qwen35_hauhaucs.yaml": 64,
        "configs/grpo_tool_qwen35_hauhaucs.yaml": 64,
    }
    for rel, max_steps in expected_max_steps.items():
        payload = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        assert payload["total_steps"] <= max_steps, rel
        assert payload["log_every"] == 1, rel
        assert payload["save_every"] <= 16, rel
        assert payload["eval_every"] <= 16, rel
        if rel != "configs/posttrain_detection_sft_huihui_qwen36.yaml":
            assert payload["data"]["seq_len"] <= 1024, rel


def test_side_lora_configs_avoid_bitsandbytes_and_nvme_state() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "configs/qwen35_4b_side_lora_code_sft.yaml",
        "configs/qwen35_4b_side_lora_math_sft.yaml",
        "configs/qwen35_4b_side_lora_stem_sft.yaml",
        "configs/qwen35_4b_side_lora_tool_sft.yaml",
        "configs/qwen35_4b_side_lora_code_ilsd_l2.yaml",
    ]:
        payload = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        assert payload["optim"]["kind"] == "adamw", rel
        assert payload["model"]["hf_trainable_mode"] == "lora", rel
        assert payload["model"]["hf_adapter_base_ckpt"].endswith("/last.pt"), rel


def test_huihui_35b_distill_config_is_oom_conservative() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "configs/gguf_distill_huihui_qwen36.yaml",
        "configs/gguf_distill_huihui_qwen36_resume_lowmem.yaml",
        "configs/gguf_distill_huihui_qwen36_resume_smoke.yaml",
    ]:
        payload = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        assert payload["teacher"]["ctx_size"] <= 2048, rel
        assert payload["teacher"]["n_gpu_layers"] <= 16, rel
