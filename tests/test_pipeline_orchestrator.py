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
        "03a_stem_sft_val_eval",
        "04_kl_grpo",
        "05_side_lora_ilsd",
        "06_export_side_lora_adapters",
        "07_eval_compare",
    ]


def test_replay_refresh_profile_runs_clean_replay_then_mixed_sft_and_grpo() -> None:
    mod = _load_pipeline_module()

    names = [stage.name for stage in mod.STAGE_PROFILES["replay-refresh"]]

    assert names == [
        "00_native_clean_replay_pretrain",
        "01_prepare_mixed_lane_sft",
        "02_native_mixed_lane_sft",
        "03_native_kl_grpo",
        "04_side_lora_mixed_sft",
        "05_eval_compare",
    ]


def test_v1_pretrain_posttrain_profile_fetches_hf_then_quality_distill() -> None:
    mod = _load_pipeline_module()

    names = [stage.name for stage in mod.STAGE_PROFILES["v1-pretrain-posttrain"]]

    assert names == [
        "00_fetch_hf_dataset_mix_v1",
        "01_build_synthetic_v1_seed",
        "02_hauhaucs_v1_multilane_distill",
        "03_prepare_hauhaucs_v1_lanes",
        "04_native_clean_replay_pretrain",
        "05_hauhaucs_v1_lane_sft",
        "06_kl_grpo_v1",
        "07_eval_compare",
    ]


def test_synthetic_v1_pretrain_posttrain_profile_skips_teacher_distill() -> None:
    mod = _load_pipeline_module()

    names = [stage.name for stage in mod.STAGE_PROFILES["synthetic-v1-pretrain-posttrain"]]

    assert names == [
        "00_fetch_hf_dataset_mix_v1",
        "01_build_synthetic_v1_seed",
        "02_native_clean_replay_pretrain",
        "03_eval_compare",
    ]


def test_synthetic_v1_seed_stage_requires_gb_target_before_skip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_pipeline_module()
    output_root = tmp_path / "synthetic_v1_seed_gb"
    output_root.mkdir()
    summary = {
        "total_bytes": 10 * 1024 * 1024,
        "lanes": {
            lane: {
                "total_records": 512,
                "sample_verifier_pass_rate": 1.0,
                "unique_text_ratio": 1.0,
            }
            for lane in ("code", "math", "stem_reasoning", "tool_use")
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    captured: list[list[str]] = []

    def fake_run_subprocess(cmd: list[str], *, dry_run: bool = False) -> int:
        captured.append(cmd)
        return 0

    monkeypatch.setattr(mod, "SYNTHETIC_V1_SEED_ROOT", output_root)
    monkeypatch.setattr(mod, "SYNTHETIC_V1_TARGET_BYTES", 1024 * 1024 * 1024)
    monkeypatch.setattr(mod, "run_subprocess", fake_run_subprocess)

    mod.stage_build_synthetic_v1_seed(mod.PipelineContext(dry_run=True))

    assert len(captured) == 1
    assert "--target-bytes" in captured[0]
    assert str(1024 * 1024 * 1024) in captured[0]


def test_synthetic_v1_seed_stage_skips_only_when_target_bytes_and_quality_pass(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_pipeline_module()
    output_root = tmp_path / "synthetic_v1_seed_gb"
    output_root.mkdir()
    summary = {
        "total_bytes": 1024 * 1024 * 1024,
        "lanes": {
            lane: {
                "total_records": 1000,
                "sample_verifier_pass_rate": 1.0,
                "unique_text_ratio": 1.0,
            }
            for lane in ("code", "math", "stem_reasoning", "tool_use")
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    captured: list[list[str]] = []

    monkeypatch.setattr(mod, "SYNTHETIC_V1_SEED_ROOT", output_root)
    monkeypatch.setattr(mod, "SYNTHETIC_V1_TARGET_BYTES", 1024 * 1024 * 1024)
    monkeypatch.setattr(mod, "run_subprocess", lambda cmd, dry_run=False: captured.append(cmd))

    mod.stage_build_synthetic_v1_seed(mod.PipelineContext(dry_run=True))

    assert captured == []


def test_v1_queue_config_points_only_to_v1_lane_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (root / "configs/gguf_distill_qwen35_hauhaucs_multilane_v1_queue.yaml").read_text(
            encoding="utf-8"
        )
    )
    configs = [stage["config"] for stage in payload["stages"]]

    assert configs == [
        "gguf_distill_code_qwen35_hauhaucs_v1.yaml",
        "gguf_distill_math_qwen35_hauhaucs_v1.yaml",
        "gguf_distill_stem_qwen35_hauhaucs_v1.yaml",
        "gguf_distill_tool_qwen35_hauhaucs_v1.yaml",
    ]
    assert all("_v1.yaml" in item for item in configs)


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


def test_stem_sft_val_eval_builds_format_verifier_anytime_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_pipeline_module()
    run_dir = tmp_path / "stem_run"
    eval_dir = run_dir / "eval"
    run_dir.mkdir()
    (run_dir / "last.pt").write_bytes(b"checkpoint")
    (run_dir / "metrics.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "checkpoint", "kind": "final", "step": 40}),
            json.dumps({"event": "run_end"}),
        ]),
        encoding="utf-8",
    )
    cfg = tmp_path / "stem.yaml"
    cfg.write_text(
        f"run_dir: {run_dir.as_posix()}\ntotal_steps: 40\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "stem_manifest.yaml"
    manifest.write_text("benchmarks: []\n", encoding="utf-8")
    out_json = eval_dir / "stem_val_format_verifier_summary.json"
    out_csv = eval_dir / "stem_val_format_verifier_anytime.csv"
    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    clean_cases = bench_dir / "gguf_stem_reasoning_val_cases.jsonl"
    clean_cases.write_text(
        "\n".join([
            json.dumps({
                "prompt": "Question: real A\nChoices:\nA. alpha\nB. beta\nC. gamma\nD. delta",
                "reference": "A",
            }),
            json.dumps({
                "prompt": "Question: real B\nChoices:\nA. alpha\nB. beta\nC. gamma\nD. delta",
                "reference": "B",
            }),
        ]),
        encoding="utf-8",
    )
    manifest = bench_dir / "stem_manifest.yaml"
    manifest.write_text("benchmarks: []\n", encoding="utf-8")
    captured: list[list[str]] = []

    def fake_run_subprocess(cmd: list[str], *, dry_run: bool = False) -> int:
        captured.append(cmd)
        assert dry_run is True
        return 0

    monkeypatch.setattr(mod, "STEM_SFT_CONFIG", str(cfg))
    monkeypatch.setattr(mod, "STEM_VAL_MANIFEST", manifest)
    monkeypatch.setattr(mod, "STEM_VAL_EVAL_DIR", eval_dir)
    monkeypatch.setattr(mod, "STEM_VAL_EVAL_JSON", out_json)
    monkeypatch.setattr(mod, "STEM_VAL_EVAL_CSV", out_csv)
    monkeypatch.setattr(mod, "run_subprocess", fake_run_subprocess)

    mod.stage_stem_sft_val_eval(mod.PipelineContext(dry_run=True))

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:4] == ["uv", "run", "--no-sync", "elt-anytime"]
    assert "--benchmark-manifest" in cmd
    assert str(manifest) in cmd
    assert "--out-json" in cmd
    assert str(out_json) in cmd
    assert "--out-csv" in cmd
    assert str(out_csv) in cmd
    assert "--L-list" in cmd
    assert "1,2,3,4" in cmd


def test_stem_sft_val_eval_skips_placeholder_v0_quality_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_pipeline_module()
    run_dir = tmp_path / "stem_run"
    eval_dir = run_dir / "eval"
    run_dir.mkdir()
    (run_dir / "last.pt").write_bytes(b"checkpoint")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"event": "checkpoint", "kind": "final", "step": 40}),
        encoding="utf-8",
    )
    cfg = tmp_path / "stem.yaml"
    cfg.write_text(
        f"run_dir: {run_dir.as_posix()}\ntotal_steps: 40\n",
        encoding="utf-8",
    )
    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    cases = bench_dir / "gguf_stem_reasoning_val_cases.jsonl"
    cases.write_text(
        "\n".join([
            json.dumps({
                "prompt": "Question: duplicate\nChoices:\nA. Option A\nB. Option B\nC. Option C\nD. Option D",
                "reference": "A",
            }),
            json.dumps({
                "prompt": "Question: duplicate\nChoices:\nA. Option A\nB. Option B\nC. Option C\nD. Option D",
                "reference": "A",
            }),
        ]),
        encoding="utf-8",
    )
    manifest = bench_dir / "stem_manifest.yaml"
    manifest.write_text("benchmarks: []\n", encoding="utf-8")
    out_json = eval_dir / "stem_val_format_verifier_summary.json"
    out_csv = eval_dir / "stem_val_format_verifier_anytime.csv"
    captured: list[list[str]] = []

    monkeypatch.setattr(mod, "STEM_SFT_CONFIG", str(cfg))
    monkeypatch.setattr(mod, "STEM_VAL_MANIFEST", manifest)
    monkeypatch.setattr(mod, "STEM_VAL_EVAL_DIR", eval_dir)
    monkeypatch.setattr(mod, "STEM_VAL_EVAL_JSON", out_json)
    monkeypatch.setattr(mod, "STEM_VAL_EVAL_CSV", out_csv)
    monkeypatch.setattr(mod, "run_subprocess", lambda cmd, dry_run=False: captured.append(cmd))

    mod.stage_stem_sft_val_eval(mod.PipelineContext(dry_run=True))

    assert captured == []
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["state"] == "skipped_quality_gate"
    assert payload["quality"]["placeholder_choice_count"] == 2
    assert out_csv.exists()


def test_v0_lane_quality_gate_blocks_smoke_fallbacks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_pipeline_module()
    input_root = tmp_path / "code_distill"
    input_root.mkdir()
    row = {
        "prompt": "write a function",
        "response": "```python\ndef solve() -> None:\n    return None\n```",
        "reference": "result = locals().get('solve')\nassert callable(result)",
        "text": "Assistant: ```python\ndef solve() -> None:\n    return None\n```",
    }
    for name in ("distill_train.jsonl", "distill_val.jsonl"):
        (input_root / name).write_text(json.dumps(row) + "\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    monkeypatch.setattr(mod, "STATE_DIR", state_dir)

    try:
        mod.enforce_v0_lane_quality("code", input_root)
    except mod.PipelineError as exc:
        assert "refusing v0 code SFT" in str(exc)
    else:
        raise AssertionError("expected PipelineError")

    payload = json.loads((state_dir / "v0_code_quality_gate.json").read_text(encoding="utf-8"))
    assert payload["state"] == "failed_quality_gate"
    assert payload["quality"]["fallback_return_none"] == 4


def test_v0_lane_quality_gate_can_be_overridden_for_explicit_smoke(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_pipeline_module()
    input_root = tmp_path / "tool_distill"
    input_root.mkdir()
    row = {
        "prompt": "select tool",
        "response": '{"tool_name": "shell_command", "arguments": {}}',
        "reference": '{"tool_name": "shell_command", "arguments": {}}',
        "text": '{"tool_name": "shell_command", "arguments": {}}',
    }
    for name in ("distill_train.jsonl", "distill_val.jsonl"):
        (input_root / name).write_text(json.dumps(row) + "\n", encoding="utf-8")

    monkeypatch.setenv("ELT_ALLOW_V0_SMOKE_TRAINING", "1")

    mod.enforce_v0_lane_quality("tool_use", input_root)


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


def test_pipeline_subprocess_env_prefers_h_drive(monkeypatch, tmp_path: Path) -> None:
    mod = _load_pipeline_module()
    cache_root = tmp_path / "cache"
    temp_dir = cache_root / "tmp"
    monkeypatch.setattr(mod, "H_CACHE_ROOT", cache_root)
    monkeypatch.setattr(mod, "H_TEMP_DIR", temp_dir)
    monkeypatch.setattr(
        mod,
        "H_DRIVE_ENV",
        {
            "TMP": str(temp_dir),
            "TEMP": str(temp_dir),
            "TMPDIR": str(temp_dir),
            "UV_CACHE_DIR": str(cache_root / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(cache_root / "uv" / "python"),
            "HF_HOME": str(cache_root / "hf"),
            "HF_DATASETS_CACHE": str(cache_root / "hf" / "datasets"),
            "TORCH_HOME": str(cache_root / "torch"),
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
            "CUDA_CACHE_PATH": str(cache_root / "cuda"),
            "PYTHONPYCACHEPREFIX": str(cache_root / "pycache"),
        },
    )

    env = mod.h_drive_subprocess_env()

    assert env["TMP"] == str(temp_dir)
    assert env["TEMP"] == str(temp_dir)
    assert env["UV_CACHE_DIR"] == str(cache_root / "uv")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(cache_root / "uv" / "python")
    assert env["HF_HOME"] == str(cache_root / "hf")
    assert env["HF_DATASETS_CACHE"] == str(cache_root / "hf" / "datasets")
    assert env["TORCH_HOME"] == str(cache_root / "torch")
    assert env["CUDA_CACHE_PATH"] == str(cache_root / "cuda")
    assert env["PYTHONPYCACHEPREFIX"] == str(cache_root / "pycache")
    assert (cache_root / "hf").is_dir()


def test_vsdev_command_sets_h_drive_cache_env() -> None:
    mod = _load_pipeline_module()

    cmd = mod.vsdev_command(["uv", "run", "python", "-V"])

    assert cmd[:2] == ["cmd.exe", "/c"]
    assert "set HF_HOME=" in cmd[2]
    assert "set UV_CACHE_DIR=" in cmd[2]
    assert "set TEMP=" in cmd[2]
    assert "set CUDA_CACHE_PATH=" in cmd[2]
    assert "set PYTHONPYCACHEPREFIX=" in cmd[2]


def test_training_run_complete_detects_final_checkpoint(tmp_path: Path) -> None:
    mod = _load_pipeline_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "last.pt").write_bytes(b"checkpoint")
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "train_step", "step": 47}),
                json.dumps({"event": "checkpoint", "kind": "final", "step": 48}),
                json.dumps({"event": "run_end"}),
            ]
        ),
        encoding="utf-8",
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"run_dir: {run_dir.as_posix()}\ntotal_steps: 48\n",
        encoding="utf-8",
    )

    assert mod.training_run_complete(str(cfg)) is True


def test_training_run_complete_rejects_partial_checkpoint(tmp_path: Path) -> None:
    mod = _load_pipeline_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "last.pt").write_bytes(b"checkpoint")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"event": "checkpoint", "kind": "rolling", "step": 47}),
        encoding="utf-8",
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"run_dir: {run_dir.as_posix()}\ntotal_steps: 48\n",
        encoding="utf-8",
    )

    assert mod.training_run_complete(str(cfg)) is False


def test_prune_completed_checkpoints_keeps_last_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_pipeline_module()
    allowed = tmp_path / "elt_data" / "runs"
    run_dir = allowed / "complete_run"
    run_dir.mkdir(parents=True)
    (run_dir / "last.pt").write_bytes(b"last")
    (run_dir / "rolling_0.pt").write_bytes(b"rolling")
    (run_dir / "step_0000048.pt").write_bytes(b"step")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"event": "checkpoint", "kind": "final", "step": 48}),
        encoding="utf-8",
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"run_dir: {run_dir.as_posix()}\ntotal_steps: 48\n",
        encoding="utf-8",
    )

    def fake_allowed_path(value: str) -> Path:
        if value == "H:/elt_data/runs":
            return allowed
        return Path(value)

    monkeypatch.setattr(mod, "Path", fake_allowed_path)

    mod.prune_completed_checkpoints(str(cfg), dry_run=False)

    assert (run_dir / "last.pt").exists()
    assert not (run_dir / "rolling_0.pt").exists()
    assert not (run_dir / "step_0000048.pt").exists()


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
        "configs/posttrain_code_sft_qwen35_hauhaucs_v1.yaml",
        "configs/posttrain_math_sft_qwen35_hauhaucs_v1.yaml",
        "configs/posttrain_stem_sft_qwen35_hauhaucs_v1.yaml",
        "configs/posttrain_tool_sft_qwen35_hauhaucs_v1.yaml",
        "configs/grpo_code_qwen35_hauhaucs_v1.yaml",
        "configs/grpo_math_qwen35_hauhaucs_v1.yaml",
        "configs/grpo_tool_qwen35_hauhaucs_v1.yaml",
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
        "configs/posttrain_code_sft_qwen35_hauhaucs_v1.yaml",
        "configs/posttrain_math_sft_qwen35_hauhaucs_v1.yaml",
        "configs/posttrain_stem_sft_qwen35_hauhaucs_v1.yaml",
        "configs/posttrain_tool_sft_qwen35_hauhaucs_v1.yaml",
        "configs/grpo_code_qwen35_hauhaucs_v1.yaml",
        "configs/grpo_math_qwen35_hauhaucs_v1.yaml",
        "configs/grpo_tool_qwen35_hauhaucs_v1.yaml",
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
        "configs/posttrain_code_sft_qwen35_hauhaucs_v1.yaml": 96,
        "configs/posttrain_math_sft_qwen35_hauhaucs_v1.yaml": 96,
        "configs/posttrain_stem_sft_qwen35_hauhaucs_v1.yaml": 80,
        "configs/posttrain_tool_sft_qwen35_hauhaucs_v1.yaml": 80,
        "configs/grpo_code_qwen35_hauhaucs_v1.yaml": 64,
        "configs/grpo_math_qwen35_hauhaucs_v1.yaml": 64,
        "configs/grpo_tool_qwen35_hauhaucs_v1.yaml": 64,
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
