from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

from elt_lm.gguf_distill import (
    build_teacher_instruction,
    build_hf_cli_plan,
    build_status_snapshot,
    build_sft_record,
    build_task_specs,
    evaluate_distill_records,
    extract_json_object,
    guard_against_unsafe_reset,
    load_gguf_distill_config,
    acquire_run_lock,
    normalize_teacher_example,
    assert_quality_gate,
    DistillTask,
    DistillQualityError,
    QualityGateError,
    write_status_artifacts,
    validate_distill_record_quality,
)


def _quality_cfg(tmp_path: Path, lane: str):
    cfg_path = tmp_path / f"gguf_{lane}_v1.yaml"
    cfg_path.write_text(
        f"""
teacher:
  model_path: C:/models/teacher.gguf
pipeline:
  output_root: H:/elt_data/gguf_distill/{lane}_v1
  samples_per_task: 2
  quality_profile: v1
  reject_fallback_outputs: true
  min_unique_text_ratio: 0.75
  max_exact_duplicate_ratio: 0.10
  max_generation_retries: 3
lane: {lane}
""".strip(),
        encoding="utf-8",
    )
    return load_gguf_distill_config(cfg_path)


def _lane_task(lane: str, variant_index: int = 0) -> DistillTask:
    target_kind = {
        "code": "python_exec",
        "math": "exact_math",
        "stem_reasoning": "mcq_reasoning",
        "tool_use": "json_match",
    }[lane]
    return DistillTask(
        lane=lane,  # type: ignore[arg-type]
        domain=f"{lane}_quality",
        description=f"{lane} quality task",
        target_kind=target_kind,
        tags=["shell_command"] if lane == "tool_use" else [lane],
        target_label="",
        risk_tags=[],
        variant_index=variant_index,
        mode="standard",
        variant="quality unit test",
    )


def test_load_config_and_build_task_specs(tmp_path: Path) -> None:
    cfg_path = tmp_path / "gguf_distill.yaml"
    cfg_path.write_text(
        """
teacher:
  model_path: C:/models/teacher.gguf
  mmproj_path: C:/models/teacher.mmproj.gguf
  host: 127.0.0.1
  port: 8091
pipeline:
  output_root: H:/elt_data/gguf_distill/test
  repo_id: zapabobouj/test-distill
  samples_per_domain: 3
domains:
  - name: drug_detection
    description: detect drug sale or usage content
    target_label: review
    risk_tags: [drug_reference, procurement]
  - name: nsfw_detection
    description: detect adult sexual content
    target_label: block
    risk_tags: [adult_content]
""".strip(),
        encoding="utf-8",
    )

    cfg = load_gguf_distill_config(cfg_path)
    tasks = build_task_specs(cfg)

    assert cfg.teacher.model_path.endswith("teacher.gguf")
    assert cfg.pipeline.repo_id == "zapabobouj/test-distill"
    assert len(tasks) == 6
    assert {task.domain for task in tasks} == {"drug_detection", "nsfw_detection"}


def test_extract_json_object_handles_fenced_payload() -> None:
    payload = """
Here is the example.

```json
{"input_text":"buy pills discreetly","policy_label":"review","severity":"high","risk_tags":["drug_reference"],"rationale":"procurement request"}
```
"""
    obj = extract_json_object(payload)
    assert obj is not None
    assert obj["policy_label"] == "review"
    assert obj["risk_tags"] == ["drug_reference"]


def test_normalize_teacher_example_falls_back_to_line_fields() -> None:
    task = DistillTask(
        lane="detection",
        domain="drug_detection",
        description="detect drug sale or usage content",
        target_kind="json_match",
        tags=["drug_reference"],
        target_label="review",
        risk_tags=["drug_reference"],
        variant_index=0,
        mode="positive",
    )
    text = """
input_text: DM me if you know where to buy some.
policy_label: review
severity: high
risk_tags: drug_reference, procurement
rationale: asks for procurement
"""
    example = normalize_teacher_example(text, task, None)
    assert example["policy_label"] == "review"
    assert example["severity"] == "high"
    assert example["risk_tags"] == ["drug_reference", "procurement"]


def test_build_sft_record_renders_chat_shape() -> None:
    example = {
        "input_text": "DM me for where to buy it cheap",
        "policy_label": "review",
        "severity": "medium",
        "risk_tags": ["drug_reference", "procurement"],
        "rationale": "mentions buying a controlled substance",
    }

    record = build_sft_record(
        domain="drug_detection",
        example=example,
        teacher_name="huihui-qwen36-gguf",
        split="train",
    )

    assert record["mode"] == "sft"
    assert record["source"] == "huihui-qwen36-gguf"
    assert "Analyze the following content" in record["prompt"]
    assert "Assistant:" in record["text"]
    response = json.loads(record["response"])
    assert response["domain"] == "drug_detection"
    assert response["policy_label"] == "review"
    assert record["metadata"]["split"] == "train"


def test_evaluate_distill_records_counts_domains_and_duplicates() -> None:
    records = [
        {
            "prompt": "p1",
            "response": json.dumps({"domain": "drug_detection", "policy_label": "review"}),
            "metadata": {"split": "train"},
        },
        {
            "prompt": "p2",
            "response": json.dumps({"domain": "drug_detection", "policy_label": "review"}),
            "metadata": {"split": "val"},
        },
        {
            "prompt": "p1",
            "response": json.dumps({"domain": "nsfw_detection", "policy_label": "block"}),
            "metadata": {"split": "train"},
        },
    ]

    summary = evaluate_distill_records(records)

    assert summary["total_records"] == 3
    assert summary["duplicate_prompt_count"] == 1
    assert summary["domain_counts"]["drug_detection"] == 2
    assert summary["label_counts"]["review"] == 2
    assert summary["split_counts"]["train"] == 2


def test_load_lane_config_uses_samples_per_task(tmp_path: Path) -> None:
    cfg_path = tmp_path / "gguf_code.yaml"
    cfg_path.write_text(
        """
teacher:
  model_path: C:/models/teacher.gguf
pipeline:
  output_root: H:/elt_data/gguf_distill/code
  samples_per_task: 2
lane: code
tasks:
  - name: function_implementation
    description: write Python functions
    target_kind: python_exec
    tags: [python]
    variants: [one, two]
""".strip(),
        encoding="utf-8",
    )

    cfg = load_gguf_distill_config(cfg_path)
    tasks = build_task_specs(cfg)

    assert cfg.lane == "code"
    assert cfg.pipeline.samples_per_task == 2
    assert len(tasks) == 2
    assert all(task.lane == "code" for task in tasks)
    assert {task.variant for task in tasks} == {"one", "two"}


def test_v1_teacher_prompt_varies_by_variant_index() -> None:
    task0 = _lane_task("math", variant_index=0)
    task1 = _lane_task("math", variant_index=1)

    prompt0 = build_teacher_instruction(task0, quality_profile="v1")
    prompt1 = build_teacher_instruction(task1, quality_profile="v1")

    assert prompt0 != prompt1
    assert "DIVERSITY V1 REQUIREMENTS" in prompt0
    assert "- difficulty:" in prompt0
    assert "- reasoning_style:" in prompt0
    assert "fallback answer 0" in prompt0


def test_tool_v1_teacher_prompt_requires_mcp_or_agent_harness() -> None:
    task = _lane_task("tool_use", variant_index=2)

    prompt = build_teacher_instruction(task, quality_profile="v1")

    assert "MCP / AI-agent harness" in prompt
    assert "tool_name must be an MCP-style or agent-harness tool name" in prompt
    assert "DIVERSITY V1 REQUIREMENTS" in prompt
    assert "read_only" in prompt or "dry_run" in prompt


def test_build_code_math_and_tool_records_include_reference_and_lane_metadata() -> None:
    code_task = DistillTask(
        lane="code",
        domain="function_implementation",
        description="write Python functions",
        target_kind="python_exec",
        tags=["python"],
        target_label="",
        risk_tags=[],
        variant_index=0,
        mode="standard",
        variant="pure function",
    )
    math_task = DistillTask(
        lane="math",
        domain="algebra_reasoning",
        description="solve algebra problems",
        target_kind="exact_math",
        tags=["algebra"],
        target_label="",
        risk_tags=[],
        variant_index=0,
        mode="standard",
        variant="linear equation",
    )
    tool_task = DistillTask(
        lane="tool_use",
        domain="shell_selection",
        description="choose a shell tool",
        target_kind="json_match",
        tags=["shell_command"],
        target_label="",
        risk_tags=[],
        variant_index=0,
        mode="standard",
        variant="search files",
    )

    code_record = build_sft_record(
        task=code_task,
        example={
            "user_request": "Write a function add(a, b).",
            "assistant_code": "def add(a, b):\n    return a + b",
            "verifier_snippet": "assert add(2, 3) == 5",
        },
        teacher_name="hauhaucs",
        split="train",
    )
    math_record = build_sft_record(
        task=math_task,
        example={
            "question": "Solve x + 1 = 3.",
            "reasoning": "Subtract 1 from both sides.",
            "final_answer": "2",
            "reference": "2",
        },
        teacher_name="hauhaucs",
        split="val",
    )
    tool_record = build_sft_record(
        task=tool_task,
        example={
            "user_request": "List Python files recursively.",
            "tool_name": "shell_command",
            "arguments": {"command": "Get-ChildItem -Recurse -Filter *.py"},
            "reference": {"tool_name": "shell_command", "arguments": {"command": "Get-ChildItem -Recurse -Filter *.py"}},
        },
        teacher_name="hauhaucs",
        split="train",
    )

    assert code_record["task"] == "python_exec"
    assert code_record["reference"] == "assert add(2, 3) == 5"
    assert code_record["metadata"]["lane"] == "code"
    assert math_record["task"] == "exact_math"
    assert math_record["reference"] == "2"
    assert "<answer>2</answer>" in math_record["response"]
    assert tool_record["task"] == "json_match"
    assert json.loads(tool_record["response"]) == json.loads(tool_record["reference"])
    assert tool_record["metadata"]["lane"] == "tool_use"


def test_build_hf_cli_plan_for_dataset_upload(tmp_path: Path) -> None:
    out_dir = tmp_path / "bundle"
    plan = build_hf_cli_plan(
        output_dir=out_dir,
        repo_id="zapabobouj/elt-lm-distill-dataset",
        private=True,
    )

    assert plan[0] == [
        "hf", "repos", "create", "zapabobouj/elt-lm-distill-dataset",
        "--type", "dataset", "--private", "--exist-ok",
    ]
    assert plan[1][:4] == ["hf", "upload-large-folder", "zapabobouj/elt-lm-distill-dataset", str(out_dir)]
    assert "--type" in plan[1]


def test_build_status_snapshot_tracks_progress_and_eta() -> None:
    snapshot = build_status_snapshot(
        teacher_name="huihui-qwen36-gguf",
        repo_id="zapabobouj/demo",
        current_stage="teacher_generation",
        state="running",
        started_at=100.0,
        updated_at=130.0,
        processed_tasks=3,
        total_tasks=10,
        train_records=2,
        val_records=1,
        error_count=0,
        domain_counts={"drug_detection": 2, "nsfw_detection": 1},
        label_counts={"review": 2, "block": 1},
        split_counts={"train": 2, "val": 1},
        last_domain="nsfw_detection",
        last_policy_label="block",
        last_latency_sec=9.5,
        last_error="",
        student_eval_path="",
    )

    assert snapshot["progress_pct"] == 30.0
    assert snapshot["eta_sec"] == 70.0
    assert snapshot["domain_counts"]["drug_detection"] == 2
    assert snapshot["last_policy_label"] == "block"


def test_write_status_artifacts_writes_status_and_heartbeat(tmp_path: Path) -> None:
    snapshot = build_status_snapshot(
        teacher_name="huihui-qwen36-gguf",
        repo_id="zapabobouj/demo",
        current_stage="teacher_generation",
        state="running",
        started_at=100.0,
        updated_at=130.0,
        processed_tasks=3,
        total_tasks=10,
        train_records=2,
        val_records=1,
        error_count=0,
        domain_counts={"drug_detection": 2},
        label_counts={"review": 2},
        split_counts={"train": 2, "val": 1},
        last_domain="drug_detection",
        last_policy_label="review",
        last_latency_sec=8.0,
        last_error="",
        student_eval_path="",
    )

    write_status_artifacts(tmp_path, snapshot)
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))

    assert status["current_stage"] == "teacher_generation"
    assert heartbeat["state"] == "running"
    assert heartbeat["processed_tasks"] == 3


def test_acquire_run_lock_rejects_live_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "run.lock"
    release = acquire_run_lock(lock_path)
    try:
        with pytest.raises(RuntimeError):
            acquire_run_lock(lock_path)
    finally:
        release()


def test_guard_against_unsafe_reset_rejects_completed_bundle_summary(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_teacher_examples.jsonl"
    train_path = tmp_path / "distill_train.jsonl"
    val_path = tmp_path / "distill_val.jsonl"
    raw_path.write_text("", encoding="utf-8")
    train_path.write_text("", encoding="utf-8")
    val_path.write_text("", encoding="utf-8")
    (tmp_path / "eval_summary.json").write_text(
        json.dumps({"total_records": 384}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Refusing to reset existing GGUF distill output"):
        guard_against_unsafe_reset(tmp_path, (raw_path, train_path, val_path))


def test_guard_against_unsafe_reset_allows_explicit_force_reset(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_teacher_examples.jsonl"
    train_path = tmp_path / "distill_train.jsonl"
    val_path = tmp_path / "distill_val.jsonl"
    train_path.write_text('{"record": 1}\n', encoding="utf-8")

    guard_against_unsafe_reset(
        tmp_path,
        (raw_path, train_path, val_path),
        force_reset=True,
    )


def test_v1_rejects_code_return_none_fallback_and_callable_only_verifier(tmp_path: Path) -> None:
    cfg = _quality_cfg(tmp_path, "code")
    cfg.pipeline.reject_fallback_outputs = False
    task = _lane_task("code")
    fallback_example = {
        "user_request": "Write a function solve().",
        "assistant_code": "def solve() -> None:\n    return None",
        "verifier_snippet": "result = locals().get('solve')\nassert callable(result)",
        "rationale": "placeholder",
    }
    fallback_record = build_sft_record(
        task=task,
        example=fallback_example,
        teacher_name="hauhaucs",
        split="train",
    )

    with pytest.raises(DistillQualityError, match="fallback_code_stub"):
        validate_distill_record_quality(fallback_record, fallback_example, task, cfg)

    callable_example = {
        "user_request": "Write an add function.",
        "assistant_code": "def add(a: int, b: int) -> int:\n    total = a + b\n    return total",
        "verifier_snippet": "result = locals().get('add')\nassert callable(result)",
        "rationale": "valid code but weak verifier",
    }
    callable_record = build_sft_record(
        task=task,
        example=callable_example,
        teacher_name="hauhaucs",
        split="train",
    )
    with pytest.raises(DistillQualityError, match="callable_only_verifier"):
        validate_distill_record_quality(callable_record, callable_example, task, cfg)

    summary = evaluate_distill_records(
        [fallback_record],
        quality_counters={"fallback_reject_count": 1},
        run_verifiers=False,
    )
    assert summary["fallback_reject_count"] > 0


def test_v1_rejects_untyped_code_even_when_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elt_lm.gguf_distill as gguf_distill

    monkeypatch.setattr(gguf_distill, "python_exec_correctness", lambda *_args, **_kwargs: 1.0)
    cfg = _quality_cfg(tmp_path, "code")
    task = _lane_task("code")
    example = {
        "user_request": "Write add(a, b) returning the sum, including negative values.",
        "assistant_code": (
            "def add(a, b):\n"
            "    total = a + b\n"
            "    if total == 0:\n"
            "        return 0\n"
            "    return total\n"
        ),
        "verifier_snippet": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
        "rationale": "executable but not MILSPEC-style typed code",
    }
    record = build_sft_record(task=task, example=example, teacher_name="hauhaucs", split="train")

    with pytest.raises(DistillQualityError, match="missing_typed_public_callable"):
        validate_distill_record_quality(record, example, task, cfg)


def test_v1_accepts_code_with_assert_verifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elt_lm.gguf_distill as gguf_distill

    monkeypatch.setattr(gguf_distill, "python_exec_correctness", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(gguf_distill, "ruff_check_score", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gguf_distill, "mypy_strict_score", lambda *_args, **_kwargs: None)
    cfg = _quality_cfg(tmp_path, "code")
    task = _lane_task("code")
    example = {
        "user_request": "Write add(a, b) returning the sum, including negative values.",
        "assistant_code": "def add(a: int, b: int) -> int:\n    return a + b",
        "verifier_snippet": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
        "rationale": "asserts nominal and edge cases",
    }
    record = build_sft_record(task=task, example=example, teacher_name="hauhaucs", split="train")

    validate_distill_record_quality(record, example, task, cfg)


def test_v1_rejects_math_zero_fallback_and_accepts_equivalent_answer(tmp_path: Path) -> None:
    cfg = _quality_cfg(tmp_path, "math")
    task = _lane_task("math")
    zero_example = {
        "question": "What is 1 - 1?",
        "reasoning": "Subtracting gives zero.",
        "final_answer": "0",
        "reference": "0",
        "rationale": "computed answer",
    }
    zero_record = build_sft_record(task=task, example=zero_example, teacher_name="hauhaucs", split="train")
    with pytest.raises(DistillQualityError, match="fallback_zero_answer"):
        validate_distill_record_quality(zero_record, zero_example, task, cfg)

    equivalent_example = {
        "question": "A fair coin is flipped once. What is the probability of heads?",
        "reasoning": "There is one favorable outcome out of two equally likely outcomes.",
        "final_answer": "1/2",
        "reference": "0.5",
        "rationale": "numeric equivalence should pass",
    }
    equivalent_record = build_sft_record(
        task=task,
        example=equivalent_example,
        teacher_name="hauhaucs",
        split="train",
    )
    validate_distill_record_quality(equivalent_record, equivalent_example, task, cfg)


def test_v1_rejects_stem_placeholder_choices_and_biased_distribution(tmp_path: Path) -> None:
    cfg = _quality_cfg(tmp_path, "stem_reasoning")
    task = _lane_task("stem_reasoning", variant_index=0)
    placeholder_example = {
        "question": "Which option is correct?",
        "choices": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"],
        "reasoning": "Pick A.",
        "final_choice": "A",
        "reference": "A",
        "rationale": "placeholder choices",
    }
    placeholder_record = build_sft_record(
        task=task,
        example=placeholder_example,
        teacher_name="hauhaucs",
        split="train",
    )
    with pytest.raises(DistillQualityError, match="placeholder_stem_choices"):
        validate_distill_record_quality(placeholder_record, placeholder_example, task, cfg)

    records = []
    for idx in range(4):
        example = {
            "question": f"Question {idx}: what is the SI unit of force?",
            "choices": ["A. newton", "B. joule", "C. watt", "D. pascal"],
            "reasoning": "Force is measured in newtons.",
            "final_choice": "A",
            "reference": "A",
            "rationale": "real choices but biased label",
        }
        records.append(build_sft_record(task=task, example=example, teacher_name="hauhaucs", split="train"))
    summary = evaluate_distill_records(records, quality_counters={"attempted_tasks": 4}, run_verifiers=True)
    with pytest.raises(QualityGateError, match="answer distribution"):
        assert_quality_gate(summary, cfg)


def test_v1_rejects_empty_and_non_mcp_tool_arguments_and_accepts_exact_json(tmp_path: Path) -> None:
    cfg = _quality_cfg(tmp_path, "tool_use")
    task = _lane_task("tool_use")
    empty_example = {
        "user_request": "List files.",
        "tool_name": "mcp.shell_command",
        "arguments": {},
        "reference": {"tool_name": "mcp.shell_command", "arguments": {}},
        "rationale": "empty arguments are not useful",
    }
    empty_record = build_sft_record(task=task, example=empty_example, teacher_name="hauhaucs", split="train")
    with pytest.raises(DistillQualityError, match="empty_tool_arguments"):
        validate_distill_record_quality(empty_record, empty_example, task, cfg)

    non_mcp_example = {
        "user_request": "Search Python files for TODO comments.",
        "tool_name": "shell_command",
        "arguments": {"command": "Select-String -Path ./**/*.py -Pattern TODO", "timeout_ms": 20000},
        "reference": {
            "tool_name": "shell_command",
            "arguments": {"command": "Select-String -Path ./**/*.py -Pattern TODO", "timeout_ms": 20000},
        },
        "rationale": "legacy direct tool call without MCP or agent harness prefix",
    }
    non_mcp_record = build_sft_record(task=task, example=non_mcp_example, teacher_name="hauhaucs", split="train")
    with pytest.raises(DistillQualityError, match="non_mcp_agent_tool_name"):
        validate_distill_record_quality(non_mcp_record, non_mcp_example, task, cfg)

    good_example = {
        "user_request": "Search Python files for TODO comments.",
        "tool_name": "mcp.shell_command",
        "arguments": {
            "server": "local-powershell",
            "tool": "shell_command",
            "input": {"command": "Select-String -Path ./**/*.py -Pattern TODO"},
            "cwd": "C:/repo",
            "timeout_ms": 20000,
            "expected_observation": "matching file paths and lines",
            "safety": "read_only",
        },
        "reference": {
            "tool_name": "mcp.shell_command",
            "arguments": {
                "server": "local-powershell",
                "tool": "shell_command",
                "input": {"command": "Select-String -Path ./**/*.py -Pattern TODO"},
                "cwd": "C:/repo",
                "timeout_ms": 20000,
                "expected_observation": "matching file paths and lines",
                "safety": "read_only",
            },
        },
        "rationale": "exact non-empty JSON call",
    }
    good_record = build_sft_record(task=task, example=good_example, teacher_name="hauhaucs", split="train")
    validate_distill_record_quality(good_record, good_example, task, cfg)


def test_v1_quality_gate_fails_duplicate_text_ratio(tmp_path: Path) -> None:
    cfg = _quality_cfg(tmp_path, "tool_use")
    task = _lane_task("tool_use")
    example = {
        "user_request": "Search Python files for TODO comments.",
        "tool_name": "mcp.shell_command",
        "arguments": {
            "server": "local-powershell",
            "tool": "shell_command",
            "input": {"command": "Select-String -Path ./**/*.py -Pattern TODO"},
            "cwd": "C:/repo",
        },
        "reference": {
            "tool_name": "mcp.shell_command",
            "arguments": {
                "server": "local-powershell",
                "tool": "shell_command",
                "input": {"command": "Select-String -Path ./**/*.py -Pattern TODO"},
                "cwd": "C:/repo",
            },
        },
        "rationale": "exact non-empty JSON call",
    }
    record = build_sft_record(task=task, example=example, teacher_name="hauhaucs", split="train")
    summary = evaluate_distill_records([record, dict(record)], quality_counters={"attempted_tasks": 2}, run_verifiers=True)

    assert summary["exact_duplicate_count"] == 1
    assert summary["unique_text_ratio"] == 0.5
    with pytest.raises(QualityGateError, match="unique_text_ratio"):
        assert_quality_gate(summary, cfg)


def test_hauhaucs_v1_configs_use_separate_quality_output_roots() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_names = [
        "gguf_distill_code_qwen35_hauhaucs_v1.yaml",
        "gguf_distill_math_qwen35_hauhaucs_v1.yaml",
        "gguf_distill_stem_qwen35_hauhaucs_v1.yaml",
        "gguf_distill_tool_qwen35_hauhaucs_v1.yaml",
    ]

    for name in config_names:
        cfg = load_gguf_distill_config(project_root / "configs" / name)
        assert cfg.pipeline.quality_profile == "v1"
        assert cfg.pipeline.reject_fallback_outputs is True
        assert cfg.pipeline.max_generation_retries == 3
        assert cfg.pipeline.output_root.endswith("_v1")
        assert cfg.pipeline.output_root != cfg.pipeline.output_root.removesuffix("_v1")


def test_v1_teacher_instructions_encode_lane_quality_requirements() -> None:
    code_prompt = build_teacher_instruction(_lane_task("code"), quality_profile="v1")
    assert "MILSPEC-style Python" in code_prompt
    assert "mypy --strict" in code_prompt
    assert "complete parameter and return type annotations" in code_prompt

    math_prompt = build_teacher_instruction(_lane_task("math"), quality_profile="v1")
    assert "MATH/AIME/GPQA-style" in math_prompt
    assert "ELT loop refinement" in math_prompt

    stem_prompt = build_teacher_instruction(_lane_task("stem_reasoning"), quality_profile="v1")
    assert "multi-perspective STEM reasoning" in stem_prompt
    assert "patient-specific diagnosis" in stem_prompt

    tool_prompt = build_teacher_instruction(_lane_task("tool_use"), quality_profile="v1")
    assert "MCP / AI-agent harness" in tool_prompt
    assert "beginning with 'mcp.' or 'agent.'" in tool_prompt


def test_hf_dataset_mix_v1_manifest_documents_lane_sources_and_sensitive_routing() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((project_root / "configs" / "hf_dataset_mix_v1.yaml").read_text(encoding="utf-8"))

    assert manifest["policy"]["broad_knowledge_default"] == "preserve"
    assert "directly actionable crime, procurement, evasion, or weaponization instructions" in manifest["policy"]["excluded_from_sft_targets"]
    assert "AI-MO/NuminaMath-1.5" in {
        source["repo_id"] for source in manifest["lanes"]["math"]["primary_hf_sources"]
    }
    assert "glaiveai/glaive-function-calling-v2" in {
        source["repo_id"] for source in manifest["lanes"]["tool_use"]["primary_hf_sources"]
    }
    assert "allenai/real-toxicity-prompts" in {
        source["repo_id"]
        for source in manifest["lanes"]["safety_and_sensitive_understanding"]["primary_hf_sources"]
    }
