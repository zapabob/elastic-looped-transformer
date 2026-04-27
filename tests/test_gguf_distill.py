from __future__ import annotations

import json
from pathlib import Path
import pytest

from elt_lm.gguf_distill import (
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
    DistillTask,
    write_status_artifacts,
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
