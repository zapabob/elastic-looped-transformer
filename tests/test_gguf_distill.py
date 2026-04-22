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
        domain="drug_detection",
        description="detect drug sale or usage content",
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
