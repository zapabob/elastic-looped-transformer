"""GGUF-teacher distillation pipeline for moderation and detection data."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import yaml

from elt_lm.posttrain_data import render_chat_text


@dataclass
class GGUFTeacherConfig:
    model_path: str
    mmproj_path: str | None = None
    use_mmproj: bool = False
    server_bin: str = "llama-server.exe"
    name: str = "gguf-teacher"
    host: str = "127.0.0.1"
    port: int = 8091
    ctx_size: int = 8192
    n_gpu_layers: int = 99
    threads: int = 8
    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 384
    timeout_sec: int = 120


@dataclass
class DistillDomain:
    name: str
    description: str
    target_label: str = "review"
    risk_tags: list[str] = field(default_factory=list)


@dataclass
class GGUFDistillPipelineConfig:
    output_root: str
    repo_id: str = ""
    private: bool = True
    repo_type: str = "dataset"
    samples_per_domain: int = 32
    student_ckpt: str = ""
    benchmark_manifest: str = ""
    benchmark_out: str = "student_eval.csv"


@dataclass
class GGUFDistillConfig:
    teacher: GGUFTeacherConfig
    pipeline: GGUFDistillPipelineConfig
    domains: list[DistillDomain]


@dataclass
class DistillTask:
    domain: str
    description: str
    target_label: str
    risk_tags: list[str]
    variant_index: int
    mode: str


DEFAULT_DOMAINS: list[DistillDomain] = [
    DistillDomain("drug_detection", "detect drug sale, procurement, or usage promotion", "review", ["drug_reference", "procurement"]),
    DistillDomain("nsfw_detection", "detect adult sexual or explicit content", "block", ["adult_content"]),
    DistillDomain("violence_detection", "detect violent threats or graphic harm references", "review", ["violence"]),
    DistillDomain("weapons_detection", "detect weapon acquisition or threat content", "review", ["weapon_reference"]),
    DistillDomain("self_harm_detection", "detect self-harm ideation or encouragement", "block", ["self_harm"]),
    DistillDomain("fraud_detection", "detect scams, impersonation, or financial deception", "review", ["fraud"]),
    DistillDomain("malware_detection", "detect malware, phishing, or credential theft intent", "review", ["malware"]),
    DistillDomain("pii_detection", "detect exposure of personal or secret data", "review", ["pii"]),
    DistillDomain("hate_harassment_detection", "detect hateful or harassing language", "review", ["harassment"]),
    DistillDomain("medical_risk_detection", "detect unsafe medical advice or diagnosis claims", "review", ["medical_risk"]),
    DistillDomain("legal_risk_detection", "detect legal evasion or wrongdoing planning", "review", ["legal_risk"]),
    DistillDomain("benign_control", "generate safe benign controls across the same topical areas", "allow", ["benign"]),
]


def load_gguf_distill_config(path: str | Path) -> GGUFDistillConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    teacher = GGUFTeacherConfig(**(raw.get("teacher") or {}))
    pipeline = GGUFDistillPipelineConfig(**(raw.get("pipeline") or {}))
    domains_raw = raw.get("domains") or [asdict(domain) for domain in DEFAULT_DOMAINS]
    domains = [DistillDomain(**domain_raw) for domain_raw in domains_raw]
    return GGUFDistillConfig(teacher=teacher, pipeline=pipeline, domains=domains)


def build_task_specs(config: GGUFDistillConfig) -> list[DistillTask]:
    tasks: list[DistillTask] = []
    for domain in config.domains:
        for index in range(config.pipeline.samples_per_domain):
            mode = "positive" if index % 2 == 0 else "benign_control"
            tasks.append(DistillTask(
                domain=domain.name,
                description=domain.description,
                target_label=domain.target_label,
                risk_tags=list(domain.risk_tags),
                variant_index=index,
                mode=mode,
            ))
    return tasks


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if "```json" in cleaned:
        start = cleaned.find("```json")
        end = cleaned.find("```", start + 7)
        if end != -1:
            cleaned = cleaned[start + 7:end].strip()
    elif "```" in cleaned:
        start = cleaned.find("```")
        end = cleaned.find("```", start + 3)
        if end != -1:
            cleaned = cleaned[start + 3:end].strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        parsed = json.loads(cleaned[first:last + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_structured_fields(text: str) -> dict[str, Any] | None:
    fields: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key not in {"input_text", "policy_label", "severity", "risk_tags", "rationale"}:
            continue
        if key == "risk_tags":
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = [part.strip() for part in value.strip("[]").split(",") if part.strip()]
                fields[key] = parsed
            else:
                fields[key] = [part.strip() for part in value.split(",") if part.strip()]
        else:
            fields[key] = value
    return fields or None


def normalize_teacher_example(text: str, task: DistillTask, parsed: dict[str, Any] | None) -> dict[str, Any]:
    obj = dict(parsed or {})
    fallback = extract_structured_fields(text)
    if fallback:
        for key, value in fallback.items():
            obj.setdefault(key, value)

    if not str(obj.get("input_text", "")).strip():
        compact = " ".join(part.strip() for part in text.splitlines() if part.strip())
        obj["input_text"] = compact[:280] if compact else f"Synthetic example for {task.domain}"
    obj.setdefault("policy_label", "allow" if task.mode == "benign_control" else task.target_label)
    obj.setdefault("severity", "low" if str(obj.get("policy_label")) == "allow" else "medium")
    obj.setdefault("risk_tags", task.risk_tags or (["benign"] if task.mode == "benign_control" else []))
    obj.setdefault("rationale", "normalized from teacher output")
    return obj


def _normalized_response(domain: str, example: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": domain,
        "policy_label": str(example.get("policy_label", "review")).strip() or "review",
        "severity": str(example.get("severity", "medium")).strip() or "medium",
        "risk_tags": [
            str(tag).strip()
            for tag in example.get("risk_tags", [])
            if str(tag).strip()
        ],
        "rationale": str(example.get("rationale", "")).strip(),
    }


def build_detection_prompt(input_text: str) -> str:
    return (
        "Analyze the following content for moderation and safety detection.\n"
        "Return strict JSON with keys: domain, policy_label, severity, risk_tags, rationale.\n"
        "Use concise labels and do not add extra prose.\n\n"
        f"Content:\n{input_text.strip()}"
    )


def build_sft_record(
    domain: str,
    example: dict[str, Any],
    teacher_name: str,
    split: str,
) -> dict[str, Any]:
    input_text = str(example.get("input_text", "")).strip()
    response_obj = _normalized_response(domain, example)
    response = json.dumps(response_obj, ensure_ascii=False, sort_keys=True)
    prompt = build_detection_prompt(input_text)
    return {
        "bucket": "gguf_detection_distill",
        "mode": "sft",
        "source": teacher_name,
        "prompt": prompt,
        "response": response,
        "system": "",
        "text": render_chat_text(prompt, response),
        "metadata": {
            "domain": domain,
            "split": split,
            "teacher": teacher_name,
        },
    }


def evaluate_distill_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    domain_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    valid_json = 0

    for record in records:
        prompt = str(record.get("prompt", ""))
        prompt_counts[prompt] += 1
        split = str((record.get("metadata") or {}).get("split", "train"))
        split_counts[split] += 1
        try:
            response = json.loads(str(record.get("response", "{}")))
        except json.JSONDecodeError:
            continue
        if not isinstance(response, dict):
            continue
        valid_json += 1
        domain_counts[str(response.get("domain", "unknown"))] += 1
        label_counts[str(response.get("policy_label", "unknown"))] += 1

    duplicate_prompt_count = sum(count - 1 for count in prompt_counts.values() if count > 1)
    total_records = len(records)
    return {
        "total_records": total_records,
        "valid_json_records": valid_json,
        "schema_valid_rate": (valid_json / total_records) if total_records else 0.0,
        "duplicate_prompt_count": duplicate_prompt_count,
        "domain_counts": dict(domain_counts),
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
    }


def build_hf_cli_plan(output_dir: Path, repo_id: str, private: bool) -> list[list[str]]:
    return [
        ["hf", "repos", "create", repo_id, "--type", "dataset", *(["--private"] if private else []), "--exist-ok"],
        ["hf", "upload-large-folder", repo_id, str(output_dir), "--type", "dataset"],
    ]


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_llama_server(host: str, port: int, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            with urlrequest.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"llama-server did not become healthy at {url}")


def launch_llama_server(cfg: GGUFTeacherConfig, log_path: Path) -> subprocess.Popen[str]:
    server_bin = shutil.which(cfg.server_bin) or cfg.server_bin
    cmd = [
        server_bin,
        "-m", cfg.model_path,
        "--host", cfg.host,
        "--port", str(cfg.port),
        "-c", str(cfg.ctx_size),
        "-ngl", str(cfg.n_gpu_layers),
        "-t", str(cfg.threads),
    ]
    if cfg.use_mmproj and cfg.mmproj_path:
        cmd.extend(["--mmproj", cfg.mmproj_path])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_llama_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def build_teacher_instruction(task: DistillTask) -> str:
    label_hint = "allow" if task.mode == "benign_control" else task.target_label
    risk_tags = ", ".join(task.risk_tags) if task.risk_tags else "benign"
    return (
        "Create one synthetic moderation-training example for detection and classification.\n"
        "The output is for policy detection only, not for operational guidance.\n"
        "Do not include recipes, dosages, exploit steps, explicit sexual detail, or real personal data.\n"
        "Return JSON only with keys: input_text, policy_label, severity, risk_tags, rationale.\n"
        f"Domain: {task.domain}\n"
        f"Description: {task.description}\n"
        f"Preferred policy_label: {label_hint}\n"
        f"Preferred risk_tags: [{risk_tags}]\n"
        f"Variant: {task.variant_index}\n"
        f"Mode: {task.mode}\n"
        "Make the input_text short, realistic, and suitable for classifier training."
    )


def request_teacher_example(cfg: GGUFTeacherConfig, task: DistillTask) -> dict[str, Any]:
    payload = {
        "model": cfg.name,
        "messages": [{"role": "user", "content": build_teacher_instruction(task)}],
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_new_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    data = _post_json(
        f"http://{cfg.host}:{cfg.port}/v1/chat/completions",
        payload,
        timeout_sec=cfg.timeout_sec,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("teacher returned no choices")
    message = choices[0].get("message") or {}
    content = str(message.get("content", "")).strip()
    parsed = normalize_teacher_example(content, task, extract_json_object(content))
    return {
        "task": asdict(task),
        "teacher_response": content,
        "parsed_example": parsed,
    }


def _split_for_index(index: int) -> str:
    return "val" if index % 10 == 0 else "train"


def write_bundle_card(output_dir: Path, cfg: GGUFDistillConfig, summary: dict[str, Any]) -> None:
    text = (
        "---\n"
        "language:\n"
        "- en\n"
        "- ja\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "- text-generation\n"
        "- text-classification\n"
        "pretty_name: GGUF Distillation Bundle\n"
        "---\n\n"
        f"# {cfg.teacher.name} GGUF Distillation Bundle\n\n"
        "This dataset bundle is generated from a local GGUF teacher through llama.cpp\n"
        "for moderation and policy-detection training. It targets detection and\n"
        "classification use cases, including drug, NSFW, fraud, malware, and related\n"
        "safety domains, without requesting operational harmful instructions.\n\n"
        f"- Teacher model: `{cfg.teacher.model_path}`\n"
        f"- MMProj: `{cfg.teacher.mmproj_path or ''}`\n"
        f"- Total records: `{summary['total_records']}`\n"
        f"- Schema-valid rate: `{summary['schema_valid_rate']:.3f}`\n"
        f"- Repo target: `{cfg.pipeline.repo_id}`\n"
    )
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run_subprocess(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with rc={proc.returncode}: {' '.join(cmd)}")


def run_hf_cli_plan(plan: list[list[str]]) -> None:
    for cmd in plan:
        run_subprocess(cmd)


def maybe_run_student_eval(cfg: GGUFDistillConfig, output_dir: Path) -> Path | None:
    ckpt = Path(cfg.pipeline.student_ckpt) if cfg.pipeline.student_ckpt else None
    manifest = Path(cfg.pipeline.benchmark_manifest) if cfg.pipeline.benchmark_manifest else None
    if ckpt is None or manifest is None:
        return None
    if not ckpt.exists() or not manifest.exists():
        return None
    out_path = output_dir / cfg.pipeline.benchmark_out
    run_subprocess([
        "uv", "run", "elt-anytime",
        "--ckpt", str(ckpt),
        "--benchmark-manifest", str(manifest),
        "--out-csv", str(out_path),
    ])
    return out_path


def run_pipeline(
    cfg: GGUFDistillConfig,
    *,
    output_dir: Path | None = None,
    max_tasks: int = 0,
    skip_upload: bool = False,
    skip_student_eval: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    out_dir = output_dir or Path(cfg.pipeline.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(cfg.teacher.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if cfg.teacher.use_mmproj and cfg.teacher.mmproj_path and not Path(cfg.teacher.mmproj_path).exists():
        raise FileNotFoundError(cfg.teacher.mmproj_path)
    if not shutil.which(cfg.teacher.server_bin) and not Path(cfg.teacher.server_bin).exists():
        raise FileNotFoundError(cfg.teacher.server_bin)

    tasks = build_task_specs(cfg)
    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    plan = {
        "teacher": asdict(cfg.teacher),
        "pipeline": asdict(cfg.pipeline),
        "task_count": len(tasks),
        "output_dir": str(out_dir),
    }
    (out_dir / "pipeline_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if dry_run:
        return plan

    log_path = out_dir / "llama_server.log"
    proc = launch_llama_server(cfg.teacher, log_path)
    records: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    try:
        wait_for_llama_server(cfg.teacher.host, cfg.teacher.port, cfg.teacher.timeout_sec)
        for index, task in enumerate(tasks):
            raw = request_teacher_example(cfg.teacher, task)
            split = _split_for_index(index)
            record = build_sft_record(
                domain=task.domain,
                example=raw["parsed_example"],
                teacher_name=cfg.teacher.name,
                split=split,
            )
            raw["normalized_record"] = record
            raw_rows.append(raw)
            records.append(record)
    finally:
        stop_llama_server(proc)

    raw_path = out_dir / "raw_teacher_examples.jsonl"
    train_path = out_dir / "distill_train.jsonl"
    val_path = out_dir / "distill_val.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(train_path, "w", encoding="utf-8") as train_f, open(val_path, "w", encoding="utf-8") as val_f:
        for record in records:
            target = val_f if record["metadata"]["split"] == "val" else train_f
            target.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = evaluate_distill_records(records)
    summary["teacher_name"] = cfg.teacher.name
    summary["raw_teacher_examples"] = str(raw_path)
    summary["train_path"] = str(train_path)
    summary["val_path"] = str(val_path)
    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_bundle_card(out_dir, cfg, summary)
    maybe_run_student_eval(cfg, out_dir) if not skip_student_eval else None

    if cfg.pipeline.repo_id and not skip_upload:
        plan = build_hf_cli_plan(out_dir, cfg.pipeline.repo_id, cfg.pipeline.private)
        run_hf_cli_plan(plan)

    return summary


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config for GGUF distillation")
    p.add_argument("--output-dir", default="", help="override output directory")
    p.add_argument("--max-tasks", type=int, default=0, help="limit generated tasks for smoke runs")
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument("--skip-student-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_gguf_distill_config(args.config)
    summary = run_pipeline(
        cfg,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_tasks=args.max_tasks,
        skip_upload=args.skip_upload,
        skip_student_eval=args.skip_student_eval,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
