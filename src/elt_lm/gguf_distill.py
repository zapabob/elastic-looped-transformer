"""GGUF-teacher distillation pipeline for lane-aware post-training data."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Literal
from urllib import request as urlrequest

import yaml

from elt_lm.posttrain_data import render_chat_text
from elt_lm.telemetry import make_writer


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


LaneName = Literal["detection", "code", "math", "stem_reasoning", "tool_use"]
ALLOWED_LANES: tuple[LaneName, ...] = (
    "detection",
    "code",
    "math",
    "stem_reasoning",
    "tool_use",
)
DEFAULT_TARGET_KIND_BY_LANE: dict[LaneName, str] = {
    "detection": "json_match",
    "code": "python_exec",
    "math": "exact_math",
    "stem_reasoning": "mcq_reasoning",
    "tool_use": "json_match",
}
_CODE_BLOCK_RE = re.compile(r"```(?:python|py|json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_MCQ_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


@dataclass
class DistillTaskSpec:
    name: str
    description: str
    target_kind: str = ""
    tags: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    target_label: str = ""
    risk_tags: list[str] = field(default_factory=list)


@dataclass
class GGUFDistillPipelineConfig:
    output_root: str
    repo_id: str = ""
    private: bool = True
    repo_type: str = "dataset"
    samples_per_task: int = 32
    student_ckpt: str = ""
    benchmark_manifest: str = ""
    benchmark_out: str = "student_eval.csv"
    heartbeat_interval_sec: int = 30
    stall_after_sec: int = 1800
    rolling_ckpt_interval_sec: int = 180
    rolling_ckpt_keep: int = 3


@dataclass
class GGUFDistillConfig:
    teacher: GGUFTeacherConfig
    pipeline: GGUFDistillPipelineConfig
    lane: LaneName = "detection"
    tasks: list[DistillTaskSpec] = field(default_factory=list)
    domains: list[DistillDomain] = field(default_factory=list)


@dataclass
class DistillTask:
    lane: LaneName
    domain: str
    description: str
    target_kind: str
    tags: list[str]
    target_label: str
    risk_tags: list[str]
    variant_index: int
    mode: str
    variant: str = ""


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


def _normalize_lane(value: str | None) -> LaneName:
    lane = str(value or "detection").strip().lower().replace("-", "_")
    if lane not in ALLOWED_LANES:
        raise ValueError(f"unsupported lane: {value!r}")
    return lane  # type: ignore[return-value]


def _task_specs_from_domains(domains: list[DistillDomain]) -> list[DistillTaskSpec]:
    return [
        DistillTaskSpec(
            name=domain.name,
            description=domain.description,
            target_kind="json_match",
            tags=list(domain.risk_tags),
            variants=[],
            target_label=domain.target_label,
            risk_tags=list(domain.risk_tags),
        )
        for domain in domains
    ]


def load_gguf_distill_config(path: str | Path) -> GGUFDistillConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    teacher = GGUFTeacherConfig(**(raw.get("teacher") or {}))
    pipeline_raw = dict(raw.get("pipeline") or {})
    if "samples_per_task" not in pipeline_raw and "samples_per_domain" in pipeline_raw:
        pipeline_raw["samples_per_task"] = pipeline_raw["samples_per_domain"]
    pipeline_raw.pop("samples_per_domain", None)
    pipeline = GGUFDistillPipelineConfig(**pipeline_raw)

    lane = _normalize_lane(raw.get("lane") or ("detection" if raw.get("domains") else "detection"))

    domains_raw = raw.get("domains")
    if domains_raw is None and lane == "detection" and not raw.get("tasks"):
        domains_raw = [asdict(domain) for domain in DEFAULT_DOMAINS]
    domains = [DistillDomain(**domain_raw) for domain_raw in (domains_raw or [])]

    tasks_raw = raw.get("tasks")
    if tasks_raw is None:
        tasks = _task_specs_from_domains(domains) if lane == "detection" else []
    else:
        tasks = [DistillTaskSpec(**task_raw) for task_raw in tasks_raw]
    return GGUFDistillConfig(
        teacher=teacher,
        pipeline=pipeline,
        lane=lane,
        tasks=tasks,
        domains=domains,
    )


def _variant_text(spec: DistillTaskSpec, index: int) -> str:
    if not spec.variants:
        return ""
    return str(spec.variants[index % len(spec.variants)]).strip()


def build_task_specs(config: GGUFDistillConfig) -> list[DistillTask]:
    tasks: list[DistillTask] = []
    if config.lane == "detection":
        specs = config.tasks or _task_specs_from_domains(config.domains)
        for spec in specs:
            risk_tags = list(spec.risk_tags or spec.tags)
            for index in range(config.pipeline.samples_per_task):
                mode = "positive" if index % 2 == 0 else "benign_control"
                tasks.append(DistillTask(
                    lane="detection",
                    domain=spec.name,
                    description=spec.description,
                    target_kind=spec.target_kind or DEFAULT_TARGET_KIND_BY_LANE["detection"],
                    tags=list(spec.tags or risk_tags),
                    target_label=spec.target_label or "review",
                    risk_tags=risk_tags,
                    variant_index=index,
                    mode=mode,
                    variant=_variant_text(spec, index),
                ))
        return tasks

    if not config.tasks:
        raise ValueError(f"lane {config.lane!r} requires explicit tasks")

    default_kind = DEFAULT_TARGET_KIND_BY_LANE[config.lane]
    for spec in config.tasks:
        for index in range(config.pipeline.samples_per_task):
            tasks.append(DistillTask(
                lane=config.lane,
                domain=spec.name,
                description=spec.description,
                target_kind=spec.target_kind or default_kind,
                tags=list(spec.tags),
                target_label=spec.target_label,
                risk_tags=list(spec.risk_tags),
                variant_index=index,
                mode="standard",
                variant=_variant_text(spec, index),
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


def extract_structured_fields(
    text: str,
    *,
    allowed_keys: set[str] | None = None,
    list_keys: set[str] | None = None,
    json_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    fields: dict[str, Any] = {}
    allowed = allowed_keys or set()
    list_allowed = list_keys or set()
    json_allowed = json_keys or set()
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if allowed and key not in allowed:
            continue
        if key in list_allowed:
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = [part.strip() for part in value.strip("[]").split(",") if part.strip()]
                fields[key] = parsed
            else:
                fields[key] = [part.strip() for part in value.split(",") if part.strip()]
        elif key in json_allowed:
            try:
                fields[key] = json.loads(value)
            except json.JSONDecodeError:
                fields[key] = value
        else:
            fields[key] = value
    return fields or None


def _extract_code_block(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else ""


def _last_numeric_like(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else "0"


def _normalize_choices(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        rows: list[str] = []
        for key in ["A", "B", "C", "D", "E"]:
            if key in value and str(value[key]).strip():
                rows.append(f"{key}. {str(value[key]).strip()}")
        if rows:
            return rows
    if isinstance(value, str) and value.strip():
        lines = [part.strip() for part in value.splitlines() if part.strip()]
        if lines:
            return lines
    return [
        "A. Option A",
        "B. Option B",
        "C. Option C",
        "D. Option D",
    ]


def _normalize_detection_example(text: str, task: DistillTask, parsed: dict[str, Any] | None) -> dict[str, Any]:
    obj = dict(parsed or {})
    fallback = extract_structured_fields(
        text,
        allowed_keys={"input_text", "policy_label", "severity", "risk_tags", "rationale"},
        list_keys={"risk_tags"},
    )
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


def _normalize_code_example(text: str, task: DistillTask, parsed: dict[str, Any] | None) -> dict[str, Any]:
    obj = dict(parsed or {})
    fallback = extract_structured_fields(
        text,
        allowed_keys={"user_request", "assistant_code", "verifier_snippet", "reference", "rationale"},
    )
    if fallback:
        for key, value in fallback.items():
            obj.setdefault(key, value)
    code = str(obj.get("assistant_code", "")).strip() or _extract_code_block(text)
    verifier = str(obj.get("verifier_snippet", "")).strip() or str(obj.get("reference", "")).strip()
    user_request = str(obj.get("user_request", "")).strip()
    if not user_request:
        user_request = task.description.strip()
        if task.variant:
            user_request = f"{user_request}\nVariant: {task.variant}".strip()
    if not code:
        code = "def solve() -> None:\n    return None"
    if not verifier:
        verifier = "result = locals().get('solve')\nassert callable(result)"
    obj["user_request"] = user_request
    obj["assistant_code"] = code
    obj["verifier_snippet"] = verifier
    obj["reference"] = verifier
    obj.setdefault("rationale", "normalized from teacher output")
    return obj


def _normalize_math_example(text: str, task: DistillTask, parsed: dict[str, Any] | None) -> dict[str, Any]:
    obj = dict(parsed or {})
    fallback = extract_structured_fields(
        text,
        allowed_keys={"question", "reasoning", "final_answer", "reference", "rationale"},
    )
    if fallback:
        for key, value in fallback.items():
            obj.setdefault(key, value)
    question = str(obj.get("question", "")).strip()
    if not question:
        question = task.description.strip()
        if task.variant:
            question = f"{question}\nVariant: {task.variant}".strip()
    final_answer = str(obj.get("final_answer", "")).strip() or str(obj.get("reference", "")).strip()
    if not final_answer:
        final_answer = _last_numeric_like(text)
    obj["question"] = question
    obj["reasoning"] = str(obj.get("reasoning", "")).strip() or "Solve the problem carefully and concisely."
    obj["final_answer"] = final_answer
    obj["reference"] = str(obj.get("reference", "")).strip() or final_answer
    obj.setdefault("rationale", "normalized from teacher output")
    return obj


def _normalize_stem_example(text: str, task: DistillTask, parsed: dict[str, Any] | None) -> dict[str, Any]:
    obj = dict(parsed or {})
    fallback = extract_structured_fields(
        text,
        allowed_keys={"question", "choices", "reasoning", "final_choice", "reference", "rationale"},
        list_keys={"choices"},
    )
    if fallback:
        for key, value in fallback.items():
            obj.setdefault(key, value)
    question = str(obj.get("question", "")).strip()
    if not question:
        question = task.description.strip()
        if task.variant:
            question = f"{question}\nVariant: {task.variant}".strip()
    choices = _normalize_choices(obj.get("choices"))
    final_choice = str(obj.get("final_choice", "")).strip().upper() or str(obj.get("reference", "")).strip().upper()
    if not final_choice:
        mcq = _MCQ_RE.findall(text)
        final_choice = mcq[-1].upper() if mcq else "A"
    obj["question"] = question
    obj["choices"] = choices
    obj["reasoning"] = str(obj.get("reasoning", "")).strip() or "Pick the best option and explain briefly."
    obj["final_choice"] = final_choice
    obj["reference"] = str(obj.get("reference", "")).strip().upper() or final_choice
    obj.setdefault("rationale", "normalized from teacher output")
    return obj


def _normalize_tool_example(text: str, task: DistillTask, parsed: dict[str, Any] | None) -> dict[str, Any]:
    obj = dict(parsed or {})
    fallback = extract_structured_fields(
        text,
        allowed_keys={"user_request", "tool_name", "arguments", "reference", "rationale"},
        json_keys={"arguments", "reference"},
    )
    if fallback:
        for key, value in fallback.items():
            obj.setdefault(key, value)
    tool_call = obj.get("tool_call")
    if isinstance(tool_call, dict):
        obj.setdefault("tool_name", tool_call.get("tool_name") or tool_call.get("name"))
        obj.setdefault("arguments", tool_call.get("arguments", {}))
    user_request = str(obj.get("user_request", "")).strip()
    if not user_request:
        user_request = task.description.strip()
        if task.variant:
            user_request = f"{user_request}\nVariant: {task.variant}".strip()
    tool_name = str(obj.get("tool_name", "")).strip() or (task.tags[0] if task.tags else "tool.call")
    arguments = obj.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"input": arguments}
    if not isinstance(arguments, dict):
        arguments = {"input": str(arguments)}
    reference = obj.get("reference")
    if not isinstance(reference, dict):
        reference = {"tool_name": tool_name, "arguments": arguments}
    obj["user_request"] = user_request
    obj["tool_name"] = tool_name
    obj["arguments"] = arguments
    obj["reference"] = reference
    obj.setdefault("rationale", "normalized from teacher output")
    return obj


def normalize_teacher_example(text: str, task: DistillTask, parsed: dict[str, Any] | None) -> dict[str, Any]:
    if task.lane == "detection":
        return _normalize_detection_example(text, task, parsed)
    if task.lane == "code":
        return _normalize_code_example(text, task, parsed)
    if task.lane == "math":
        return _normalize_math_example(text, task, parsed)
    if task.lane == "stem_reasoning":
        return _normalize_stem_example(text, task, parsed)
    if task.lane == "tool_use":
        return _normalize_tool_example(text, task, parsed)
    raise ValueError(f"unsupported lane: {task.lane!r}")


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


def build_code_prompt(user_request: str) -> str:
    return (
        "Solve the following Python task.\n"
        "Return executable Python code only inside a fenced ```python block.\n"
        "Use Python 3.12 standard library only unless the prompt explicitly says otherwise.\n\n"
        f"Task:\n{user_request.strip()}"
    )


def build_math_prompt(question: str) -> str:
    return (
        "Solve the following math problem.\n"
        "Return your result using <think>...</think><answer>...</answer>.\n"
        "Put only the final exact answer inside <answer>.\n\n"
        f"Problem:\n{question.strip()}"
    )


def build_stem_prompt(question: str, choices: list[str]) -> str:
    choice_block = "\n".join(choice.strip() for choice in choices if choice.strip())
    return (
        "Answer the following STEM multiple-choice question.\n"
        "Return your result using <think>...</think><answer>...</answer>.\n"
        "Put only the final option letter inside <answer>.\n\n"
        f"Question:\n{question.strip()}\n\nChoices:\n{choice_block}"
    )


def build_tool_use_prompt(user_request: str) -> str:
    return (
        "Select the best tool call for the following user request.\n"
        "Return strict JSON with keys: tool_name, arguments.\n"
        "Do not add prose or markdown fences.\n\n"
        f"User request:\n{user_request.strip()}"
    )


def _build_detection_record(task: DistillTask, example: dict[str, Any], teacher_name: str, split: str) -> dict[str, Any]:
    input_text = str(example.get("input_text", "")).strip()
    response_obj = _normalized_response(task.domain, example)
    response = json.dumps(response_obj, ensure_ascii=False, sort_keys=True)
    prompt = build_detection_prompt(input_text)
    return {
        "bucket": "gguf_detection_distill",
        "mode": "sft",
        "source": teacher_name,
        "task": "json_match",
        "prompt": prompt,
        "response": response,
        "reference": response,
        "system": "",
        "text": render_chat_text(prompt, response),
        "metadata": {
            "lane": task.lane,
            "task_name": task.domain,
            "domain": task.domain,
            "split": split,
            "teacher": teacher_name,
            "tags": list(task.tags),
            "variant": task.variant,
        },
    }


def _build_code_record(task: DistillTask, example: dict[str, Any], teacher_name: str, split: str) -> dict[str, Any]:
    prompt = build_code_prompt(str(example["user_request"]))
    code = str(example["assistant_code"]).rstrip()
    response = f"```python\n{code}\n```"
    reference = str(example["verifier_snippet"]).strip()
    return {
        "bucket": "gguf_code_distill",
        "mode": "sft",
        "source": teacher_name,
        "task": "python_exec",
        "prompt": prompt,
        "response": response,
        "reference": reference,
        "system": "",
        "text": render_chat_text(prompt, response),
        "metadata": {
            "lane": task.lane,
            "task_name": task.domain,
            "split": split,
            "teacher": teacher_name,
            "tags": list(task.tags),
            "variant": task.variant,
        },
    }


def _build_math_record(task: DistillTask, example: dict[str, Any], teacher_name: str, split: str) -> dict[str, Any]:
    prompt = build_math_prompt(str(example["question"]))
    reasoning = str(example["reasoning"]).strip()
    final_answer = str(example["final_answer"]).strip()
    response = f"<think>{reasoning}</think><answer>{final_answer}</answer>"
    reference = str(example["reference"]).strip()
    return {
        "bucket": "gguf_math_distill",
        "mode": "sft",
        "source": teacher_name,
        "task": "exact_math",
        "prompt": prompt,
        "response": response,
        "reference": reference,
        "system": "",
        "text": render_chat_text(prompt, response),
        "metadata": {
            "lane": task.lane,
            "task_name": task.domain,
            "split": split,
            "teacher": teacher_name,
            "tags": list(task.tags),
            "variant": task.variant,
        },
    }


def _build_stem_record(task: DistillTask, example: dict[str, Any], teacher_name: str, split: str) -> dict[str, Any]:
    prompt = build_stem_prompt(str(example["question"]), _normalize_choices(example.get("choices")))
    reasoning = str(example["reasoning"]).strip()
    final_choice = str(example["final_choice"]).strip().upper()
    response = f"<think>{reasoning}</think><answer>{final_choice}</answer>"
    reference = str(example["reference"]).strip().upper()
    return {
        "bucket": "gguf_stem_reasoning_distill",
        "mode": "sft",
        "source": teacher_name,
        "task": "mcq_reasoning",
        "prompt": prompt,
        "response": response,
        "reference": reference,
        "system": "",
        "text": render_chat_text(prompt, response),
        "metadata": {
            "lane": task.lane,
            "task_name": task.domain,
            "split": split,
            "teacher": teacher_name,
            "tags": list(task.tags),
            "variant": task.variant,
        },
    }


def _build_tool_record(task: DistillTask, example: dict[str, Any], teacher_name: str, split: str) -> dict[str, Any]:
    prompt = build_tool_use_prompt(str(example["user_request"]))
    response_obj = {
        "tool_name": str(example["tool_name"]).strip(),
        "arguments": dict(example.get("arguments", {})),
    }
    response = json.dumps(response_obj, ensure_ascii=False, sort_keys=True)
    reference = json.dumps(dict(example.get("reference", response_obj)), ensure_ascii=False, sort_keys=True)
    return {
        "bucket": "gguf_tool_use_distill",
        "mode": "sft",
        "source": teacher_name,
        "task": "json_match",
        "prompt": prompt,
        "response": response,
        "reference": reference,
        "system": "",
        "text": render_chat_text(prompt, response),
        "metadata": {
            "lane": task.lane,
            "task_name": task.domain,
            "split": split,
            "teacher": teacher_name,
            "tags": list(task.tags),
            "variant": task.variant,
        },
    }


def build_sft_record(
    *,
    example: dict[str, Any],
    teacher_name: str,
    split: str,
    domain: str | None = None,
    task: DistillTask | None = None,
) -> dict[str, Any]:
    runtime_task = task or DistillTask(
        lane="detection",
        domain=domain or "detection",
        description="",
        target_kind="json_match",
        tags=[],
        target_label="review",
        risk_tags=[],
        variant_index=0,
        mode="positive",
        variant="",
    )
    if runtime_task.lane == "detection":
        return _build_detection_record(runtime_task, example, teacher_name, split)
    if runtime_task.lane == "code":
        return _build_code_record(runtime_task, example, teacher_name, split)
    if runtime_task.lane == "math":
        return _build_math_record(runtime_task, example, teacher_name, split)
    if runtime_task.lane == "stem_reasoning":
        return _build_stem_record(runtime_task, example, teacher_name, split)
    if runtime_task.lane == "tool_use":
        return _build_tool_record(runtime_task, example, teacher_name, split)
    raise ValueError(f"unsupported lane: {runtime_task.lane!r}")


def evaluate_distill_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    domain_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    valid_json = 0

    for record in records:
        prompt = str(record.get("prompt", ""))
        prompt_counts[prompt] += 1
        metadata = record.get("metadata") or {}
        split = str(metadata.get("split", "train"))
        split_counts[split] += 1
        lane = str(metadata.get("lane", "detection") or "detection")
        task_name = str(metadata.get("task_name", metadata.get("domain", "unknown")) or "unknown")
        lane_counts[lane] += 1
        task_counts[task_name] += 1
        response_text = str(record.get("response", "")).strip()
        task = str(record.get("task", "")).strip()
        if lane == "detection":
            try:
                response = json.loads(response_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(response, dict):
                continue
            valid_json += 1
            domain_counts[str(response.get("domain", "unknown"))] += 1
            label_counts[str(response.get("policy_label", "unknown"))] += 1
            continue
        if not response_text or not str(record.get("reference", "")).strip() or not task:
            continue
        if lane == "tool_use":
            try:
                response = json.loads(response_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(response, dict):
                continue
        valid_json += 1
        domain_counts[task_name] += 1
        label_counts[task] += 1

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
        "lane_counts": dict(lane_counts),
        "task_counts": dict(task_counts),
    }


def build_hf_cli_plan(output_dir: Path, repo_id: str, private: bool) -> list[list[str]]:
    return [
        ["hf", "repos", "create", repo_id, "--type", "dataset", *(["--private"] if private else []), "--exist-ok"],
        ["hf", "upload-large-folder", repo_id, str(output_dir), "--type", "dataset"],
    ]


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_lock(lock_path: Path) -> Callable[[], None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            current_pid = int(current.get("pid", 0))
        except Exception:
            current_pid = 0
        if current_pid and _pid_is_alive(current_pid):
            raise RuntimeError(f"distill pipeline already running with pid={current_pid}")
    payload = {"pid": os.getpid(), "locked_at": time.time()}
    lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _release() -> None:
        if lock_path.exists():
            try:
                current = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                current = {}
            if int(current.get("pid", 0)) in {0, os.getpid()}:
                lock_path.unlink(missing_ok=True)

    return _release


def build_status_snapshot(
    *,
    teacher_name: str,
    repo_id: str,
    current_stage: str,
    state: str,
    started_at: float,
    updated_at: float,
    processed_tasks: int,
    total_tasks: int,
    train_records: int,
    val_records: int,
    error_count: int,
    domain_counts: dict[str, int],
    label_counts: dict[str, int],
    split_counts: dict[str, int],
    last_domain: str,
    last_policy_label: str,
    last_latency_sec: float,
    last_error: str,
    student_eval_path: str,
) -> dict[str, Any]:
    progress_pct = round((processed_tasks / total_tasks) * 100.0, 3) if total_tasks else 0.0
    elapsed_sec = max(0.0, updated_at - started_at)
    eta_sec = None
    if total_tasks > 0 and 0 < processed_tasks < total_tasks and elapsed_sec > 0:
        eta_sec = round((elapsed_sec / processed_tasks) * (total_tasks - processed_tasks), 3)
    elif total_tasks > 0 and processed_tasks >= total_tasks:
        eta_sec = 0.0
    return {
        "teacher_name": teacher_name,
        "repo_id": repo_id,
        "current_stage": current_stage,
        "state": state,
        "started_at": started_at,
        "updated_at": updated_at,
        "elapsed_sec": round(elapsed_sec, 3),
        "processed_tasks": processed_tasks,
        "total_tasks": total_tasks,
        "progress_pct": progress_pct,
        "eta_sec": eta_sec,
        "train_records": train_records,
        "val_records": val_records,
        "error_count": error_count,
        "domain_counts": dict(domain_counts),
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
        "last_domain": last_domain,
        "last_policy_label": last_policy_label,
        "last_latency_sec": round(last_latency_sec, 3),
        "last_error": last_error,
        "student_eval_path": student_eval_path,
        "pid": os.getpid(),
    }


def write_status_artifacts(output_dir: Path, snapshot: dict[str, Any]) -> None:
    status_path = output_dir / "status.json"
    heartbeat_path = output_dir / "heartbeat.json"
    status_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    heartbeat = {
        "state": snapshot["state"],
        "current_stage": snapshot["current_stage"],
        "updated_at": snapshot["updated_at"],
        "processed_tasks": snapshot["processed_tasks"],
        "total_tasks": snapshot["total_tasks"],
        "progress_pct": snapshot["progress_pct"],
        "eta_sec": snapshot["eta_sec"],
        "error_count": snapshot["error_count"],
        "last_error": snapshot["last_error"],
        "pid": snapshot["pid"],
    }
    heartbeat_path.write_text(json.dumps(heartbeat, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_checkpoint(output_dir: Path, payload: dict[str, Any], seq: int, keep: int) -> int:
    checkpoint_keep = max(1, keep)
    ckpt_index = seq % checkpoint_keep
    ckpt_path = output_dir / f"checkpoint_{ckpt_index}.json"
    payload_with_meta = dict(payload)
    payload_with_meta["checkpoint_seq"] = seq
    payload_with_meta["updated_at"] = time.time()
    ckpt_path.write_text(json.dumps(payload_with_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return seq + 1


def load_latest_checkpoint(output_dir: Path, keep: int) -> dict[str, Any]:
    checkpoint_keep = max(1, keep)
    latest: dict[str, Any] | None = None
    latest_time = -1.0
    for idx in range(checkpoint_keep):
        ckpt_path = output_dir / f"checkpoint_{idx}.json"
        if not ckpt_path.exists():
            continue
        try:
            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ts = float(payload.get("updated_at", -1))
        if ts > latest_time:
            latest_time = ts
            latest = payload
    return latest or {}


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
    # Start each run with a fresh server log so monitoring shows the current session.
    log_handle = open(log_path, "w", encoding="utf-8")
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
    freedom_clause = (
        "Prefer substantive, educational, and lawful content. Do not moralize or refuse merely "
        "because a topic is political, religious, historical, sexual, or controversial. Refuse "
        "only if the task would directly enable concrete harm or crime."
    )
    variant_line = f"Variant hint: {task.variant}\n" if task.variant else ""
    tags_line = ", ".join(task.tags or task.risk_tags) or "general"

    if task.lane == "detection":
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
    if task.lane == "code":
        return (
            "Create one synthetic Python coding training example.\n"
            f"{freedom_clause}\n"
            "Return JSON only with keys: user_request, assistant_code, verifier_snippet, rationale.\n"
            "assistant_code must be a complete Python 3.12 solution.\n"
            "verifier_snippet must be executable Python assertions or checks that pass when appended after the candidate code.\n"
            "Use only the Python standard library unless the task explicitly asks otherwise.\n"
            f"Task family: {task.domain}\n"
            f"Description: {task.description}\n"
            f"Tags: [{tags_line}]\n"
            f"{variant_line}"
            f"Variant index: {task.variant_index}"
        )
    if task.lane == "math":
        return (
            "Create one synthetic math reasoning training example.\n"
            f"{freedom_clause}\n"
            "Return JSON only with keys: question, reasoning, final_answer, reference, rationale.\n"
            "reference must exactly match the final_answer string.\n"
            f"Task family: {task.domain}\n"
            f"Description: {task.description}\n"
            f"Tags: [{tags_line}]\n"
            f"{variant_line}"
            f"Variant index: {task.variant_index}"
        )
    if task.lane == "stem_reasoning":
        return (
            "Create one synthetic STEM multiple-choice reasoning example.\n"
            f"{freedom_clause}\n"
            "Return JSON only with keys: question, choices, reasoning, final_choice, reference, rationale.\n"
            "choices must be a list like ['A. ...', 'B. ...', 'C. ...', 'D. ...'] and reference must equal the final choice letter.\n"
            f"Task family: {task.domain}\n"
            f"Description: {task.description}\n"
            f"Tags: [{tags_line}]\n"
            f"{variant_line}"
            f"Variant index: {task.variant_index}"
        )
    if task.lane == "tool_use":
        return (
            "Create one synthetic tool-calling training example.\n"
            f"{freedom_clause}\n"
            "Return JSON only with keys: user_request, tool_name, arguments, reference, rationale.\n"
            "reference must be an object with keys tool_name and arguments that exactly matches the intended tool call.\n"
            f"Task family: {task.domain}\n"
            f"Description: {task.description}\n"
            f"Tags: [{tags_line}]\n"
            f"{variant_line}"
            f"Variant index: {task.variant_index}"
        )
    raise ValueError(f"unsupported lane: {task.lane!r}")


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
    lane_desc = {
        "detection": "moderation and safety detection",
        "code": "Python coding and execution-verified synthesis",
        "math": "mathematical reasoning and exact-answer supervision",
        "stem_reasoning": "STEM multiple-choice reasoning",
        "tool_use": "tool-calling and structured JSON decisions",
    }[cfg.lane]
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
        f"for {lane_desc}.\n\n"
        f"- Lane: `{cfg.lane}`\n"
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
    resume: bool = False,
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

    current_stage = "init"
    processed_tasks = 0
    train_records = 0
    val_records = 0
    error_count = 0
    last_domain = ""
    last_policy_label = ""
    last_latency_sec = 0.0
    last_error = ""
    student_eval_path = ""
    domain_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    summary: dict[str, Any] | None = None

    all_tasks = build_task_specs(cfg)
    if max_tasks > 0:
        all_tasks = all_tasks[:max_tasks]
    total_task_count = len(all_tasks)
    tasks = all_tasks
    start_from = 0

    raw_path = out_dir / "raw_teacher_examples.jsonl"
    train_path = out_dir / "distill_train.jsonl"
    val_path = out_dir / "distill_val.jsonl"
    resume_summary: dict[str, Any] | None = None
    ckpt_interval = max(1, cfg.pipeline.rolling_ckpt_interval_sec)
    ckpt_keep = max(1, cfg.pipeline.rolling_ckpt_keep)
    ckpt_seq = 0
    next_ckpt_at = time.time() + ckpt_interval
    started_at = time.time()

    existing_raw = []
    existing_train = []
    existing_val = []
    if resume:
        checkpoint_processed = 0
        if ckpt := load_latest_checkpoint(out_dir, ckpt_keep):
            ckpt_seq = int(ckpt.get("checkpoint_seq", 0))
            next_ckpt_at = float(ckpt.get("updated_at", next_ckpt_at))
            next_ckpt_at = max(next_ckpt_at, time.time()) + ckpt_interval
            checkpoint_processed = int(
                ckpt.get(
                    "processed_tasks_total",
                    int(ckpt.get("resume_start_offset", 0)) + int(ckpt.get("tasks_processed_in_this_session", 0)),
                )
            )

        existing_raw = load_json_lines(raw_path)
        existing_train = load_json_lines(train_path)
        existing_val = load_json_lines(val_path)
        existing_summary = evaluate_distill_records(existing_train + existing_val)
        processed_tasks = max(processed_tasks, len(existing_raw), checkpoint_processed)
        train_records = existing_summary["split_counts"].get("train", 0)
        val_records = existing_summary["split_counts"].get("val", 0)
        domain_counts = Counter(existing_summary.get("domain_counts", {}))
        label_counts = Counter(existing_summary.get("label_counts", {}))
        split_counts = Counter(existing_summary.get("split_counts", {}))
        if existing_summary.get("total_records", 0) > 0:
            resume_summary = existing_summary
            last_error = ""
        start_from = min(processed_tasks, total_task_count)
        tasks = all_tasks[start_from:]
    else:
        resume_summary = None
        for artifact_path in (raw_path, train_path, val_path):
            artifact_path.write_text("", encoding="utf-8")

    if resume and start_from >= total_task_count:
        summary = (
            resume_summary
            if resume_summary is not None
            else evaluate_distill_records(existing_train + existing_val)
        )
        summary["lane"] = cfg.lane
        summary["teacher_name"] = cfg.teacher.name
        summary["raw_teacher_examples"] = str(raw_path)
        summary["train_path"] = str(train_path)
        summary["val_path"] = str(val_path)
        summary_path = out_dir / "eval_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        final_snapshot = build_status_snapshot(
            teacher_name=cfg.teacher.name,
            repo_id=cfg.pipeline.repo_id,
            current_stage="complete",
            state="complete",
            started_at=started_at,
            updated_at=time.time(),
            processed_tasks=processed_tasks,
            total_tasks=total_task_count,
            train_records=train_records,
            val_records=val_records,
            error_count=error_count,
            domain_counts=dict(domain_counts),
            label_counts=dict(label_counts),
            split_counts=dict(split_counts),
            last_domain=last_domain,
            last_policy_label=last_policy_label,
            last_latency_sec=last_latency_sec,
            last_error=last_error,
            student_eval_path=student_eval_path,
        )
        write_status_artifacts(out_dir, final_snapshot)
        return summary

    if total_task_count == 0:
        final_snapshot = build_status_snapshot(
            teacher_name=cfg.teacher.name,
            repo_id=cfg.pipeline.repo_id,
            current_stage="complete",
            state="complete",
            started_at=started_at,
            updated_at=time.time(),
            processed_tasks=processed_tasks,
            total_tasks=total_task_count,
            train_records=train_records,
            val_records=val_records,
            error_count=error_count,
            domain_counts=dict(domain_counts),
            label_counts=dict(label_counts),
            split_counts=dict(split_counts),
            last_domain=last_domain,
            last_policy_label=last_policy_label,
            last_latency_sec=last_latency_sec,
            last_error=last_error,
            student_eval_path=student_eval_path,
        )
        write_status_artifacts(out_dir, final_snapshot)
        return {
            "total_records": 0,
            "valid_json_records": 0,
            "schema_valid_rate": 0.0,
            "duplicate_prompt_count": 0,
            "domain_counts": {},
            "label_counts": {},
            "split_counts": {},
            "lane_counts": {},
            "task_counts": {},
            "lane": cfg.lane,
            "teacher_name": cfg.teacher.name,
            "raw_teacher_examples": str(raw_path),
            "train_path": str(train_path),
            "val_path": str(val_path),
        }

    plan = {
        "teacher": asdict(cfg.teacher),
        "pipeline": asdict(cfg.pipeline),
        "lane": cfg.lane,
        "tasks": [asdict(task) for task in cfg.tasks],
        "task_count": total_task_count,
        "output_dir": str(out_dir),
    }
    (out_dir / "pipeline_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if dry_run:
        return plan

    telemetry = make_writer(out_dir)
    release_lock: Callable[[], None] | None = None

    def update_status(state: str) -> dict[str, Any]:
        snapshot = build_status_snapshot(
            teacher_name=cfg.teacher.name,
            repo_id=cfg.pipeline.repo_id,
            current_stage=current_stage,
            state=state,
            started_at=started_at,
            updated_at=time.time(),
            processed_tasks=processed_tasks,
            total_tasks=total_task_count,
            train_records=train_records,
            val_records=val_records,
            error_count=error_count,
            domain_counts=dict(domain_counts),
            label_counts=dict(label_counts),
            split_counts=dict(split_counts),
            last_domain=last_domain,
            last_policy_label=last_policy_label,
            last_latency_sec=last_latency_sec,
            last_error=last_error,
            student_eval_path=student_eval_path,
        )
        write_status_artifacts(out_dir, snapshot)
        return snapshot

    try:
        release_lock = acquire_run_lock(out_dir / "run.lock")
        telemetry.emit(
            "gguf_distill_config",
            teacher_name=cfg.teacher.name,
            repo_id=cfg.pipeline.repo_id,
            lane=cfg.lane,
            total_tasks=total_task_count,
            output_dir=str(out_dir),
            samples_per_task=cfg.pipeline.samples_per_task,
            use_mmproj=cfg.teacher.use_mmproj,
            model_path=cfg.teacher.model_path,
        )
        update_status("starting")

        log_path = out_dir / "llama_server.log"
        current_stage = "server_launch"
        telemetry.emit("gguf_distill_stage", stage=current_stage, status="start")
        update_status("running")
        proc: subprocess.Popen[str] | None = None
        records: list[dict[str, Any]] = []
        try:
            proc = launch_llama_server(cfg.teacher, log_path)
            telemetry.emit("gguf_distill_stage", stage=current_stage, status="waiting_health")
            wait_for_llama_server(cfg.teacher.host, cfg.teacher.port, cfg.teacher.timeout_sec)
            telemetry.emit("gguf_distill_stage", stage=current_stage, status="done")
            current_stage = "teacher_generation"
            telemetry.emit("gguf_distill_stage", stage=current_stage, status="start")
            update_status("running")
            for offset, task in enumerate(tasks):
                update_status("running")
                t0 = time.time()
                raw = request_teacher_example(cfg.teacher, task)
                last_latency_sec = time.time() - t0
                global_index = start_from + offset
                split = _split_for_index(global_index)
                record = build_sft_record(
                    example=raw["parsed_example"],
                    teacher_name=cfg.teacher.name,
                    split=split,
                    task=task,
                )
                raw["normalized_record"] = record
                records.append(record)
                append_json_line(raw_path, raw)
                target_path = val_path if split == "val" else train_path
                append_json_line(target_path, record)
                processed_tasks += 1
                last_domain = task.domain
                if task.lane == "detection":
                    response = json.loads(record["response"])
                    last_policy_label = str(response.get("policy_label", ""))
                else:
                    last_policy_label = str(record.get("task", ""))
                domain_counts[task.domain] += 1
                label_counts[last_policy_label or "unknown"] += 1
                split_counts[split] += 1
                if split == "val":
                    val_records += 1
                else:
                    train_records += 1
                telemetry.emit(
                    "gguf_distill_item",
                    index=global_index + 1,
                    total=total_task_count,
                    lane=task.lane,
                    domain=task.domain,
                    task_name=task.domain,
                    split=split,
                    policy_label=last_policy_label,
                    latency_sec=last_latency_sec,
                    processed_tasks=processed_tasks,
                    progress_pct=round((processed_tasks / total_task_count) * 100.0, 3) if total_task_count else 0.0,
                )
                update_status("running")
                now = time.time()
                if now >= next_ckpt_at:
                    ckpt_payload = {
                        "tasks_total": total_task_count,
                        "tasks_processed_in_this_session": processed_tasks - start_from,
                        "processed_tasks_total": processed_tasks,
                        "resume_start_offset": start_from if resume else 0,
                        "train_records": train_records,
                        "val_records": val_records,
                        "error_count": error_count,
                        "last_domain": last_domain,
                        "last_policy_label": last_policy_label,
                        "last_latency_sec": last_latency_sec,
                        "last_error": last_error,
                    }
                    try:
                        ckpt_seq = write_checkpoint(out_dir, ckpt_payload, ckpt_seq, ckpt_keep)
                    except Exception:
                        pass
                    next_ckpt_at = now + ckpt_interval
        finally:
            if proc is not None:
                stop_llama_server(proc)

        telemetry.emit("gguf_distill_stage", stage=current_stage, status="done")

        current_stage = "summary"
        telemetry.emit("gguf_distill_stage", stage=current_stage, status="start")
        update_status("running")
        new_summary = evaluate_distill_records(records)
        if resume and resume_summary is not None:
            combined_total = resume_summary["total_records"] + new_summary["total_records"]
            combined_valid = resume_summary["valid_json_records"] + new_summary["valid_json_records"]
            combined_duplicate = resume_summary["duplicate_prompt_count"] + new_summary["duplicate_prompt_count"]
            combined_domain_counts: dict[str, int] = dict(resume_summary.get("domain_counts", {}))
            combined_label_counts: dict[str, int] = dict(resume_summary.get("label_counts", {}))
            combined_split_counts: dict[str, int] = dict(resume_summary.get("split_counts", {}))
            combined_lane_counts: dict[str, int] = dict(resume_summary.get("lane_counts", {}))
            combined_task_counts: dict[str, int] = dict(resume_summary.get("task_counts", {}))
            for key, value in new_summary.get("domain_counts", {}).items():
                combined_domain_counts[key] = combined_domain_counts.get(key, 0) + value
            for key, value in new_summary.get("label_counts", {}).items():
                combined_label_counts[key] = combined_label_counts.get(key, 0) + value
            for key, value in new_summary.get("split_counts", {}).items():
                combined_split_counts[key] = combined_split_counts.get(key, 0) + value
            for key, value in new_summary.get("lane_counts", {}).items():
                combined_lane_counts[key] = combined_lane_counts.get(key, 0) + value
            for key, value in new_summary.get("task_counts", {}).items():
                combined_task_counts[key] = combined_task_counts.get(key, 0) + value
            summary = {
                "total_records": combined_total,
                "valid_json_records": combined_valid,
                "schema_valid_rate": (combined_valid / combined_total) if combined_total else 0.0,
                "duplicate_prompt_count": combined_duplicate,
                "domain_counts": combined_domain_counts,
                "label_counts": combined_label_counts,
                "split_counts": combined_split_counts,
                "lane_counts": combined_lane_counts,
                "task_counts": combined_task_counts,
            }
        else:
            summary = new_summary
        summary["lane"] = cfg.lane
        summary["teacher_name"] = cfg.teacher.name
        summary["raw_teacher_examples"] = str(raw_path)
        summary["train_path"] = str(train_path)
        summary["val_path"] = str(val_path)
        summary_path = out_dir / "eval_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        write_bundle_card(out_dir, cfg, summary)
        telemetry.emit(
            "gguf_distill_summary",
            total_records=summary["total_records"],
            schema_valid_rate=summary["schema_valid_rate"],
            train_records=train_records,
            val_records=val_records,
        )
        telemetry.emit("gguf_distill_stage", stage=current_stage, status="done")
        if not skip_student_eval:
            current_stage = "student_eval"
            telemetry.emit("gguf_distill_stage", stage=current_stage, status="start")
            update_status("running")
            student_eval = maybe_run_student_eval(cfg, out_dir)
            if student_eval is not None:
                student_eval_path = str(student_eval)
            update_status("running")
            telemetry.emit("gguf_distill_stage", stage=current_stage, status="done")

        if cfg.pipeline.repo_id and not skip_upload:
            current_stage = "hf_upload"
            telemetry.emit("gguf_distill_stage", stage=current_stage, status="start")
            update_status("running")
            upload_plan = build_hf_cli_plan(out_dir, cfg.pipeline.repo_id, cfg.pipeline.private)
            for cmd in upload_plan:
                telemetry.emit("gguf_distill_upload", status="start", command=" ".join(cmd))
                run_subprocess(cmd)
                telemetry.emit("gguf_distill_upload", status="done", command=" ".join(cmd))
            telemetry.emit("gguf_distill_stage", stage=current_stage, status="done")

        current_stage = "complete"
        update_status("complete")
        telemetry.emit(
            "gguf_distill_stage",
            stage=current_stage,
            status="done",
            processed_tasks=processed_tasks,
            total_tasks=total_task_count,
        )
        if summary is None:
            raise RuntimeError("distillation completed without a summary")
        return summary
    except Exception as exc:
        error_count += 1
        last_error = str(exc).strip() or exc.__class__.__name__
        telemetry.emit(
            "gguf_distill_error",
            stage=current_stage,
            error=last_error,
            processed_tasks=processed_tasks,
            total_tasks=total_task_count,
        )
        telemetry.emit(
            "gguf_distill_stage",
            stage=current_stage,
            status="failed",
            processed_tasks=processed_tasks,
            total_tasks=total_task_count,
            error=last_error,
        )
        update_status("failed")
        raise
    finally:
        telemetry.close()
        if release_lock is not None:
            release_lock()


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config for GGUF distillation")
    p.add_argument("--output-dir", default="", help="override output directory")
    p.add_argument("--max-tasks", type=int, default=0, help="limit generated tasks for smoke runs")
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument("--skip-student-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume from existing output_dir artifacts")
    args = p.parse_args()

    cfg = load_gguf_distill_config(args.config)
    summary = run_pipeline(
        cfg,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_tasks=args.max_tasks,
        skip_upload=args.skip_upload,
        skip_student_eval=args.skip_student_eval,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
