"""Manifest-driven post-training data normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Iterator, Literal

import yaml


SourceKind = Literal["jsonl", "hf"]
BucketMode = Literal["sft", "preference"]


@dataclass
class PostTrainSource:
    name: str
    kind: SourceKind
    mode: BucketMode | None = None
    path: str | None = None
    dataset: str | None = None
    config: str | None = None
    split: str = "train"
    prompt_field: str | None = None
    response_field: str | None = None
    system_field: str | None = None
    text_field: str | None = None
    chosen_field: str | None = None
    rejected_field: str | None = None
    prompt_template: str | None = None
    response_template: str | None = None
    text_template: str | None = None
    chosen_template: str | None = None
    rejected_template: str | None = None
    limit: int = 0
    metadata_fields: list[str] = field(default_factory=list)


@dataclass
class PostTrainBucket:
    name: str
    mode: BucketMode
    output_path: str
    sources: list[PostTrainSource]


@dataclass
class PostTrainManifest:
    buckets: list[PostTrainBucket]


def load_posttrain_manifest(path: str | Path) -> PostTrainManifest:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    buckets: list[PostTrainBucket] = []
    for bucket_raw in raw.get("buckets", []):
        bucket_mode = bucket_raw["mode"]
        sources = []
        for src_raw in bucket_raw.get("sources", []):
            src = PostTrainSource(**src_raw)
            if src.mode is None:
                src.mode = bucket_mode
            sources.append(src)
        buckets.append(PostTrainBucket(
            name=bucket_raw["name"],
            mode=bucket_mode,
            output_path=bucket_raw["output_path"],
            sources=sources,
        ))
    return PostTrainManifest(buckets=buckets)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, dict) for v in value):
            parts = []
            for item in value:
                role = item.get("role") or item.get("from") or ""
                content = item.get("content") or item.get("value") or ""
                if not content:
                    continue
                if role:
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(content))
            if parts:
                return "\n\n".join(parts)
        return "\n".join(_stringify(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _render_template(template: str, row: dict[str, Any]) -> str:
    rendered = template
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        rendered = rendered.replace(
            "{" + field_name + "}",
            _stringify(row.get(field_name, "")),
        )
    return rendered


def _extract_value(
    row: dict[str, Any],
    field_name: str | None,
    template: str | None,
) -> str:
    if template:
        return _render_template(template, row).strip()
    if field_name:
        return _stringify(row.get(field_name)).strip()
    return ""


def render_chat_text(prompt: str, response: str, system: str = "") -> str:
    parts = []
    if system:
        parts.append(f"System: {system.strip()}")
    parts.append(f"User: {prompt.strip()}")
    parts.append(f"Assistant: {response.strip()}")
    return "\n\n".join(parts)


def normalize_row(
    bucket: PostTrainBucket,
    source: PostTrainSource,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    metadata = {
        key: row.get(key)
        for key in source.metadata_fields
        if key in row
    }
    if bucket.mode == "sft":
        text = _extract_value(row, source.text_field, source.text_template)
        prompt = _extract_value(row, source.prompt_field, source.prompt_template)
        response = _extract_value(row, source.response_field, source.response_template)
        system = _extract_value(row, source.system_field, None)
        if text:
            return {
                "bucket": bucket.name,
                "mode": bucket.mode,
                "source": source.name,
                "prompt": prompt,
                "response": response,
                "system": system,
                "text": text,
                "metadata": metadata,
            }
        if not prompt or not response:
            return None
        return {
            "bucket": bucket.name,
            "mode": bucket.mode,
            "source": source.name,
            "prompt": prompt,
            "response": response,
            "system": system,
            "text": render_chat_text(prompt, response, system),
            "metadata": metadata,
        }

    prompt = _extract_value(row, source.prompt_field, source.prompt_template)
    chosen = _extract_value(row, source.chosen_field, source.chosen_template)
    rejected = _extract_value(row, source.rejected_field, source.rejected_template)
    system = _extract_value(row, source.system_field, None)
    if not prompt or not chosen or not rejected:
        return None
    return {
        "bucket": bucket.name,
        "mode": bucket.mode,
        "source": source.name,
        "prompt": prompt,
        "system": system,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_text": render_chat_text(prompt, chosen, system),
        "rejected_text": render_chat_text(prompt, rejected, system),
        "metadata": metadata,
    }


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _iter_hf(source: PostTrainSource) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    if source.dataset is None:
        raise ValueError(f"{source.name}: hf source requires `dataset`")
    ds = load_dataset(
        source.dataset,
        source.config,
        split=source.split,
        streaming=True,
    )
    for row in ds:
        if isinstance(row, dict):
            yield row


def iter_source_rows(source: PostTrainSource) -> Iterable[dict[str, Any]]:
    if source.kind == "jsonl":
        if source.path is None:
            raise ValueError(f"{source.name}: jsonl source requires `path`")
        return _iter_jsonl(Path(source.path))
    if source.kind == "hf":
        return _iter_hf(source)
    raise ValueError(f"unsupported source kind: {source.kind!r}")


def iter_normalized_bucket(bucket: PostTrainBucket) -> Iterator[dict[str, Any]]:
    for source in bucket.sources:
        emitted = 0
        for row in iter_source_rows(source):
            normalized = normalize_row(bucket, source, row)
            if normalized is None:
                continue
            yield normalized
            emitted += 1
            if source.limit and emitted >= source.limit:
                break


def resolve_output_path(path: str, output_root: str | Path = "") -> Path:
    out = Path(path)
    if out.is_absolute() or not output_root:
        return out
    return Path(output_root) / out


def write_bucket(bucket: PostTrainBucket, output_root: str | Path = "") -> tuple[Path, int]:
    out_path = resolve_output_path(bucket.output_path, output_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for record in iter_normalized_bucket(bucket):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return out_path, count


def write_manifest(manifest: PostTrainManifest, output_root: str | Path = "") -> list[tuple[str, Path, int]]:
    written = []
    for bucket in manifest.buckets:
        path, count = write_bucket(bucket, output_root=output_root)
        written.append((bucket.name, path, count))
    return written
