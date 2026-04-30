from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from elt_lm.gguf_distill import DEFAULT_TARGET_KIND_BY_LANE, LaneName, _normalize_lane
from elt_lm.prepare_gguf_lane_sft import (
    infer_lane,
    write_lane_benchmark_cases,
    write_lane_benchmark_manifest,
)
from elt_lm.tokenize_data import tokenize_to_bin


TOKENIZER_PATH = "H:/Qwen3.5-9B-official-hf"


@dataclass(frozen=True)
class LaneMixSpec:
    distill: float
    original: float
    clean: float
    original_sources: tuple[str, ...]
    clean_sources: tuple[str, ...]


LANE_MIX_SPECS: dict[LaneName, LaneMixSpec] = {
    "code": LaneMixSpec(
        distill=0.70,
        original=0.25,
        clean=0.05,
        original_sources=("H:/elt_data/posttrain/raw/code.jsonl",),
        clean_sources=(
            "H:/elt_data/clean/webdataset_coding.jsonl",
            "H:/elt_data/clean/webdataset_coding_train.jsonl",
            "H:/elt_data/clean/aegis_local.jsonl",
        ),
    ),
    "math": LaneMixSpec(
        distill=0.60,
        original=0.35,
        clean=0.05,
        original_sources=("H:/elt_data/posttrain/raw/reasoning.jsonl",),
        clean_sources=(
            "H:/elt_data/clean/aegis_local.jsonl",
            "H:/elt_data/clean/camel_sci.jsonl",
            "H:/elt_data/clean/webdataset_phi35.jsonl",
        ),
    ),
    "stem_reasoning": LaneMixSpec(
        distill=0.60,
        original=0.35,
        clean=0.05,
        original_sources=(
            "H:/elt_data/clean/camel_sci.jsonl",
            "H:/elt_data/clean/webdataset_domain_knowledge.jsonl",
            "H:/elt_data/clean/webdataset_phi35.jsonl",
        ),
        clean_sources=(
            "H:/elt_data/clean/tulu3.jsonl",
            "H:/elt_data/clean/webdataset_integrated.jsonl",
        ),
    ),
    "tool_use": LaneMixSpec(
        distill=0.70,
        original=0.25,
        clean=0.05,
        original_sources=("H:/elt_data/posttrain/raw/tool_call.jsonl",),
        clean_sources=(
            "H:/elt_data/clean/webdataset_integrated.jsonl",
            "H:/elt_data/clean/tulu3.jsonl",
        ),
    ),
}


@dataclass
class MixedLanePrepSummary:
    input_root: str
    output_root: str
    lane: str
    train_records: int
    val_records: int
    distill_train_records: int
    distill_val_records: int
    original_train_records: int
    original_val_records: int
    clean_train_records: int
    clean_val_records: int
    train_tokens: int
    val_tokens: int
    benchmark_cases: int
    train_jsonl: str
    val_jsonl: str
    train_bin: str
    val_bin: str
    benchmark_cases_path: str
    benchmark_manifest_path: str
    source_counts: dict[str, int]


def _load_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _row_text(row: dict[str, Any]) -> str:
    text = str(row.get("text", "")).strip()
    if text:
        return text
    prompt = str(row.get("prompt", "")).strip()
    response = str(row.get("response", "")).strip()
    if prompt and response:
        return f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
    return ""


def _record_from_text(
    *,
    text: str,
    lane: LaneName,
    split: str,
    source: str,
    bucket: str,
) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "mode": "sft_replay",
        "source": source,
        "prompt": "",
        "response": "",
        "reference": "",
        "task": DEFAULT_TARGET_KIND_BY_LANE[lane],
        "text": text,
        "metadata": {
            "lane": lane,
            "task_name": bucket,
            "split": split,
            "teacher": "original_ltf_replay",
            "source": source,
        },
    }


def _distill_rows(rows: list[dict[str, Any]], *, lane: LaneName, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        text = _row_text(row)
        if not text:
            continue
        copied = dict(row)
        copied["text"] = text
        metadata = dict(copied.get("metadata") or {})
        metadata.setdefault("lane", lane)
        metadata["split"] = split
        copied["metadata"] = metadata
        out.append(copied)
    return out


def _target_counts(distill_count: int, spec: LaneMixSpec) -> tuple[int, int]:
    if distill_count <= 0:
        return (0, 0)
    total = math.ceil(distill_count / spec.distill)
    original = max(0, round(total * spec.original))
    clean = max(0, round(total * spec.clean))
    return original, clean


def _read_replay_rows(
    sources: Iterable[str],
    *,
    lane: LaneName,
    split: str,
    bucket: str,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    if limit <= 0:
        return out, counts
    for source in sources:
        path = Path(source)
        if not path.exists() or path.stat().st_size == 0:
            counts[source] = 0
            continue
        for row in _load_jsonl(path):
            text = _row_text(row)
            if not text:
                continue
            out.append(
                _record_from_text(
                    text=text,
                    lane=lane,
                    split=split,
                    source=source,
                    bucket=bucket,
                )
            )
            counts[source] = counts.get(source, 0) + 1
            if len(out) >= limit:
                return out, counts
        counts.setdefault(source, 0)
    return out, counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_mixed_lane_sft_bundle(
    *,
    input_root: Path,
    output_root: Path,
    tokenizer_path: str,
    lane: str,
    max_distill_records: int | None = None,
) -> MixedLanePrepSummary:
    train_jsonl = input_root / "distill_train.jsonl"
    val_jsonl = input_root / "distill_val.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(train_jsonl)
    if not val_jsonl.exists():
        raise FileNotFoundError(val_jsonl)

    raw_train = _load_jsonl(train_jsonl, limit=max_distill_records)
    raw_val_limit = None if max_distill_records is None else max(1, max_distill_records // 10)
    raw_val = _load_jsonl(val_jsonl, limit=raw_val_limit)
    resolved_lane = infer_lane(raw_train + raw_val, default=lane)
    if resolved_lane not in LANE_MIX_SPECS:
        raise ValueError(f"mixed replay is not defined for lane: {resolved_lane}")
    spec = LANE_MIX_SPECS[resolved_lane]

    distill_train = _distill_rows(raw_train, lane=resolved_lane, split="train")
    distill_val = _distill_rows(raw_val, lane=resolved_lane, split="val")
    original_train_target, clean_train_target = _target_counts(len(distill_train), spec)
    original_val_target, clean_val_target = _target_counts(len(distill_val), spec)

    original_train, original_train_counts = _read_replay_rows(
        spec.original_sources,
        lane=resolved_lane,
        split="train",
        bucket="original_replay",
        limit=original_train_target,
    )
    clean_train, clean_train_counts = _read_replay_rows(
        spec.clean_sources,
        lane=resolved_lane,
        split="train",
        bucket="clean_replay",
        limit=clean_train_target,
    )
    original_val, original_val_counts = _read_replay_rows(
        spec.original_sources,
        lane=resolved_lane,
        split="val",
        bucket="original_replay",
        limit=original_val_target,
    )
    clean_val, clean_val_counts = _read_replay_rows(
        spec.clean_sources,
        lane=resolved_lane,
        split="val",
        bucket="clean_replay",
        limit=clean_val_target,
    )

    mixed_train = distill_train + original_train + clean_train
    mixed_val = distill_val + original_val + clean_val
    if not mixed_train:
        raise ValueError(f"no train records produced for lane: {resolved_lane}")
    if not mixed_val:
        mixed_val = distill_val[:1] or mixed_train[:1]

    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_root = output_root / "jsonl"
    bin_root = output_root / "bin"
    bench_root = output_root / "benchmarks"
    mixed_train_jsonl = jsonl_root / "train.jsonl"
    mixed_val_jsonl = jsonl_root / "val.jsonl"
    train_bin = bin_root / "train.bin"
    val_bin = bin_root / "val.bin"
    benchmark_cases_path = bench_root / f"mixed_{resolved_lane}_val_cases.jsonl"
    benchmark_manifest_path = bench_root / f"mixed_{resolved_lane}_val_manifest.yaml"

    _write_jsonl(mixed_train_jsonl, mixed_train)
    _write_jsonl(mixed_val_jsonl, mixed_val)
    train_tokens = tokenize_to_bin(
        tokenizer_path=tokenizer_path,
        inputs=[mixed_train_jsonl],
        output=train_bin,
        exts=[".jsonl"],
        append_eos=True,
    )
    val_tokens = tokenize_to_bin(
        tokenizer_path=tokenizer_path,
        inputs=[mixed_val_jsonl],
        output=val_bin,
        exts=[".jsonl"],
        append_eos=True,
    )

    benchmark_cases = write_lane_benchmark_cases(distill_val, benchmark_cases_path, lane=resolved_lane)
    write_lane_benchmark_manifest(
        benchmark_cases_path,
        benchmark_manifest_path,
        lane=resolved_lane,
        rows=distill_val,
    )

    source_counts: dict[str, int] = {}
    for counts in (original_train_counts, clean_train_counts, original_val_counts, clean_val_counts):
        for source, count in counts.items():
            source_counts[source] = source_counts.get(source, 0) + count

    summary = MixedLanePrepSummary(
        input_root=str(input_root),
        output_root=str(output_root),
        lane=resolved_lane,
        train_records=len(mixed_train),
        val_records=len(mixed_val),
        distill_train_records=len(distill_train),
        distill_val_records=len(distill_val),
        original_train_records=len(original_train),
        original_val_records=len(original_val),
        clean_train_records=len(clean_train),
        clean_val_records=len(clean_val),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        benchmark_cases=benchmark_cases,
        train_jsonl=str(mixed_train_jsonl),
        val_jsonl=str(mixed_val_jsonl),
        train_bin=str(train_bin),
        val_bin=str(val_bin),
        benchmark_cases_path=str(benchmark_cases_path),
        benchmark_manifest_path=str(benchmark_manifest_path),
        source_counts=source_counts,
    )
    (output_root / "prep_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER_PATH)
    p.add_argument("--lane", required=True)
    p.add_argument("--max-distill-records", type=int, default=None)
    args = p.parse_args()
    summary = prepare_mixed_lane_sft_bundle(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        tokenizer_path=args.tokenizer,
        lane=args.lane,
        max_distill_records=args.max_distill_records,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
