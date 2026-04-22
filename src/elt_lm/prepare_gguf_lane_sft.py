from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from elt_lm.gguf_distill import DEFAULT_TARGET_KIND_BY_LANE, LaneName, _normalize_lane
from elt_lm.tokenize_data import tokenize_to_bin


@dataclass
class GGUFLanePrepSummary:
    input_root: str
    output_root: str
    lane: str
    train_records: int
    val_records: int
    train_tokens: int
    val_tokens: int
    benchmark_cases: int
    train_bin: str
    val_bin: str
    benchmark_cases_path: str
    benchmark_manifest_path: str


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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
                rows.append(row)
    return rows


def infer_lane(rows: list[dict[str, Any]], default: str = "detection") -> LaneName:
    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and metadata.get("lane"):
            return _normalize_lane(str(metadata["lane"]))
    return _normalize_lane(default)


def lane_benchmark_task(lane: str, rows: list[dict[str, Any]] | None = None) -> str:
    if rows:
        for row in rows:
            task = str(row.get("task", "")).strip()
            if task:
                return task
    return DEFAULT_TARGET_KIND_BY_LANE[_normalize_lane(lane)]


def write_lane_benchmark_cases(
    val_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    lane: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    default_task = lane_benchmark_task(lane, val_rows)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in val_rows:
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            reference = str(row.get("reference", response)).strip()
            task = str(row.get("task", default_task)).strip() or default_task
            if not prompt or not response or not reference:
                continue
            payload = {
                "prompt": prompt,
                "reference": reference,
                "task": task,
                "bucket": row.get("bucket", ""),
                "source": row.get("source", ""),
                "metadata": row.get("metadata", {}),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_lane_benchmark_manifest(
    cases_path: Path,
    output_path: Path,
    *,
    lane: str,
    rows: list[dict[str, Any]] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task = lane_benchmark_task(lane, rows)
    manifest = {
        "benchmarks": [
            {
                "name": f"gguf_{lane}_val",
                "kind": "jsonl",
                "task": task,
                "path": str(cases_path),
                "prompt_field": "prompt",
                "reference_field": "reference",
            }
        ]
    }
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return output_path


def prepare_lane_sft_bundle(
    *,
    input_root: Path,
    output_root: Path,
    tokenizer_path: str,
    lane: str = "",
) -> GGUFLanePrepSummary:
    train_jsonl = input_root / "distill_train.jsonl"
    val_jsonl = input_root / "distill_val.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(train_jsonl)
    if not val_jsonl.exists():
        raise FileNotFoundError(val_jsonl)

    output_root.mkdir(parents=True, exist_ok=True)
    bin_root = output_root / "bin"
    bench_root = output_root / "benchmarks"
    summary_path = output_root / "prep_summary.json"

    train_rows = _load_jsonl(train_jsonl)
    val_rows = _load_jsonl(val_jsonl)
    resolved_lane = infer_lane(train_rows + val_rows, default=lane or "detection")

    train_bin = bin_root / "train.bin"
    val_bin = bin_root / "val.bin"
    benchmark_cases_path = bench_root / f"gguf_{resolved_lane}_val_cases.jsonl"
    benchmark_manifest_path = bench_root / f"gguf_{resolved_lane}_val_manifest.yaml"

    train_tokens = tokenize_to_bin(
        tokenizer_path=tokenizer_path,
        inputs=[train_jsonl],
        output=train_bin,
        exts=[".jsonl"],
        append_eos=True,
    )
    val_tokens = tokenize_to_bin(
        tokenizer_path=tokenizer_path,
        inputs=[val_jsonl],
        output=val_bin,
        exts=[".jsonl"],
        append_eos=True,
    )

    benchmark_cases = write_lane_benchmark_cases(val_rows, benchmark_cases_path, lane=resolved_lane)
    write_lane_benchmark_manifest(
        benchmark_cases_path,
        benchmark_manifest_path,
        lane=resolved_lane,
        rows=val_rows,
    )

    summary = GGUFLanePrepSummary(
        input_root=str(input_root),
        output_root=str(output_root),
        lane=resolved_lane,
        train_records=len(train_rows),
        val_records=len(val_rows),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        benchmark_cases=benchmark_cases,
        train_bin=str(train_bin),
        val_bin=str(val_bin),
        benchmark_cases_path=str(benchmark_cases_path),
        benchmark_manifest_path=str(benchmark_manifest_path),
    )
    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True, help="GGUF distill output dir with distill_train.jsonl / distill_val.jsonl")
    p.add_argument("--output-root", required=True, help="Prepared lane SFT output dir")
    p.add_argument("--tokenizer", required=True, help="HF tokenizer path")
    p.add_argument("--lane", default="", help="optional lane override; inferred from metadata if omitted")
    args = p.parse_args()

    summary = prepare_lane_sft_bundle(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        tokenizer_path=args.tokenizer,
        lane=args.lane,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
