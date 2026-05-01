from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass
class HFDatasetFetchResult:
    lane: str
    repo_id: str
    role: str
    gate: str
    status: str
    rows_written: int = 0
    sample_path: str = ""
    error: str = ""


def _sanitize_repo_id(repo_id: str) -> str:
    return (
        repo_id.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def ensure_h_drive_hf_cache_env() -> None:
    cache_root = Path("H:/elt_data/cache/hf")
    defaults = {
        "HF_HOME": str(cache_root),
        "HF_HUB_CACHE": str(cache_root / "hub"),
        "HF_DATASETS_CACHE": str(cache_root / "datasets"),
        "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def _json_default(value: object) -> str:
    return str(value)


def _iter_hf_sources(manifest: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    lanes = manifest.get("lanes") or {}
    if not isinstance(lanes, dict):
        return
    for lane, lane_payload in lanes.items():
        if not isinstance(lane_payload, dict):
            continue
        for source in lane_payload.get("primary_hf_sources") or []:
            if isinstance(source, dict) and source.get("repo_id"):
                yield str(lane), source


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _take_streaming_rows(repo_id: str, max_rows: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(repo_id, split="train", streaming=True, trust_remote_code=False)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        if isinstance(row, dict):
            rows.append(row)
        else:
            rows.append({"value": row})
        if len(rows) >= max_rows:
            break
    return rows


def fetch_hf_dataset_mix(
    *,
    config_path: str | Path,
    output_root: str | Path,
    max_rows_per_source: int = 32,
    metadata_only: bool = False,
    min_sampled_sources: int = 1,
) -> dict[str, Any]:
    ensure_h_drive_hf_cache_env()
    config = Path(config_path)
    out = Path(output_root)
    manifest = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    out.mkdir(parents=True, exist_ok=True)

    results: list[HFDatasetFetchResult] = []
    total_rows = 0
    for lane, source in _iter_hf_sources(manifest):
        repo_id = str(source["repo_id"])
        role = str(source.get("role") or "")
        gate = str(source.get("gate") or "")
        result = HFDatasetFetchResult(
            lane=lane,
            repo_id=repo_id,
            role=role,
            gate=gate,
            status="metadata_only" if metadata_only else "pending",
        )
        if metadata_only:
            results.append(result)
            continue
        sample_path = out / "samples" / lane / f"{_sanitize_repo_id(repo_id)}.jsonl"
        try:
            rows = _take_streaming_rows(repo_id, max_rows=max_rows_per_source)
        except Exception as exc:
            result.status = "error"
            result.error = str(exc)[:800]
            results.append(result)
            continue
        wrapped_rows = [
            {
                "source_repo_id": repo_id,
                "lane": lane,
                "role": role,
                "gate": gate,
                "row": row,
            }
            for row in rows
        ]
        _write_jsonl(sample_path, wrapped_rows)
        result.status = "sampled"
        result.rows_written = len(wrapped_rows)
        result.sample_path = str(sample_path)
        total_rows += len(wrapped_rows)
        results.append(result)

    summary = {
        "config_path": str(config),
        "output_root": str(out),
        "metadata_only": metadata_only,
        "max_rows_per_source": max_rows_per_source,
        "total_sources": len(results),
        "sampled_sources": sum(1 for item in results if item.status == "sampled"),
        "error_sources": sum(1 for item in results if item.status == "error"),
        "total_rows_written": total_rows,
        "results": [asdict(item) for item in results],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(out / "sources.jsonl", [asdict(item) for item in results])
    if not metadata_only and summary["sampled_sources"] < min_sampled_sources:
        raise RuntimeError(
            "HF dataset mix fetch did not sample enough sources: "
            f"{summary['sampled_sources']} < {min_sampled_sources}. "
            f"See {out / 'summary.json'} for per-source errors."
        )
    return summary


def cli() -> None:
    parser = argparse.ArgumentParser(description="Fetch reviewed HF dataset mix samples to H: for v1 distill planning.")
    parser.add_argument("--config", default="configs/hf_dataset_mix_v1.yaml")
    parser.add_argument("--output-root", default="H:/elt_data/hf_dataset_mix_v1")
    parser.add_argument("--max-rows-per-source", type=int, default=32)
    parser.add_argument("--min-sampled-sources", type=int, default=1)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    summary = fetch_hf_dataset_mix(
        config_path=args.config,
        output_root=args.output_root,
        max_rows_per_source=max(0, args.max_rows_per_source),
        metadata_only=args.metadata_only,
        min_sampled_sources=max(0, args.min_sampled_sources),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
