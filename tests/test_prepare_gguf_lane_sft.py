from __future__ import annotations

import json
from pathlib import Path

import yaml

from elt_lm.prepare_gguf_lane_sft import (
    infer_lane,
    prepare_lane_sft_bundle,
    write_lane_benchmark_cases,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_infer_lane_prefers_metadata() -> None:
    lane = infer_lane([{"metadata": {"lane": "tool_use"}}], default="detection")
    assert lane == "tool_use"


def test_write_lane_benchmark_cases_preserves_reference_and_task(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "P1",
            "response": "<think>x</think><answer>42</answer>",
            "reference": "42",
            "task": "exact_math",
            "metadata": {"lane": "math", "split": "val"},
        },
        {
            "prompt": "P2",
            "response": "{\"tool_name\":\"shell\",\"arguments\":{\"cmd\":\"pwd\"}}",
            "reference": "{\"tool_name\":\"shell\",\"arguments\":{\"cmd\":\"pwd\"}}",
            "task": "json_match",
            "metadata": {"lane": "tool_use", "split": "val"},
        },
    ]
    out = tmp_path / "cases.jsonl"
    count = write_lane_benchmark_cases(rows, out, lane="math")
    assert count == 2
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert written[0]["reference"] == "42"
    assert written[0]["task"] == "exact_math"
    assert written[1]["task"] == "json_match"


def test_prepare_lane_sft_bundle_writes_lane_manifest(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "distill"
    output_root = tmp_path / "prepared"
    input_root.mkdir(parents=True)
    _write_jsonl(
        input_root / "distill_train.jsonl",
        [{
            "prompt": "Solve x+1=3",
            "response": "<think>subtract 1</think><answer>2</answer>",
            "reference": "2",
            "task": "exact_math",
            "text": "User: ...",
            "metadata": {"lane": "math", "split": "train"},
        }],
    )
    _write_jsonl(
        input_root / "distill_val.jsonl",
        [{
            "prompt": "Solve x+2=5",
            "response": "<think>subtract 2</think><answer>3</answer>",
            "reference": "3",
            "task": "exact_math",
            "text": "User: ...",
            "metadata": {"lane": "math", "split": "val"},
        }],
    )

    calls: list[Path] = []

    def _fake_tokenize_to_bin(*, tokenizer_path: str, inputs: list[Path], output: Path, exts: list[str], append_eos: bool, chunk_chars: int = 1_000_000) -> int:
        del tokenizer_path, exts, append_eos, chunk_chars
        calls.append(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"\x01\x00\x00\x00")
        return len(inputs)

    monkeypatch.setattr("elt_lm.prepare_gguf_lane_sft.tokenize_to_bin", _fake_tokenize_to_bin)

    summary = prepare_lane_sft_bundle(
        input_root=input_root,
        output_root=output_root,
        tokenizer_path="unused-tokenizer",
    )

    assert summary.lane == "math"
    assert summary.benchmark_cases == 1
    assert calls == [
        output_root / "bin" / "train.bin",
        output_root / "bin" / "val.bin",
    ]
    manifest = yaml.safe_load((output_root / "benchmarks" / "gguf_math_val_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["benchmarks"][0]["task"] == "exact_math"
