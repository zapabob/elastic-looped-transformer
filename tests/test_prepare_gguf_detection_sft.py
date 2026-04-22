from __future__ import annotations

import json
from pathlib import Path

import yaml

from elt_lm.prepare_gguf_detection_sft import (
    prepare_detection_sft_bundle,
    write_detection_benchmark_cases,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_write_detection_benchmark_cases_uses_prompt_and_response(tmp_path: Path) -> None:
    rows = [
        {"prompt": "P1", "response": "{\"a\":1}", "metadata": {"split": "val"}},
        {"prompt": "P2", "response": "{\"b\":2}", "metadata": {"split": "val"}},
    ]
    out = tmp_path / "cases.jsonl"
    count = write_detection_benchmark_cases(rows, out)
    assert count == 2
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert written[0]["prompt"] == "P1"
    assert written[0]["reference"] == "{\"a\":1}"
    assert written[1]["task"] == "json_match"


def test_prepare_detection_sft_bundle_writes_bins_and_manifest(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "distill"
    output_root = tmp_path / "prepared"
    input_root.mkdir(parents=True)
    _write_jsonl(
        input_root / "distill_train.jsonl",
        [{"prompt": "Train", "response": "{\"label\":\"review\"}", "text": "User: Train\n\nAssistant: ok"}],
    )
    _write_jsonl(
        input_root / "distill_val.jsonl",
        [{"prompt": "Val", "response": "{\"label\":\"allow\"}", "text": "User: Val\n\nAssistant: ok"}],
    )

    calls: list[Path] = []

    def _fake_tokenize_to_bin(*, tokenizer_path: str, inputs: list[Path], output: Path, exts: list[str], append_eos: bool, chunk_chars: int = 1_000_000) -> int:
        del tokenizer_path, exts, append_eos, chunk_chars
        calls.append(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"\x01\x00\x00\x00")
        return len(inputs)

    monkeypatch.setattr("elt_lm.prepare_gguf_lane_sft.tokenize_to_bin", _fake_tokenize_to_bin)

    summary = prepare_detection_sft_bundle(
        input_root=input_root,
        output_root=output_root,
        tokenizer_path="unused-tokenizer",
    )

    assert summary.train_records == 1
    assert summary.val_records == 1
    assert summary.benchmark_cases == 1
    assert summary.lane == "detection"
    assert calls == [
        output_root / "bin" / "train.bin",
        output_root / "bin" / "val.bin",
    ]
    manifest = yaml.safe_load((output_root / "benchmarks" / "gguf_detection_val_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["benchmarks"][0]["task"] == "json_match"
    summary_json = json.loads((output_root / "prep_summary.json").read_text(encoding="utf-8"))
    assert summary_json["benchmark_cases"] == 1
