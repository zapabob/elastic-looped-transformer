from __future__ import annotations

import json
from pathlib import Path

from elt_lm.prepare_mixed_lane_sft import prepare_mixed_lane_sft_bundle


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_prepare_mixed_lane_sft_respects_math_replay_ratios(tmp_path: Path, monkeypatch) -> None:
    distill = tmp_path / "distill"
    output = tmp_path / "mixed"
    _write_jsonl(
        distill / "distill_train.jsonl",
        [
            {
                "prompt": f"p{i}",
                "response": f"<answer>{i}</answer>",
                "reference": str(i),
                "task": "exact_math",
                "text": f"distill train {i}",
                "metadata": {"lane": "math", "split": "train"},
            }
            for i in range(6)
        ],
    )
    _write_jsonl(
        distill / "distill_val.jsonl",
        [
            {
                "prompt": "vp",
                "response": "<answer>1</answer>",
                "reference": "1",
                "task": "exact_math",
                "text": "distill val",
                "metadata": {"lane": "math", "split": "val"},
            }
        ],
    )
    original = tmp_path / "posttrain" / "raw" / "reasoning.jsonl"
    clean = tmp_path / "clean" / "aegis_local.jsonl"
    _write_jsonl(original, [{"text": f"original {i}"} for i in range(10)])
    _write_jsonl(clean, [{"text": f"clean {i}"} for i in range(10)])

    import elt_lm.prepare_mixed_lane_sft as mod

    monkeypatch.setitem(
        mod.LANE_MIX_SPECS,
        "math",
        mod.LaneMixSpec(
            distill=0.60,
            original=0.35,
            clean=0.05,
            original_sources=(str(original),),
            clean_sources=(str(clean),),
        ),
    )

    def _fake_tokenize_to_bin(*, tokenizer_path: str, inputs: list[Path], output: Path, exts: list[str], append_eos: bool, chunk_chars: int = 1_000_000) -> int:
        del tokenizer_path, inputs, exts, append_eos, chunk_chars
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"\x01\x00\x00\x00")
        return 1

    monkeypatch.setattr(mod, "tokenize_to_bin", _fake_tokenize_to_bin)

    summary = prepare_mixed_lane_sft_bundle(
        input_root=distill,
        output_root=output,
        tokenizer_path="unused",
        lane="math",
    )

    assert summary.distill_train_records == 6
    assert summary.original_train_records == 4
    assert summary.clean_train_records == 0
    assert summary.train_records == 10
    assert summary.benchmark_cases == 1
    assert Path(summary.train_bin).exists()
    assert Path(summary.val_bin).exists()


def test_prepare_mixed_lane_sft_skips_empty_replay_sources(tmp_path: Path, monkeypatch) -> None:
    distill = tmp_path / "distill"
    output = tmp_path / "mixed"
    _write_jsonl(
        distill / "distill_train.jsonl",
        [
            {
                "text": f"code distill {i}",
                "prompt": "write code",
                "response": "```python\nprint(1)\n```",
                "metadata": {"lane": "code", "split": "train"},
            }
            for i in range(3)
        ],
    )
    _write_jsonl(
        distill / "distill_val.jsonl",
        [{
            "text": "code val",
            "prompt": "write code",
            "response": "```python\nprint(2)\n```",
            "metadata": {"lane": "code", "split": "val"},
        }],
    )
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    import elt_lm.prepare_mixed_lane_sft as mod

    monkeypatch.setitem(
        mod.LANE_MIX_SPECS,
        "code",
        mod.LaneMixSpec(
            distill=0.70,
            original=0.25,
            clean=0.05,
            original_sources=(str(empty),),
            clean_sources=(str(empty),),
        ),
    )

    def _fake_tokenize_to_bin(*, tokenizer_path: str, inputs: list[Path], output: Path, exts: list[str], append_eos: bool, chunk_chars: int = 1_000_000) -> int:
        del tokenizer_path, inputs, exts, append_eos, chunk_chars
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"\x01\x00\x00\x00")
        return 1

    monkeypatch.setattr(mod, "tokenize_to_bin", _fake_tokenize_to_bin)

    summary = prepare_mixed_lane_sft_bundle(
        input_root=distill,
        output_root=output,
        tokenizer_path="unused",
        lane="code",
    )

    assert summary.train_records == 3
    assert summary.original_train_records == 0
    assert summary.clean_train_records == 0
    assert summary.source_counts[str(empty)] == 0
