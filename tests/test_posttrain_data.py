from __future__ import annotations

import json
from pathlib import Path

import yaml

from elt_lm.posttrain_data import (
    load_posttrain_manifest,
    normalize_row,
    render_chat_text,
    write_manifest,
)
from elt_lm.synthetic_preferences import generate_synthetic_preference_pairs


def test_render_chat_text_includes_roles() -> None:
    text = render_chat_text("Ping", "Pong", "Keep logs")
    assert "System: Keep logs" in text
    assert "User: Ping" in text
    assert "Assistant: Pong" in text


def test_load_manifest_and_normalize_sft_jsonl(tmp_path: Path) -> None:
    source_path = tmp_path / "code.jsonl"
    source_path.write_text(
        json.dumps({"prompt": "Q", "response": "A"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({
        "buckets": [{
            "name": "code",
            "mode": "sft",
            "output_path": "posttrain/raw/code.jsonl",
            "sources": [{
                "name": "local_code",
                "kind": "jsonl",
                "path": str(source_path),
                "prompt_field": "prompt",
                "response_field": "response",
            }],
        }],
    }), encoding="utf-8")

    manifest = load_posttrain_manifest(manifest_path)
    bucket = manifest.buckets[0]
    row = {"prompt": "Write tests", "response": "Use pytest."}
    normalized = normalize_row(bucket, bucket.sources[0], row)
    assert normalized is not None
    assert normalized["bucket"] == "code"
    assert normalized["text"] == "User: Write tests\n\nAssistant: Use pytest."


def test_normalize_preference_row_with_conversation_lists(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({
        "buckets": [{
            "name": "prefs",
            "mode": "preference",
            "output_path": "posttrain/raw/prefs.jsonl",
            "sources": [{
                "name": "prefs_local",
                "kind": "jsonl",
                "path": "unused",
                "prompt_field": "prompt",
                "chosen_field": "chosen",
                "rejected_field": "rejected",
            }],
        }],
    }), encoding="utf-8")
    bucket = load_posttrain_manifest(manifest_path).buckets[0]
    row = {
        "prompt": "Answer carefully",
        "chosen": [
            {"role": "user", "content": "Answer carefully"},
            {"role": "assistant", "content": "Validate input first."},
        ],
        "rejected": [
            {"role": "assistant", "content": "Just use eval()."},
        ],
    }
    normalized = normalize_row(bucket, bucket.sources[0], row)
    assert normalized is not None
    assert "assistant: validate input first." in normalized["chosen"].lower()
    assert "eval()" in normalized["rejected"]


def test_write_manifest_writes_bucket_outputs(tmp_path: Path) -> None:
    source_path = tmp_path / "seed.jsonl"
    source_path.write_text(
        json.dumps({"prompt": "Q", "response": "A"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({
        "buckets": [{
            "name": "security_style",
            "mode": "sft",
            "output_path": "raw/security_style.jsonl",
            "sources": [{
                "name": "seed",
                "kind": "jsonl",
                "path": str(source_path),
                "prompt_field": "prompt",
                "response_field": "response",
            }],
        }],
    }), encoding="utf-8")
    manifest = load_posttrain_manifest(manifest_path)
    written = write_manifest(manifest, output_root=tmp_path / "out")
    assert written[0][2] == 1
    assert written[0][1].is_file()


def test_generate_synthetic_preference_pairs_count_and_shape() -> None:
    pairs = generate_synthetic_preference_pairs(10, seed=7)
    assert len(pairs) == 10
    assert all(pair.prompt for pair in pairs)
    assert all(pair.chosen != pair.rejected for pair in pairs)
    assert all("MIL-STD-498" in " ".join(pair.standard_refs) for pair in pairs)
