from __future__ import annotations

import json

import pytest

from elt_lm.unsloth_qwen35_qlora_sft import (
    build_parser,
    coerce_messages,
    dataset_source_label,
    latest_eval_metrics_from_state,
    looks_like_mojibake,
    passes_adult_consent_filter,
    resolve_parquet_data_files,
    resolve_model_path,
    validate_jsonl_file,
)


def test_resolve_model_path_uses_hf_cache_ref(tmp_path):
    cache = tmp_path / "models--owner--repo"
    snapshot = cache / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    refs = cache / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123\n", encoding="utf-8")

    assert resolve_model_path(cache) == snapshot


def test_coerce_messages_from_prompt_response_schema():
    messages = coerce_messages({
        "system": "Be concise.",
        "prompt": "Hello",
        "response": "Hi",
    })

    assert messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]


def test_adult_consent_filter_blocks_obvious_high_risk_terms():
    assert passes_adult_consent_filter({"text": "adult consensual roleplay"})
    assert not passes_adult_consent_filter({"text": "未成年を含む設定"})
    assert not passes_adult_consent_filter({"text": "拒否を無視する設定"})


def test_mojibake_heuristic_flags_common_utf8_cp932_artifacts():
    assert looks_like_mojibake("縺ゅ↑縺溘・繧ｭ繝ｼ")
    assert not looks_like_mojibake("自然な日本語の短い文")


def test_adult_consent_filter_blocks_readable_high_risk_terms():
    assert not passes_adult_consent_filter({"text": "未成年を含む記録"})
    assert not passes_adult_consent_filter({"text": "JSを含む記録"})
    assert not passes_adult_consent_filter({"text": "薬物を含む記録"})


def test_validate_jsonl_file_rejects_multiline_or_broken_json(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"prompt": "hello"\n{"response": "world"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSONL"):
        validate_jsonl_file(path)


def test_validate_jsonl_file_accepts_one_object_per_line(tmp_path):
    path = tmp_path / "ok.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"prompt": "hello", "response": "world"}),
            json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        ]),
        encoding="utf-8",
    )

    validate_jsonl_file(path)


def test_resolve_parquet_data_files_accepts_glob_and_lists(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    first.write_bytes(b"")
    second.write_bytes(b"")

    assert resolve_parquet_data_files(str(tmp_path / "*.parquet")) == [
        str(first),
        str(second),
    ]
    assert resolve_parquet_data_files(f"{first};{second}") == [
        str(first),
        str(second),
    ]


def test_dataset_source_label_includes_extra_sources():
    args = build_parser().parse_args([
        "--parquet",
        "base/*.parquet",
        "--extra-jsonl",
        "extra.jsonl",
        "--extra-parquet",
        "more/*.parquet",
    ])

    assert dataset_source_label(args) == "base/*.parquet;extra.jsonl;more/*.parquet"


def test_latest_eval_metrics_from_state_prefers_most_recent_eval_row():
    metrics = latest_eval_metrics_from_state([
        {"step": 10, "loss": 2.0},
        {"step": 20, "eval_loss": 1.5, "eval_runtime": 12.0, "epoch": 0.1},
        {"step": 30, "loss": 1.7},
        {"step": 40, "eval_loss": 1.25, "eval_samples_per_second": 0.5, "epoch": 0.2},
    ])

    assert metrics == {
        "eval_loss": 1.25,
        "eval_samples_per_second": 0.5,
        "step": 40,
        "epoch": 0.2,
    }


def test_latest_eval_metrics_from_state_returns_none_without_eval_rows():
    assert latest_eval_metrics_from_state([{"step": 10, "loss": 2.0}]) is None
