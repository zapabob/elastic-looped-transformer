from __future__ import annotations

from pathlib import Path

from elt_lm.eval.gguf_quant_report import (
    MetricRow,
    build_eval_corpus,
    compare_metrics,
    parse_bench_rows,
    parse_perplexity_output,
    quantization_lane_boundaries,
    summarize_metric_rows,
)


def test_parse_bench_rows_splits_prompt_and_decode() -> None:
    rows = parse_bench_rows("TQ4_1S", [
        {
            "type_k": "f16",
            "type_v": "f16",
            "n_prompt": 16,
            "n_gen": 0,
            "samples_ts": [10.0, 12.0],
        },
        {
            "type_k": "q8_0",
            "type_v": "q8_0",
            "n_prompt": 0,
            "n_gen": 4,
            "samples_ts": [7.0],
        },
    ])

    assert [row.metric for row in rows] == ["prompt_eval_tps", "prompt_eval_tps", "decode_tps"]
    assert rows[0].block == "pp16_kf16_vf16_rep0"
    assert rows[-1].block == "tg4_kq8_0_vq8_0_rep0"


def test_parse_perplexity_output_extracts_quality_and_cache() -> None:
    output = """
       0  29.3171  3.378171  1.690071
llama_kv_cache:        CPU KV buffer size =   384.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =   128.00 MiB
llama_kv_cache: size =  512.00 MiB (   256 cells), K (f16): 256.00 MiB, V (f16): 256.00 MiB
llama_memory_recurrent: size = 3216.00 MiB (    64 cells,  32 layers, 64 seqs)
"""

    rows = parse_perplexity_output("Q8_0", output)
    by_metric = {row.metric: row for row in rows}

    assert by_metric["perplexity"].value == 29.3171
    assert by_metric["negative_log_likelihood"].value == 3.378171
    assert by_metric["kv_cache_mib"].value == 512.0
    assert by_metric["recurrent_state_mib"].value == 3216.0


def test_parse_perplexity_output_extracts_kl_table() -> None:
    output = """
chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              X RMS            Same top p
   1    13677.2349 +/-    0.0000       0.18974 +/-    0.00000       1.64035 +/-    0.00000     2.546 +/-  0.000 %    14.286 +/- 14.286 %
"""

    rows = parse_perplexity_output("Q8_0", output)
    by_metric = {row.metric: row for row in rows}

    assert by_metric["perplexity"].value == 13677.2349
    assert by_metric["logits_kl_vs_bf16"].value == 1.64035


def test_compare_metrics_reports_paired_group_tests() -> None:
    rows = [
        MetricRow("prompt_eval_tps", "BF16", "a", 1.0, "tok/s", "fixture", True),
        MetricRow("prompt_eval_tps", "BF16", "b", 2.0, "tok/s", "fixture", True),
        MetricRow("prompt_eval_tps", "Q8_0", "a", 2.0, "tok/s", "fixture", True),
        MetricRow("prompt_eval_tps", "Q8_0", "b", 3.0, "tok/s", "fixture", True),
        MetricRow("prompt_eval_tps", "TQ4_1S", "a", 1.5, "tok/s", "fixture", True),
        MetricRow("prompt_eval_tps", "TQ4_1S", "b", 2.5, "tok/s", "fixture", True),
    ]

    summaries = summarize_metric_rows(rows)
    report = compare_metrics(rows, seed=3)

    assert any(row.group == "Q8_0" and row.mean == 2.5 for row in summaries)
    assert report["prompt_eval_tps"]["n_blocks"] == 2
    assert report["prompt_eval_tps"]["pairwise"]


def test_build_eval_corpus_uses_text_or_prompt_reference(tmp_path: Path) -> None:
    source = tmp_path / "cases.jsonl"
    source.write_text(
        '{"prompt":"Q1","reference":"A1"}\n{"text":"User: Q2\\nAssistant: A2"}\n',
        encoding="utf-8",
    )
    out = build_eval_corpus([source], tmp_path / "corpus.txt", max_records=2, min_chars=10)

    text = out.read_text(encoding="utf-8")
    assert "User: Q1" in text
    assert "Assistant: A2" in text


def test_quantization_lane_boundaries_keep_kv_and_dflash_separate() -> None:
    boundaries = quantization_lane_boundaries()

    assert "weight-compression artifact" in boundaries["weight_compression"]
    assert "does not claim Google TurboQuant KV-cache serving performance" in boundaries["kv_compression"]
    assert "speculative-decoding lane" in boundaries["speculative_decoding"]
