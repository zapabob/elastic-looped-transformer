from __future__ import annotations

import json
from pathlib import Path

from elt_lm.eval.benchmarks import BenchmarkSpec, load_benchmark_cases
from elt_lm.eval.benchmark_comparison import compare_group_scores, render_markdown
from elt_lm.eval.statistics import (
    fold_accuracy_stats,
    friedman_permutation_test,
    paired_permutation_pvalue,
    pairwise_group_comparisons,
    summarize_scores,
)
from elt_lm.release_readiness import build_release_manifest


def test_fold_accuracy_stats_reports_mean_sem_and_ci() -> None:
    stats = fold_accuracy_stats([1, 0, 1, 1, 0, 1], folds=3)

    assert stats.fold_count == 3
    assert len(stats.fold_scores) == 3
    assert 0.0 <= stats.ci95_low <= stats.mean <= stats.ci95_high <= 1.0
    assert stats.sem >= 0.0


def test_group_statistics_report_paired_p_values() -> None:
    groups = {
        "vanilla": [0, 0, 0, 1, 0, 0],
        "sft": [0, 1, 0, 1, 0, 1],
        "grpo": [1, 1, 0, 1, 1, 1],
    }

    summary = summarize_scores("grpo", groups["grpo"])
    pairwise = pairwise_group_comparisons(groups, permutations=256, seed=7)
    omnibus = friedman_permutation_test(groups, permutations=256, seed=7)

    assert summary.mean > 0.0
    assert any(item.left == "vanilla" and item.right == "grpo" for item in pairwise)
    assert 0.0 <= omnibus.p_value <= 1.0
    assert omnibus.n_blocks == 6


def test_paired_permutation_handles_equal_scores() -> None:
    assert paired_permutation_pvalue([1, 0, 1], [1, 0, 1]) == 1.0


def test_benchmark_comparison_renders_markdown() -> None:
    report = compare_group_scores(
        "toy_cv",
        {"vanilla": [0, 1, 0, 1], "complete": [1, 1, 0, 1]},
        permutations=128,
        seed=0,
    )

    md = render_markdown(report)

    assert "toy_cv" in md
    assert "vanilla - complete" in md


def test_release_readiness_reports_blockers(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_bytes(b"stub")

    manifest = build_release_manifest(
        hf_dir=hf_dir,
        gguf_path=tmp_path / "model.gguf",
        repo_id="org/model",
        llama_cpp_dir=tmp_path / "llama.cpp",
    )

    assert not manifest["hf_safetensors_ready"]
    assert not manifest["gguf_ready"]
    assert manifest["blocking_notes"]


def test_release_readiness_reports_turboquant_artifact(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    llama_cpp_dir = tmp_path / "llama.cpp"
    turboquant_dir = tmp_path / "Turboquant-CUDA"
    hf_dir.mkdir()
    llama_cpp_dir.mkdir()
    (turboquant_dir / "scripts").mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_bytes(b"stub")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (llama_cpp_dir / "convert_hf_to_gguf.py").write_text("# stub\n", encoding="utf-8")
    (turboquant_dir / "scripts" / "convert_weight_turboquant_gguf.py").write_text("# stub\n", encoding="utf-8")
    gguf = tmp_path / "model.gguf"
    q8_gguf = tmp_path / "model-Q8_0.gguf"
    tq_gguf = tmp_path / "model-TQ4_1S.gguf"
    gguf.write_bytes(b"stub")
    q8_gguf.write_bytes(b"stub")
    tq_gguf.write_bytes(b"stub")

    manifest = build_release_manifest(
        hf_dir=hf_dir,
        gguf_path=gguf,
        repo_id="org/model",
        llama_cpp_dir=llama_cpp_dir,
        turboquant_gguf_path=tq_gguf,
        turboquant_source_gguf_path=q8_gguf,
        turboquant_cuda_dir=turboquant_dir,
    )

    assert manifest["turboquant_ready"]
    assert manifest["turboquant_converter_exists"]
    assert manifest["turboquant_source_gguf_path"] == str(q8_gguf)
    assert f"--input-gguf {q8_gguf}" in manifest["commands"]["turboquant_convert"]
    assert "--replace-existing-turboquant-metadata" in manifest["commands"]["turboquant_convert"]
    assert not manifest["blocking_notes"]


def test_jsonl_case_task_overrides_manifest_default(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({
            "prompt": "implement static checked code",
            "reference": "Run with cargo test --edition 2024",
            "task": "code_static_spec",
        }) + "\n",
        encoding="utf-8",
    )
    spec = BenchmarkSpec(
        name="mixed_code",
        task="python_exec",
        kind="jsonl",
        path=str(path),
        prompt_field="prompt",
        reference_field="reference",
    )

    cases = load_benchmark_cases(spec)

    assert cases[0].task == "code_static_spec"
