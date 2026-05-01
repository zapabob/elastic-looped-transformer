from __future__ import annotations

import json
from pathlib import Path

from elt_lm.eval.benchmarks import BenchmarkSpec, load_benchmark_cases
from elt_lm.eval.statistics import fold_accuracy_stats


def test_fold_accuracy_stats_reports_mean_sem_and_ci() -> None:
    stats = fold_accuracy_stats([1, 0, 1, 1, 0, 1], folds=3)

    assert stats.fold_count == 3
    assert len(stats.fold_scores) == 3
    assert 0.0 <= stats.ci95_low <= stats.mean <= stats.ci95_high <= 1.0
    assert stats.sem >= 0.0


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
