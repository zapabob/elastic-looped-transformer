from __future__ import annotations

import json
from pathlib import Path

from elt_lm import hf_dataset_mix


def test_fetch_hf_dataset_mix_writes_metadata_only_summary(tmp_path: Path) -> None:
    manifest = tmp_path / "mix.yaml"
    manifest.write_text(
        """
version: 1
lanes:
  code:
    primary_hf_sources:
      - repo_id: owner/code-data
        role: code seed
        gate: unit tests
  safety_and_sensitive_understanding:
    primary_hf_sources:
      - repo_id: owner/safety-data
        role: boundary eval
        gate: classification only
""".strip(),
        encoding="utf-8",
    )

    summary = hf_dataset_mix.fetch_hf_dataset_mix(
        config_path=manifest,
        output_root=tmp_path / "out",
        metadata_only=True,
    )

    assert summary["total_sources"] == 2
    assert summary["sampled_sources"] == 0
    assert summary["results"][0]["lane"] == "code"
    assert (tmp_path / "out" / "summary.json").exists()
    sources = (tmp_path / "out" / "sources.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(sources) == 2


def test_fetch_hf_dataset_mix_samples_with_monkeypatched_loader(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "mix.yaml"
    manifest.write_text(
        """
version: 1
lanes:
  math:
    primary_hf_sources:
      - repo_id: owner/math-data
        role: math seed
        gate: exact verifier
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        hf_dataset_mix,
        "_take_streaming_rows",
        lambda repo_id, max_rows: [{"question": "2+2", "answer": 4}],
    )

    summary = hf_dataset_mix.fetch_hf_dataset_mix(
        config_path=manifest,
        output_root=tmp_path / "out",
        max_rows_per_source=1,
    )

    assert summary["sampled_sources"] == 1
    assert summary["total_rows_written"] == 1
    sample_path = Path(summary["results"][0]["sample_path"])
    rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_repo_id"] == "owner/math-data"
    assert rows[0]["lane"] == "math"
    assert rows[0]["row"]["answer"] == 4
