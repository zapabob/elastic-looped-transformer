from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_audit_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "audit_clean_corpus.py"
    spec = importlib.util.spec_from_file_location("audit_clean_corpus", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: list[dict[str, str] | str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, str):
                f.write(record + "\n")
            else:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_audit_clean_dir_counts_duplicates_and_quality(tmp_path: Path) -> None:
    mod = _load_audit_module()
    duplicate = "これは十分な長さの日本語テキストです。" * 5
    low_quality = "!!!!!!!! " * 20
    _write_jsonl(
        tmp_path / "sample.jsonl",
        [
            {"text": duplicate},
            {"text": duplicate},
            {"text": low_quality},
            "{not-json",
            {"prompt": "Question " * 20, "response": "Answer " * 20},
        ],
    )
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")

    report = mod.audit_clean_dir(tmp_path)

    assert report.files == 2
    assert len(report.zero_byte_files) == 1
    assert report.total_docs == 4
    assert report.parse_errors == 1
    assert report.low_quality == 1
    assert report.prefix_duplicates == 1
    assert report.exact_duplicates == 1
    assert report.top_quality_reasons["low_alnum_cjk_ratio"] == 1


def test_audit_report_writers(tmp_path: Path) -> None:
    mod = _load_audit_module()
    _write_jsonl(tmp_path / "sample.jsonl", [{"text": "A useful document " * 8}])
    report = mod.audit_clean_dir(tmp_path)
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    mod.write_json(report, json_path)
    mod.write_markdown(report, md_path)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["total_docs"] == 1
    assert "Clean Corpus Audit" in md_path.read_text(encoding="utf-8")
