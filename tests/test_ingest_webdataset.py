from __future__ import annotations

import builtins
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_ingest_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "ingest_webdataset.py"
    spec = importlib.util.spec_from_file_location("ingest_webdataset", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_iter_jsonl_zst_reads_jsonl_records(tmp_path: Path) -> None:
    mod = _load_ingest_module()
    zstd = importlib.import_module("zstandard")

    raw_lines = [
        json.dumps({"text": "first record"}),
        json.dumps({"text": "second record"}),
    ]
    src = tmp_path / "sample.jsonl.zst"
    compressor = zstd.ZstdCompressor()
    src.write_bytes(compressor.compress(("\n".join(raw_lines) + "\n").encode("utf-8")))

    texts = list(mod.iter_jsonl_zst(src, mod.ex_text))
    assert texts == ["first record", "second record"]


def test_iter_jsonl_zst_reports_missing_zstandard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_ingest_module()
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "zstandard":
            raise ImportError("No module named 'zstandard'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="zstandard is required"):
        next(mod.iter_jsonl_zst(tmp_path / "missing.jsonl.zst", mod.ex_text))
