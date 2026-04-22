from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


def _load_build_train_bin_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "build_train_bin.py"
    spec = importlib.util.spec_from_file_location("build_train_bin", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_corpus_manifest_clean_has_unique_paths() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "scripts" / "corpus_manifest_clean.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    paths = [entry["path"] for entry in manifest["sources"]]
    duplicates = sorted({
        path
        for path in paths
        if paths.count(path) > 1
    })
    assert duplicates == []


def test_iter_weighted_texts_downsamples_deterministically() -> None:
    mod = _load_build_train_bin_module()
    texts = ["doc-1", "doc-2", "doc-3", "doc-4"]
    assert list(mod.iter_weighted_texts(texts, weight=0.5)) == ["doc-2", "doc-4"]


def test_iter_weighted_texts_upsamples_deterministically() -> None:
    mod = _load_build_train_bin_module()
    texts = ["doc-1", "doc-2", "doc-3"]
    assert list(mod.iter_weighted_texts(texts, weight=2.5)) == [
        "doc-1",
        "doc-1",
        "doc-2",
        "doc-2",
        "doc-2",
        "doc-3",
        "doc-3",
    ]
