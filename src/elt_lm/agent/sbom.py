"""Minimal SPDX-style SBOM emission for generated artifacts."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Iterable


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_spdx_sbom(paths: Iterable[str | Path]) -> dict[str, object]:
    files = []
    for item in paths:
        path = Path(item)
        files.append({
            "SPDXID": f"SPDXRef-{path.name}",
            "fileName": str(path),
            "checksums": [{"algorithm": "SHA256", "checksumValue": _sha256(path)}],
        })
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "name": "elt-lm-generated-artifacts",
        "files": files,
    }


def write_spdx_sbom(paths: Iterable[str | Path], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_spdx_sbom(paths), indent=2), encoding="utf-8")
    return out
