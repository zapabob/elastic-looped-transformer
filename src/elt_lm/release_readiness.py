"""Write a release-readiness manifest for HF safetensors and GGUF handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _has_any(path: Path, patterns: list[str]) -> bool:
    return any(any(path.glob(pattern)) for pattern in patterns)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _load_elt_loop_metadata(hf_dir: Path) -> dict[str, Any]:
    config = _read_json(hf_dir / "config.json")
    manifest = _read_json(hf_dir / "elt_export_manifest.json")
    config_elt = config.get("elt_config")
    elt_config = dict(config_elt) if isinstance(config_elt, dict) else {}

    l_min = _int_or_none(elt_config.get("L_min", manifest.get("L_min")))
    l_max = _int_or_none(elt_config.get("L_max", manifest.get("L_max")))
    looped_required = bool(
        elt_config.get("looped_runtime_required")
        or manifest.get("looped_runtime_required")
        or (l_max is not None and l_max > 1)
        or (l_min is not None and l_min > 1)
    )

    return {
        "present": bool(elt_config or manifest.get("format", "").startswith("elt_")),
        "schema": elt_config.get("schema", ""),
        "backbone_kind": elt_config.get("backbone_kind", ""),
        "source_model_id": elt_config.get("source_model_id", manifest.get("source_model_id", "")),
        "L_min": l_min,
        "L_max": l_max,
        "L_default": elt_config.get("L_default", manifest.get("L_default", l_max)),
        "loop_unit": elt_config.get("loop_unit", manifest.get("loop_unit", "")),
        "looped_runtime_required": looped_required,
        "gguf_runtime_status": elt_config.get("gguf_runtime_status", manifest.get("gguf_runtime_status", "")),
        "turboquant_model_family": elt_config.get(
            "turboquant_model_family",
            manifest.get("turboquant_model_family", ""),
        ),
    }


def build_release_manifest(
    *,
    hf_dir: str | Path,
    gguf_path: str | Path,
    repo_id: str,
    llama_cpp_dir: str | Path,
    turboquant_gguf_path: str | Path | None = None,
    turboquant_source_gguf_path: str | Path | None = None,
    turboquant_cuda_dir: str | Path | None = None,
    turboquant_model_family: str | None = None,
    loop_runtime_supported: bool = False,
    turboquant_loop_metadata_supported: bool = False,
) -> dict[str, Any]:
    hf = Path(hf_dir)
    gguf = Path(gguf_path)
    llama_cpp = Path(llama_cpp_dir)
    turboquant_gguf = Path(turboquant_gguf_path) if turboquant_gguf_path is not None else None
    turboquant_source_gguf = Path(turboquant_source_gguf_path) if turboquant_source_gguf_path is not None else gguf
    turboquant_cuda = Path(turboquant_cuda_dir) if turboquant_cuda_dir is not None else None
    turboquant_converter = (
        turboquant_cuda / "scripts" / "convert_weight_turboquant_gguf.py"
        if turboquant_cuda is not None
        else None
    )
    elt_loop = _load_elt_loop_metadata(hf)
    resolved_turboquant_model_family = (
        turboquant_model_family
        or (str(elt_loop.get("turboquant_model_family") or "") if elt_loop["looped_runtime_required"] else "")
        or "Qwen/Qwen3.5-4B"
    )
    hf_ready = (
        hf.exists()
        and (hf / "config.json").exists()
        and _has_any(hf, ["*.safetensors", "model*.safetensors"])
        and _has_any(hf, ["tokenizer.json", "tokenizer.model"])
    )
    converter = llama_cpp / "convert_hf_to_gguf.py"
    manifest = {
        "repo_id": repo_id,
        "hf_dir": str(hf),
        "gguf_path": str(gguf),
        "hf_safetensors_ready": hf_ready,
        "gguf_ready": gguf.exists() and gguf.suffix == ".gguf",
        "llama_cpp_converter": str(converter),
        "llama_cpp_converter_exists": converter.exists(),
        "elt_loop": elt_loop,
        "loop_runtime_supported": bool(loop_runtime_supported),
        "turboquant_loop_metadata_supported": bool(turboquant_loop_metadata_supported),
        "commands": {
            "hf_upload": f"hf upload {repo_id} {hf} . --repo-type model",
            "gguf_convert": (
                f"python {converter} {hf} --outfile {gguf}"
            ),
            "gguf_upload": f"hf upload {repo_id}-GGUF {gguf} {gguf.name} --repo-type model",
        },
        "blocking_notes": [],
    }
    if turboquant_gguf is not None:
        manifest["turboquant_gguf_path"] = str(turboquant_gguf)
        manifest["turboquant_source_gguf_path"] = str(turboquant_source_gguf)
        manifest["turboquant_ready"] = turboquant_gguf.exists() and turboquant_gguf.suffix == ".gguf"
        manifest["commands"]["turboquant_upload"] = (
            f"hf upload {repo_id}-GGUF {turboquant_gguf} {turboquant_gguf.name} --repo-type model"
        )
    if turboquant_converter is not None:
        manifest["turboquant_converter"] = str(turboquant_converter)
        manifest["turboquant_converter_exists"] = turboquant_converter.exists()
        if turboquant_gguf is not None:
            manifest["commands"]["turboquant_convert"] = (
                f"uv run --no-sync python {turboquant_converter} "
                f"--input-gguf {turboquant_source_gguf} --output-gguf {turboquant_gguf} "
                f"--model-family {resolved_turboquant_model_family} "
                "--replace-existing-turboquant-metadata --force"
            )
    if not hf_ready:
        manifest["blocking_notes"].append(
            "HF directory must contain config.json, tokenizer files, and safetensors weights before GGUF conversion."
        )
    if not converter.exists():
        manifest["blocking_notes"].append(
            "Set --llama-cpp-dir to a checkout containing convert_hf_to_gguf.py."
        )
    if not manifest["gguf_ready"]:
        manifest["blocking_notes"].append(
            "GGUF is not present yet; run commands.gguf_convert after HF safetensors export is ready."
        )
    if turboquant_gguf is not None and not manifest["turboquant_ready"]:
        manifest["blocking_notes"].append(
            "Turboquant GGUF is not present yet; run commands.turboquant_convert after Q8/plain GGUF is ready."
        )
    if turboquant_gguf is not None and not turboquant_source_gguf.exists():
        manifest["blocking_notes"].append(
            "Turboquant source GGUF is not present yet; produce the source GGUF before running commands.turboquant_convert."
        )
    if turboquant_converter is not None and not manifest["turboquant_converter_exists"]:
        manifest["blocking_notes"].append(
            "Set --turboquant-cuda-dir to a checkout containing scripts/convert_weight_turboquant_gguf.py."
        )
    if elt_loop["looped_runtime_required"] and not loop_runtime_supported:
        manifest["blocking_notes"].append(
            "ELT loop metadata declares L>=2; use a loop-aware llama.cpp runtime before publishing GGUF runtime claims."
        )
    if (
        turboquant_gguf is not None
        and elt_loop["looped_runtime_required"]
        and not turboquant_loop_metadata_supported
    ):
        manifest["blocking_notes"].append(
            "ELT loop metadata declares L>=2; use a Turboquant converter that preserves elt.* loop metadata."
        )
    return manifest


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--llama-cpp-dir", default="llama.cpp")
    parser.add_argument("--turboquant-gguf-path", default=None)
    parser.add_argument("--turboquant-source-gguf-path", default=None)
    parser.add_argument("--turboquant-cuda-dir", default=None)
    parser.add_argument("--turboquant-model-family", default=None)
    parser.add_argument("--loop-runtime-supported", action="store_true")
    parser.add_argument("--turboquant-loop-metadata-supported", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = build_release_manifest(
        hf_dir=args.hf_dir,
        gguf_path=args.gguf_path,
        repo_id=args.repo_id,
        llama_cpp_dir=args.llama_cpp_dir,
        turboquant_gguf_path=args.turboquant_gguf_path,
        turboquant_source_gguf_path=args.turboquant_source_gguf_path,
        turboquant_cuda_dir=args.turboquant_cuda_dir,
        turboquant_model_family=args.turboquant_model_family,
        loop_runtime_supported=args.loop_runtime_supported,
        turboquant_loop_metadata_supported=args.turboquant_loop_metadata_supported,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    cli()
