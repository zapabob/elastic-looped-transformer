"""Write a release-readiness manifest for HF safetensors and GGUF handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _has_any(path: Path, patterns: list[str]) -> bool:
    return any(any(path.glob(pattern)) for pattern in patterns)


def build_release_manifest(
    *,
    hf_dir: str | Path,
    gguf_path: str | Path,
    repo_id: str,
    llama_cpp_dir: str | Path,
    turboquant_gguf_path: str | Path | None = None,
    turboquant_source_gguf_path: str | Path | None = None,
    turboquant_cuda_dir: str | Path | None = None,
    turboquant_model_family: str = "Qwen/Qwen3.5-4B",
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
                f"--model-family {turboquant_model_family} "
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
    parser.add_argument("--turboquant-model-family", default="Qwen/Qwen3.5-4B")
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
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    cli()
