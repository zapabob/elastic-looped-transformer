"""Export adapter-only LoRA artifacts from HF-backed Qwen3.5 ELT checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _adapter_state_from_checkpoint(state: dict[str, Any]) -> tuple[dict[str, torch.Tensor], str]:
    if state.get("adapter_only"):
        model_state = state.get("model")
        if not isinstance(model_state, dict):
            raise ValueError("adapter-only checkpoint is missing model state")
        base = str(state.get("base_checkpoint") or "")
        return {
            str(k): v.detach().cpu()
            for k, v in model_state.items()
            if isinstance(v, torch.Tensor) and ".lora_" in str(k)
        }, base

    model_state = state.get("model", state)
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint does not contain a model state_dict")
    return {
        str(k): v.detach().cpu()
        for k, v in model_state.items()
        if isinstance(v, torch.Tensor) and ".lora_" in str(k)
    }, ""


def _write_adapter_card(out: Path, metadata: dict[str, Any]) -> None:
    text = f"""---
license: apache-2.0
library_name: transformers
tags:
  - elastic-looped-transformer
  - lora
  - adapter
---

# ELT Qwen3.5 LoRA adapter

This directory contains adapter-only LoRA tensors exported from an
`hf_qwen35_looped` ELT checkpoint. The portable tensor payload is
`adapter_model.safetensors`; `adapter.pt` is kept for local ELT runtime
compatibility.

Base checkpoint: `{metadata.get("base_checkpoint") or "unspecified"}`

Tensor count: `{metadata["num_tensors"]}`

Parameter count: `{metadata["num_parameters"]}`
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def export_lora_adapter(ckpt_path: str | Path, out_dir: str | Path) -> Path:
    ckpt = Path(ckpt_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"unsupported checkpoint payload: {type(state)!r}")
    adapter_state, base_checkpoint = _adapter_state_from_checkpoint(state)
    if not adapter_state:
        raise ValueError(f"no LoRA adapter tensors found in {ckpt}")

    adapter_path = out / "adapter.pt"
    torch.save(adapter_state, adapter_path)
    safetensors_path = out / "adapter_model.safetensors"
    save_file(adapter_state, safetensors_path, metadata={"format": "pt"})

    cfg = state.get("cfg")
    model_cfg = getattr(cfg, "model", None)
    metadata = {
        "format": "elt_hf_qwen35_lora_v1",
        "source_checkpoint": str(ckpt),
        "base_checkpoint": base_checkpoint,
        "num_tensors": len(adapter_state),
        "num_parameters": int(sum(t.numel() for t in adapter_state.values())),
        "adapter_pt": str(adapter_path),
        "adapter_safetensors": str(safetensors_path),
        "hf_upload_command": "hf upload <repo-id> . . --repo-type model",
        "rank": getattr(model_cfg, "hf_lora_rank", None),
        "alpha": getattr(model_cfg, "hf_lora_alpha", None),
        "top_layers": getattr(model_cfg, "hf_lora_top_layers", None),
        "target_modules": getattr(model_cfg, "hf_lora_target_modules", None),
    }
    (out / "adapter_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_adapter_card(out, metadata)
    return adapter_path


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="LoRA full or adapter-only checkpoint")
    parser.add_argument("--out-dir", required=True, help="Output directory for adapter.pt and metadata")
    args = parser.parse_args()
    adapter_path = export_lora_adapter(args.ckpt, args.out_dir)
    print(f"exported {adapter_path}")


if __name__ == "__main__":
    cli()
