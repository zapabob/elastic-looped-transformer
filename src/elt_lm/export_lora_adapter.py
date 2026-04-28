"""Export adapter-only LoRA artifacts from HF-backed Qwen3.5 ELT checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


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

    cfg = state.get("cfg")
    model_cfg = getattr(cfg, "model", None)
    metadata = {
        "format": "elt_hf_qwen35_lora_v1",
        "source_checkpoint": str(ckpt),
        "base_checkpoint": base_checkpoint,
        "num_tensors": len(adapter_state),
        "num_parameters": int(sum(t.numel() for t in adapter_state.values())),
        "rank": getattr(model_cfg, "hf_lora_rank", None),
        "alpha": getattr(model_cfg, "hf_lora_alpha", None),
        "top_layers": getattr(model_cfg, "hf_lora_top_layers", None),
        "target_modules": getattr(model_cfg, "hf_lora_target_modules", None),
    }
    (out / "adapter_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
