"""Export merged Qwen3.5 LoRA checkpoints as HF safetensors.

The side-LoRA training path stores small adapter-only checkpoints for local
iteration. GGUF conversion needs a normal Hugging Face directory instead:
``config.json``, tokenizer files, and full ``model.safetensors`` weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from elt_lm.config import ModelConfig, TrainConfig
from elt_lm.hf_qwen35_looped import load_qwen35_text_config


_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
}


def _load_torch_state(path: str | Path) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"unsupported checkpoint payload from {path}: {type(state)!r}")
    return state


def _model_cfg(state: dict[str, Any]) -> ModelConfig:
    cfg = state.get("cfg")
    if isinstance(cfg, TrainConfig):
        return cfg.model
    model_cfg = getattr(cfg, "model", None)
    if isinstance(model_cfg, ModelConfig):
        return model_cfg
    raise ValueError("checkpoint is missing TrainConfig/ModelConfig metadata")


def _state_model(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    model_state = state.get("model", state)
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint does not contain a model state_dict")
    return {
        str(k): v.detach().cpu()
        for k, v in model_state.items()
        if isinstance(v, torch.Tensor)
    }


def _adapter_base_state(state: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], str]:
    if not state.get("adapter_only"):
        model_state = _state_model(state)
        adapter = {k: v for k, v in model_state.items() if ".lora_" in k}
        return model_state, adapter, ""

    base_path = str(state.get("base_checkpoint") or "")
    if not base_path:
        raise ValueError("adapter-only checkpoint is missing base_checkpoint")
    base_state = _load_torch_state(base_path)
    return _state_model(base_state), _state_model(state), base_path


def _standard_qwen_key(key: str) -> str:
    if key.startswith("qwen."):
        return key[len("qwen.") :]
    return key


def _merge_lora_weights(
    base_state: dict[str, torch.Tensor],
    adapter_state: dict[str, torch.Tensor],
    *,
    alpha: float,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    merged: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    merged_modules: list[str] = []

    lora_a_keys = sorted(k for k in adapter_state if k.endswith(".lora_A"))
    lora_delta_by_weight_key: dict[str, torch.Tensor] = {}
    for a_key in lora_a_keys:
        module = a_key[: -len(".lora_A")]
        b_key = module + ".lora_B"
        if b_key not in adapter_state:
            raise ValueError(f"missing LoRA B tensor for {a_key}")
        weight_key = module + ".weight"
        base_weight_key = module + ".base.weight"
        if weight_key not in base_state and base_weight_key in base_state:
            weight_key = base_weight_key
        if weight_key not in base_state:
            raise ValueError(f"missing base weight for adapter module {module}")

        a = adapter_state[a_key].float()
        b = adapter_state[b_key].float()
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"LoRA tensors must be matrices: {a_key}, {b_key}")
        rank = a.shape[0]
        if rank <= 0 or b.shape[1] != rank:
            raise ValueError(f"invalid LoRA rank shapes: {a_key}={tuple(a.shape)}, {b_key}={tuple(b.shape)}")

        lora_delta_by_weight_key[weight_key] = (b @ a) * (float(alpha) / float(rank))
        consumed.update({a_key, b_key})
        merged_modules.append(module)

    for key, tensor in base_state.items():
        out_key = key
        if out_key.endswith(".base.weight"):
            out_key = out_key[: -len(".base.weight")] + ".weight"
        elif out_key.endswith(".base.bias"):
            out_key = out_key[: -len(".base.bias")] + ".bias"

        output = tensor
        if key in lora_delta_by_weight_key:
            output = tensor.float() + lora_delta_by_weight_key[key]
        if output.is_floating_point():
            output = output.to(dtype=dtype)
        merged[_standard_qwen_key(out_key)] = output.contiguous()

    extra = sorted(k for k in adapter_state if ".lora_" in k and k not in consumed)
    if extra:
        raise ValueError(f"unconsumed LoRA tensors: {extra[:5]}")
    return merged, merged_modules


def _clone_shared_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    seen: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], str] = {}
    result: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        ident = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        if ident in seen:
            result[key] = tensor.clone()
        else:
            seen[ident] = key
            result[key] = tensor
    return result


def _copy_tokenizer(tokenizer_path: str | Path, out_dir: Path, *, required: bool) -> bool:
    tok_path = Path(tokenizer_path)
    if tok_path.exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True, use_fast=True)
            tokenizer.save_pretrained(out_dir)
            return True
        except Exception:
            pass
        copied = False
        for child in tok_path.iterdir():
            if child.is_file() and child.name in _TOKENIZER_FILES:
                shutil.copy2(child, out_dir / child.name)
                copied = True
        if copied:
            return True
        if required:
            raise ValueError(f"could not copy tokenizer files from {tokenizer_path}")
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True, use_fast=True)
        tokenizer.save_pretrained(out_dir)
        return True
    except Exception as exc:
        if required:
            raise ValueError(f"could not load tokenizer {tokenizer_path}") from exc
        return False


def _write_readme(out_dir: Path, metadata: dict[str, Any]) -> None:
    readme = f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: {metadata["source_model_id"]}
tags:
  - elastic-looped-transformer
  - qwen3.5
  - lora-merged
  - gguf-ready
---

# {metadata["repo_id"] or out_dir.name}

Merged Qwen3.5 side-LoRA export from the ELT-LM workflow.

- Source model: `{metadata["source_model_id"]}`
- Source checkpoint: `{metadata["source_checkpoint"]}`
- Base checkpoint: `{metadata["base_checkpoint"] or "embedded/full checkpoint"}`
- ELT loop range: `L={metadata["L_min"]}..{metadata["L_max"]}`
- ELT loop unit: `{metadata["loop_unit"]}`
- GGUF runtime status: `{metadata["gguf_runtime_status"]}`
- Merged LoRA modules: `{metadata["merged_lora_modules"]}`

The weights are saved as standard Hugging Face safetensors so llama.cpp can run
`convert_hf_to_gguf.py`. ELT loop metadata is also recorded in
`elt_export_manifest.json` and in `config.json` under `elt_config`.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def export_merged_qwen35_hf(
    *,
    ckpt_path: str | Path,
    out_dir: str | Path,
    tokenizer_path: str | Path,
    repo_id: str = "",
    dtype_name: str = "bf16",
    require_tokenizer: bool = True,
) -> dict[str, Any]:
    if dtype_name not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype_name!r}; expected one of {sorted(_DTYPES)}")
    state = _load_torch_state(ckpt_path)
    cfg = _model_cfg(state)
    if cfg.backbone_kind != "hf_qwen35_looped":
        raise ValueError(f"expected hf_qwen35_looped checkpoint, got {cfg.backbone_kind!r}")

    base_state, adapter_state, base_checkpoint = _adapter_base_state(state)
    merged_state, merged_modules = _merge_lora_weights(
        base_state,
        adapter_state,
        alpha=cfg.hf_lora_alpha,
        dtype=_DTYPES[dtype_name],
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(_clone_shared_tensors(merged_state), out / "model.safetensors", metadata={"format": "pt"})

    text_cfg = load_qwen35_text_config(cfg.hf_model_path)
    text_cfg.architectures = ["Qwen3_5ForCausalLM"]
    looped_runtime_required = int(cfg.L_max) > 1 or int(cfg.L_min) > 1
    loop_unit = "qwen3.5_text_model_pass"
    gguf_runtime_status = (
        "requires_looped_qwen35_runtime"
        if looped_runtime_required
        else "plain_qwen35_compatible"
    )
    turboquant_model_family = (
        "ELT/Qwen3.5-looped"
        if looped_runtime_required
        else cfg.source_model_id or cfg.hf_model_path
    )
    elt_config = {
        "schema": "elt.looped_qwen35.v1",
        "backbone_kind": cfg.backbone_kind,
        "source_model_id": cfg.source_model_id or cfg.hf_model_path,
        "L_min": cfg.L_min,
        "L_max": cfg.L_max,
        "L_default": cfg.L_max,
        "loop_unit": loop_unit,
        "looped_runtime_required": looped_runtime_required,
        "gguf_architecture": "qwen35",
        "gguf_runtime_status": gguf_runtime_status,
        "turboquant_model_family": turboquant_model_family,
        "loop_bootstrap_L_max": cfg.loop_bootstrap_L_max,
        "adapter_format": state.get("adapter_format", ""),
        "merged_lora": bool(merged_modules),
        "hf_lora_rank": cfg.hf_lora_rank,
        "hf_lora_alpha": cfg.hf_lora_alpha,
        "hf_lora_target_modules": cfg.hf_lora_target_modules,
    }
    text_cfg.update({"elt_config": elt_config})
    text_cfg.save_pretrained(out)
    tokenizer_ready = _copy_tokenizer(tokenizer_path, out, required=require_tokenizer)

    metadata = {
        "format": "elt_qwen35_merged_hf_v1",
        "repo_id": repo_id,
        "source_model_id": cfg.source_model_id or cfg.hf_model_path,
        "source_checkpoint": str(ckpt_path),
        "base_checkpoint": base_checkpoint or str(state.get("base_checkpoint") or ""),
        "out_dir": str(out),
        "model_safetensors": str(out / "model.safetensors"),
        "tokenizer_ready": tokenizer_ready,
        "dtype": dtype_name,
        "num_tensors": len(merged_state),
        "num_parameters": int(sum(t.numel() for t in merged_state.values())),
        "merged_lora_modules": len(merged_modules),
        "L_min": cfg.L_min,
        "L_max": cfg.L_max,
        "L_default": cfg.L_max,
        "loop_unit": loop_unit,
        "looped_runtime_required": looped_runtime_required,
        "gguf_runtime_ready": not looped_runtime_required,
        "gguf_runtime_status": gguf_runtime_status,
        "turboquant_model_family": turboquant_model_family,
        "llama_cpp_command": f"python convert_hf_to_gguf.py {out} --outfile <out.gguf>",
    }
    (out / "elt_export_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(out, metadata)
    return metadata


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="adapter-only or full hf_qwen35_looped checkpoint")
    parser.add_argument("--out-dir", required=True, help="HF output directory")
    parser.add_argument("--tokenizer", default="H:/Qwen3.5-9B-official-hf", help="tokenizer path or repo id")
    parser.add_argument("--repo-id", default="", help="target Hugging Face repo id for metadata")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--allow-missing-tokenizer", action="store_true")
    args = parser.parse_args()

    manifest = export_merged_qwen35_hf(
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        tokenizer_path=args.tokenizer,
        repo_id=args.repo_id,
        dtype_name=args.dtype,
        require_tokenizer=not args.allow_missing_tokenizer,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
