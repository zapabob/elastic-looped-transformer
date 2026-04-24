from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from elt_lm.config import TrainConfig, load_train_config
from elt_lm.hf_qwen35_looped import HFQwen35LoopedLM, sync_model_config_from_qwen35, load_qwen35_text_config
from elt_lm.model import build_model


def build_bootstrap_train_config(
    *,
    hf_model_path: str,
    tokenizer_path: str,
    source_model_id: str = "",
    template_config: str | None = None,
    loop_bootstrap_L_max: int = 1,
) -> TrainConfig:
    cfg = load_train_config(template_config) if template_config else TrainConfig()
    text_cfg = load_qwen35_text_config(hf_model_path)
    model_cfg = replace(cfg.model)
    model_cfg.backbone_kind = "hf_qwen35_looped"
    model_cfg.hf_model_path = hf_model_path
    model_cfg.source_model_id = source_model_id or hf_model_path
    model_cfg.language_only = True
    model_cfg.freeze_vision = True
    model_cfg.import_lm_head = True
    model_cfg.L_min = 1
    model_cfg.L_max = loop_bootstrap_L_max
    model_cfg.loop_bootstrap_L_max = loop_bootstrap_L_max
    sync_model_config_from_qwen35(model_cfg, text_cfg)
    cfg.model = model_cfg
    cfg.data = replace(cfg.data, tokenizer_path=tokenizer_path)
    return cfg


@torch.no_grad()
def compare_l1_parity(
    *,
    model: HFQwen35LoopedLM,
    source_model_path: str,
    tokenizer_path: str,
    prompt: str,
) -> dict[str, float | bool]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    source = AutoModelForCausalLM.from_pretrained(
        source_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    model = model.to(dtype=torch.float32).eval()

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    ref = source(input_ids=input_ids, use_cache=False).logits
    got = model(input_ids=input_ids, L=1).logits
    diff = (ref - got).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "allclose_atol_1e-5": bool(torch.allclose(ref, got, atol=1e-5, rtol=1e-5)),
    }


def bootstrap_qwen35_elt_checkpoint(
    *,
    hf_model_path: str,
    out_path: str | Path,
    tokenizer_path: str,
    source_model_id: str = "",
    template_config: str | None = None,
    loop_bootstrap_L_max: int = 1,
    parity_prompt: str | None = None,
) -> tuple[Path, dict[str, float | bool] | None]:
    cfg = build_bootstrap_train_config(
        hf_model_path=hf_model_path,
        tokenizer_path=tokenizer_path,
        source_model_id=source_model_id,
        template_config=template_config,
        loop_bootstrap_L_max=loop_bootstrap_L_max,
    )
    model = build_model(cfg.model)
    if not isinstance(model, HFQwen35LoopedLM):
        raise TypeError(f"expected HFQwen35LoopedLM, got {type(model)!r}")
    model.load_pretrained_from_source()

    parity = None
    if parity_prompt:
        parity = compare_l1_parity(
            model=model,
            source_model_path=hf_model_path,
            tokenizer_path=tokenizer_path,
            prompt=parity_prompt,
        )

    payload = {
        "step": 0,
        "model": model.state_dict(),
        "cfg": cfg,
        "wall_time": time.time(),
        "source_model_id": cfg.model.source_model_id,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out, parity


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="HF repo id or local model dir")
    p.add_argument("--out", required=True, help="output .pt checkpoint path")
    p.add_argument("--tokenizer", default="H:/Qwen3.5-9B-official-hf")
    p.add_argument("--source-model-id", default="")
    p.add_argument("--template-config", default="")
    p.add_argument("--loop-bootstrap-L-max", type=int, default=1)
    p.add_argument("--parity-prompt", default="")
    args = p.parse_args()

    out, parity = bootstrap_qwen35_elt_checkpoint(
        hf_model_path=args.hf_model,
        out_path=args.out,
        tokenizer_path=args.tokenizer,
        source_model_id=args.source_model_id,
        template_config=args.template_config or None,
        loop_bootstrap_L_max=args.loop_bootstrap_L_max,
        parity_prompt=args.parity_prompt or None,
    )
    result = {"checkpoint": str(out), "parity": parity}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
