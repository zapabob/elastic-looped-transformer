"""Any-Time inference CLI: generate with a user-chosen loop count L."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from elt_lm.config import TrainConfig
from elt_lm.model import ELTLanguageModel


def load_checkpoint(ckpt_path: str | Path, device: torch.device) -> tuple[ELTLanguageModel, TrainConfig]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: TrainConfig = ckpt["cfg"]
    model = ELTLanguageModel(cfg.model).to(device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.ckpt, device)

    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_path, use_fast=True)
    prompt_ids = tok.encode(args.prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    L = args.L if args.L > 0 else cfg.model.L_max
    assert cfg.model.L_min <= L <= cfg.model.L_max, \
        f"L={L} outside training range [{cfg.model.L_min}, {cfg.model.L_max}]"

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
        out_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            L=L,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
    print(decoded)


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--L", type=int, default=0, help="loop count (0 => use L_max)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    cli()
