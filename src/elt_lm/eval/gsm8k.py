"""GSM8K chain-of-thought eval at a specific loop count L.

Loads test JSONL (schema: {"question": "...", "answer": "... #### 42"}), builds
an 8-shot CoT prompt, generates answers, extracts the final number after "####",
and reports exact-match accuracy.

Usage:
    uv run python -m elt_lm.eval.gsm8k --ckpt runs/base_100M/last.pt \
        --test H:/from_D/dataset/gsm8k/test.jsonl --L 4 --n 200
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from elt_lm.config import TrainConfig
from elt_lm.model import ELTLanguageModel

FEW_SHOT = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
     "clips in May. How many clips did Natalia sell altogether in April and May?",
     "In April she sold 48 clips. In May she sold 48/2 = 24 clips. "
     "Total = 48 + 24 = 72 clips. #### 72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
     "How much did she earn?",
     "50 minutes = 50/60 = 5/6 hour. She earns 12 * 5/6 = $10. #### 10"),
]

ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def build_prompt(q: str) -> str:
    parts = []
    for qi, ai in FEW_SHOT:
        parts.append(f"Question: {qi}\nAnswer: {ai}")
    parts.append(f"Question: {q}\nAnswer:")
    return "\n\n".join(parts)


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1)
    # fallback: last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg: TrainConfig = ckpt["cfg"]
    model = ELTLanguageModel(cfg.model).to(device=device).eval()
    model.load_state_dict(ckpt["model"])
    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_path, use_fast=True)

    test_items: list[dict] = []
    with open(args.test, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_items.append(json.loads(line))
    if args.n > 0:
        test_items = test_items[: args.n]

    correct = 0
    for item in tqdm(test_items, desc=f"gsm8k L={args.L}"):
        prompt = build_prompt(item["question"])
        ids = tok.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out_ids = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, L=args.L,
            temperature=0.0 if args.greedy else args.temperature,
            top_k=1 if args.greedy else args.top_k,
            eos_token_id=tok.eos_token_id,
        )
        gen_text = tok.decode(out_ids[0, input_ids.size(1):].tolist(), skip_special_tokens=True)
        pred = extract_answer(gen_text)
        gold = extract_answer(item["answer"])
        if pred is not None and gold is not None and pred.strip() == gold.strip():
            correct += 1

    acc = correct / max(1, len(test_items))
    print(f"GSM8K accuracy @ L={args.L}: {acc:.4f}  ({correct}/{len(test_items)})")

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps({"L": args.L, "accuracy": acc, "n": len(test_items)}, indent=2),
            encoding="utf-8",
        )


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test", required=True, help="GSM8K test.jsonl path")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n", type=int, default=0, help="limit number of examples (0=all)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    cli()
