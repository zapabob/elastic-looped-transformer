"""Build GRPO prompts JSONL from cleaned GSM8K.

Input : H:/elt_data/raw/gsm8k.jsonl (schema: {"text", "source"} where text is
        "question ... #### gold_answer")
Output: H:/elt_data/grpo/gsm8k_prompts.jsonl
        Each line: {"prompt": "...<think-format instruction>...",
                    "reference": "#### <gold>",
                    "task": "gsm8k"}

The prompt instructs the policy to emit a `<think>...</think><answer>...</answer>`
block so that verifiers.format_score can gate credit; the reference is the raw
`#### N` line used by verifiers.gsm8k_correctness.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROMPT_TEMPLATE = (
    "Solve the following math word problem. Show your reasoning inside "
    "<think>...</think>, then give the final numeric answer inside "
    "<answer>...</answer>.\n\n"
    "Problem:\n{question}\n"
)

_GSM_SPLIT = re.compile(r"####\s*")


def convert_one(text: str) -> tuple[str, str] | None:
    parts = _GSM_SPLIT.split(text, maxsplit=1)
    if len(parts) != 2:
        return None
    question = parts[0].strip()
    gold = parts[1].strip().splitlines()[0].strip()
    if not question or not gold:
        return None
    prompt = PROMPT_TEMPLATE.format(question=question)
    reference = f"#### {gold}"
    return prompt, reference


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="H:/elt_data/raw/gsm8k.jsonl")
    p.add_argument("--out", dest="out", default="H:/elt_data/grpo/gsm8k_prompts.jsonl")
    p.add_argument("--limit", type=int, default=0, help="cap; 0 = all")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    with open(args.inp, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or ""
            got = convert_one(text)
            if got is None:
                continue
            prompt, reference = got
            fout.write(json.dumps(
                {"prompt": prompt, "reference": reference, "task": "gsm8k"},
                ensure_ascii=False,
            ) + "\n")
            n_out += 1
            if args.limit and n_out >= args.limit:
                break
    print(f"  {args.inp}  rows_in={n_in:,}  rows_out={n_out:,}  -> {out_path}")


if __name__ == "__main__":
    main()
