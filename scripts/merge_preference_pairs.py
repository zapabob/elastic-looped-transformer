"""Merge normalized preference-pair JSONL files into one training file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--inputs", nargs="+", required=True)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for item in args.inputs:
            with open(item, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "chosen_text" not in row or "rejected_text" not in row:
                        continue
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
    print(f"merged {written:,} preference pairs -> {out_path}")


if __name__ == "__main__":
    cli()
