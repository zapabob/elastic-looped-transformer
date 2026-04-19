"""Tokenize a raw-text corpus into a packed uint32 token stream on disk.

Usage:
    uv run elt-tokenize \
        --tokenizer H:/Qwen3.5-9B-official-hf \
        --input  path/to/text/dir \
        --output data_bin/train.bin \
        --ext .txt .md .jsonl

For .jsonl we assume each line is {"text": "..."} (fallback: str(obj)).

Format: a flat uint32 binary stream (little-endian). Nothing else is needed —
the dataset class mmaps this file and slices windows of `seq_len+1`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def iter_texts(paths: Iterable[Path]) -> Iterator[str]:
    for p in paths:
        suffix = p.suffix.lower()
        try:
            if suffix == ".jsonl":
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict) and "text" in obj:
                                yield str(obj["text"])
                            else:
                                yield str(obj)
                        except json.JSONDecodeError:
                            yield line
            else:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    yield f.read()
        except OSError as e:
            print(f"[warn] skip {p}: {e}", file=sys.stderr)


def gather_files(roots: list[Path], exts: list[str]) -> list[Path]:
    exts_norm = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.suffix.lower() in exts_norm:
                out.append(root)
            continue
        for sub in root.rglob("*"):
            if sub.is_file() and sub.suffix.lower() in exts_norm:
                out.append(sub)
    return sorted(out)


def tokenize_to_bin(
    tokenizer_path: str,
    inputs: list[Path],
    output: Path,
    exts: list[str],
    append_eos: bool = True,
    chunk_chars: int = 1_000_000,
) -> int:
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    eos_id = tok.eos_token_id

    files = gather_files(inputs, exts)
    if not files:
        raise SystemExit(f"no files matched under {inputs} with exts {exts}")

    output.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0

    with open(output, "wb") as out_f:
        for text in tqdm(iter_texts(files), total=len(files), desc="tokenize"):
            # Chunk very large documents to keep tokenizer memory bounded.
            for start in range(0, len(text), chunk_chars):
                piece = text[start: start + chunk_chars]
                if not piece:
                    continue
                ids = tok.encode(piece, add_special_tokens=False)
                if append_eos and start + chunk_chars >= len(text) and eos_id is not None:
                    ids.append(eos_id)
                if not ids:
                    continue
                arr = np.asarray(ids, dtype=np.uint32)
                # sanity: vocab must fit in uint32 — Qwen3.5 is 248,320, fine.
                out_f.write(arr.tobytes())
                total_tokens += arr.size

    size_gb = output.stat().st_size / (1024**3)
    print(f"wrote {total_tokens:,} tokens -> {output} ({size_gb:.2f} GB)")
    return total_tokens


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="path to HF tokenizer dir")
    p.add_argument("--input", required=True, nargs="+", help="input files or dirs")
    p.add_argument("--output", required=True)
    p.add_argument("--ext", nargs="+", default=[".txt", ".md", ".jsonl"])
    p.add_argument("--no-eos", action="store_true")
    args = p.parse_args()

    tokenize_to_bin(
        tokenizer_path=args.tokenizer,
        inputs=[Path(x) for x in args.input],
        output=Path(args.output),
        exts=args.ext,
        append_eos=not args.no_eos,
    )


if __name__ == "__main__":
    cli()

# Suppress unused-import warnings for attrs used by CLI entry point only.
_ = os
