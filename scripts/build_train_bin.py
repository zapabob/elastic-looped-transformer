"""Build the tokenized packed bin files that `PackedTokenDataset` consumes.

Sources we understand (auto-detected by filename or --source-type):
  - "flat"   : JSONL with {"text": ...}   (our HF downloader + wikipedia dumps)
  - "aegis_reasoning"
             : aegis reasoning schema — {"id","text","category",...}
  - "aegis_sft"
             : aegis SFT schema — {"instruction": "<str of dict with 'messages'>"}
  - "txt"    : a single plain-text file (utf-8)

The aggregator writes one uint32 stream to H:/elt_data/bin/train.bin and
val.bin. EOS is appended between documents so the model sees explicit
document boundaries. Manifest entries may also specify `weight`:

  - `weight = 1.0` keeps every document once
  - `0 < weight < 1` downsamples deterministically
  - `weight > 1` repeats documents deterministically

Usage:
    uv run python scripts/build_train_bin.py \
        --tokenizer H:/Qwen3.5-9B-official-hf \
        --out-dir   H:/elt_data/bin \
        --config    scripts/corpus_manifest.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class Source:
    path: Path
    type: str
    weight: float = 1.0
    max_docs: int | None = None


def iter_weighted_texts(texts: Iterable[str], weight: float) -> Iterator[str]:
    """Scale a source's contribution deterministically.

    `weight=1.0` keeps every document once.
    `0 < weight < 1` keeps a stable subset.
    `weight > 1` repeats documents in a stable spread.

    We use a Bresenham-style accumulator instead of RNG so repeated corpus
    builds are bit-for-bit reproducible for the same manifest order.
    """
    if weight < 0:
        raise ValueError(f"source weight must be >= 0, got {weight}")
    if weight == 0:
        return

    target_emissions = 0.0
    emitted = 0
    for text in texts:
        target_emissions += weight
        copies = int(target_emissions) - emitted
        emitted += copies
        for _ in range(copies):
            yield text


def _parse_aegis_sft_instruction(raw: str) -> str | None:
    """aegis_v21_sft_50k_final.jsonl has `instruction` = str(dict) with a
    `messages` list [{role, content}, ...]. Recover the conversation as plain
    text."""
    try:
        obj = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(obj, dict):
        return None
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if not content:
                continue
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts) if parts else None
    # fallback: problem + solution fields
    prob = obj.get("problem") or obj.get("question") or ""
    sol = obj.get("solution") or obj.get("answer") or ""
    if prob and sol:
        return f"{prob}\n\n{sol}"
    return None


def _iter_source_texts(src: Source) -> Iterator[str]:
    p = src.path
    t = src.type
    count = 0
    if t == "txt":
        try:
            yield p.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            print(f"[warn] {p}: {e}", file=sys.stderr)
        return
    if not p.exists():
        print(f"[warn] missing: {p}", file=sys.stderr)
        return
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t == "flat":
                text = obj.get("text") if isinstance(obj, dict) else None
            elif t == "aegis_reasoning":
                text = obj.get("text") if isinstance(obj, dict) else None
            elif t == "aegis_sft":
                raw = obj.get("instruction") if isinstance(obj, dict) else None
                text = _parse_aegis_sft_instruction(raw) if isinstance(raw, str) else None
            else:
                text = None
            if not text:
                continue
            text = text.strip()
            if len(text) < 16:
                continue
            yield text
            count += 1
            if src.max_docs is not None and count >= src.max_docs:
                break


def iter_source(src: Source) -> Iterator[str]:
    """Yield source texts after deterministic source-level weighting."""
    yield from iter_weighted_texts(_iter_source_texts(src), src.weight)


def expand_sources(manifest: dict) -> list[Source]:
    out: list[Source] = []
    for entry in manifest.get("sources", []):
        path = Path(entry["path"])
        typ = entry["type"]
        weight = float(entry.get("weight", 1.0))
        max_docs = entry.get("max_docs")
        if path.is_dir():
            # directory: glob according to type
            if typ == "txt":
                for f in sorted(path.rglob("*.txt")):
                    out.append(Source(f, "txt", weight, max_docs))
            else:
                for f in sorted(path.rglob("*.jsonl")):
                    out.append(Source(f, typ, weight, max_docs))
        else:
            out.append(Source(path, typ, weight, max_docs))
    return out


def tokenize_and_write(
    tokenizer_path: str,
    manifest: dict,
    out_dir: Path,
    val_fraction: float = 0.02,
    chunk_chars: int = 200_000,
) -> None:
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("tokenizer has no eos_token_id; aborting")

    sources = expand_sources(manifest)
    if not sources:
        raise SystemExit("manifest produced no sources")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "_all.bin.tmp"
    total = 0
    with open(tmp_path, "wb") as out_f:
        for src in sources:
            pbar = tqdm(desc=f"{src.type}:{src.path.name}", unit="tok", unit_scale=True)
            for doc in iter_source(src):
                # chunk long docs so the tokenizer stays within reasonable RAM
                for start in range(0, len(doc), chunk_chars):
                    piece = doc[start : start + chunk_chars]
                    if not piece:
                        continue
                    ids = tok.encode(piece, add_special_tokens=False)
                    if not ids:
                        continue
                    arr = np.asarray(ids, dtype=np.uint32)
                    out_f.write(arr.tobytes())
                    total += arr.size
                    pbar.update(arr.size)
                # EOS between docs
                out_f.write(np.asarray([eos], dtype=np.uint32).tobytes())
                total += 1
                pbar.update(1)
            pbar.close()

    # split into train / val by trailing slice (cheap, deterministic)
    val_tokens = int(total * val_fraction)
    train_tokens = total - val_tokens
    print(f"total tokens: {total:,}  train={train_tokens:,}  val={val_tokens:,}")

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    # stream copy to avoid loading the whole thing in RAM
    buf = 64 * 1024 * 1024  # 64MB
    with open(tmp_path, "rb") as src, open(train_path, "wb") as tf:
        remaining = train_tokens * 4
        while remaining > 0:
            chunk = src.read(min(buf, remaining))
            if not chunk:
                break
            tf.write(chunk)
            remaining -= len(chunk)
        with open(val_path, "wb") as vf:
            while True:
                chunk = src.read(buf)
                if not chunk:
                    break
                vf.write(chunk)
    tmp_path.unlink()
    print(f"wrote {train_path} ({train_path.stat().st_size/1e9:.2f}GB)")
    print(f"wrote {val_path}   ({val_path.stat().st_size/1e9:.2f}GB)")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", required=True, help="YAML manifest of sources")
    p.add_argument("--val-fraction", type=float, default=0.02)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    tokenize_and_write(
        tokenizer_path=args.tokenizer,
        manifest=manifest,
        out_dir=Path(args.out_dir),
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    cli()
