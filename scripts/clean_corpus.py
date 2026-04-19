"""Clean and normalize the raw corpus before tokenization.

Pipeline per document:
  1. Decode + type-specific extraction (flat, aegis_reasoning, aegis_sft, txt).
  2. **NFKC unicode normalization** — critical for Japanese full-width/half-width
     consistency and for canonicalizing compatibility characters.
  3. Whitespace collapse (runs of 3+ newlines → 2, tabs → space, strip).
  4. Length filter: 64 ≤ len(text) ≤ 500_000 characters.
  5. Quality filter:
        - (alnum + CJK) ratio ≥ 0.4         (reject mostly-symbol / mostly-empty)
        - no single-char share ≥ 0.3        (reject "AAAAA..." pathologies)
        - no line repeated ≥ 30 times       (reject degenerate loops)
  6. **Dedup** via SHA1 of the first 512 normalized chars — shared hash set
     across all sources, so Wikipedia/CC/aegis overlap gets collapsed.

Output: `H:/elt_data/clean/<name>.jsonl` with flat schema `{"text": ..., "source": ...}`.

Usage:
    uv run python scripts/clean_corpus.py \
        --config scripts/corpus_manifest.yaml \
        --out H:/elt_data/clean
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml
from tqdm import tqdm


MIN_CHARS = 64
MAX_CHARS = 500_000
MIN_ALNUM_CJK_RATIO = 0.4
MAX_SINGLE_CHAR_RATIO = 0.3
MAX_LINE_REPEAT = 30
DEDUP_HASH_PREFIX = 512

_ws_runs = re.compile(r"\n{3,}")
_tab = re.compile(r"[\t\u00a0]+")
_cjk_re = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff66-\uff9f]"
)


@dataclass
class CleanStats:
    seen: int = 0
    kept: int = 0
    drop_short: int = 0
    drop_long: int = 0
    drop_ratio: int = 0
    drop_singlechar: int = 0
    drop_linerepeat: int = 0
    drop_dup: int = 0
    drop_parse: int = 0

    def line(self) -> str:
        return (
            f"seen={self.seen:,} kept={self.kept:,} "
            f"short={self.drop_short:,} long={self.drop_long:,} "
            f"ratio={self.drop_ratio:,} single={self.drop_singlechar:,} "
            f"linerep={self.drop_linerepeat:,} dup={self.drop_dup:,} "
            f"parse={self.drop_parse:,}"
        )


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _tab.sub(" ", text)
    text = _ws_runs.sub("\n\n", text)
    return text.strip()


def quality_ok(text: str, stats: CleanStats) -> bool:
    n = len(text)
    if n < MIN_CHARS:
        stats.drop_short += 1
        return False
    if n > MAX_CHARS:
        stats.drop_long += 1
        return False

    alnum = sum(c.isalnum() for c in text)
    cjk = len(_cjk_re.findall(text))
    if (alnum + cjk) / n < MIN_ALNUM_CJK_RATIO:
        stats.drop_ratio += 1
        return False

    counts = Counter(text)
    top_share = counts.most_common(1)[0][1] / n
    if top_share >= MAX_SINGLE_CHAR_RATIO:
        stats.drop_singlechar += 1
        return False

    # repeated-line check (catches "okokok..." style loops after whitespace split)
    line_counts: Counter[str] = Counter()
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 4:
            continue
        line_counts[line] += 1
        if line_counts[line] >= MAX_LINE_REPEAT:
            stats.drop_linerepeat += 1
            return False

    return True


def doc_hash(text: str) -> bytes:
    return hashlib.sha1(text[:DEDUP_HASH_PREFIX].encode("utf-8")).digest()


def _parse_aegis_sft_instruction(raw: str) -> str | None:
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
    prob = obj.get("problem") or obj.get("question") or ""
    sol = obj.get("solution") or obj.get("answer") or ""
    if prob and sol:
        return f"{prob}\n\n{sol}"
    return None


def iter_raw_texts(path: Path, type_: str, stats: CleanStats) -> Iterator[str]:
    if type_ == "txt":
        try:
            yield path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            print(f"[warn] {path}: {e}", file=sys.stderr)
        return
    if not path.exists():
        print(f"[warn] missing: {path}", file=sys.stderr)
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats.drop_parse += 1
                continue
            if type_ in ("flat", "aegis_reasoning"):
                text = obj.get("text") if isinstance(obj, dict) else None
            elif type_ == "aegis_sft":
                raw = obj.get("instruction") if isinstance(obj, dict) else None
                text = _parse_aegis_sft_instruction(raw) if isinstance(raw, str) else None
            else:
                text = None
            if text:
                yield text
            else:
                stats.drop_parse += 1


def expand_source_files(entry: dict) -> list[tuple[Path, str]]:
    path = Path(entry["path"])
    typ = entry["type"]
    if path.is_dir():
        if typ == "txt":
            return [(p, "txt") for p in sorted(path.rglob("*.txt"))]
        return [(p, typ) for p in sorted(path.rglob("*.jsonl"))]
    return [(path, typ)]


def process(manifest: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[bytes] = set()
    # group outputs by top-level manifest name (derived from path stem or dir name)
    for entry in manifest.get("sources", []):
        ep = Path(entry["path"])
        group = entry.get("name") or (ep.stem if ep.is_file() else ep.name)
        stats = CleanStats()
        out_path = out_dir / f"{group}.jsonl"
        files = expand_source_files(entry)
        if not files:
            print(f"[warn] {group}: no files expanded", file=sys.stderr)
            continue
        print(f"[{group}] {len(files)} file(s) -> {out_path}")
        with open(out_path, "w", encoding="utf-8", errors="ignore") as out_f:
            for path, typ in files:
                pbar = tqdm(
                    desc=f"{group}:{path.name}", unit="doc", unit_scale=True
                )
                for raw in iter_raw_texts(path, typ, stats):
                    stats.seen += 1
                    pbar.update(1)
                    norm = normalize(raw)
                    if not quality_ok(norm, stats):
                        continue
                    h = doc_hash(norm)
                    if h in seen_hashes:
                        stats.drop_dup += 1
                        continue
                    seen_hashes.add(h)
                    line = json.dumps(
                        {"text": norm, "source": group}, ensure_ascii=False
                    )
                    out_f.write(line + "\n")
                    stats.kept += 1
                pbar.close()
        print(f"[{group}] {stats.line()}")
    print(f"total unique docs: {len(seen_hashes):,}")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    process(manifest, Path(args.out))


if __name__ == "__main__":
    cli()
