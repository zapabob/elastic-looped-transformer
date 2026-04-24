"""Audit cleaned JSONL corpus files for duplicate and quality issues.

This is intentionally non-destructive. It scans `H:/elt_data/clean` by default
and writes a compact JSON/Markdown report that can guide later dedup or
source-weight changes.

Usage:
    uv run python scripts/audit_clean_corpus.py \
        --clean-dir H:/elt_data/clean \
        --report-json runs/dashboard_runtime/clean_corpus_audit_2026-04-24.json \
        --report-md _docs/2026-04-24-clean-corpus-audit-gpt-5.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


MIN_CHARS = 64
MAX_CHARS = 500_000
MIN_ALNUM_CJK_RATIO = 0.4
MAX_SINGLE_CHAR_RATIO = 0.3
MAX_LINE_REPEAT = 30
DEDUP_HASH_PREFIX = 512
SIMHASH_BITS = 64

_ws_runs = re.compile(r"\n{3,}")
_tab = re.compile(r"[\t\u00a0]+")
_cjk_re = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff66-\uff9f]"
)
_feature_re = re.compile(
    r"[A-Za-z0-9_]{2,}|[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff66-\uff9f]"
)


@dataclass
class FileAudit:
    path: str
    bytes: int
    docs: int = 0
    parse_errors: int = 0
    empty_text: int = 0
    low_quality: int = 0
    exact_duplicates: int = 0
    prefix_duplicates: int = 0
    simhash_duplicates: int = 0
    avg_chars: float = 0.0
    quality_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class CorpusAudit:
    clean_dir: str
    files: int
    zero_byte_files: list[str]
    total_bytes: int = 0
    total_docs: int = 0
    parse_errors: int = 0
    empty_text: int = 0
    low_quality: int = 0
    exact_duplicates: int = 0
    prefix_duplicates: int = 0
    simhash_duplicates: int = 0
    top_quality_reasons: dict[str, int] = field(default_factory=dict)
    top_duplicate_prefixes: list[dict[str, Any]] = field(default_factory=list)
    files_by_issue_rate: list[FileAudit] = field(default_factory=list)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _tab.sub(" ", text)
    text = _ws_runs.sub("\n\n", text)
    return text.strip()


def quality_reason(text: str) -> str | None:
    n = len(text)
    if n < MIN_CHARS:
        return "short"
    if n > MAX_CHARS:
        return "long"

    alnum = sum(c.isalnum() for c in text)
    cjk = len(_cjk_re.findall(text))
    if (alnum + cjk) / n < MIN_ALNUM_CJK_RATIO:
        return "low_alnum_cjk_ratio"

    counts = Counter(text)
    if counts and counts.most_common(1)[0][1] / n >= MAX_SINGLE_CHAR_RATIO:
        return "single_char_dominant"

    line_counts: Counter[str] = Counter()
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 4:
            continue
        line_counts[line] += 1
        if line_counts[line] >= MAX_LINE_REPEAT:
            return "line_repeat"
    return None


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def exact_hash(text: str) -> str:
    return _sha1_hex(text)


def prefix_hash(text: str) -> str:
    return _sha1_hex(text[:DEDUP_HASH_PREFIX])


def simhash64(text: str, *, max_features: int = 512) -> int:
    """Return a lightweight high-confidence duplicate fingerprint.

    Equal simhashes are counted as duplicates. We do not use this as a
    near-neighbor semantic dedup decision; it is only a cheap audit signal.
    """
    features = _feature_re.findall(text.lower())
    if not features:
        return 0
    weights = [0] * SIMHASH_BITS
    for feature in features[:max_features]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(SIMHASH_BITS):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    out = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            out |= 1 << bit
    return out


def _extract_text(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    text = obj.get("text")
    if isinstance(text, str):
        return text
    prompt = obj.get("prompt")
    response = obj.get("response")
    if isinstance(prompt, str) and isinstance(response, str):
        return f"{prompt}\n\n{response}"
    return None


def _iter_jsonl(path: Path) -> Iterable[tuple[str | None, bool]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield None, False
                continue
            yield _extract_text(obj), True


def audit_clean_dir(
    clean_dir: Path,
    *,
    max_docs_per_file: int | None = None,
    enable_simhash: bool = False,
) -> CorpusAudit:
    files = sorted(clean_dir.glob("*.jsonl"))
    zero_byte = [str(p) for p in files if p.stat().st_size == 0]
    total_bytes = sum(p.stat().st_size for p in files)

    seen_exact: set[str] = set()
    seen_prefix: set[str] = set()
    seen_simhash: set[int] = set()
    prefix_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    file_audits: list[FileAudit] = []

    for path in files:
        fa = FileAudit(path=str(path), bytes=path.stat().st_size)
        chars = 0
        for idx, (raw_text, parsed) in enumerate(_iter_jsonl(path)):
            if max_docs_per_file is not None and idx >= max_docs_per_file:
                break
            if not parsed:
                fa.parse_errors += 1
                continue
            if not raw_text:
                fa.empty_text += 1
                continue
            text = normalize(raw_text)
            if not text:
                fa.empty_text += 1
                continue
            fa.docs += 1
            chars += len(text)

            reason = quality_reason(text)
            if reason is not None:
                fa.low_quality += 1
                fa.quality_reasons[reason] = fa.quality_reasons.get(reason, 0) + 1
                quality_counts[reason] += 1

            eh = exact_hash(text)
            ph = prefix_hash(text)
            prefix_counts[ph] += 1
            if eh in seen_exact:
                fa.exact_duplicates += 1
            else:
                seen_exact.add(eh)
            if ph in seen_prefix:
                fa.prefix_duplicates += 1
            else:
                seen_prefix.add(ph)
            if enable_simhash:
                sh = simhash64(text)
                if sh in seen_simhash:
                    fa.simhash_duplicates += 1
                else:
                    seen_simhash.add(sh)

        fa.avg_chars = round(chars / fa.docs, 3) if fa.docs else 0.0
        file_audits.append(fa)

    def issue_rate(fa: FileAudit) -> float:
        if fa.docs <= 0:
            return 1.0 if fa.bytes == 0 else 0.0
        issues = (
            fa.parse_errors
            + fa.empty_text
            + fa.low_quality
            + fa.prefix_duplicates
            + fa.simhash_duplicates
        )
        return issues / fa.docs

    top_prefixes = [
        {"prefix_hash": key, "count": count}
        for key, count in prefix_counts.most_common(20)
        if count > 1
    ]
    sorted_files = sorted(file_audits, key=issue_rate, reverse=True)[:20]

    return CorpusAudit(
        clean_dir=str(clean_dir),
        files=len(files),
        zero_byte_files=zero_byte,
        total_bytes=total_bytes,
        total_docs=sum(f.docs for f in file_audits),
        parse_errors=sum(f.parse_errors for f in file_audits),
        empty_text=sum(f.empty_text for f in file_audits),
        low_quality=sum(f.low_quality for f in file_audits),
        exact_duplicates=sum(f.exact_duplicates for f in file_audits),
        prefix_duplicates=sum(f.prefix_duplicates for f in file_audits),
        simhash_duplicates=sum(f.simhash_duplicates for f in file_audits),
        top_quality_reasons=dict(quality_counts.most_common()),
        top_duplicate_prefixes=top_prefixes,
        files_by_issue_rate=sorted_files,
    )


def write_json(report: CorpusAudit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_markdown(report: CorpusAudit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    issue_rows = "\n".join(
        "| {name} | {docs:,} | {low:,} | {pref:,} | {sim:,} | {parse:,} |".format(
            name=Path(f.path).name,
            docs=f.docs,
            low=f.low_quality,
            pref=f.prefix_duplicates,
            sim=f.simhash_duplicates,
            parse=f.parse_errors,
        )
        for f in report.files_by_issue_rate[:12]
    )
    zero_list = "\n".join(f"- `{Path(p).name}`" for p in report.zero_byte_files)
    if not zero_list:
        zero_list = "- none"
    reasons = "\n".join(
        f"- `{k}`: {v:,}" for k, v in report.top_quality_reasons.items()
    )
    if not reasons:
        reasons = "- none"

    body = f"""---
date: 2026-04-24
slug: clean-corpus-audit
ai: gpt-5
---

# Clean Corpus Audit

## Goal

Recheck `H:/elt_data/clean` for duplicate and low-quality residue before the next
native ELT continued pretraining and Huihui/Qwen lane-distill stages.

## Summary

- files: {report.files:,}
- zero-byte files: {len(report.zero_byte_files):,}
- bytes: {report.total_bytes:,}
- docs scanned: {report.total_docs:,}
- parse errors: {report.parse_errors:,}
- empty text records: {report.empty_text:,}
- low-quality records: {report.low_quality:,}
- exact duplicate records: {report.exact_duplicates:,}
- prefix duplicate records: {report.prefix_duplicates:,}
- simhash duplicate records: {report.simhash_duplicates:,}

## Top Quality Reasons

{reasons}

## Files Touched

- `scripts/audit_clean_corpus.py`
- `tests/test_audit_clean_corpus.py`
- `runs/dashboard_runtime/clean_corpus_audit_2026-04-24.json`
- `_docs/2026-04-24-clean-corpus-audit-gpt-5.md`

## Tests / Commands

```powershell
uv run --no-sync pytest -q tests/test_audit_clean_corpus.py
uv run --no-sync pyright tests/test_audit_clean_corpus.py
uv run --no-sync python scripts/audit_clean_corpus.py --clean-dir H:/elt_data/clean --report-json runs/dashboard_runtime/clean_corpus_audit_2026-04-24.json --report-md _docs/2026-04-24-clean-corpus-audit-gpt-5.md
```

## Zero-Byte Files

{zero_list}

## Files With Highest Issue Rates

| file | docs | low_quality | prefix_dups | simhash_dups | parse_errors |
|---|---:|---:|---:|---:|---:|
{issue_rows}

## Decisions

- This pass is audit-only and does not delete or rewrite corpus files.
- `prefix_dups` uses the same first-512 normalized-character idea as
  `scripts/clean_corpus.py`, so it is compatible with the existing clean stage.
- `simhash_dups` is optional because it is slower on the full corpus. When
  enabled, it is a high-confidence audit signal only and should guide a later
  semantic dedup stage, not silently remove records by itself.

## Next Session Notes

1. Remove zero-byte manifest outputs from packing or regenerate their sources.
2. Rebuild a filtered clean directory only after reviewing this report.
3. Then rebuild `H:/elt_data/bin/{{train,val}}.bin` and rerun token inventory.
"""
    path.write_text(body, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-dir", type=Path, default=Path("H:/elt_data/clean"))
    p.add_argument(
        "--report-json",
        type=Path,
        default=Path("runs/dashboard_runtime/clean_corpus_audit_2026-04-24.json"),
    )
    p.add_argument(
        "--report-md",
        type=Path,
        default=Path("_docs/2026-04-24-clean-corpus-audit-gpt-5.md"),
    )
    p.add_argument("--max-docs-per-file", type=int, default=None)
    p.add_argument(
        "--enable-simhash",
        action="store_true",
        help="Also compute lightweight simhash duplicate counts. Slower.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_clean_dir(
        args.clean_dir,
        max_docs_per_file=args.max_docs_per_file,
        enable_simhash=args.enable_simhash,
    )
    write_json(report, args.report_json)
    write_markdown(report, args.report_md)
    print(f"wrote {args.report_json}")
    print(f"wrote {args.report_md}")
    print(
        "docs={docs:,} low_quality={low:,} prefix_dups={dups:,} zero_files={zero:,}".format(
            docs=report.total_docs,
            low=report.low_quality,
            dups=report.prefix_duplicates,
            zero=len(report.zero_byte_files),
        )
    )


if __name__ == "__main__":
    main()
