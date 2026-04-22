"""Normalize H:/from_D/webdataset/* heterogeneous sources into flat
H:/elt_data/raw/webdataset_*.jsonl (schema: {"text": "..."}), so the existing
corpus_manifest.yaml can consume them as `type: flat`.

Only high-value, license-clean sources are included. NSFW / drug / Nikkei225
copyrighted subtrees are explicitly skipped.

Run:

    uv run python scripts/ingest_webdataset.py \\
        --src H:/from_D/webdataset \\
        --out H:/elt_data/raw

Idempotent: re-runs overwrite. Missing sources are logged and skipped.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

MIN_CHARS = 32


def _write_jsonl(out_path: Path, texts: Iterable[str]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            t = (t or "").strip()
            if len(t) < MIN_CHARS:
                continue
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            n += 1
    return n


def _conversations_to_text(convs: list[dict]) -> str | None:
    """ShareGPT/Alpaca `conversations` → "User: ...\\nAssistant: ..." text."""
    parts: list[str] = []
    for turn in convs or []:
        frm = str(turn.get("from") or turn.get("role") or "").lower()
        val = str(turn.get("value") or turn.get("content") or "").strip()
        if not val:
            continue
        tag = {"human": "User", "user": "User",
               "gpt": "Assistant", "assistant": "Assistant",
               "system": "System"}.get(frm, frm.capitalize() or "User")
        parts.append(f"{tag}: {val}")
    return "\n\n".join(parts) if parts else None


def iter_json_array(path: Path, key: str = "conversations") -> Iterator[str]:
    """Big JSON array of {id, conversations:[...]}; we stream via ijson when avail."""
    try:
        import ijson  # type: ignore
    except ImportError:
        data = json.loads(path.read_text(encoding="utf-8"))
        for obj in data:
            t = _conversations_to_text(obj.get(key, []))
            if t: yield t
        return
    with open(path, "rb") as f:
        for obj in ijson.items(f, "item"):
            t = _conversations_to_text(obj.get(key, []))
            if t: yield t


def iter_jsonl_gz(path: Path, extractor: Callable[[dict], str | None]) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try: obj = json.loads(line)
            except json.JSONDecodeError: continue
            t = extractor(obj)
            if t: yield t


def iter_jsonl_zst(path: Path, extractor: Callable[[dict], str | None]) -> Iterator[str]:
    try:
        import zstandard as zstd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "zstandard is required to ingest .jsonl.zst sources. "
            "Install project dependencies with `uv sync`."
        ) from exc
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f, dctx.stream_reader(f) as r:
        buf = b""
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                if buf:
                    try:
                        obj = json.loads(buf)
                        t = extractor(obj)
                        if t: yield t
                    except json.JSONDecodeError: pass
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line: continue
                try: obj = json.loads(line)
                except json.JSONDecodeError: continue
                t = extractor(obj)
                if t: yield t


def iter_parquet(path: Path, extractor: Callable[[dict], str | None]) -> Iterator[str]:
    import pyarrow.parquet as pq  # type: ignore
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            t = extractor(row)
            if t: yield t


def iter_jsonl(path: Path, extractor: Callable[[dict], str | None]) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except json.JSONDecodeError: continue
            t = extractor(obj)
            if t: yield t


# --- Schema adapters ---------------------------------------------------------

def ex_text(o: dict) -> str | None:
    return o.get("text") or o.get("original_text")

def ex_instruction_output(o: dict) -> str | None:
    ins, out = o.get("instruction"), o.get("output") or o.get("response")
    if ins and out: return f"Q: {ins}\n\nA: {out}"
    return ins or out

def ex_oasst_message(o: dict) -> str | None:
    # oasst2 ready.messages schema: {"text","role","lang",...}
    role = o.get("role", "user")
    txt = o.get("text")
    if not txt: return None
    tag = {"prompter": "User", "assistant": "Assistant"}.get(role, role.capitalize())
    return f"{tag}: {txt}"

def ex_wizard_conv(o: dict) -> str | None:
    return _conversations_to_text(o.get("conversations", []))

def ex_fujiki(o: dict) -> str | None:
    # fujiki_japanese_hh-rlhf-49k parquet: {prompt, chosen, rejected} → use chosen
    p = o.get("prompt", "")
    c = o.get("chosen") or o.get("text")
    if p and c: return f"User: {p}\n\nAssistant: {c}"
    return c


# --- Registry ---------------------------------------------------------------

def build_sources(src_root: Path) -> list[tuple[str, Iterator[str]]]:
    """(out_stem, text_iterator) for each source we ingest."""
    out: list[tuple[str, Iterator[str]]] = []

    # 1. Freedom JP alpaca + sharegpt (big JSON arrays of conversations)
    p = src_root / "datasets/FreedomIntelligence_alpaca-gpt4-japanese/alpaca-gpt4-japanese.json"
    if p.exists(): out.append(("webdataset_alpaca_ja", iter_json_array(p)))
    p = src_root / "datasets/FreedomIntelligence_sharegpt-japanese/sharegpt-japanese.json"
    if p.exists(): out.append(("webdataset_sharegpt_ja", iter_json_array(p)))

    # 2. fujiki hh-rlhf JP (parquet)
    for p in (src_root / "datasets/fujiki_japanese_hh-rlhf-49k/data").glob("*.parquet"):
        out.append(("webdataset_fujiki_ja", iter_parquet(p, ex_fujiki)))

    # 3. OpenAssistant oasst2 (gz jsonl, "ready" split only — spam-filtered upstream)
    p = src_root / "datasets/OpenAssistant_oasst2/2023-11-05_oasst2_ready.messages.jsonl.gz"
    if p.exists(): out.append(("webdataset_oasst2", iter_jsonl_gz(p, ex_oasst_message)))

    # 4. Wizard-Vicuna 70k (EN instruction, big json)
    p = src_root / "datasets/ehartford_wizard_vicuna_70k_unfiltered/wizard_vicuna_dataset_unfiltered.json"
    if p.exists(): out.append(("webdataset_wizard_vicuna", iter_json_array(p)))

    # 5. SlimPajama 627B (EN pretrain, zstd jsonl)
    slim = src_root / "datasets/cerebras_SlimPajama-627B/test/chunk1"
    for p in sorted(slim.glob("*.jsonl.zst"))[:200]:
        out.append(("webdataset_slimpajama", iter_jsonl_zst(p, ex_text)))

    # 6. Local coding_dataset/*.jsonl — already flat {text}
    for p in sorted((src_root / "coding_dataset").glob("*.jsonl")):
        out.append(("webdataset_coding", iter_jsonl(p, ex_text)))

    # 7. Local coding_training_data/*.jsonl — {instruction, output}
    for p in sorted((src_root / "coding_training_data").glob("*.jsonl")):
        out.append(("webdataset_coding_train", iter_jsonl(p, ex_instruction_output)))

    # 8. domain_knowledge_collected/*_cleaned.jsonl — {instruction, output}
    for p in sorted((src_root / "domain_knowledge_collected").glob("*_cleaned.jsonl")):
        out.append(("webdataset_domain_knowledge", iter_jsonl(p, ex_instruction_output)))

    # 9. phi3.5 PPO-optimized integrated (large, already cleaned upstream)
    p = src_root / "phi35_integrated/phi35_ppo_optimized_integrated.jsonl"
    if p.exists(): out.append(("webdataset_phi35", iter_jsonl(p, ex_text)))

    return out


# --- Detection-only sources (labeled) -----------------------------------
# These feed a future safety classifier head / verifier, NOT the pretrain LM.
# Output schema: {"text": "...", "label": "...", "category": "...", ...}
# Kept in H:/elt_data/detection/ so nothing leaks into the pretrain bin.

def _write_labeled_jsonl(out_path: Path, records: Iterable[dict]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            text = (r.get("text") or "").strip()
            if len(text) < MIN_CHARS:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_nsfw_detection(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try: o = json.loads(line)
            except json.JSONDecodeError: continue
            yield {"text": o.get("text", ""),
                   "label": o.get("nsfw_label", "unknown"),
                   "category": o.get("category", ""),
                   "severity": o.get("nsfw_severity"),
                   "confidence": o.get("nsfw_confidence"),
                   "language": o.get("language"),
                   "source_dataset": "nsfw_detection"}


def iter_drug_detection(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try: o = json.loads(line)
            except json.JSONDecodeError: continue
            yield {"text": o.get("text", ""),
                   "label": o.get("category", "unknown"),
                   "language": o.get("language"),
                   "source_dataset": "drug_detection"}


def iter_qlora_classification(path: Path) -> Iterator[dict]:
    # 4-way classifier SFT data: instruction + input → output with four_class_label.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try: o = json.loads(line)
            except json.JSONDecodeError: continue
            body = (o.get("input") or "").strip()
            if not body:
                continue
            yield {"text": body,
                   "label": o.get("four_class_label", "unknown"),
                   "category": o.get("category", ""),
                   "nsfw_label": o.get("nsfw_label"),
                   "instruction": o.get("instruction"),
                   "target": o.get("output"),
                   "language": o.get("language"),
                   "source_dataset": "nsfw_drug_qlora"}


def iter_elizezen_nsfw(path: Path) -> Iterator[dict]:
    # Raw NSFW syosetsu text as negative-class examples for the detector.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] elizezen: {e}")
        return
    items = data if isinstance(data, list) else data.get("data", [])
    for o in items:
        txt = o.get("text") or o.get("content") or ""
        if not txt: continue
        yield {"text": txt, "label": "nsfw_raw",
               "category": "japanese_nsfw_fiction",
               "source_dataset": "elizezen_nsfw_syosetsu"}


def iter_eliasalbouzidi_parquet(path: Path) -> Iterator[dict]:
    import pyarrow.parquet as pq  # type: ignore
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            txt = row.get("text") or row.get("content") or ""
            if not txt: continue
            yield {"text": txt,
                   "label": str(row.get("label", "unknown")),
                   "source_dataset": "eliasalbouzidi_nsfw_safe"}


def build_detection_sources(src_root: Path) -> list[tuple[str, Iterator[dict]]]:
    out: list[tuple[str, Iterator[dict]]] = []

    # nsfw_detection_{train,val}.jsonl (labeled)
    for split in ("train", "val"):
        p = src_root / f"nsfw_detection_dataset/nsfw_detection_{split}.jsonl"
        if p.exists(): out.append((f"detection_nsfw_{split}", iter_nsfw_detection(p)))

    # drug_pharmaceutical_detection_{train,val}.jsonl
    for split in ("train", "val"):
        p = src_root / f"drug_pharmaceutical_detection_dataset/drug_pharmaceutical_detection_{split}.jsonl"
        if p.exists(): out.append((f"detection_drug_{split}", iter_drug_detection(p)))

    # nsfw_drug_detection_qlora_training_data/*.jsonl (4-class SFT)
    for p in sorted((src_root / "nsfw_drug_detection_qlora_training_data").glob("*.jsonl")):
        out.append(("detection_4class_qlora", iter_qlora_classification(p)))

    # Elizezen NSFW syosetsu (raw negative-class)
    p = src_root / "datasets/Elizezen_japanese-nsfw-syosetsu-dataset/nsfw_0.json"
    if p.exists(): out.append(("detection_elizezen_nsfw", iter_elizezen_nsfw(p)))

    # eliasalbouzidi NSFW-Safe parquet (EN classifier data)
    for p in sorted((src_root / "datasets/eliasalbouzidi_NSFW-Safe-Dataset/data").glob("train-*.parquet")):
        out.append(("detection_nsfw_safe_train", iter_eliasalbouzidi_parquet(p)))
    for p in sorted((src_root / "datasets/eliasalbouzidi_NSFW-Safe-Dataset/data").glob("test-*.parquet")):
        out.append(("detection_nsfw_safe_test", iter_eliasalbouzidi_parquet(p)))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="H:/from_D/webdataset root")
    ap.add_argument("--out", required=True,
                    help="pretrain: H:/elt_data/raw  |  detection: H:/elt_data/detection")
    ap.add_argument("--mode", choices=("pretrain", "detection"), default="pretrain",
                    help="pretrain: flat {text}  |  detection: labeled {text,label,...}")
    args = ap.parse_args()

    src_root, out_dir = Path(args.src), Path(args.out)

    if args.mode == "pretrain":
        groups_t: dict[str, list[Iterator[str]]] = {}
        for stem, it in build_sources(src_root):
            groups_t.setdefault(stem, []).append(it)
        for stem, iters in groups_t.items():
            def _chain_t(iters_: list[Iterator[str]]) -> Iterator[str]:
                for i in iters_: yield from i
            out_path = out_dir / f"{stem}.jsonl"
            try:
                n = _write_jsonl(out_path, _chain_t(iters))
                size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
                print(f"  {stem}: {n:,} docs  ({size_mb:.1f} MB) -> {out_path}")
            except Exception as e:
                print(f"  [warn] {stem}: {e}")
    else:  # detection
        groups_d: dict[str, list[Iterator[dict]]] = {}
        for stem, it in build_detection_sources(src_root):
            groups_d.setdefault(stem, []).append(it)
        for stem, iters in groups_d.items():
            def _chain_d(iters_: list[Iterator[dict]]) -> Iterator[dict]:
                for i in iters_: yield from i
            out_path = out_dir / f"{stem}.jsonl"
            try:
                n = _write_labeled_jsonl(out_path, _chain_d(iters))
                size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
                print(f"  {stem}: {n:,} labeled docs  ({size_mb:.1f} MB) -> {out_path}")
            except Exception as e:
                print(f"  [warn] {stem}: {e}")


if __name__ == "__main__":
    main()
