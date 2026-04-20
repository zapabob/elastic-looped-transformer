"""Offline teacher-response generation for SFT distillation.

Loads `huihui-ai/Huihui-Qwopus3.5-4B-v3-abliterated` in bf16, runs greedy
decoding over a prompt bank, and appends {"text": "Q: ...\\n\\nA: ..."} lines
to `H:/elt_data/distill/teacher_sft.jsonl` so the SFT phase can consume it via
the normal flat-JSONL path.

Design notes
------------
- **Offline, not online KD.** On 12GB VRAM we can't afford to hold teacher +
  student activations simultaneously. We generate once, cache, reuse.
- **Resumable.** A sidecar `teacher_sft.progress.json` stores the last-completed
  prompt index. Kill -9 the process and re-run; it picks up where it left off.
- **Prompt bank** = GSM8K train + a sample of MetaMathQA + OpenCodeInstruct,
  read from the existing `H:/elt_data/raw/*.jsonl`. Avoids re-downloading.

Run:

    uv run python scripts/distill_teacher_gen.py \\
        --teacher huihui-ai/Huihui-Qwopus3.5-4B-v3-abliterated \\
        --prompts H:/elt_data/raw/gsm8k.jsonl H:/elt_data/raw/metamath.jsonl \\
        --out H:/elt_data/distill/teacher_sft.jsonl \\
        --max-prompts 20000 --max-new-tokens 512
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator


def _iter_prompts(paths: list[Path], max_prompts: int) -> Iterator[str]:
    """Yield up to `max_prompts` prompt strings from flat JSONL files."""
    n = 0
    for p in paths:
        if not p.exists():
            print(f"[warn] missing: {p}")
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: o = json.loads(line)
                except json.JSONDecodeError: continue
                txt = (o.get("text") or o.get("prompt") or "").strip()
                if len(txt) < 32:
                    continue
                # Heuristic: the "prompt" is the first 1-2 lines; answers come
                # after. For GSM8K we keep only the question part.
                q = txt.split("\n\n", 1)[0].strip()
                if len(q) > 1024:
                    q = q[:1024]
                yield q
                n += 1
                if n >= max_prompts:
                    return


def _load_progress(path: Path) -> int:
    if path.exists():
        try: return int(json.loads(path.read_text())["next_idx"])
        except Exception: return 0
    return 0


def _save_progress(path: Path, next_idx: int) -> None:
    path.write_text(json.dumps({"next_idx": next_idx,
                                "wall_time": time.time()}),
                    encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="huihui-ai/Huihui-Qwopus3.5-4B-v3-abliterated")
    ap.add_argument("--prompts", nargs="+", required=True,
                    help="input JSONL files with {text}/{prompt}")
    ap.add_argument("--out", required=True, help="output JSONL (appended)")
    ap.add_argument("--max-prompts", type=int, default=20_000)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Heavy imports happen only here so `--help` is fast.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(".progress.json")
    start_idx = _load_progress(progress_path)

    prompts = list(_iter_prompts([Path(p) for p in args.prompts], args.max_prompts))
    print(f"  total prompts: {len(prompts):,}  (resuming at {start_idx:,})")
    if start_idx >= len(prompts):
        print("  already complete")
        return

    print(f"  loading teacher: {args.teacher}")
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=dtype, trust_remote_code=True,
        device_map=args.device,
    ).eval()

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Append mode: prior runs' outputs are preserved.
    with open(out_path, "a", encoding="utf-8") as out_f:
        t0 = time.time()
        for i in range(start_idx, len(prompts)):
            q = prompts[i]
            chat = [{"role": "user", "content": q}]
            # Fall back to raw concat if the tokenizer lacks a chat template.
            try:
                ids = tok.apply_chat_template(chat, tokenize=True,
                                              add_generation_prompt=True,
                                              return_tensors="pt")
            except Exception:
                ids = tok(q, return_tensors="pt").input_ids
            ids = ids.to(args.device)

            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            response_ids = out[0, ids.shape[1]:]
            response = tok.decode(response_ids, skip_special_tokens=True).strip()
            if not response:
                _save_progress(progress_path, i + 1)
                continue

            rec = {"text": f"Q: {q}\n\nA: {response}"}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            _save_progress(progress_path, i + 1)

            if (i - start_idx) % 50 == 0 and i > start_idx:
                elapsed = time.time() - t0
                rate = (i - start_idx) / elapsed
                eta = (len(prompts) - i) / rate if rate > 0 else 0.0
                print(f"  [{i:,}/{len(prompts):,}] {rate:.2f} prompts/s  "
                      f"eta {eta/60:.1f} min")

    print(f"  done -> {out_path}")


if __name__ == "__main__":
    main()
