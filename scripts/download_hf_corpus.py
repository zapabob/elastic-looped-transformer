"""Download a curated Japanese + English + math + code corpus from HuggingFace
Hub and stage it on H:\\elt_data\\raw as flat JSONL (one `{"text": ...}` per line).

Designed to respect the user's 18 GB free H: budget:
- each source has a hard --max-bytes cap (streaming, so we never pull the whole
  dump for giant datasets like full Wikipedia).
- we normalize every row to a single `text` field so the downstream aggregator
  does not need per-source schema knowledge.

Usage:
    uv run python scripts/download_hf_corpus.py --out H:/elt_data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset
from tqdm import tqdm


def _write_stream(
    out_path: Path,
    iterator: Any,
    text_fn: Callable[[dict], str | None],
    max_bytes: int,
    source: str,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    n = 0
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        pbar = tqdm(desc=source, unit="MB", unit_scale=True)
        for row in iterator:
            try:
                text = text_fn(row)
            except (KeyError, TypeError, ValueError):
                continue
            if not text:
                continue
            text = text.strip()
            if len(text) < 32:
                continue
            line = json.dumps({"text": text, "source": source}, ensure_ascii=False)
            enc = (line + "\n").encode("utf-8")
            f.write(line + "\n")
            written += len(enc)
            n += 1
            pbar.update(len(enc) / (1024 * 1024))
            if written >= max_bytes:
                break
        pbar.close()
    print(f"  -> {out_path}  rows={n:,}  bytes={written/1e9:.2f}GB")
    return written


def pull_wiki_ja(out_dir: Path, max_bytes: int) -> None:
    # Japanese Wikipedia. range3/wikipedia-ja-20230101 ships as a clean `text` field.
    ds = load_dataset("range3/wikipedia-ja-20230101", split="train", streaming=True)
    _write_stream(
        out_dir / "wiki_ja.jsonl",
        ds,
        lambda r: r.get("text"),
        max_bytes,
        "wiki_ja",
    )


def pull_wiki_en(out_dir: Path, max_bytes: int) -> None:
    # English Wikipedia — streamed so we can cap. 20231101.en config is canonical.
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )
    _write_stream(
        out_dir / "wiki_en.jsonl",
        ds,
        lambda r: r.get("text"),
        max_bytes,
        "wiki_en",
    )


def pull_metamath(out_dir: Path, max_bytes: int) -> None:
    # Math: MetaMathQA — "query" + "response" pairs (CoT math).
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)

    def _tx(r: dict) -> str:
        q = r.get("query", "").strip()
        a = r.get("response", "").strip()
        if not q or not a:
            return ""
        return f"Problem: {q}\n\nSolution: {a}"

    _write_stream(
        out_dir / "metamath.jsonl", ds, _tx, max_bytes, "metamath"
    )


def pull_gsm8k(out_dir: Path, max_bytes: int) -> None:
    # Canonical grade-school math with chain-of-thought (`####` final answer).
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)

    def _tx(r: dict) -> str:
        q = r.get("question", "").strip()
        a = r.get("answer", "").strip()
        if not q or not a:
            return ""
        return f"Question: {q}\n\nAnswer: {a}"

    _write_stream(out_dir / "gsm8k.jsonl", ds, _tx, max_bytes, "gsm8k")


def pull_magicoder(out_dir: Path, max_bytes: int) -> None:
    # Code instruction-following. Schema: {"problem": ..., "solution": ...}
    ds = load_dataset(
        "ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        p = r.get("problem", "").strip()
        s = r.get("solution", "").strip()
        if not p or not s:
            return ""
        return f"{p}\n\n{s}"

    _write_stream(
        out_dir / "magicoder.jsonl", ds, _tx, max_bytes, "magicoder"
    )


def pull_magicoder_evol(out_dir: Path, max_bytes: int) -> None:
    # Companion to Magicoder-OSS: evol-instruct variant, broader instruction complexity.
    ds = load_dataset(
        "ise-uiuc/Magicoder-Evol-Instruct-110K", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        inst = r.get("instruction", "").strip()
        resp = r.get("response", "").strip()
        if not inst or not resp:
            return ""
        return f"{inst}\n\n{resp}"

    _write_stream(
        out_dir / "magicoder_evol.jsonl", ds, _tx, max_bytes, "magicoder_evol"
    )


def pull_stack_smol(out_dir: Path, max_bytes: int) -> None:
    # BigCode The Stack smol: curated multi-language code corpus.
    # Schema includes `content` (raw source) + `lang` + `path`.
    ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)

    def _tx(r: dict) -> str:
        content = r.get("content", "")
        lang = r.get("lang", "") or r.get("language", "")
        if not content or len(content) < 64:
            return ""
        header = f"# language: {lang}\n" if lang else ""
        return f"{header}{content}"

    _write_stream(
        out_dir / "stack_smol.jsonl", ds, _tx, max_bytes, "stack_smol"
    )


def pull_codeparrot(out_dir: Path, max_bytes: int) -> None:
    # Large cleaned GitHub python corpus from codeparrot. Streamed to respect cap.
    ds = load_dataset(
        "codeparrot/github-code-clean", "all-all", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        code = r.get("code", "")
        lang = r.get("language", "")
        if not code or len(code) < 64:
            return ""
        header = f"# language: {lang}\n" if lang else ""
        return f"{header}{code}"

    _write_stream(
        out_dir / "codeparrot.jsonl", ds, _tx, max_bytes, "codeparrot"
    )


def pull_glaive_tools(out_dir: Path, max_bytes: int) -> None:
    # Glaive function-calling v2: standard tool-use training corpus.
    # Schema: {"system": "...tools...", "chat": "USER: ... ASSISTANT: <functioncall> {...}"}
    ds = load_dataset(
        "glaiveai/glaive-function-calling-v2", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        sys = r.get("system", "").strip()
        chat = r.get("chat", "").strip()
        if not chat:
            return ""
        if sys:
            return f"<|system|>\n{sys}\n\n{chat}"
        return chat

    _write_stream(
        out_dir / "glaive_tools.jsonl", ds, _tx, max_bytes, "glaive_tools"
    )


def _chat_from_shareGPT(convs: list, tools_str: str | None = None) -> str | None:
    """Render a ShareGPT-style [{from, value}] (or [{role, content}]) list into a
    flat prompt. Used for Hermes / ToolACE schemas."""
    parts: list[str] = []
    if tools_str:
        parts.append(f"<|tools|>\n{tools_str}")
    for m in convs:
        role = m.get("from") or m.get("role") or ""
        content = m.get("value") or m.get("content") or ""
        if not content:
            continue
        parts.append(f"<|{role}|>\n{content}")
    return "\n\n".join(parts) if parts else None


def pull_codefeedback(out_dir: Path, max_bytes: int) -> None:
    # m-a-p CodeFeedback: error -> fix conversations.
    # Schema: {"query": "...", "answer": "..."}
    ds = load_dataset(
        "m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        q = r.get("query", "").strip()
        a = r.get("answer", "").strip()
        if not q or not a:
            return ""
        return f"{q}\n\n{a}"

    _write_stream(
        out_dir / "codefeedback.jsonl", ds, _tx, max_bytes, "codefeedback"
    )


def pull_opencoder_sft(out_dir: Path, max_bytes: int) -> None:
    # OpenCoder stage2 SFT. Schema varies; fall back over common fields.
    ds = load_dataset(
        "OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train",
        streaming=True,
    )

    def _tx(r: dict) -> str:
        inst = r.get("instruction") or r.get("input") or r.get("prompt") or ""
        out = r.get("output") or r.get("response") or r.get("answer") or ""
        inst = inst.strip() if isinstance(inst, str) else ""
        out = out.strip() if isinstance(out, str) else ""
        if not inst or not out:
            return ""
        return f"{inst}\n\n{out}"

    _write_stream(
        out_dir / "opencoder_sft.jsonl", ds, _tx, max_bytes, "opencoder_sft"
    )


def pull_commitpackft(out_dir: Path, max_bytes: int) -> None:
    # BigCode commitpackft: source-pair edits driven by commit messages.
    # Stream all langs by requesting config 'all'. Schema: {message, subject,
    # old_contents, new_contents, lang, ...}
    ds = load_dataset("bigcode/commitpackft", "python", split="train", streaming=True)

    def _tx(r: dict) -> str:
        msg = (r.get("message") or r.get("subject") or "").strip()
        old = (r.get("old_contents") or "").strip()
        new = (r.get("new_contents") or "").strip()
        lang = r.get("lang") or r.get("language") or ""
        if not new or not msg:
            return ""
        header = f"# commit: {msg}\n# language: {lang}" if lang else f"# commit: {msg}"
        blocks = [header]
        if old:
            blocks.append(f"<|before|>\n{old}")
        blocks.append(f"<|after|>\n{new}")
        return "\n\n".join(blocks)

    _write_stream(
        out_dir / "commitpackft.jsonl", ds, _tx, max_bytes, "commitpackft"
    )


def pull_hermes_tools(out_dir: Path, max_bytes: int) -> None:
    # Hermes function-calling v1. ShareGPT-style conversations.
    ds = load_dataset(
        "NousResearch/hermes-function-calling-v1", "func_calling",
        split="train", streaming=True,
    )

    def _tx(r: dict) -> str:
        convs = r.get("conversations") or []
        tools = r.get("tools") or r.get("system") or ""
        if not isinstance(convs, list) or not convs:
            return ""
        rendered = _chat_from_shareGPT(
            convs, tools_str=tools if isinstance(tools, str) else None
        )
        return rendered or ""

    _write_stream(
        out_dir / "hermes_tools.jsonl", ds, _tx, max_bytes, "hermes_tools"
    )


def pull_toolace(out_dir: Path, max_bytes: int) -> None:
    # Team-ACE ToolACE multi-turn tool-use reasoning.
    ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)

    def _tx(r: dict) -> str:
        convs = r.get("conversations") or r.get("messages") or []
        tools = r.get("system") or r.get("tools") or ""
        if not isinstance(convs, list) or not convs:
            return ""
        return _chat_from_shareGPT(
            convs, tools_str=tools if isinstance(tools, str) else None
        ) or ""

    _write_stream(
        out_dir / "toolace.jsonl", ds, _tx, max_bytes, "toolace"
    )


def pull_apibench(out_dir: Path, max_bytes: int) -> None:
    # Gorilla APIBench — small but historically the canonical API retrieval set.
    ds = load_dataset("gorilla-llm/APIBench", split="train", streaming=True)

    def _tx(r: dict) -> str:
        inst = r.get("instruction") or r.get("question") or r.get("input") or ""
        out = r.get("output") or r.get("answer") or ""
        inst = inst.strip() if isinstance(inst, str) else ""
        out = out.strip() if isinstance(out, str) else ""
        if not inst or not out:
            return ""
        return f"{inst}\n\n{out}"

    _write_stream(
        out_dir / "apibench.jsonl", ds, _tx, max_bytes, "apibench"
    )


def pull_orca_agent(out_dir: Path, max_bytes: int) -> None:
    # Microsoft Orca AgentInstruct 1M (2024): broad synthetic agentic tasks.
    ds = load_dataset(
        "microsoft/orca-agentinstruct-1M-v1", split="creative_content",
        streaming=True,
    )

    def _tx(r: dict) -> str:
        msgs_raw = r.get("messages")
        if isinstance(msgs_raw, str):
            # some configs ship messages as JSON-encoded string
            try:
                msgs = json.loads(msgs_raw)
            except json.JSONDecodeError:
                return ""
        else:
            msgs = msgs_raw
        if not isinstance(msgs, list) or not msgs:
            return ""
        parts = []
        for m in msgs:
            role = m.get("role") or ""
            content = m.get("content") or ""
            if not content:
                continue
            parts.append(f"<|{role}|>\n{content}")
        return "\n\n".join(parts) if parts else ""

    _write_stream(
        out_dir / "orca_agent.jsonl", ds, _tx, max_bytes, "orca_agent"
    )


def pull_openhermes(out_dir: Path, max_bytes: int) -> None:
    # teknium OpenHermes-2.5: large general-assistant corpus (~1M ShareGPT).
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

    def _tx(r: dict) -> str:
        convs = r.get("conversations") or []
        if not isinstance(convs, list) or not convs:
            return ""
        return _chat_from_shareGPT(convs) or ""

    _write_stream(
        out_dir / "openhermes.jsonl", ds, _tx, max_bytes, "openhermes"
    )


def pull_wildchat(out_dir: Path, max_bytes: int) -> None:
    # allenai WildChat-1M: real user-assistant conversations with GPT-3.5/4.
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    def _tx(r: dict) -> str:
        convs = r.get("conversation") or []
        if not isinstance(convs, list) or not convs:
            return ""
        parts = []
        for m in convs:
            role = m.get("role") or ""
            content = m.get("content") or ""
            if not content:
                continue
            parts.append(f"<|{role}|>\n{content}")
        return "\n\n".join(parts) if parts else ""

    _write_stream(
        out_dir / "wildchat.jsonl", ds, _tx, max_bytes, "wildchat"
    )


def pull_general_thought(out_dir: Path, max_bytes: int) -> None:
    # 2025 GeneralReasoning GeneralThought-430K: broad reasoning traces.
    ds = load_dataset(
        "GeneralReasoning/GeneralThought-430K", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        q = (r.get("question") or r.get("prompt") or r.get("instruction") or "").strip()
        th = (r.get("thought") or r.get("reasoning") or "").strip()
        a = (r.get("answer") or r.get("output") or r.get("response") or "").strip()
        if not q or (not th and not a):
            return ""
        blocks = [f"<|question|>\n{q}"]
        if th:
            blocks.append(f"<|thought|>\n{th}")
        if a:
            blocks.append(f"<|answer|>\n{a}")
        return "\n\n".join(blocks)

    _write_stream(
        out_dir / "general_thought.jsonl", ds, _tx, max_bytes, "general_thought"
    )


def pull_slim_orca(out_dir: Path, max_bytes: int) -> None:
    # Open-Orca SlimOrca-Dedup: distilled broad chat, deduped.
    ds = load_dataset(
        "Open-Orca/SlimOrca-Dedup", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        convs = r.get("conversations") or []
        if not isinstance(convs, list) or not convs:
            return ""
        return _chat_from_shareGPT(convs) or ""

    _write_stream(
        out_dir / "slim_orca.jsonl", ds, _tx, max_bytes, "slim_orca"
    )


def pull_swe_gym(out_dir: Path, max_bytes: int) -> None:
    # SWE-Gym 2024-25: real bug-fix tasks with reproducible tests.
    # Schema: {instance_id, problem_statement, patch, test_patch, ...}
    ds = load_dataset("SWE-Gym/SWE-Gym", split="train", streaming=True)

    def _tx(r: dict) -> str:
        prob = (r.get("problem_statement") or "").strip()
        patch = (r.get("patch") or "").strip()
        test_patch = (r.get("test_patch") or "").strip()
        if not prob or not patch:
            return ""
        parts = [f"<|problem|>\n{prob}", f"<|patch|>\n{patch}"]
        if test_patch:
            parts.append(f"<|tests|>\n{test_patch}")
        return "\n\n".join(parts)

    _write_stream(out_dir / "swe_gym.jsonl", ds, _tx, max_bytes, "swe_gym")


def pull_codeact(out_dir: Path, max_bytes: int) -> None:
    # CodeAct (Liu et al., 2024): multi-turn tool-using code-execution agents.
    ds = load_dataset("xingyaoww/code-act", split="codeact", streaming=True)

    def _tx(r: dict) -> str:
        convs = r.get("conversations") or r.get("messages") or []
        if not isinstance(convs, list) or not convs:
            return ""
        return _chat_from_shareGPT(convs) or ""

    _write_stream(out_dir / "codeact.jsonl", ds, _tx, max_bytes, "codeact")


def pull_agent_instruct(out_dir: Path, max_bytes: int) -> None:
    # THUDM AgentInstruct: 6 agentic tasks (OS, DB, web, ...).
    ds = load_dataset("THUDM/AgentInstruct", "os", split="train", streaming=True)

    def _tx(r: dict) -> str:
        convs = r.get("conversations") or []
        if not isinstance(convs, list) or not convs:
            return ""
        return _chat_from_shareGPT(convs) or ""

    _write_stream(
        out_dir / "agent_instruct.jsonl", ds, _tx, max_bytes, "agent_instruct"
    )


def pull_self_oss_instruct(out_dir: Path, max_bytes: int) -> None:
    # BigCode self-oss-instruct sc2 execution-filtered (pytest verified).
    ds = load_dataset(
        "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        split="train", streaming=True,
    )

    def _tx(r: dict) -> str:
        inst = (r.get("instruction") or r.get("prompt") or "").strip()
        resp = (r.get("response") or r.get("output") or r.get("completion") or "").strip()
        if not inst or not resp:
            return ""
        return f"{inst}\n\n{resp}"

    _write_stream(
        out_dir / "self_oss_instruct.jsonl", ds, _tx, max_bytes, "self_oss_instruct"
    )


def pull_openthoughts(out_dir: Path, max_bytes: int) -> None:
    # 2025 CoT reasoning distillation (math + code + science), ShareGPT format.
    ds = load_dataset(
        "open-thoughts/OpenThoughts-114k", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        convs = r.get("conversations") or []
        if not isinstance(convs, list) or not convs:
            return ""
        return _chat_from_shareGPT(convs) or ""

    _write_stream(
        out_dir / "openthoughts.jsonl", ds, _tx, max_bytes, "openthoughts"
    )


def pull_cosmopedia(out_dir: Path, max_bytes: int) -> None:
    # HuggingFaceTB cosmopedia-v2: synthetic high-quality educational text
    # covering STEM, textbook-style. Streamed, capped.
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
        split="train", streaming=True,
    )

    def _tx(r: dict) -> str:
        text = r.get("text") or ""
        return text if isinstance(text, str) else ""

    _write_stream(
        out_dir / "cosmopedia.jsonl", ds, _tx, max_bytes, "cosmopedia"
    )


def pull_finemath(out_dir: Path, max_bytes: int) -> None:
    # 2025 HuggingFaceTB finemath-4plus: top-quality math-heavy web.
    ds = load_dataset(
        "HuggingFaceTB/finemath", "finemath-4plus",
        split="train", streaming=True,
    )

    def _tx(r: dict) -> str:
        text = r.get("text") or ""
        return text if isinstance(text, str) else ""

    _write_stream(
        out_dir / "finemath.jsonl", ds, _tx, max_bytes, "finemath"
    )


def pull_openwebmath(out_dir: Path, max_bytes: int) -> None:
    # open-web-math: huge math-weighted web crawl (12B tokens total). Stream + cap.
    ds = load_dataset(
        "open-web-math/open-web-math", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        text = r.get("text") or ""
        return text if isinstance(text, str) else ""

    _write_stream(
        out_dir / "openwebmath.jsonl", ds, _tx, max_bytes, "openwebmath"
    )


def pull_camel_sci(out_dir: Path, max_bytes: int) -> None:
    # CAMEL physics + chemistry + biology concatenated (~20k each).
    # Schema: {"message_1": "<user>", "message_2": "<assistant>", ...}
    subsets = [("camel-ai/physics", "physics"),
               ("camel-ai/chemistry", "chemistry"),
               ("camel-ai/biology", "biology")]

    out_path = out_dir / "camel_sci.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        pbar = tqdm(desc="camel_sci", unit="MB", unit_scale=True)
        for ds_name, domain in subsets:
            try:
                ds = load_dataset(ds_name, split="train", streaming=True)
            except Exception as e:
                print(f"[camel_sci/{domain}] skip: {e}", file=sys.stderr)
                continue
            for row in ds:
                if not isinstance(row, dict):
                    continue
                u = str(row.get("message_1", "")).strip()
                a = str(row.get("message_2", "")).strip()
                if not u or not a:
                    continue
                text = f"[{domain}]\n<|user|>\n{u}\n\n<|assistant|>\n{a}"
                line = json.dumps({"text": text, "source": "camel_sci"},
                                  ensure_ascii=False)
                enc = (line + "\n").encode("utf-8")
                f.write(line + "\n")
                total += len(enc)
                pbar.update(len(enc) / (1024 * 1024))
                if total >= max_bytes:
                    break
            if total >= max_bytes:
                break
        pbar.close()
    print(f"  -> {out_path}  bytes={total/1e9:.2f}GB")


def pull_tulu3(out_dir: Path, max_bytes: int) -> None:
    # Allen AI Tulu 3 SFT mix (2024-25): broad STEM + instruction coverage.
    ds = load_dataset(
        "allenai/tulu-3-sft-mixture", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        msgs = r.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            return ""
        parts = []
        for m in msgs:
            role = m.get("role") or ""
            content = m.get("content") or ""
            if not content:
                continue
            parts.append(f"<|{role}|>\n{content}")
        return "\n\n".join(parts) if parts else ""

    _write_stream(out_dir / "tulu3.jsonl", ds, _tx, max_bytes, "tulu3")


def pull_openmath2(out_dir: Path, max_bytes: int) -> None:
    # NVIDIA OpenMathInstruct-2: 14M problems, tool-integrated solutions.
    ds = load_dataset(
        "nvidia/OpenMathInstruct-2", split="train_1M", streaming=True
    )

    def _tx(r: dict) -> str:
        p = (r.get("problem") or "").strip()
        s = (r.get("generated_solution") or r.get("solution") or "").strip()
        if not p or not s:
            return ""
        return f"Problem: {p}\n\nSolution: {s}"

    _write_stream(
        out_dir / "openmath2.jsonl", ds, _tx, max_bytes, "openmath2"
    )


def pull_opencode_reasoning(out_dir: Path, max_bytes: int) -> None:
    # NVIDIA OpenCodeReasoning (2025): long-chain-of-thought coding reasoning.
    # Parquet-backed (non-gated). Schema: {input, output, ...} — we fold
    # problem + reasoning trace into one text blob.
    ds = load_dataset(
        "nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True
    )

    def _tx(r: dict) -> str:
        prob = (r.get("input") or r.get("problem") or "").strip()
        sol = (r.get("output") or r.get("solution") or "").strip()
        if not prob or not sol:
            return ""
        return f"<|problem|>\n{prob}\n\n<|solution|>\n{sol}"

    _write_stream(
        out_dir / "opencode_reasoning.jsonl", ds, _tx, max_bytes,
        "opencode_reasoning",
    )


def pull_opencode_instruct(out_dir: Path, max_bytes: int) -> None:
    # NVIDIA OpenCodeInstruct (2025): instruction-tuned coding SFT. Parquet.
    ds = load_dataset(
        "nvidia/OpenCodeInstruct", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        inst = (r.get("input") or r.get("instruction") or "").strip()
        out = (r.get("output") or r.get("response") or "").strip()
        if not inst or not out:
            return ""
        return f"<|user|>\n{inst}\n\n<|assistant|>\n{out}"

    _write_stream(
        out_dir / "opencode_instruct.jsonl", ds, _tx, max_bytes,
        "opencode_instruct",
    )


def pull_xlam_tools(out_dir: Path, max_bytes: int) -> None:
    # Salesforce xLAM APIGen: single/parallel function-calling, modern schema.
    ds = load_dataset(
        "Salesforce/xlam-function-calling-60k", split="train", streaming=True
    )

    def _tx(r: dict) -> str:
        q = r.get("query", "").strip()
        tools = r.get("tools", "").strip()
        answers = r.get("answers", "").strip()
        if not q or not answers:
            return ""
        blocks = [f"<|user|>\n{q}"]
        if tools:
            blocks.append(f"<|tools|>\n{tools}")
        blocks.append(f"<|assistant|>\n{answers}")
        return "\n\n".join(blocks)

    _write_stream(
        out_dir / "xlam_tools.jsonl", ds, _tx, max_bytes, "xlam_tools"
    )


SOURCES: dict[str, tuple[Callable[[Path, int], None], int]] = {
    # name: (fn, default_max_bytes)
    "wiki_ja":        (pull_wiki_ja,         2_500_000_000),  # 2.5 GB
    "wiki_en":        (pull_wiki_en,         2_500_000_000),  # 2.5 GB
    "metamath":       (pull_metamath,          500_000_000),  # 0.5 GB
    "gsm8k":          (pull_gsm8k,              50_000_000),  # full (tiny)
    # --- coding (instruction + raw) ---
    "magicoder":      (pull_magicoder,       1_000_000_000),  # 1.0 GB
    "magicoder_evol": (pull_magicoder_evol,    500_000_000),  # 0.5 GB
    "stack_smol":     (pull_stack_smol,      2_000_000_000),  # 2.0 GB multi-lang code
    "codeparrot":     (pull_codeparrot,      2_000_000_000),  # 2.0 GB py/js/cpp
    "codefeedback":   (pull_codefeedback,      500_000_000),  # 0.5 GB error->fix
    "opencoder_sft":  (pull_opencoder_sft,     500_000_000),  # 0.5 GB OpenCoder SFT
    "commitpackft":   (pull_commitpackft,      500_000_000),  # 0.5 GB commit diffs
    "opencode_reasoning": (pull_opencode_reasoning, 1_000_000_000),  # 1.0 GB NVIDIA 2025 CoT code
    "opencode_instruct":  (pull_opencode_instruct,    800_000_000),  # 0.8 GB NVIDIA 2025 code SFT
    # --- tool / function calling ---
    "glaive_tools":   (pull_glaive_tools,      500_000_000),  # 0.5 GB
    "xlam_tools":     (pull_xlam_tools,        200_000_000),  # 0.2 GB APIGen
    "hermes_tools":   (pull_hermes_tools,      300_000_000),  # 0.3 GB Hermes
    "toolace":        (pull_toolace,           500_000_000),  # 0.5 GB ToolACE
    "apibench":       (pull_apibench,          100_000_000),  # 0.1 GB Gorilla
    # --- extra math ---
    "openmath2":      (pull_openmath2,         800_000_000),  # 0.8 GB OpenMathInstruct-2
    # --- general AI agent (2024-25) ---
    "orca_agent":       (pull_orca_agent,      1_000_000_000),  # 1.0 GB MS Orca Agent 1M
    "openhermes":       (pull_openhermes,        800_000_000),  # 0.8 GB OpenHermes 2.5
    "wildchat":         (pull_wildchat,          500_000_000),  # 0.5 GB real user logs
    "general_thought":  (pull_general_thought,   500_000_000),  # 0.5 GB 2025 reasoning
    "slim_orca":        (pull_slim_orca,         400_000_000),  # 0.4 GB distilled chat
    # --- coding agent (2024-25) ---
    "swe_gym":          (pull_swe_gym,           500_000_000),  # 0.5 GB SWE-Gym
    "codeact":          (pull_codeact,           500_000_000),  # 0.5 GB CodeAct multi-turn
    "agent_instruct":   (pull_agent_instruct,    300_000_000),  # 0.3 GB THUDM AgentInstruct
    "self_oss_instruct":(pull_self_oss_instruct, 500_000_000),  # 0.5 GB exec-filtered
    # --- STEM 2024-2025 ---
    "openthoughts":   (pull_openthoughts,      800_000_000),  # 0.8 GB 2025 CoT reasoning
    "cosmopedia":     (pull_cosmopedia,      1_500_000_000),  # 1.5 GB synthetic STEM textbooks
    "finemath":       (pull_finemath,        1_000_000_000),  # 1.0 GB 2025 math web
    "openwebmath":    (pull_openwebmath,     1_500_000_000),  # 1.5 GB math-heavy web
    "camel_sci":      (pull_camel_sci,         300_000_000),  # 0.3 GB physics/chem/bio
    "tulu3":          (pull_tulu3,             800_000_000),  # 0.8 GB broad STEM SFT
}


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="H:/elt_data/raw")
    p.add_argument(
        "--only",
        nargs="+",
        choices=list(SOURCES.keys()),
        default=list(SOURCES.keys()),
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiplier on default max-bytes (e.g. 0.1 for a smoke run)",
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        fn, base = SOURCES[name]
        budget = int(base * args.scale)
        print(f"[{name}] downloading up to {budget/1e9:.2f}GB to {out_dir}")
        try:
            fn(out_dir, budget)
        except Exception as e:
            print(f"[{name}] FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    cli()
