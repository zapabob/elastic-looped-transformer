"""End-to-end training pipeline orchestrator with boot-time auto-resume.

Stages run top-to-bottom; each writes a `.done` marker on success so the next
boot skips completed work. On the *final* stage's completion the orchestrator
calls `pipeline_unregister.ps1`, removing its own Task Scheduler entry.

Install (once, from an admin PowerShell in the project root):

    powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1

Uninstall (or manual):

    powershell -ExecutionPolicy Bypass -File scripts/pipeline_unregister.ps1

Manual run (any user):

    uv run python scripts/pipeline.py            # full pipeline, respect done markers
    uv run python scripts/pipeline.py --only pretrain,sft    # subset
    uv run python scripts/pipeline.py --dry-run  # print plan, don't execute
    uv run python scripts/pipeline.py --reset    # clear markers, start over
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path("H:/elt_data/pipeline_state")
DISTILL_OUT = Path("H:/elt_data/distill/teacher_sft.jsonl")
TELEMETRY_PATH = STATE_DIR / "pipeline.jsonl"


def _sh(cmd: list[str] | str, check: bool = True) -> int:
    """Run a command, streaming output to our stdout. Returns exit code."""
    if isinstance(cmd, str):
        print(f"  $ {cmd}")
        r = subprocess.run(cmd, shell=True, cwd=ROOT)
    else:
        print("  $ " + " ".join(cmd))
        r = subprocess.run(cmd, cwd=ROOT)
    if check and r.returncode != 0:
        raise SystemExit(f"stage failed (rc={r.returncode})")
    return r.returncode


# --- Stage implementations --------------------------------------------------

def stage_download() -> None:
    _sh(["uv", "run", "python", "scripts/download_hf_corpus.py",
         "--out", "H:/elt_data/raw", "--scale", "1.0"])

def stage_ingest_webdataset() -> None:
    _sh(["uv", "run", "python", "scripts/ingest_webdataset.py",
         "--src", "H:/from_D/webdataset", "--out", "H:/elt_data/raw",
         "--mode", "pretrain"])
    _sh(["uv", "run", "python", "scripts/ingest_webdataset.py",
         "--src", "H:/from_D/webdataset", "--out", "H:/elt_data/detection",
         "--mode", "detection"])

def stage_clean() -> None:
    _sh(["uv", "run", "python", "scripts/clean_corpus.py",
         "--in", "H:/elt_data/raw", "--out", "H:/elt_data/clean"])

def stage_tokenize() -> None:
    _sh(["uv", "run", "python", "scripts/build_train_bin.py",
         "--tokenizer", "H:/Qwen3.5-9B-official-hf",
         "--out-dir", "H:/elt_data/bin",
         "--config", "scripts/corpus_manifest_clean.yaml"])

def stage_smoke() -> None:
    _sh(["uv", "run", "elt-train", "--config", "configs/tiny_10M.yaml"])

def stage_pretrain() -> None:
    last = ROOT / "runs" / "base_100M" / "last.pt"
    cmd = ["uv", "run", "elt-train", "--config", "configs/base_100M.yaml"]
    if last.exists():
        cmd += ["--resume", str(last)]
    _sh(cmd)

def stage_distill_gen() -> None:
    DISTILL_OUT.parent.mkdir(parents=True, exist_ok=True)
    prompts = ["H:/elt_data/raw/gsm8k.jsonl",
               "H:/elt_data/raw/metamath.jsonl",
               "H:/elt_data/raw/opencode_instruct.jsonl"]
    prompts = [p for p in prompts if Path(p).exists()]
    _sh(["uv", "run", "python", "scripts/distill_teacher_gen.py",
         "--teacher", "huihui-ai/Huihui-Qwopus3.5-4B-v3-abliterated",
         "--prompts", *prompts,
         "--out", str(DISTILL_OUT),
         "--max-prompts", "20000",
         "--max-new-tokens", "512"])

def stage_sft() -> None:
    # SFT bin may not exist yet (no dedicated SFT manifest built); fall back to
    # the full clean bin so this stage never blocks. Users who want a CoT-only
    # mix should add scripts/sft_manifest.yaml before running.
    last = ROOT / "runs" / "sft_cot" / "last.pt"
    cmd = ["uv", "run", "elt-train", "--config", "configs/sft_cot.yaml"]
    if last.exists():
        cmd += ["--resume", str(last)]
    elif (ROOT / "runs" / "base_100M" / "last.pt").exists():
        cmd += ["--resume", str(ROOT / "runs" / "base_100M" / "last.pt")]
    _sh(cmd)

def stage_grpo() -> None:
    last = ROOT / "runs" / "grpo_gsm8k" / "last.pt"
    cmd = ["uv", "run", "elt-train-grpo", "--config", "configs/grpo_gsm8k.yaml"]
    if last.exists():
        cmd += ["--resume", str(last)]
    elif (ROOT / "runs" / "sft_cot" / "last.pt").exists():
        cmd += ["--resume", str(ROOT / "runs" / "sft_cot" / "last.pt")]
    _sh(cmd)

def stage_eval() -> None:
    ckpt = ROOT / "runs" / "grpo_gsm8k" / "last.pt"
    if not ckpt.exists():
        ckpt = ROOT / "runs" / "sft_cot" / "last.pt"
    if not ckpt.exists():
        ckpt = ROOT / "runs" / "base_100M" / "last.pt"
    _sh(["uv", "run", "elt-anytime",
         "--ckpt", str(ckpt),
         "--val-bin", "H:/elt_data/bin/val.bin",
         "--L-list", "1,2,3,4"])

def stage_export_hf() -> None:
    ckpt = ROOT / "runs" / "grpo_gsm8k" / "last.pt"
    if not ckpt.exists():
        ckpt = ROOT / "runs" / "base_100M" / "last.pt"
    _sh(["uv", "run", "python", "scripts/export_to_hf.py",
         "--ckpt", str(ckpt),
         "--out", "hf_export/elt-lm-base-275m",
         "--tokenizer", "H:/Qwen3.5-9B-official-hf",
         "--repo-id", "zapabob/elt-lm-base-275m"])


# --- Orchestration ----------------------------------------------------------

@dataclass
class Stage:
    name: str
    run: Callable[[], None]

    def marker(self) -> Path:
        return STATE_DIR / f"{self.name}.done"

    def is_done(self) -> bool:
        return self.marker().exists()

    def mark_done(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.marker().write_text(f"completed_at={time.time()}\n", encoding="utf-8")


STAGES: list[Stage] = [
    Stage("00_download",   stage_download),
    Stage("01_ingest",     stage_ingest_webdataset),
    Stage("02_clean",      stage_clean),
    Stage("03_tokenize",   stage_tokenize),
    Stage("04_smoke",      stage_smoke),
    Stage("05_pretrain",   stage_pretrain),
    Stage("06_distill",    stage_distill_gen),
    Stage("07_sft",        stage_sft),
    Stage("08_grpo",       stage_grpo),
    Stage("09_eval",       stage_eval),
    Stage("10_export_hf",  stage_export_hf),
]


def unregister_startup() -> None:
    """Remove the Task Scheduler entry that boots this pipeline."""
    script = ROOT / "scripts" / "pipeline_unregister.ps1"
    if not script.exists():
        print(f"  [warn] unregister script missing: {script}")
        return
    try:
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass",
                        "-File", str(script)], check=False)
        print("  pipeline self-removed from Windows startup")
    except Exception as e:
        print(f"  [warn] failed to unregister startup: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated stage name substrings")
    ap.add_argument("--skip", help="comma-separated stage name substrings")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true",
                    help="delete all stage-done markers before running")
    ap.add_argument("--no-unregister", action="store_true",
                    help="do not remove the startup task on completion")
    args = ap.parse_args()

    if args.reset:
        for s in STAGES:
            m = s.marker()
            if m.exists(): m.unlink()
        print("  cleared all stage markers")

    only = set((args.only or "").split(",")) if args.only else set()
    skip = set((args.skip or "").split(",")) if args.skip else set()

    plan = []
    for s in STAGES:
        if only and not any(tag and tag in s.name for tag in only): continue
        if skip and any(tag and tag in s.name for tag in skip): continue
        plan.append(s)

    print(f"  STATE_DIR = {STATE_DIR}")
    print("  plan:")
    for s in plan:
        print(f"    {'[done]' if s.is_done() else '      '} {s.name}")

    if args.dry_run:
        return

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    def emit(event: str, **fields: object) -> None:
        try:
            with open(TELEMETRY_PATH, "a", encoding="utf-8", buffering=1) as f:
                f.write(json.dumps({"ts": time.time(), "event": event, **fields}) + "\n")
        except OSError:
            pass

    emit("pipeline_start", plan=[s.name for s in plan])

    all_ok = True
    for s in plan:
        if s.is_done():
            print(f"  skip: {s.name}")
            emit("pipeline_stage", name=s.name, status="skipped")
            continue
        print(f"\n========== {s.name} ==========")
        emit("pipeline_stage", name=s.name, status="start")
        t0 = time.time()
        try:
            s.run()
        except SystemExit as e:
            print(f"  stage {s.name} aborted: {e}")
            emit("pipeline_stage", name=s.name, status="aborted",
                 elapsed_sec=time.time() - t0, error=str(e))
            all_ok = False
            break
        except Exception as e:
            print(f"  stage {s.name} crashed: {e}")
            emit("pipeline_stage", name=s.name, status="crashed",
                 elapsed_sec=time.time() - t0, error=str(e))
            all_ok = False
            break
        s.mark_done()
        elapsed = time.time() - t0
        emit("pipeline_stage", name=s.name, status="done", elapsed_sec=elapsed)
        print(f"  {s.name} done in {elapsed/60:.1f} min")

    if all_ok and plan and plan[-1] is STAGES[-1] and not args.no_unregister:
        print("\n  all stages complete — removing boot registration")
        unregister_startup()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
