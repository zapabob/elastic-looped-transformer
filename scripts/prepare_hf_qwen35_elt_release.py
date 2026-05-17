"""Prepare and optionally upload the Qwen3.5 ELT L3 Hugging Face release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "zapabobouj/qwen3.5-elt-l3"
DEFAULT_HF_EXPORT = Path("H:/elt_data/hf_exports/elt-lm-qwen35-side-stem-aha-ilsd-l3-merged")
DEFAULT_OUT = Path("H:/elt_data/hf_publish/qwen3.5-elt-l3")
DEFAULT_GGUFS = {
    "elt-lm-qwen35-side-stem-aha-ilsd-l3-BF16.gguf": Path(
        "H:/elt_data/releases/elt-lm-qwen35-side-stem-aha-ilsd-l3.gguf"
    ),
    "elt-lm-qwen35-side-stem-aha-ilsd-l3-Q8_0.gguf": Path(
        "H:/elt_data/releases/elt-lm-qwen35-side-stem-aha-ilsd-l3-Q8_0.gguf"
    ),
    "elt-lm-qwen35-side-stem-aha-ilsd-l3-TQ4_1S.gguf": Path(
        "H:/elt_data/releases/elt-lm-qwen35-side-stem-aha-ilsd-l3-TQ4_1S.gguf"
    ),
}
README_STATS = ROOT / "_docs/assets/2026-05-17-l3-thetom-k-protected/l3_readme_stats.json"
README_GRAPH = ROOT / "_docs/assets/2026-05-17-l3-thetom-k-protected/l3_readme_accuracy_errorbars.png"
README_STATS_MD = ROOT / "_docs/assets/2026-05-17-l3-thetom-k-protected/l3_readme_stats.md"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _link_or_copy(src: Path, dst: Path) -> None:
    _require(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _summaries() -> dict:
    return json.loads(_require(README_STATS).read_text(encoding="utf-8"))


def _model_card(repo_id: str, stats: dict) -> str:
    eval_rows = stats["evaluation_summaries"]
    loop_rows = stats["loop_summaries"]
    fisher = {item["comparison"]: item["p"] for item in stats["fisher_exact"]}
    mcnemar = {item["comparison"]: item for item in stats["loop_mcnemar_exact"]}

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    def ci(row: dict) -> str:
        return f"[{row['ci95_low'] * 100:.1f}, {row['ci95_high'] * 100:.1f}]"

    lines = [
        "---",
        "license: apache-2.0",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "base_model: huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated",
        "tags:",
        "  - qwen3.5",
        "  - elastic-looped-transformer",
        "  - gguf",
        "  - safetensors",
        "  - lora-merged",
        "  - research-artifact",
        "---",
        "",
        "# Qwen3.5 ELT L3",
        "",
        "This repository publishes the L3 Qwen3.5 Elastic Looped Transformer",
        "research handoff as Hugging Face safetensors plus GGUF artifacts.",
        "",
        "The strongest supported claim is narrow: this artifact is strong on the",
        "local STEM bridge task and loop depth helps in the loop-aware HF runtime.",
        "It is not yet a broad LLM benchmark or GSM8K/math-generalization claim.",
        "",
        "## Files",
        "",
        "- `model-00001-of-00007.safetensors` ... `model-00007-of-00007.safetensors` - merged BF16 Hugging Face shards.",
        "- `model.safetensors.index.json` - HF sharded-weight index.",
        "- `elt-lm-qwen35-side-stem-aha-ilsd-l3-BF16.gguf` - BF16 GGUF handoff.",
        "- `elt-lm-qwen35-side-stem-aha-ilsd-l3-Q8_0.gguf` - Q8_0 GGUF handoff.",
        "- `elt-lm-qwen35-side-stem-aha-ilsd-l3-TQ4_1S.gguf` - TurboQuant-style offline weight-compression GGUF.",
        "- `elt_export_manifest.json` and `publish_manifest.json` - export and publication metadata.",
        "",
        "## Runtime boundary",
        "",
        "The GGUF files carry ELT loop metadata and are useful for structural and",
        "single-pass runtime handoff, but stock GGUF execution is not the loop-aware",
        "L>=2 runtime. The config records",
        "`elt.gguf.runtime_status=requires_looped_qwen35_runtime`; use the",
        "loop-aware Qwen3.5 HF runtime for L-depth quality experiments.",
        "",
        "The TQ4_1S file is an offline weight-compression artifact. It is not a",
        "claim about TurboQuant KV-cache serving performance.",
        "",
        "## Evaluation snapshot",
        "",
        "![L3 Qwen3.5 ELT accuracy error bars](eval/l3_readme_accuracy_errorbars.png)",
        "",
        "| evaluation | n | correct | accuracy | Wilson 95% CI | prompt tok/s | decode tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in eval_rows:
        lines.append(
            f"| {row['name']} | {row['total']} | {row['correct']} | {pct(row['accuracy'])} | "
            f"{ci(row)} | {row['mean_prompt_tok_s']:.2f} | {row['mean_decode_tok_s']:.2f} |"
        )
    lines.extend([
        "",
        "Two-sided Fisher exact p-values over correct/incorrect counts:",
        "",
        "| comparison | p |",
        "|---|---:|",
        f"| Local STEM bridge vs MMLU-STEM heldout | {fisher['Local STEM bridge vs MMLU-STEM heldout']:.4g} |",
        f"| Local STEM bridge vs GSM8K heldout | {fisher['Local STEM bridge vs GSM8K heldout']:.4g} |",
        f"| MMLU-STEM heldout vs GSM8K heldout | {fisher['MMLU-STEM heldout vs GSM8K heldout']:.4g} |",
        "",
        "Loop-aware HF runtime quality uses paired case IDs on 32 local STEM bridge",
        "questions and MCQ log-probability scoring:",
        "",
        "| L | n | correct | accuracy | Wilson 95% CI | mean margin | wall sec/case |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in loop_rows:
        lines.append(
            f"| {row['L']} | {row['total']} | {row['correct']} | {pct(row['mean'])} | "
            f"{ci(row)} | {row['mean_margin_logprob']:.4f} | {row['mean_wall_sec']:.3f} |"
        )
    l13 = mcnemar["L1_vs_L3"]
    lines.extend([
        "",
        "Paired exact McNemar/binomial result for L1 vs L3:",
        f"{l13['improved']} improved, {l13['regressed']} regressed, p={l13['mcnemar_exact_p']:.4g}.",
        "",
        "## Loading",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f"repo_id = \"{repo_id}\"",
        "tok = AutoTokenizer.from_pretrained(repo_id)",
        "model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=\"auto\", device_map=\"auto\")",
        "```",
        "",
        "For GGUF downloads:",
        "",
        "```bash",
        f"hf download {repo_id} elt-lm-qwen35-side-stem-aha-ilsd-l3-Q8_0.gguf",
        "```",
        "",
        "## Provenance",
        "",
        "- Source method: Elastic Looped Transformer / ILSD workflow.",
        "- Base model: `huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated`.",
        "- Export date: 2026-05-17.",
        "- README statistics: see `eval/l3_readme_stats.json` and `eval/l3_readme_stats.md`.",
        "",
    ])
    return "\n".join(lines)


def prepare(repo_id: str, hf_export: Path, out_dir: Path, compute_sha256: bool) -> None:
    publish_root = Path("H:/elt_data/hf_publish")
    if not _is_within(out_dir, publish_root):
        raise ValueError(f"Refusing to write outside {publish_root}: {out_dir}")

    _require(hf_export)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(hf_export.iterdir()):
        if src.is_file() and src.name != "README.md":
            _link_or_copy(src, out_dir / src.name)

    for dst_name, src in DEFAULT_GGUFS.items():
        _link_or_copy(src, out_dir / dst_name)

    eval_dir = out_dir / "eval"
    _link_or_copy(README_GRAPH, eval_dir / README_GRAPH.name)
    _link_or_copy(README_STATS, eval_dir / README_STATS.name)
    _link_or_copy(README_STATS_MD, eval_dir / README_STATS_MD.name)

    stats = _summaries()
    (out_dir / "README.md").write_text(_model_card(repo_id, stats), encoding="utf-8")

    artifacts = []
    for path in sorted(item for item in out_dir.rglob("*") if item.is_file()):
        rel = path.relative_to(out_dir).as_posix()
        entry = {"path": rel, "bytes": path.stat().st_size}
        if compute_sha256 or path.suffix.lower() not in {".gguf", ".safetensors"}:
            entry["sha256"] = _sha256(path)
        artifacts.append(entry)
    manifest = {
        "repo_id": repo_id,
        "prepared_at": "2026-05-18",
        "artifact": "qwen3.5-elt-l3",
        "public_claim_boundary": "local STEM bridge strong; small external heldouts; GSM8K not solved; loop-aware runtime required for L>=2 quality",
        "files": artifacts,
    }
    (out_dir / "publish_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def upload(repo_id: str, out_dir: Path, private: bool, num_workers: int) -> None:
    from huggingface_hub import HfApi

    try:
        import truststore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "truststore is required for reliable Hugging Face TLS on this Windows host. "
            "Install with: uv --native-tls pip install --python ./.venv/Scripts/python3.exe truststore"
        ) from exc
    truststore.inject_into_ssl()
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=out_dir,
        private=private,
        num_workers=num_workers,
        print_report=True,
        print_report_every=30,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--hf-export", type=Path, default=DEFAULT_HF_EXPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--compute-sha256", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(args.repo_id, args.hf_export, args.out_dir, args.compute_sha256)
    if args.upload:
        upload(args.repo_id, args.out_dir, private=args.private, num_workers=args.num_workers)
    print(args.out_dir.resolve())


if __name__ == "__main__":
    main()
