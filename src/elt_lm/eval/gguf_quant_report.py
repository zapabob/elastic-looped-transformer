"""GGUF quantization benchmark, perplexity, logits, and CV report helpers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

from elt_lm.eval.statistics import (
    friedman_permutation_test,
    paired_permutation_pvalue,
)


DEFAULT_MODELS = {
    "BF16": "H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge.gguf",
    "Q8_0": "H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge-Q8_0.gguf",
    "TQ4_1S": "H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge-TQ4_1S.gguf",
}

DEFAULT_EVAL_SOURCES = [
    "training_data/synthetic_v2_hard/stem_reasoning/distill_val.jsonl",
    "training_data/synthetic_v2_hard/math/distill_val.jsonl",
    "training_data/synthetic_v2_hard/code/distill_val.jsonl",
    "training_data/synthetic_v2_hard/tool_use/distill_val.jsonl",
]


@dataclass(frozen=True)
class MetricRow:
    metric: str
    group: str
    block: str
    value: float
    unit: str
    source: str
    higher_is_better: bool


@dataclass(frozen=True)
class SummaryRow:
    metric: str
    group: str
    n: int
    mean: float
    sd: float
    sem: float
    ci95_low: float
    ci95_high: float
    unit: str
    higher_is_better: bool


def quantization_lane_boundaries() -> dict[str, str]:
    """Return the claim boundaries that keep weight, KV, and serving lanes separate."""

    return {
        "weight_compression": (
            "TQ4_1S is reported as a local GGUF weight-compression artifact."
        ),
        "kv_compression": (
            "TurboQuant-style KV cache compression is tracked as a separate runtime lane; "
            "this report does not claim Google TurboQuant KV-cache serving performance."
        ),
        "speculative_decoding": (
            "DFlash is tracked as a separate speculative-decoding lane and should be "
            "evaluated with target equivalence, acceptance length, accept rate, and tok/s."
        ),
    }


def _read_jsonl_texts(paths: Sequence[Path], *, max_records: int) -> list[str]:
    texts: list[str] = []
    per_path = max(1, math.ceil(max_records / max(1, len(paths))))
    for path in paths:
        if not path.exists():
            continue
        taken = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if taken >= per_path or len(texts) >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = item.get("text")
                if not text:
                    prompt = item.get("prompt", "")
                    response = item.get("response") or item.get("reference", "")
                    text = f"User: {prompt}\nAssistant: {response}"
                texts.append(str(text))
                taken += 1
    if not texts:
        raise ValueError("no evaluation text records were found")
    return texts


def build_eval_corpus(
    sources: Sequence[str | Path],
    out_path: str | Path,
    *,
    max_records: int = 24,
    min_chars: int = 4096,
) -> Path:
    """Build a deterministic local held-out text file for llama-perplexity."""

    paths = [Path(item) for item in sources]
    texts = _read_jsonl_texts(paths, max_records=max_records)
    corpus = "\n\n".join(texts)
    while len(corpus) < min_chars:
        corpus = corpus + "\n\n" + "\n\n".join(texts)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(corpus, encoding="utf-8")
    return out


def _run_command(command: Sequence[str], *, cwd: Path, timeout_sec: int) -> str:
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(
        list(command),
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
        check=False,
    )
    output = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(command)}\n{output[-4000:]}"
        )
    return output


def run_llama_bench(
    *,
    llama_bin_dir: str | Path,
    model_name: str,
    model_path: str | Path,
    out_dir: str | Path,
    repetitions: int,
    prompt_tokens: int,
    gen_tokens: int,
    cache_pairs: Sequence[tuple[str, str]],
    timeout_sec: int,
    cwd: str | Path = ".",
) -> list[dict[str, Any]]:
    exe = Path(llama_bin_dir) / "llama-bench.exe"
    if not exe.exists():
        raise FileNotFoundError(exe)
    rows: list[dict[str, Any]] = []
    raw_dir = Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for cache_k, cache_v in cache_pairs:
        command = [
            str(exe),
            "-m",
            str(model_path),
            "-o",
            "json",
            "-r",
            str(repetitions),
            "-p",
            str(prompt_tokens),
            "-n",
            str(gen_tokens),
            "-ctk",
            cache_k,
            "-ctv",
            cache_v,
        ]
        output = _run_command(command, cwd=Path(cwd), timeout_sec=timeout_sec)
        (raw_dir / f"bench_{model_name}_{cache_k}_{cache_v}.json").write_text(
            output,
            encoding="utf-8",
        )
        parsed = json.loads(_extract_json_array(output))
        rows.extend(parsed)
    return rows


def run_llama_perplexity(
    *,
    llama_bin_dir: str | Path,
    model_name: str,
    model_path: str | Path,
    corpus_path: str | Path,
    out_dir: str | Path,
    ctx_size: int,
    chunks: int,
    timeout_sec: int,
    cache_k: str = "f16",
    cache_v: str = "f16",
    logits_base: str | Path | None = None,
    save_logits_to: str | Path | None = None,
    cwd: str | Path = ".",
) -> str:
    exe = Path(llama_bin_dir) / "llama-perplexity.exe"
    if not exe.exists():
        raise FileNotFoundError(exe)
    command = [
        str(exe),
        "-m",
        str(model_path),
        "-f",
        str(corpus_path),
        "--chunks",
        str(chunks),
        "-c",
        str(ctx_size),
        "-np",
        "1",
        "--ppl-output-type",
        "1",
        "--no-warmup",
        "-ctk",
        cache_k,
        "-ctv",
        cache_v,
    ]
    if save_logits_to is not None:
        command.extend(["--save-all-logits", str(save_logits_to)])
    if logits_base is not None:
        command.extend(["--kl-divergence-base", str(logits_base), "--kl-divergence"])
    output = _run_command(command, cwd=Path(cwd), timeout_sec=timeout_sec)
    raw_dir = Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"perplexity_{model_name}.txt").write_text(output, encoding="utf-8")
    return output


def _extract_json_array(output: str) -> str:
    start = output.find("[")
    end = output.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"no JSON array found in output: {output[:400]}")
    return output[start : end + 1]


def parse_bench_rows(model_name: str, rows: Sequence[Mapping[str, Any]]) -> list[MetricRow]:
    parsed: list[MetricRow] = []
    for row in rows:
        cache = f"k{row.get('type_k', '')}_v{row.get('type_v', '')}"
        prompt = int(row.get("n_prompt") or 0)
        gen = int(row.get("n_gen") or 0)
        samples = [float(item) for item in row.get("samples_ts") or [row.get("avg_ts", 0.0)]]
        if prompt > 0:
            metric = "prompt_eval_tps"
            block_prefix = f"pp{prompt}_{cache}"
        elif gen > 0:
            metric = "decode_tps"
            block_prefix = f"tg{gen}_{cache}"
        else:
            continue
        for idx, value in enumerate(samples):
            parsed.append(MetricRow(
                metric=metric,
                group=model_name,
                block=f"{block_prefix}_rep{idx}",
                value=value,
                unit="tok/s",
                source="llama-bench",
                higher_is_better=True,
            ))
    return parsed


_PPL_LINE = re.compile(
    r"^\s*(?P<idx>\d+)\s+(?P<ppl>[-+0-9.eE]+)\s+(?P<nll>[-+0-9.eE]+)(?:\s+(?P<err>[-+0-9.eE]+))?",
    re.MULTILINE,
)
_KV_LINE = re.compile(r"^llama_kv_cache:\s+size\s+=\s*(?P<mib>[-+0-9.]+)\s*MiB", re.MULTILINE)
_RS_LINE = re.compile(r"llama_memory_recurrent:\s+size\s+=\s*(?P<mib>[-+0-9.]+)\s*MiB")
_FLOAT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _numeric_table_rows(output: str) -> Iterable[tuple[str, list[float]]]:
    for line in output.splitlines():
        if not re.match(r"^\s*\d+\s+", line):
            continue
        values = [float(item) for item in _FLOAT.findall(line)]
        if len(values) >= 3:
            yield str(int(values[0])), values


def parse_perplexity_output(model_name: str, output: str) -> list[MetricRow]:
    rows: list[MetricRow] = []
    seen_ppl_blocks: set[str] = set()
    for block_idx, values in _numeric_table_rows(output):
        block = f"chunk{block_idx}"
        if block in seen_ppl_blocks:
            continue
        rows.append(MetricRow(
            metric="perplexity",
            group=model_name,
            block=block,
            value=float(values[1]),
            unit="ppl",
            source="llama-perplexity",
            higher_is_better=False,
        ))
        # Plain PPL rows are: chunk, ppl, nll, stderr. KL rows are:
        # chunk, ppl, ppl_err, ln_ratio, ln_ratio_err, kl, kl_err, ...
        if "KL Divergence" in output and len(values) >= 6:
            rows.append(MetricRow(
                metric="logits_kl_vs_bf16",
                group=model_name,
                block=block,
                value=float(values[5]),
                unit="KL",
                source="llama-perplexity-logits",
                higher_is_better=False,
            ))
        else:
            rows.append(MetricRow(
                metric="negative_log_likelihood",
                group=model_name,
                block=block,
                value=float(values[2]),
                unit="nll",
                source="llama-perplexity",
                higher_is_better=False,
            ))
        seen_ppl_blocks.add(block)
    for match in _PPL_LINE.finditer(output):
        block = f"chunk{match.group('idx')}"
        if block in seen_ppl_blocks:
            continue
        rows.append(MetricRow(
            metric="perplexity",
            group=model_name,
            block=block,
            value=float(match.group("ppl")),
            unit="ppl",
            source="llama-perplexity",
            higher_is_better=False,
        ))
        seen_ppl_blocks.add(block)
        rows.append(MetricRow(
            metric="negative_log_likelihood",
            group=model_name,
            block=block,
            value=float(match.group("nll")),
            unit="nll",
            source="llama-perplexity",
            higher_is_better=False,
        ))
    kv_match = _KV_LINE.search(output)
    if kv_match:
        rows.append(MetricRow(
            metric="kv_cache_mib",
            group=model_name,
            block="load",
            value=float(kv_match.group("mib")),
            unit="MiB",
            source="llama-perplexity-log",
            higher_is_better=False,
        ))
    rs_match = _RS_LINE.search(output)
    if rs_match:
        rows.append(MetricRow(
            metric="recurrent_state_mib",
            group=model_name,
            block="load",
            value=float(rs_match.group("mib")),
            unit="MiB",
            source="llama-perplexity-log",
            higher_is_better=False,
        ))
    return rows


def file_size_rows(models: Mapping[str, str | Path]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    bf16_size = None
    if "BF16" in models and Path(models["BF16"]).exists():
        bf16_size = Path(models["BF16"]).stat().st_size
    for name, path_raw in models.items():
        path = Path(path_raw)
        if not path.exists():
            continue
        size = path.stat().st_size
        rows.append(MetricRow(
            metric="file_size_gib",
            group=name,
            block="artifact",
            value=size / (1024 ** 3),
            unit="GiB",
            source="filesystem",
            higher_is_better=False,
        ))
        if bf16_size:
            rows.append(MetricRow(
                metric="size_ratio_vs_bf16",
                group=name,
                block="artifact",
                value=size / bf16_size,
                unit="ratio",
                source="filesystem",
                higher_is_better=False,
            ))
    return rows


def _values_by_metric(rows: Sequence[MetricRow], metric: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.metric != metric:
            continue
        grouped.setdefault(row.group, {})[row.block] = row.value
    return grouped


def _summarize(metric_rows: Sequence[MetricRow]) -> SummaryRow:
    values = [row.value for row in metric_rows]
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    sem = sd / math.sqrt(max(1, len(values)))
    ci = 1.96 * sem
    return SummaryRow(
        metric=metric_rows[0].metric,
        group=metric_rows[0].group,
        n=len(values),
        mean=avg,
        sd=sd,
        sem=sem,
        ci95_low=avg - ci,
        ci95_high=avg + ci,
        unit=metric_rows[0].unit,
        higher_is_better=metric_rows[0].higher_is_better,
    )


def summarize_metric_rows(rows: Sequence[MetricRow]) -> list[SummaryRow]:
    by_key: dict[tuple[str, str], list[MetricRow]] = {}
    for row in rows:
        by_key.setdefault((row.metric, row.group), []).append(row)
    return [_summarize(items) for items in by_key.values()]


def _scipy_tests(groups: Mapping[str, Sequence[float]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    try:
        from scipy import stats
    except Exception:
        return None, []
    names = list(groups)
    arrays = [[float(x) for x in groups[name]] for name in names]
    pairwise: list[dict[str, Any]] = []
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if j <= i:
                continue
            a = arrays[i]
            b = arrays[j]
            if len(a) < 2 or all(abs(x - y) < 1e-12 for x, y in zip(a, b)):
                p_value = 1.0
                statistic = 0.0
            else:
                result = stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
                p_value = float(result.pvalue)
                statistic = float(result.statistic)
            pairwise.append({
                "left": left,
                "right": right,
                "statistic": statistic,
                "mean_delta": mean(a) - mean(b),
                "p_value": p_value,
                "method": "scipy.stats.wilcoxon_zsplit_two_sided",
            })
    omnibus = None
    if len(arrays) >= 3 and len(arrays[0]) >= 2:
        result = stats.friedmanchisquare(*arrays)
        omnibus = {
            "groups": names,
            "n_blocks": len(arrays[0]),
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method": "scipy.stats.friedmanchisquare",
        }
    return omnibus, pairwise


def compare_metrics(rows: Sequence[MetricRow], *, seed: int = 0) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for metric in sorted({row.metric for row in rows}):
        grouped = _values_by_metric(rows, metric)
        if len(grouped) < 2:
            continue
        common_blocks = sorted(set.intersection(*(set(blocks) for blocks in grouped.values())))
        if not common_blocks:
            continue
        groups = {
            group: [blocks[block] for block in common_blocks]
            for group, blocks in grouped.items()
        }
        omnibus, pairwise = _scipy_tests(groups)
        if not pairwise:
            pairwise = []
            names = list(groups)
            for i, left in enumerate(names):
                for j, right in enumerate(names):
                    if j <= i:
                        continue
                    pairwise.append({
                        "left": left,
                        "right": right,
                        "statistic": None,
                        "mean_delta": mean(groups[left]) - mean(groups[right]),
                        "p_value": paired_permutation_pvalue(
                            groups[left],
                            groups[right],
                            permutations=10000,
                            seed=seed + i * 1009 + j,
                        ),
                        "method": "paired_permutation_10000",
                    })
        if omnibus is None and len(groups) >= 3 and len(common_blocks) >= 2:
            item = friedman_permutation_test(groups, permutations=10000, seed=seed)
            omnibus = item.__dict__
        reports[metric] = {
            "n_blocks": len(common_blocks),
            "blocks": common_blocks,
            "groups": groups,
            "omnibus": omnibus,
            "pairwise": pairwise,
        }
    return reports


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def render_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "## GGUF BF16 / Q8_0 / TQ4_1S cross-validation report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Runtime: `{report['runtime']}`",
        f"- Corpus: `{report['corpus_path']}`",
        f"- Scope: short local GGUF validation; not a broad lm-eval leaderboard claim.",
        "",
        "### Lane boundaries",
        "",
    ]
    for lane, boundary in report.get("lane_boundaries", {}).items():
        lines.append(f"- `{lane}`: {boundary}")
    lines.extend([
        "",
        "### Summary",
        "",
        "| metric | group | n | mean | sd | sem | 95% CI | unit |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in report["summaries"]:
        lines.append(
            "| {metric} | {group} | {n} | {mean:.4f} | {sd:.4f} | {sem:.4f} | "
            "[{ci95_low:.4f}, {ci95_high:.4f}] | {unit} |".format(**row)
        )
    lines.extend(["", "### Multi-group tests", ""])
    for metric, stats_report in report["comparisons"].items():
        lines.extend([
            f"#### {metric}",
            "",
            f"- paired blocks: `{stats_report['n_blocks']}`",
        ])
        omnibus = stats_report.get("omnibus")
        if omnibus:
            lines.append(
                "- omnibus: {method}, statistic={statistic:.4f}, p={p_value:.6f}".format(**omnibus)
            )
        for row in stats_report["pairwise"]:
            lines.append(
                "- {left} vs {right}: delta={mean_delta:.4f}, p={p_value:.6f}, {method}".format(**row)
            )
        lines.append("")
    lines.extend([
        "### Interpretation",
        "",
        "- `prompt_eval_tps` and `decode_tps` are llama.cpp runtime throughput blocks paired by cache type and phase.",
        "- `perplexity` and `negative_log_likelihood` are local held-out text checks from verifier-backed synthetic-v2 hard validation records.",
        "- `logits_kl_vs_bf16` is included when the optional llama-perplexity logits file run is available.",
        "- `TQ4_1S` currently reports as a mixed GGUF weight artifact with Turboquant metadata; do not read it as a TurboQuant KV-cache result.",
    ])
    return "\n".join(lines) + "\n"


def _summary_lookup(summaries: Sequence[Mapping[str, Any]], metric: str) -> list[Mapping[str, Any]]:
    return [row for row in summaries if row["metric"] == metric]


def write_plots(report: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("matplotlib is required to render PNG plots; install the eval extra") from exc

    plt.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    out = Path(out_dir)
    summaries = report["summaries"]
    metrics = [
        ("file_size_gib", "GGUF size", "GiB"),
        ("prompt_eval_tps", "Prompt eval", "tok/s"),
        ("decode_tps", "Decode", "tok/s"),
        ("perplexity", "Perplexity", "ppl"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6), constrained_layout=True)
    for ax, (metric, title, unit) in zip(axes.flat, metrics):
        rows = _summary_lookup(summaries, metric)
        labels = [row["group"] for row in rows]
        means = [row["mean"] for row in rows]
        errors = [row["sem"] for row in rows]
        colors = ["#2f6f9f", "#2f9f6f", "#b2642f"][: len(labels)]
        ax.bar(labels, means, yerr=errors, capsize=5, color=colors, edgecolor="#1f2933")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("GGUF BF16 / Q8_0 / TQ4_1S paired CV summary", fontsize=16)
    plot_path = out / "gguf_quant_cv_errorbars.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 3)
    ax_title = fig.add_subplot(grid[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.75,
        "GGUF量子化 評価サマリー: BF16 / Q8_0 / TQ4_1S",
        fontsize=22,
        weight="bold",
    )
    ax_title.text(
        0.0,
        0.35,
        "llama.cpp CUDA runtime, paired cache/phase blocks, SciPy Friedman/Wilcoxon where available",
        fontsize=12,
    )

    ax_size = fig.add_subplot(grid[1, 0])
    rows = _summary_lookup(summaries, "file_size_gib")
    ax_size.bar([r["group"] for r in rows], [r["mean"] for r in rows], color="#4f7cac")
    ax_size.set_title("Artifact size")
    ax_size.set_ylabel("GiB")
    ax_size.grid(axis="y", alpha=0.25)

    ax_runtime = fig.add_subplot(grid[1, 1])
    rows = _summary_lookup(summaries, "prompt_eval_tps")
    ax_runtime.errorbar(
        [r["group"] for r in rows],
        [r["mean"] for r in rows],
        yerr=[r["sem"] for r in rows],
        fmt="o",
        capsize=5,
        color="#2f9f6f",
    )
    ax_runtime.set_title("Prompt eval throughput")
    ax_runtime.set_ylabel("tok/s")
    ax_runtime.grid(axis="y", alpha=0.25)

    ax_ppl = fig.add_subplot(grid[1, 2])
    rows = _summary_lookup(summaries, "perplexity")
    ax_ppl.errorbar(
        [r["group"] for r in rows],
        [r["mean"] for r in rows],
        yerr=[r["sem"] for r in rows],
        fmt="o",
        capsize=5,
        color="#b2642f",
    )
    ax_ppl.set_title("Held-out perplexity")
    ax_ppl.set_ylabel("ppl lower is better")
    ax_ppl.grid(axis="y", alpha=0.25)

    ax_notes = fig.add_subplot(grid[2, :])
    ax_notes.axis("off")
    comparison_bits = []
    for metric in ("prompt_eval_tps", "decode_tps", "perplexity"):
        item = report["comparisons"].get(metric, {})
        omnibus = item.get("omnibus")
        if omnibus:
            comparison_bits.append(f"{metric}: Friedman p={omnibus['p_value']:.4g}")
    note = " | ".join(comparison_bits) if comparison_bits else "p-values require at least two paired blocks per metric."
    ax_notes.text(0.0, 0.72, note, fontsize=13, weight="bold")
    ax_notes.text(
        0.0,
        0.42,
        "README claim boundary: this is a local short-run release check over verifier-backed synthetic-v2 hard text, "
        "not an external lm-eval leaderboard submission.",
        fontsize=12,
    )
    ax_notes.text(
        0.0,
        0.16,
        "Use the CSV/JSON artifacts for exact means, SEM, 95% CI, pairwise p-values, cache settings, and raw command logs.",
        fontsize=12,
    )
    infographic_path = out / "gptimage_gguf_quant_cv_infographic.png"
    fig.savefig(infographic_path, dpi=160)
    plt.close(fig)
    return {
        "errorbars": str(plot_path),
        "infographic": str(infographic_path),
    }


def build_report(
    *,
    rows: Sequence[MetricRow],
    model_paths: Mapping[str, str],
    generated_at: str,
    runtime: str,
    corpus_path: str,
    out_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    summaries = [row.__dict__ for row in summarize_metric_rows(rows)]
    comparisons = compare_metrics(rows, seed=seed)
    report: dict[str, Any] = {
        "generated_at": generated_at,
        "runtime": runtime,
        "corpus_path": corpus_path,
        "model_paths": dict(model_paths),
        "metrics": [row.__dict__ for row in rows],
        "summaries": summaries,
        "comparisons": comparisons,
        "claim_boundary": (
            "Local short-run GGUF release validation over synthetic-v2 hard held-out text; "
            "not a broad external lm-eval leaderboard result."
        ),
        "lane_boundaries": quantization_lane_boundaries(),
        "next_eval_fields": [
            "model",
            "weight_quant",
            "kv_k_type",
            "kv_v_type",
            "ctx",
            "seed",
            "prompt_id",
            "loop_depth",
            "ppl",
            "kl",
            "top1_match",
            "top5_overlap",
            "argmax_flip_rate",
            "elt_exact",
            "elt_step_accuracy",
            "gsm8k_exact",
            "needle_score",
            "prompt_tok_s",
            "gen_tok_s",
            "kv_mib",
            "vram_peak_mib",
        ],
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(
        out / "gguf_quant_cv_metrics.csv",
        [row.__dict__ for row in rows],
        ["metric", "group", "block", "value", "unit", "source", "higher_is_better"],
    )
    write_csv(
        out / "gguf_quant_cv_summary.csv",
        summaries,
        ["metric", "group", "n", "mean", "sd", "sem", "ci95_low", "ci95_high", "unit", "higher_is_better"],
    )
    pairwise_rows: list[dict[str, Any]] = []
    omnibus_rows: list[dict[str, Any]] = []
    for metric, item in comparisons.items():
        for pair in item["pairwise"]:
            pairwise_rows.append({"metric": metric, **pair})
        if item.get("omnibus"):
            omnibus_rows.append({"metric": metric, **item["omnibus"]})
    write_csv(
        out / "gguf_quant_cv_pairwise.csv",
        pairwise_rows,
        ["metric", "left", "right", "statistic", "mean_delta", "p_value", "method"],
    )
    write_csv(
        out / "gguf_quant_cv_omnibus.csv",
        omnibus_rows,
        ["metric", "groups", "n_blocks", "statistic", "p_value", "method"],
    )
    (out / "gguf_quant_cv_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "gguf_quant_cv_report.md").write_text(render_markdown_report(report), encoding="utf-8")
    prompt = (
        "16:9 Japanese technical infographic. Title: GGUF量子化評価 BF16/Q8_0/TQ4_1S. "
        "Use only the measured CSV/JSON values from gguf_quant_cv_report.json. "
        "Show artifact size, prompt/decode throughput with error bars, perplexity, logits KL if present, "
        "Friedman/Wilcoxon p-values, and a clear note that this is a local short-run release validation."
    )
    (out / "gptimage_prompt.md").write_text(prompt + "\n", encoding="utf-8")
    report["plots"] = write_plots(report, out)
    (out / "gguf_quant_cv_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _parse_model_args(values: Sequence[str]) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_MODELS)
    models: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--model must be NAME=PATH")
        name, path = value.split("=", 1)
        models[name] = path
    return models


def load_existing_raw(out_dir: str | Path, models: Mapping[str, str]) -> list[MetricRow]:
    rows: list[MetricRow] = file_size_rows(models)
    raw = Path(out_dir) / "raw"
    for path in raw.glob("bench_*.json"):
        name = path.stem.split("_", 1)[1].split("_f16", 1)[0].split("_q8", 1)[0]
        parsed = json.loads(_extract_json_array(path.read_text(encoding="utf-8")))
        rows.extend(parse_bench_rows(name, parsed))
    for path in raw.glob("perplexity_*.txt"):
        name = path.stem.removeprefix("perplexity_")
        rows.extend(parse_perplexity_output(name, path.read_text(encoding="utf-8")))
    return rows


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=[], help="NAME=PATH. Default uses BF16/Q8_0/TQ4_1S release paths.")
    parser.add_argument("--llama-bin-dir", default="C:/Users/downl/AppData/Local/Programs/llama-turboquant/bin")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--corpus-path", default="")
    parser.add_argument("--eval-source", action="append", default=[])
    parser.add_argument("--max-records", type=int, default=24)
    parser.add_argument("--min-corpus-chars", type=int, default=4096)
    parser.add_argument("--run-bench", action="store_true")
    parser.add_argument("--run-perplexity", action="store_true")
    parser.add_argument("--run-logits-kl", action="store_true")
    parser.add_argument("--logits-path", default="", help="Optional external path for BF16 logits; keeps large files out of README assets.")
    parser.add_argument("--bench-repetitions", type=int, default=1)
    parser.add_argument("--bench-prompt-tokens", type=int, default=16)
    parser.add_argument("--bench-gen-tokens", type=int, default=4)
    parser.add_argument("--bench-cache-pair", action="append", default=None)
    parser.add_argument("--perplexity-ctx", type=int, default=32)
    parser.add_argument("--perplexity-chunks", type=int, default=1)
    parser.add_argument("--perplexity-cache-k", default="f16")
    parser.add_argument("--perplexity-cache-v", default="f16")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--generated-at", default="2026-05-03")
    parser.add_argument("--runtime", default="llama.cpp CUDA / RTX 3060")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    models = _parse_model_args(args.model)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = args.eval_source or DEFAULT_EVAL_SOURCES
    corpus = Path(args.corpus_path) if args.corpus_path else out / "gguf_quant_eval_corpus.txt"
    if args.run_perplexity or args.run_logits_kl or not corpus.exists():
        build_eval_corpus(sources, corpus, max_records=args.max_records, min_chars=args.min_corpus_chars)

    rows: list[MetricRow] = file_size_rows(models)
    if args.run_bench:
        cache_args = args.bench_cache_pair or ["f16:f16", "q8_0:q8_0"]
        cache_pairs = [tuple(item.split(":", 1)) for item in cache_args]
        for name, path in models.items():
            bench_rows = run_llama_bench(
                llama_bin_dir=args.llama_bin_dir,
                model_name=name,
                model_path=path,
                out_dir=out,
                repetitions=args.bench_repetitions,
                prompt_tokens=args.bench_prompt_tokens,
                gen_tokens=args.bench_gen_tokens,
                cache_pairs=cache_pairs,
                timeout_sec=args.timeout_sec,
            )
            rows.extend(parse_bench_rows(name, bench_rows))
    if args.run_perplexity:
        logits_path = Path(args.logits_path) if args.logits_path else out / "raw" / "bf16_base_logits.bin"
        if args.run_logits_kl:
            logits_path.parent.mkdir(parents=True, exist_ok=True)
        for name, path in models.items():
            save_logits_to = logits_path if args.run_logits_kl and name == "BF16" else None
            logits_base = logits_path if args.run_logits_kl and name != "BF16" and logits_path.exists() else None
            output = run_llama_perplexity(
                llama_bin_dir=args.llama_bin_dir,
                model_name=name,
                model_path=path,
                corpus_path=corpus,
                out_dir=out,
                ctx_size=args.perplexity_ctx,
                chunks=args.perplexity_chunks,
                timeout_sec=args.timeout_sec,
                cache_k=args.perplexity_cache_k,
                cache_v=args.perplexity_cache_v,
                save_logits_to=save_logits_to,
                logits_base=logits_base,
            )
            rows.extend(parse_perplexity_output(name, output))
    if not args.run_bench and not args.run_perplexity:
        rows = load_existing_raw(out, models)

    report = build_report(
        rows=rows,
        model_paths=models,
        generated_at=args.generated_at,
        runtime=args.runtime,
        corpus_path=str(corpus),
        out_dir=out,
        seed=args.seed,
    )
    print(f"wrote {out / 'gguf_quant_cv_report.json'}")
    print(f"wrote {report['plots']['infographic']}")


if __name__ == "__main__":
    cli()
