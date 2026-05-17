"""Render README-ready L3 evidence statistics and error-bar figure."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path("_docs/assets/2026-05-17-l3-thetom-k-protected")
LOCAL_PATH = BASE / "llm_quality_128/l3_128_gpu_ngl999_stem_bridge.json"
MMLU_PATH = BASE / "external_heldout/l3_external_mmlu_stem16_gpu_ngl999.json"
GSM8K_PATH = BASE / "external_heldout/l3_external_gsm8k16_gpu_ngl999.json"
LOOP_PATH = BASE / "loop_aware_l3/loop_aware_l123_logprob32_stem_bridge.json"
OUT_PNG = BASE / "l3_readme_accuracy_errorbars.png"
OUT_JSON = BASE / "l3_readme_stats.json"
OUT_MD = BASE / "l3_readme_stats.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a, b], [c, d]]."""
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2

    def hypergeom(x: int) -> float:
        return (
            math.comb(col1, x)
            * math.comb(total - col1, row1 - x)
            / math.comb(total, row1)
        )

    lo = max(0, row1 - (total - col1))
    hi = min(row1, col1)
    observed = hypergeom(a)
    return min(1.0, sum(hypergeom(x) for x in range(lo, hi + 1) if hypergeom(x) <= observed + 1e-15))


def _binom_exact_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _fmt_p(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _eval_summary(name: str, payload: dict) -> dict:
    summary = payload["summary"]
    acc = summary["accuracy"]
    total = int(summary["completed"])
    correct = int(summary["correct"])
    return {
        "name": name,
        "correct": correct,
        "total": total,
        "accuracy": acc["mean"],
        "ci95_low": acc["ci95_low"],
        "ci95_high": acc["ci95_high"],
        "sem": acc["sem"],
        "mean_prompt_tok_s": summary.get("mean_prompt_tok_s"),
        "mean_decode_tok_s": summary.get("mean_decode_tok_s"),
    }


def _loop_pair(loop_payload: dict, left: int, right: int) -> dict:
    by_l: dict[int, dict[int, int]] = {}
    for row in loop_payload["rows"]:
        by_l.setdefault(int(row["L"]), {})[int(row["case_id"])] = int(row["correct"])
    common = sorted(set(by_l[left]) & set(by_l[right]))
    left_wrong_right_right = 0
    left_right_right_wrong = 0
    for case_id in common:
        a = by_l[left][case_id]
        b = by_l[right][case_id]
        if a == 0 and b == 1:
            left_wrong_right_right += 1
        elif a == 1 and b == 0:
            left_right_right_wrong += 1
    discordant = left_wrong_right_right + left_right_right_wrong
    return {
        "comparison": f"L{left}_vs_L{right}",
        "n": len(common),
        "improved": left_wrong_right_right,
        "regressed": left_right_right_wrong,
        "discordant": discordant,
        "mcnemar_exact_p": _binom_exact_two_sided(
            min(left_wrong_right_right, left_right_right_wrong), discordant
        ),
    }


def main() -> None:
    local = _eval_summary("Local STEM bridge", _load(LOCAL_PATH))
    mmlu = _eval_summary("MMLU-STEM heldout", _load(MMLU_PATH))
    gsm8k = _eval_summary("GSM8K heldout", _load(GSM8K_PATH))
    loop = _load(LOOP_PATH)
    evals = [local, mmlu, gsm8k]

    fisher = []
    for left, right in [(local, mmlu), (local, gsm8k), (mmlu, gsm8k)]:
        p_value = _fisher_exact_two_sided(
            left["correct"],
            left["total"] - left["correct"],
            right["correct"],
            right["total"] - right["correct"],
        )
        fisher.append({
            "comparison": f"{left['name']} vs {right['name']}",
            "test": "Fisher exact two-sided",
            "p": p_value,
        })

    loop_pairs = [_loop_pair(loop, 1, 2), _loop_pair(loop, 1, 3), _loop_pair(loop, 2, 3)]
    stats = {
        "source_date": "2026-05-17",
        "model": "L3 Qwen3.5 ELT Q8_0 GGUF / RTX 3060 / --ngl 999 / K=q8_0, V=turbo3",
        "evaluation_summaries": evals,
        "fisher_exact": fisher,
        "loop_summaries": loop["summaries"],
        "loop_mcnemar_exact": loop_pairs,
        "notes": [
            "Accuracy intervals are Wilson 95% confidence intervals from the source evaluators.",
            "External heldouts are small cached slices; p-values are descriptive evidence, not leaderboard claims.",
            "Loop-aware rows use paired case ids and exact McNemar/binomial tests over discordant cases.",
        ],
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=180)
    fig.patch.set_facecolor("#0b0f14")
    colors = ["#3ddc97", "#f5b14c", "#ff5f6d"]
    text = "#f4f7fb"
    muted = "#aab4c0"
    grid = "#263242"

    ax = axes[0]
    names = ["Local\nSTEM\nn=128", "MMLU\nSTEM\nn=16", "GSM8K\nn=16"]
    means = np.array([item["accuracy"] for item in evals]) * 100
    lows = np.array([item["ci95_low"] for item in evals]) * 100
    highs = np.array([item["ci95_high"] for item in evals]) * 100
    x = np.arange(len(evals))
    ax.bar(x, means, color=colors, edgecolor="#e8eef7", linewidth=0.8)
    ax.errorbar(
        x,
        means,
        yerr=[means - lows, highs - means],
        fmt="none",
        ecolor="#e8eef7",
        elinewidth=1.5,
        capsize=6,
    )
    for idx, item in enumerate(evals):
        ax.text(idx, min(101, means[idx] + 4), f"{item['correct']}/{item['total']}\n{means[idx]:.1f}%", ha="center", va="bottom", color=text, fontsize=10)
    ax.set_title("External quality gate accuracy (95% CI)", color=text, fontsize=13, weight="bold")
    ax.set_ylabel("Accuracy (%)", color=muted)
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color=text)
    ax.tick_params(colors=muted)
    ax.grid(axis="y", color=grid, alpha=0.75)
    ax.set_facecolor("#101720")
    for spine in ax.spines.values():
        spine.set_color("#2b3545")
    ax.text(
        0.02,
        0.08,
        "Fisher exact p-values\n"
        f"Local vs MMLU: {_fmt_p(fisher[0]['p'])}\n"
        f"Local vs GSM8K: {_fmt_p(fisher[1]['p'])}\n"
        f"MMLU vs GSM8K: {_fmt_p(fisher[2]['p'])}",
        transform=ax.transAxes,
        color=muted,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#17202b", "edgecolor": "#334155"},
    )

    ax = axes[1]
    loop_summaries = loop["summaries"]
    loop_x = np.array([item["L"] for item in loop_summaries])
    loop_mean = np.array([item["mean"] for item in loop_summaries]) * 100
    loop_low = np.array([item["ci95_low"] for item in loop_summaries]) * 100
    loop_high = np.array([item["ci95_high"] for item in loop_summaries]) * 100
    ax.plot(loop_x, loop_mean, color="#b694ff", lw=3.2, marker="o", ms=8, mfc="#3ddc97", mec="#e8eef7")
    ax.fill_between(loop_x, loop_low, loop_high, color="#b694ff", alpha=0.18)
    for item, value in zip(loop_summaries, loop_mean):
        ax.text(item["L"], value + 5, f"L={item['L']}\n{item['correct']}/{item['total']}\n{value:.1f}%", ha="center", color=text, fontsize=10)
    ax.set_title("Loop-aware HF runtime: depth effect (95% CI)", color=text, fontsize=13, weight="bold")
    ax.set_xlabel("Loop depth L", color=muted)
    ax.set_ylabel("MCQ logprob accuracy (%)", color=muted)
    ax.set_xlim(0.75, 3.25)
    ax.set_ylim(20, 100)
    ax.set_xticks([1, 2, 3])
    ax.tick_params(colors=muted)
    ax.grid(color=grid, alpha=0.75)
    ax.set_facecolor("#101720")
    for spine in ax.spines.values():
        spine.set_color("#2b3545")
    ax.text(
        0.04,
        0.86,
        "Paired McNemar exact\n"
        f"L1 vs L2: p={_fmt_p(loop_pairs[0]['mcnemar_exact_p'])}\n"
        f"L1 vs L3: p={_fmt_p(loop_pairs[1]['mcnemar_exact_p'])}\n"
        f"L2 vs L3: p={_fmt_p(loop_pairs[2]['mcnemar_exact_p'])}",
        transform=ax.transAxes,
        color=muted,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#17202b", "edgecolor": "#334155"},
    )

    fig.suptitle("L3 Qwen3.5 ELT evidence gates - summary statistics and p-values", color=text, fontsize=16, weight="bold")
    fig.text(0.5, 0.02, "Evidence bounded: strong local STEM bridge / small external heldouts / GSM8K not solved", ha="center", color=muted, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.93])
    fig.savefig(OUT_PNG, facecolor="#0b0f14", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# L3 README evidence statistics",
        "",
        "## Accuracy gates",
        "",
        "| evaluation | n | correct | accuracy | Wilson 95% CI | SEM | prompt tok/s | decode tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in evals:
        lines.append(
            f"| {item['name']} | {item['total']} | {item['correct']} | {item['accuracy']*100:.1f}% | "
            f"[{item['ci95_low']*100:.1f}, {item['ci95_high']*100:.1f}] | {item['sem']*100:.2f}% | "
            f"{item['mean_prompt_tok_s']:.2f} | {item['mean_decode_tok_s']:.2f} |"
        )
    lines.extend([
        "",
        "## Pairwise p-values",
        "",
        "| comparison | test | p |",
        "|---|---|---:|",
    ])
    for item in fisher:
        lines.append(f"| {item['comparison']} | {item['test']} | {_fmt_p(item['p'])} |")
    lines.extend([
        "",
        "## Loop-aware depth",
        "",
        "| L | n | correct | accuracy | Wilson 95% CI | SEM | mean margin | wall sec/case |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in loop_summaries:
        lines.append(
            f"| {item['L']} | {item['total']} | {item['correct']} | {item['mean']*100:.1f}% | "
            f"[{item['ci95_low']*100:.1f}, {item['ci95_high']*100:.1f}] | {item['sem']*100:.2f}% | "
            f"{item['mean_margin_logprob']:.4f} | {item['mean_wall_sec']:.3f} |"
        )
    lines.extend([
        "",
        "| comparison | improved | regressed | discordant | paired exact p |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in loop_pairs:
        lines.append(
            f"| {item['comparison']} | {item['improved']} | {item['regressed']} | "
            f"{item['discordant']} | {_fmt_p(item['mcnemar_exact_p'])} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(OUT_PNG.resolve())
    print(OUT_JSON.resolve())
    print(OUT_MD.resolve())


if __name__ == "__main__":
    main()
