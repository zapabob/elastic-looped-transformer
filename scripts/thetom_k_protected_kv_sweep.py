"""Run and summarize TheTom TurboQuant KV cache sweeps with protected K.

The sweep intentionally keeps K at ``q8_0`` or ``bf16`` while V is swept across
stock and TheTom ``turbo*`` cache types. Unsupported runtime values are kept as
failed rows so the public table can distinguish "not supported by this binary"
from "measured but slow." This keeps the runtime claim separate from TQ4_1S GGUF
weight compression.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from elt_lm.eval.statistics import paired_permutation_pvalue


PROMPT_TOK_S = re.compile(r"(?m)^\s*prompt eval time\s*=.*?\(\s*(?:[0-9.]+)\s+ms per token,\s*([0-9.]+)\s+tokens per second\)")
GEN_TOK_S = re.compile(r"(?m)^\s*eval time\s*=.*?\(\s*(?:[0-9.]+)\s+ms per token,\s*([0-9.]+)\s+tokens per second\)")
KV_MIB = re.compile(r"KV buffer size\s*=\s*([0-9.]+)\s*MiB")
KV_DETAIL_MIB = re.compile(r"llama_kv_cache:\s*size\s*=\s*([0-9.]+)\s*MiB")


DEFAULT_POLICIES = [
    ("f16", "f16"),
    ("q8_0", "q8_0"),
    ("q8_0", "turbo2"),
    ("bf16", "turbo2"),
    ("q8_0", "turbo3"),
    ("bf16", "turbo3"),
    ("q8_0", "turbo4"),
    ("bf16", "turbo4"),
    ("q8_0", "turbo8"),
    ("bf16", "turbo8"),
]


def _float_or_none(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "sem": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    sem = sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": avg,
        "sd": sd,
        "sem": sem,
        "ci95_low": avg - 1.96 * sem,
        "ci95_high": avg + 1.96 * sem,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _parse_row(
    *,
    text: str,
    raw_path: Path,
    cache_k: str,
    cache_v: str,
    repetition: int,
    status_override: str | None = None,
    returncode: int | str = "",
    wall_sec: float | str = "",
) -> dict[str, Any]:
    label = f"K={cache_k}_V={cache_v}"
    kv_detail_values = [float(item) for item in KV_DETAIL_MIB.findall(text)]
    kv_values = [float(item) for item in KV_MIB.findall(text)]
    prompt_tok_s = _float_or_none(PROMPT_TOK_S, text)
    gen_tok_s = _float_or_none(GEN_TOK_S, text)
    status = status_override if status_override is not None else ("ok" if gen_tok_s is not None else "failed")
    return {
        "policy": label,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "k_protected": cache_k in {"q8_0", "bf16"},
        "repetition": repetition,
        "status": status,
        "returncode": returncode,
        "prompt_tok_s": prompt_tok_s if prompt_tok_s is not None else "",
        "gen_tok_s": gen_tok_s if gen_tok_s is not None else "",
        "kv_mib": kv_detail_values[-1] if kv_detail_values else (sum(kv_values) if kv_values else ""),
        "wall_sec": wall_sec,
        "log": str(raw_path),
    }


def _run_one(
    *,
    llama_cli: Path,
    model: Path,
    prompt: str,
    out_dir: Path,
    cache_k: str,
    cache_v: str,
    repetition: int,
    ctx_size: int,
    gen_tokens: int,
    ngl: int,
    timeout_sec: int,
    reuse_existing: bool,
) -> dict[str, Any]:
    label = f"K={cache_k}_V={cache_v}"
    raw_path = out_dir / "raw" / f"cli_{label}_rep{repetition}.log"
    verbose_path = out_dir / "raw" / f"cli_{label}_rep{repetition}.verbose.log"
    if reuse_existing and raw_path.exists():
        return _parse_row(
            text=raw_path.read_text(encoding="utf-8", errors="replace"),
            raw_path=raw_path,
            cache_k=cache_k,
            cache_v=cache_v,
            repetition=repetition,
        )
    if reuse_existing and verbose_path.exists():
        text = verbose_path.read_text(encoding="utf-8", errors="replace")
        raw_path.write_text(text, encoding="utf-8")
        return _parse_row(
            text=text,
            raw_path=raw_path,
            cache_k=cache_k,
            cache_v=cache_v,
            repetition=repetition,
            status_override="timeout",
            returncode="timeout",
        )

    command = [
        str(llama_cli),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(gen_tokens),
        "-c",
        str(ctx_size),
        "-ngl",
        str(ngl),
        "--temp",
        "0",
        "--seed",
        "42",
        "--cache-type-k",
        cache_k,
        "--cache-type-v",
        cache_v,
        "--no-display-prompt",
        "--no-warmup",
        "--simple-io",
        "--single-turn",
        "--log-file",
        str(verbose_path),
        "--log-verbose",
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
        stdout = proc.stdout
        returncode: int | str = proc.returncode
        status_override = None
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_output(exc.stdout) + _coerce_output(exc.stderr)
        returncode = "timeout"
        status_override = "timeout"
    wall_sec = time.perf_counter() - started
    verbose_text = verbose_path.read_text(encoding="utf-8", errors="replace") if verbose_path.exists() else ""
    combined = stdout + "\n" + verbose_text
    raw_path.write_text(combined, encoding="utf-8")
    row = _parse_row(
        text=combined,
        raw_path=raw_path,
        cache_k=cache_k,
        cache_v=cache_v,
        repetition=repetition,
        status_override=status_override,
        returncode=returncode,
        wall_sec=wall_sec,
    )
    if status_override is None and returncode != 0:
        row["status"] = "failed"
    return row


def _summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    policies = list(dict.fromkeys(str(row["policy"]) for row in rows))
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy and row["status"] == "ok"]
        for metric in ("prompt_tok_s", "gen_tok_s", "kv_mib", "wall_sec"):
            values = [float(row[metric]) for row in policy_rows if row[metric] != ""]
            item = {"policy": policy, "metric": metric}
            item.update(_summary(values))
            summaries.append(item)

    baseline = "K=q8_0_V=q8_0"
    baseline_rows = [row for row in rows if row["policy"] == baseline and row["status"] == "ok"]
    for policy in policies:
        if policy == baseline:
            continue
        current_rows = [row for row in rows if row["policy"] == policy and row["status"] == "ok"]
        for metric in ("gen_tok_s", "kv_mib"):
            base_by_rep = {int(row["repetition"]): float(row[metric]) for row in baseline_rows if row[metric] != ""}
            cur_by_rep = {int(row["repetition"]): float(row[metric]) for row in current_rows if row[metric] != ""}
            reps = sorted(set(base_by_rep) & set(cur_by_rep))
            if not reps:
                continue
            left = [cur_by_rep[rep] for rep in reps]
            right = [base_by_rep[rep] for rep in reps]
            pairwise.append({
                "policy": policy,
                "baseline": baseline,
                "metric": metric,
                "n": len(reps),
                "mean_delta": mean(left) - mean(right),
                "p_value": paired_permutation_pvalue(left, right, seed=17) if len(reps) > 1 else "",
                "method": "paired permutation vs q8_0/q8_0" if len(reps) > 1 else "single paired block",
            })
    return summaries, pairwise


def _write_markdown(out_dir: Path, raw_rows: list[dict[str, Any]], summaries: list[dict[str, Any]], pairwise: list[dict[str, Any]]) -> None:
    metric_rows = [row for row in summaries if row["metric"] in {"gen_tok_s", "kv_mib"} and int(row["n"]) > 0]
    lines = [
        "# TheTom TurboQuant K-Protected KV Sweep",
        "",
        "K is held at `q8_0` or `bf16`; only V is swept into TheTom `turbo*` cache types.",
        "",
        "## Run Status",
        "",
        "| policy | total | ok | failed | timeout |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in list(dict.fromkeys(str(row["policy"]) for row in raw_rows)):
        policy_rows = [row for row in raw_rows if row["policy"] == policy]
        lines.append(
            f"| `{policy}` | {len(policy_rows)} | "
            f"{sum(row['status'] == 'ok' for row in policy_rows)} | "
            f"{sum(row['status'] == 'failed' for row in policy_rows)} | "
            f"{sum(row['status'] == 'timeout' for row in policy_rows)} |"
        )
    lines.extend([
        "",
        "## Summary",
        "",
        "| policy | metric | n | mean | SEM | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in metric_rows:
        lines.append(
            f"| `{row['policy']}` | `{row['metric']}` | {int(row['n'])} | "
            f"{float(row['mean']):.4g} | {float(row['sem']):.4g} | "
            f"{float(row['ci95_low']):.4g}..{float(row['ci95_high']):.4g} |"
        )
    lines.extend(["", "## Pairwise vs K=q8_0/V=q8_0", "", "| policy | metric | n | mean delta | p |", "|---|---:|---:|---:|---:|"])
    for row in pairwise:
        p = row["p_value"]
        lines.append(
            f"| `{row['policy']}` | `{row['metric']}` | {row['n']} | "
            f"{float(row['mean_delta']):.4g} | {float(p):.4g} |" if p != "" else
            f"| `{row['policy']}` | `{row['metric']}` | {row['n']} | {float(row['mean_delta']):.4g} |  |"
        )
    (out_dir / "thetom_k_protected_kv_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = [row for row in summaries if row["metric"] == "gen_tok_s" and int(row["n"]) > 0]
    kv = [row for row in summaries if row["metric"] == "kv_mib" and int(row["n"]) > 0]
    labels = []
    colors = []
    for row in gen:
        policy = str(row["policy"])
        k_label, v_label = policy.split("_V=", 1)
        labels.append(f"{k_label}\nV={v_label}")
        if "K=bf16" in policy:
            colors.append("#d28e2b")
        elif "K=q8_0" in policy:
            colors.append("#167f7a")
        else:
            colors.append("#646a73")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    fig.patch.set_facecolor("#f7f8f5")
    for ax, data, title, ylabel in [
        (axes[0], gen, "Decode throughput", "tokens/s"),
        (axes[1], kv, "KV memory", "MiB"),
    ]:
        ax.set_facecolor("#ffffff")
        ax.bar(range(len(data)), [float(row["mean"]) for row in data], yerr=[float(row["sem"]) for row in data], capsize=4, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("ELT L=3 TheTom TurboQuant KV: K protected, V swept", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.02, "RTX 3060 was busy, so this is a CPU/offload smoke sweep; K is never turbo*, only q8_0 or bf16.", ha="center", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=0.84)
    fig.savefig(out_dir / "gptimage_l3_thetom_k_protected_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="ELT loop depth L=3. Explain why K is protected in one concise sentence.")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--policies", default="", help="comma list such as q8_0:turbo2,bf16:turbo2")
    parser.add_argument("--reuse-existing", action="store_true", help="Parse existing raw logs instead of rerunning them.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "raw").mkdir(parents=True, exist_ok=True)
    policies = DEFAULT_POLICIES
    if args.policies:
        policies = [tuple(item.split(":", 1)) for item in args.policies.split(",")]  # type: ignore[list-item]

    rows: list[dict[str, Any]] = []
    for cache_k, cache_v in policies:
        for rep in range(int(args.repetitions)):
            rows.append(_run_one(
                llama_cli=args.llama_cli,
                model=args.model,
                prompt=args.prompt,
                out_dir=args.out_dir,
                cache_k=cache_k,
                cache_v=cache_v,
                repetition=rep,
                ctx_size=args.ctx_size,
                gen_tokens=args.gen_tokens,
                ngl=args.ngl,
                timeout_sec=args.timeout_sec,
                reuse_existing=args.reuse_existing,
            ))

    raw_fields = ["policy", "cache_k", "cache_v", "k_protected", "repetition", "status", "returncode", "prompt_tok_s", "gen_tok_s", "kv_mib", "wall_sec", "log"]
    _write_csv(args.out_dir / "thetom_k_protected_kv_raw.csv", rows, raw_fields)
    summaries, pairwise = _summarize(rows)
    _write_csv(args.out_dir / "thetom_k_protected_kv_summary.csv", summaries, ["policy", "metric", "n", "mean", "sd", "sem", "ci95_low", "ci95_high"])
    _write_csv(args.out_dir / "thetom_k_protected_kv_pairwise.csv", pairwise, ["policy", "baseline", "metric", "n", "mean_delta", "p_value", "method"])
    (args.out_dir / "thetom_k_protected_kv_report.json").write_text(
        json.dumps({"raw": rows, "summary": summaries, "pairwise": pairwise}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(args.out_dir, rows, summaries, pairwise)
    _write_plot(args.out_dir, summaries)


if __name__ == "__main__":
    main()
