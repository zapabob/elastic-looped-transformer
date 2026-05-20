"""Render the 2026-05-20 KV/Triality evidence bundle.

This script stitches together four already-measured surfaces:

* K-protected TheTom KV cache sweep rows.
* Triality SO(8) rotation audit rows from the Turboquant-CUDA workspace.
* ILSD self-distillation entropy/divergence telemetry.
* Loop-aware L=1/2/3 paired correctness arrays for CV multi-group statistics.

The output is intentionally evidence-bounded. It reports unsupported runtime
values as unsupported, and it does not turn local bridge checks into broad
lm-eval leaderboard claims.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import math
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

from elt_lm.eval.benchmark_comparison import compare_group_scores, render_markdown


DEFAULT_KV_REPORT = Path("_docs/assets/2026-05-17-l3-thetom-k-protected/thetom_k_protected_kv_report.json")
DEFAULT_TRIALITY_AUDIT_DIR = Path("../Turboquant-CUDA/_docs/assets/2026-05-17-triality-so8-audit")
DEFAULT_LOOP_AWARE_REPORT = Path(
    "_docs/assets/2026-05-17-l3-thetom-k-protected/loop_aware_l3/loop_aware_l123_logprob32_stem_bridge.json"
)
DEFAULT_OUT_DIR = Path("_docs/assets/2026-05-20-kv-triality-goal")

KV_POLICIES = [
    "K=q8_0_V=turbo3",
    "K=bf16_V=turbo3",
    "K=q8_0_V=turbo4",
    "K=bf16_V=turbo4",
    "K=q8_0_V=turbo8",
    "K=bf16_V=turbo8",
]
TRIALITY_BITS = {3.0, 4.0, 8.0}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_log_reason(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "Unsupported cache type" in line:
            return line.strip()
    return ""


def load_kv_report(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    raw_rows = list(report.get("raw", []))
    summary_rows = {
        (str(row["policy"]), str(row["metric"])): row
        for row in report.get("summary", [])
    }
    pairwise_rows = {
        (str(row["policy"]), str(row["metric"])): row
        for row in report.get("pairwise", [])
    }
    rows: list[dict[str, Any]] = []
    for policy in KV_POLICIES:
        policy_raw = [row for row in raw_rows if row.get("policy") == policy]
        status_counts = {
            "total": len(policy_raw),
            "ok": sum(row.get("status") == "ok" for row in policy_raw),
            "failed": sum(row.get("status") == "failed" for row in policy_raw),
            "timeout": sum(row.get("status") == "timeout" for row in policy_raw),
        }
        gen = summary_rows.get((policy, "gen_tok_s"), {})
        kv = summary_rows.get((policy, "kv_mib"), {})
        pairwise = pairwise_rows.get((policy, "gen_tok_s"), {})
        log_reason = ""
        for item in policy_raw:
            reason = _load_log_reason(Path(str(item.get("log", ""))))
            if reason:
                log_reason = reason
                break
        rows.append(
            {
                "policy": policy,
                "total": status_counts["total"],
                "ok": status_counts["ok"],
                "failed": status_counts["failed"],
                "timeout": status_counts["timeout"],
                "decode_mean": _float_or_none(gen.get("mean")),
                "decode_sem": _float_or_none(gen.get("sem")),
                "kv_mib": _float_or_none(kv.get("mean")),
                "delta_vs_q8_q8": _float_or_none(pairwise.get("mean_delta")),
                "p_value": _float_or_none(pairwise.get("p_value")),
                "unsupported_reason": log_reason,
            }
        )
    return {"source": str(path), "rows": rows}


def load_triality_audit(audit_dir: Path) -> dict[str, Any]:
    summary_path = audit_dir / "triality_so8_rotation_audit_summary.csv"
    status_path = audit_dir / "triality_so8_rotation_audit_status.json"
    status = _read_json(status_path) if status_path.exists() else {}
    rows: list[dict[str, Any]] = []
    with summary_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            bits = float(row["bits"])
            if row.get("view") != "vector" or bits not in TRIALITY_BITS:
                continue
            rows.append(
                {
                    "bits": bits,
                    "view": row["view"],
                    "layers": int(row["layers"]),
                    "blocks": int(row["blocks"]),
                    "dtype": row["storage_dtypes"],
                    "max_orthogonality_error": float(row["max_learned_orthogonality_error"]),
                    "mean_orthogonality_error": float(row["mean_learned_orthogonality_error"]),
                    "mean_determinant": float(row["mean_effective_determinant"]),
                    "max_determinant_error": float(row["max_effective_determinant_error"]),
                    "status": row["status"],
                }
            )
    return {
        "source": str(audit_dir),
        "status": status,
        "rows": sorted(rows, key=lambda item: item["bits"]),
    }


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is not None:
            values.append(value)
    return values


def load_ilsd_runs(patterns: list[str]) -> dict[str, Any]:
    metric_paths: list[Path] = []
    for pattern in patterns:
        metric_paths.extend(Path(path) for path in glob.glob(pattern))
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(set(metric_paths)):
        events: list[dict[str, Any]] = []
        for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
        train_rows = [row for row in events if row.get("event") == "train_step"]
        val_rows = [row for row in events if row.get("event") == "val_probe"]
        metric_rows = train_rows + val_rows
        tail = metric_rows[-8:]
        run_name = metrics_path.parent.name
        lane = run_name.replace("qwen35_4b_side_lora_", "").replace("_aha_ilsd_l3", "").replace("_aha_ilsd_l2", "")
        nonfinite = False
        for row in metric_rows:
            for key in ("loss", "l_dist", "l_entropy"):
                try:
                    value = float(row[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    nonfinite = True
        l_max = next((int(row["L_max"]) for row in events if row.get("event") == "train_config" and "L_max" in row), None)
        rows.append(
            {
                "run": run_name,
                "lane": lane,
                "surface": f"{lane}/L{l_max}" if l_max is not None else lane,
                "L_max": l_max,
                "steps": len(train_rows),
                "last_step": train_rows[-1].get("step") if train_rows else None,
                "tail_loss_mean": mean(_numeric_values(tail, "loss")) if _numeric_values(tail, "loss") else None,
                "last_l_dist": _float_or_none(metric_rows[-1].get("l_dist")) if metric_rows else None,
                "max_l_dist": max(_numeric_values(metric_rows, "l_dist")) if _numeric_values(metric_rows, "l_dist") else None,
                "last_l_entropy": _float_or_none(metric_rows[-1].get("l_entropy")) if metric_rows else None,
                "max_l_entropy": max(_numeric_values(metric_rows, "l_entropy")) if _numeric_values(metric_rows, "l_entropy") else None,
                "nonfinite_detected": nonfinite,
                "path": str(metrics_path),
            }
        )
    return {"patterns": patterns, "rows": rows}


def build_loop_aware_cv(loop_report_path: Path, out_dir: Path, *, permutations: int, seed: int) -> dict[str, Any]:
    payload = _read_json(loop_report_path)
    by_l: dict[str, dict[int, float]] = {}
    for row in payload.get("rows", []):
        case_id = int(row["case_id"])
        group = f"L{int(row['L'])}"
        by_l.setdefault(group, {})[case_id] = float(row["correct"])
    common_ids = sorted(set.intersection(*(set(items) for items in by_l.values()))) if by_l else []
    groups = {
        name: [case_scores[case_id] for case_id in common_ids]
        for name, case_scores in sorted(by_l.items())
    }
    input_payload = {"benchmark": "loop_aware_l123_stem_bridge_mcq_logprob", "groups": groups}
    input_path = out_dir / "loop_aware_l123_cv_groups.json"
    stats_path = out_dir / "loop_aware_l123_cv_stats.json"
    md_path = out_dir / "loop_aware_l123_cv_stats.md"
    input_path.write_text(json.dumps(input_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = compare_group_scores(input_payload["benchmark"], groups, permutations=permutations, seed=seed)
    report["source"] = str(loop_report_path)
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"input": str(input_path), "stats": str(stats_path), "markdown": str(md_path), "report": report}


def lm_eval_status() -> dict[str, Any]:
    cli = shutil.which("lm-eval") or shutil.which("lm_eval")
    return {
        "python_module_available": importlib.util.find_spec("lm_eval") is not None,
        "cli_available": cli is not None,
        "cli_name": Path(cli).name if cli else "",
    }


def _fmt(value: Any, precision: int = 3) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    if abs(number) < 0.001 and number != 0:
        return f"{number:.3e}"
    return f"{number:.{precision}f}"


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(bundle: dict[str, Any], out_dir: Path) -> Path:
    kv_rows = bundle["kv"]["rows"]
    triality_rows = bundle["triality"]["rows"]
    ilsd_rows = bundle["ilsd"]["rows"]
    cv = bundle["loop_aware_cv"]["report"]
    lm_eval = bundle["lm_eval_status"]
    lines = [
        "# 2026-05-20 KV/Triality evidence bundle",
        "",
        "## K-protected TurboQuant KV sweep",
        "",
        "| policy | ok / total | decode tok/s mean +/- SEM | KV MiB | delta vs K=q8_0/V=q8_0 | p | status note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in kv_rows:
        note = "ok" if row["ok"] else (row["unsupported_reason"] or "no successful decode rows")
        p = "n/a" if row["p_value"] is None else _fmt(row["p_value"], 4)
        lines.append(
            f"| `{row['policy']}` | {row['ok']} / {row['total']} | "
            f"{_fmt(row['decode_mean'])} +/- {_fmt(row['decode_sem'])} | {_fmt(row['kv_mib'])} | "
            f"{_fmt(row['delta_vs_q8_q8'])} | {p} | {note} |"
        )
    lines.extend(
        [
            "",
            "## Triality SO(8) rotation audit",
            "",
            f"Audit status: `{bundle['triality']['status'].get('status', 'unknown')}`; "
            f"rows audited: `{bundle['triality']['status'].get('rows', 'n/a')}`; "
            f"outliers: `{bundle['triality']['status'].get('outliers', 'n/a')}`.",
            "",
            "| bits | max orth err | mean det | max det err | status |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in triality_rows:
        lines.append(
            f"| {row['bits']:.0f} | {row['max_orthogonality_error']:.3e} | "
            f"{row['mean_determinant']:.9f} | {row['max_determinant_error']:.3e} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## ILSD entropy and divergence monitor",
            "",
            "| lane | L_max | steps | tail loss | max L-dist | last L-entropy | nonfinite |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in ilsd_rows:
        lines.append(
            f"| `{row['lane']}` | {row['L_max']} | {row['steps']} | {_fmt(row['tail_loss_mean'])} | "
            f"{_fmt(row['max_l_dist'])} | {_fmt(row['last_l_entropy'])} | `{row['nonfinite_detected']}` |"
        )
    lines.extend(
        [
            "",
            "## Loop-aware CV multi-group comparison",
            "",
            "| group | n | mean | sd | sem | 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cv["summaries"]:
        lines.append(
            "| {name} | {n} | {mean:.4f} | {std:.4f} | {sem:.4f} | "
            "[{ci95_low:.4f}, {ci95_high:.4f}] |".format(**row)
        )
    lines.extend(["", "| comparison | mean delta | p |", "|---|---:|---:|"])
    for row in cv["pairwise"]:
        lines.append("| {left} - {right} | {mean_delta:.4f} | {p_value:.6f} |".format(**row))
    if cv.get("omnibus"):
        lines.extend(
            [
                "",
                f"Friedman within-block permutation p: `{cv['omnibus']['p_value']:.6f}` "
                f"(statistic `{cv['omnibus']['statistic']:.4f}`, n={cv['omnibus']['n_blocks']}).",
            ]
        )
    if lm_eval["python_module_available"]:
        lm_eval_note = "lm-eval is importable in this environment, but no broad logged lm-eval run is included in this bundle."
    elif lm_eval["cli_available"]:
        lm_eval_note = (
            "A global lm-eval CLI is visible, but this repo's Python environment cannot import lm_eval; "
            "no broad logged lm-eval run is included in this bundle."
        )
    else:
        lm_eval_note = "lm-eval is not available in this repo environment; no broad logged lm-eval run is included in this bundle."
    lines.extend(
        [
            "",
            "## lm-eval-harness status",
            "",
            f"- Python module available: `{lm_eval['python_module_available']}`",
            f"- CLI available: `{lm_eval['cli_available']}`"
            + (f" (`{lm_eval['cli_name']}`)" if lm_eval.get("cli_name") else ""),
            "",
            lm_eval_note,
            "",
            "The current numbers are local bridge/external-heldout evidence. Broad "
            "lm-eval-harness leaderboard claims remain blocked until the same paired "
            "task set is completed under lm-eval with logged samples.",
        ]
    )
    path = out_dir / "kv_triality_goal_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_dashboard(bundle: dict[str, Any], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kv_rows = bundle["kv"]["rows"]
    triality_rows = bundle["triality"]["rows"]
    ilsd_rows = bundle["ilsd"]["rows"]
    cv_rows = bundle["loop_aware_cv"]["report"]["summaries"]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.2))
    fig.patch.set_facecolor("#f6f7f2")

    ax = axes[0][0]
    labels = [row["policy"].replace("K=", "K ").replace("_V=", "\nV ") for row in kv_rows]
    values = [row["decode_mean"] or 0.0 for row in kv_rows]
    errors = [row["decode_sem"] or 0.0 for row in kv_rows]
    colors = ["#167f7a" if "q8_0" in row["policy"] else "#d28e2b" for row in kv_rows]
    bars = ax.bar(range(len(kv_rows)), values, yerr=errors, capsize=4, color=colors)
    for bar, row in zip(bars, kv_rows):
        if row["ok"] == 0:
            bar.set_hatch("//")
            bar.set_edgecolor("#8a2f2a")
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "unsupported", rotation=90, ha="center", va="bottom", fontsize=8)
    ax.set_title("K protected, V swept")
    ax.set_ylabel("decode tok/s")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0][1]
    bit_labels = [f"{row['bits']:.0f}-bit" for row in triality_rows]
    orth = [row["max_orthogonality_error"] for row in triality_rows]
    det = [row["max_determinant_error"] for row in triality_rows]
    x = range(len(triality_rows))
    ax.bar([i - 0.18 for i in x], orth, width=0.36, label="orth err", color="#3b6ea8")
    ax.bar([i + 0.18 for i in x], det, width=0.36, label="det err", color="#b05f3c")
    ax.axhline(0.01, color="#1e1e1e", linestyle="--", linewidth=1, label="gate 0.01")
    ax.set_title("Triality SO(8) audit")
    ax.set_ylabel("max error")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bit_labels)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1][0]
    lanes = [row["surface"] for row in ilsd_rows]
    ldist = [row["max_l_dist"] or 0.0 for row in ilsd_rows]
    lentropy = [row["last_l_entropy"] or 0.0 for row in ilsd_rows]
    ax.bar(range(len(lanes)), ldist, color="#6f5aa7", label="max L-dist")
    ax2 = ax.twinx()
    ax2.plot(range(len(lanes)), lentropy, color="#1c8f59", marker="o", label="last L-entropy")
    ax.set_title("ILSD divergence / entropy monitor")
    ax.set_xticks(range(len(lanes)))
    ax.set_xticklabels(lanes, fontsize=8)
    ax.set_ylabel("max L-dist")
    ax2.set_ylabel("last L-entropy")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1][1]
    groups = [row["name"] for row in cv_rows]
    means = [row["mean"] for row in cv_rows]
    sem = [row["sem"] for row in cv_rows]
    ax.bar(range(len(groups)), means, yerr=sem, capsize=5, color=["#778899", "#3b8c88", "#c1843a"])
    ax.set_ylim(0, 1)
    ax.set_title("Loop-aware CV accuracy")
    ax.set_ylabel("accuracy")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.grid(axis="y", alpha=0.25)

    for ax in axes.ravel():
        ax.set_facecolor("#ffffff")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Qwen3.5 ELT: KV, Triality, entropy, and CV evidence", fontsize=17, fontweight="bold")
    lm_eval = bundle["lm_eval_status"]
    lm_note = "lm-eval module available" if lm_eval["python_module_available"] else "lm-eval module absent in repo env"
    if lm_eval["cli_available"] and not lm_eval["python_module_available"]:
        lm_note = "global lm-eval CLI visible, repo module absent"
    fig.text(
        0.5,
        0.02,
        f"Evidence-bounded: Turbo8 is unsupported by the installed llama.cpp KV parser; {lm_note}; no broad logged lm-eval run in this bundle.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.1)
    path = out_dir / "gptimage2_kv_triality_goal_dashboard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def copy_triality_assets(audit_dir: Path, out_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in [
        "triality_so8_rotation_audit_summary.csv",
        "triality_so8_rotation_audit_detail.csv",
        "triality_so8_rotation_audit_status.json",
        "triality_so8_orthogonality_by_bit_view.png",
        "triality_so8_determinant_by_bit_view.png",
    ]:
        src = audit_dir / name
        if src.exists():
            dst = out_dir / name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-report", type=Path, default=DEFAULT_KV_REPORT)
    parser.add_argument("--triality-audit-dir", type=Path, default=DEFAULT_TRIALITY_AUDIT_DIR)
    parser.add_argument("--loop-aware-report", type=Path, default=DEFAULT_LOOP_AWARE_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--ilsd-run-glob",
        action="append",
        default=[
            "H:/elt_data/runs/qwen35_4b_side_lora_*_aha_ilsd_l3/metrics.jsonl",
            "H:/elt_data/runs/qwen35_4b_side_lora_*_aha_ilsd_l2/metrics.jsonl",
        ],
    )
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "kv": load_kv_report(args.kv_report),
        "triality": load_triality_audit(args.triality_audit_dir),
        "ilsd": load_ilsd_runs(args.ilsd_run_glob),
        "loop_aware_cv": build_loop_aware_cv(
            args.loop_aware_report,
            args.out_dir,
            permutations=args.permutations,
            seed=args.seed,
        ),
        "lm_eval_status": lm_eval_status(),
    }
    copied = copy_triality_assets(args.triality_audit_dir, args.out_dir)
    bundle["copied_triality_assets"] = copied
    write_csv_rows(args.out_dir / "kv_triality_goal_kv_summary.csv", bundle["kv"]["rows"])
    write_csv_rows(args.out_dir / "kv_triality_goal_triality_summary.csv", bundle["triality"]["rows"])
    write_csv_rows(args.out_dir / "kv_triality_goal_ilsd_summary.csv", bundle["ilsd"]["rows"])
    markdown = write_markdown(bundle, args.out_dir)
    dashboard = write_dashboard(bundle, args.out_dir)
    bundle["outputs"] = {"markdown": str(markdown), "dashboard": str(dashboard)}
    (args.out_dir / "kv_triality_goal_report.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "gptimage2_kv_triality_goal_prompt.md").write_text(
        "Create a clean AI-engineering benchmark dashboard for Qwen3.5 ELT showing "
        "K-protected KV cache results, Triality SO(8) orthogonality/determinant gates, "
        "self-distillation entropy monitoring, and loop-aware CV p-values. "
        "Mark Turbo8 as unsupported by the installed runtime.\n",
        encoding="utf-8",
    )
    print(f"wrote {markdown}")
    print(f"wrote {dashboard}")


if __name__ == "__main__":
    main()
