"""Compare paired benchmark scores across vanilla and tuned model groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elt_lm.eval.statistics import (
    friedman_permutation_test,
    pairwise_group_comparisons,
    summarize_scores,
)


def _as_float_scores(raw: Any, *, name: str) -> list[float]:
    if not isinstance(raw, list):
        raise ValueError(f"group {name!r} must be a list of paired scores")
    values = [float(item) for item in raw]
    if not values:
        raise ValueError(f"group {name!r} has no scores")
    return values


def load_group_scores(path: str | Path) -> tuple[str, dict[str, list[float]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    benchmark = str(payload.get("benchmark") or Path(path).stem)
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, dict) or not raw_groups:
        raise ValueError("comparison input requires a non-empty 'groups' object")
    groups = {str(name): _as_float_scores(scores, name=str(name)) for name, scores in raw_groups.items()}
    lengths = {len(scores) for scores in groups.values()}
    if len(lengths) != 1:
        raise ValueError("all groups must have the same number of paired scores")
    return benchmark, groups


def compare_group_scores(
    benchmark: str,
    groups: dict[str, list[float]],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    summaries = [summarize_scores(name, scores).__dict__ for name, scores in groups.items()]
    pairwise = [
        item.__dict__
        for item in pairwise_group_comparisons(groups, permutations=permutations, seed=seed)
    ]
    omnibus = None
    if len(groups) >= 3:
        omnibus = friedman_permutation_test(
            groups,
            permutations=permutations,
            seed=seed,
        ).__dict__
    return {
        "benchmark": benchmark,
        "n_groups": len(groups),
        "n_blocks": len(next(iter(groups.values()))),
        "summaries": summaries,
        "pairwise": pairwise,
        "omnibus": omnibus,
        "notes": [
            "Scores are paired by row/fold index across groups.",
            "p_value fields use deterministic within-pair/block permutation tests.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"### {report['benchmark']}",
        "",
        "| group | n | mean | sd | sem | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summaries"]:
        lines.append(
            "| {name} | {n} | {mean:.4f} | {std:.4f} | {sem:.4f} | "
            "[{ci95_low:.4f}, {ci95_high:.4f}] |".format(**row)
        )
    lines.extend(["", "| comparison | mean delta | p | method |", "|---|---:|---:|---|"])
    for row in report["pairwise"]:
        lines.append(
            "| {left} - {right} | {mean_delta:.4f} | {p_value:.6f} | {method} |".format(**row)
        )
    if report["omnibus"]:
        row = report["omnibus"]
        lines.extend([
            "",
            "| omnibus | statistic | p | method |",
            "|---|---:|---:|---|",
            "| Friedman | {statistic:.4f} | {p_value:.6f} | {method} |".format(**row),
        ])
    return "\n".join(lines) + "\n"


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON with {'benchmark', 'groups': {name: scores}}")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    benchmark, groups = load_group_scores(args.input)
    report = compare_group_scores(
        benchmark,
        groups,
        permutations=args.permutations,
        seed=args.seed,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown(report), encoding="utf-8")
        print(f"wrote {out_md}")


if __name__ == "__main__":
    cli()
