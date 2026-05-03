"""Bridge GRPO diagnostics grounded in completed lane telemetry.

The diagnostic is intentionally read-only. It summarizes GRPO lane metrics and
turns them into follow-up actions before another expensive run is launched.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import statistics
import time
from typing import Any


DEFAULT_LANE_RUNS: dict[str, str] = {
    "code": "grpo_side_lora_code_synthetic_v2_bridge",
    "math": "grpo_side_lora_math_synthetic_v2_bridge",
    "stem": "grpo_side_lora_stem_synthetic_v2_bridge",
    "tool": "grpo_side_lora_tool_synthetic_v2_bridge",
}


DEFAULT_POLICY: dict[str, Any] = {
    "thresholds": {
        "min_grpo_steps": 24,
        "strong_mean_correct_rate": 0.75,
        "promising_mean_correct_rate": 0.25,
        "min_mean_format_rate": 0.95,
        "min_adv_signal_steps": 4,
        "nonzero_eps": 1.0e-9,
    },
    "actions": {
        "ready_for_export_eval": (
            "export and run bounded held-out eval; do not extend RL until "
            "generalization is measured"
        ),
        "promising_but_unstable": (
            "refresh prompts and run a short low-LR GRPO continuation only "
            "after held-out verifier replay"
        ),
        "unstable_sparse_success": (
            "return to replay SFT / prompt repair before more GRPO; success "
            "exists but is too sparse"
        ),
        "blocked_no_reward_signal": (
            "stop GRPO continuation; inspect verifier, answer schema, and "
            "failure-contrast SFT until nonzero rewards appear"
        ),
        "format_repair": "repair formatting examples before reward optimization",
        "incomplete": "finish or repair the lane run before making training decisions",
        "missing": "create or restore the lane run before analysis",
    },
}


@dataclass
class LaneSummary:
    lane: str
    run_dir: str
    metrics_path: str
    exists: bool
    run_end: bool = False
    event_count: int = 0
    grpo_step_count: int = 0
    latest_grpo_step: int | None = None
    final_checkpoint_step: int | None = None
    final_correct_rate: float | None = None
    final_format_rate: float | None = None
    mean_correct_rate: float = 0.0
    max_correct_rate: float = 0.0
    mean_format_rate: float = 0.0
    mean_reward: float = 0.0
    mean_reward_std: float = 0.0
    max_reward_std: float = 0.0
    mean_adv_abs: float = 0.0
    max_adv_abs: float = 0.0
    mean_kl: float = 0.0
    max_kl: float = 0.0
    mean_clip_frac: float = 0.0
    max_clip_frac: float = 0.0
    nonzero_reward_steps: int = 0
    nonzero_correct_steps: int = 0
    adv_signal_steps: int = 0
    reward_std_signal_steps: int = 0
    checkpoint_count: int = 0
    latest_event: str = ""
    prompt_tasks: list[str] = field(default_factory=list)


@dataclass
class LaneDecision:
    lane: str
    classification: str
    priority: int
    action: str
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def summarize_lane(lane: str, run_dir: Path, *, nonzero_eps: float = 1.0e-9) -> LaneSummary:
    metrics_path = run_dir / "metrics.jsonl"
    rows = read_jsonl(metrics_path)
    summary = LaneSummary(
        lane=lane,
        run_dir=str(run_dir),
        metrics_path=str(metrics_path),
        exists=metrics_path.exists(),
    )
    if not rows:
        return summary

    grpo_steps = [row for row in rows if row.get("event") == "grpo_step"]
    checkpoints = [row for row in rows if row.get("event") == "checkpoint"]
    summary.run_end = any(row.get("event") == "run_end" for row in rows)
    summary.event_count = len(rows)
    summary.grpo_step_count = len(grpo_steps)
    summary.checkpoint_count = len(checkpoints)
    summary.latest_event = str(rows[-1].get("event", ""))

    step_values = [_as_int(row.get("step")) for row in grpo_steps]
    known_steps = [step for step in step_values if step is not None]
    summary.latest_grpo_step = max(known_steps) if known_steps else None

    final_checkpoints = [
        _as_int(row.get("step"))
        for row in checkpoints
        if row.get("kind") == "final"
    ]
    known_final_steps = [step for step in final_checkpoints if step is not None]
    if known_final_steps:
        summary.final_checkpoint_step = max(known_final_steps)

    correct = [_as_float(row.get("correct_rate")) for row in grpo_steps]
    fmt = [_as_float(row.get("format_rate")) for row in grpo_steps]
    rewards = [_as_float(row.get("reward_mean")) for row in grpo_steps]
    reward_std = [_as_float(row.get("reward_std")) for row in grpo_steps]
    adv_abs = [_as_float(row.get("adv_abs_mean")) for row in grpo_steps]
    kl = [_as_float(row.get("kl")) for row in grpo_steps]
    clip_frac = [_as_float(row.get("clip_frac")) for row in grpo_steps]

    if grpo_steps:
        final = grpo_steps[-1]
        summary.final_correct_rate = _as_float(final.get("correct_rate"))
        summary.final_format_rate = _as_float(final.get("format_rate"))
    summary.mean_correct_rate = _mean(correct)
    summary.max_correct_rate = max(correct, default=0.0)
    summary.mean_format_rate = _mean(fmt)
    summary.mean_reward = _mean(rewards)
    summary.mean_reward_std = _mean(reward_std)
    summary.max_reward_std = max(reward_std, default=0.0)
    summary.mean_adv_abs = _mean(adv_abs)
    summary.max_adv_abs = max(adv_abs, default=0.0)
    summary.mean_kl = _mean(kl)
    summary.max_kl = max(kl, default=0.0)
    summary.mean_clip_frac = _mean(clip_frac)
    summary.max_clip_frac = max(clip_frac, default=0.0)
    summary.nonzero_reward_steps = sum(abs(value) > nonzero_eps for value in rewards)
    summary.nonzero_correct_steps = sum(value > nonzero_eps for value in correct)
    summary.adv_signal_steps = sum(value > nonzero_eps for value in adv_abs)
    summary.reward_std_signal_steps = sum(value > nonzero_eps for value in reward_std)
    summary.prompt_tasks = sorted({
        str(row.get("prompt_task"))
        for row in grpo_steps
        if row.get("prompt_task")
    })
    return summary


def classify_lane(summary: LaneSummary, policy: dict[str, Any] | None = None) -> LaneDecision:
    policy = merge_policy(DEFAULT_POLICY, policy or {})
    thresholds = policy["thresholds"]
    actions = policy["actions"]

    min_steps = int(thresholds["min_grpo_steps"])
    strong_correct = float(thresholds["strong_mean_correct_rate"])
    promising_correct = float(thresholds["promising_mean_correct_rate"])
    min_format = float(thresholds["min_mean_format_rate"])
    min_adv_steps = int(thresholds["min_adv_signal_steps"])

    warnings: list[str] = []
    rationale: list[str] = []

    if not summary.exists:
        return LaneDecision(
            lane=summary.lane,
            classification="missing",
            priority=0,
            action=str(actions["missing"]),
            rationale=["metrics.jsonl is absent"],
        )

    if summary.grpo_step_count < min_steps or not summary.run_end:
        return LaneDecision(
            lane=summary.lane,
            classification="incomplete",
            priority=1,
            action=str(actions["incomplete"]),
            rationale=[
                f"run_end={summary.run_end}",
                f"grpo_steps={summary.grpo_step_count} < {min_steps}",
            ],
        )

    if summary.mean_format_rate < min_format:
        warnings.append(
            f"mean format rate {summary.mean_format_rate:.3f} is below {min_format:.3f}"
        )

    if summary.adv_signal_steps < min_adv_steps:
        warnings.append(
            f"advantage signal steps {summary.adv_signal_steps} < {min_adv_steps}"
        )

    if summary.max_correct_rate == 0.0 and summary.nonzero_reward_steps == 0:
        rationale.extend([
            "max correct rate is 0.000",
            "reward never becomes nonzero",
        ])
        return LaneDecision(
            lane=summary.lane,
            classification="blocked_no_reward_signal",
            priority=0,
            action=str(actions["blocked_no_reward_signal"]),
            rationale=rationale,
            warnings=warnings,
        )

    if summary.mean_correct_rate >= strong_correct and (
        summary.final_correct_rate or 0.0
    ) >= strong_correct:
        rationale.extend([
            f"mean correct rate {summary.mean_correct_rate:.3f} >= {strong_correct:.3f}",
            f"final correct rate {(summary.final_correct_rate or 0.0):.3f} >= {strong_correct:.3f}",
        ])
        return LaneDecision(
            lane=summary.lane,
            classification="ready_for_export_eval",
            priority=3,
            action=str(actions["ready_for_export_eval"]),
            rationale=rationale,
            warnings=warnings,
        )

    if summary.mean_correct_rate >= promising_correct:
        rationale.extend([
            f"mean correct rate {summary.mean_correct_rate:.3f} >= {promising_correct:.3f}",
            f"max correct rate {summary.max_correct_rate:.3f}",
        ])
        return LaneDecision(
            lane=summary.lane,
            classification="promising_but_unstable",
            priority=2,
            action=str(actions["promising_but_unstable"]),
            rationale=rationale,
            warnings=warnings,
        )

    rationale.extend([
        f"mean correct rate {summary.mean_correct_rate:.3f} < {promising_correct:.3f}",
        f"max correct rate {summary.max_correct_rate:.3f}",
    ])
    return LaneDecision(
        lane=summary.lane,
        classification="unstable_sparse_success",
        priority=1,
        action=str(actions["unstable_sparse_success"]),
        rationale=rationale,
        warnings=warnings,
    )


def merge_policy(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def load_policy(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"policy config must be a mapping: {path}")
    policy = payload.get("diagnostics", payload)
    if not isinstance(policy, dict):
        raise ValueError(f"diagnostics policy must be a mapping: {path}")
    return policy


def analyze_bridge_runs(
    run_root: Path,
    *,
    lane_runs: dict[str, str] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = merge_policy(DEFAULT_POLICY, policy or {})
    lane_runs = lane_runs or DEFAULT_LANE_RUNS
    eps = float(policy["thresholds"]["nonzero_eps"])
    summaries: dict[str, LaneSummary] = {}
    decisions: dict[str, LaneDecision] = {}
    for lane, run_name in lane_runs.items():
        summary = summarize_lane(lane, run_root / run_name, nonzero_eps=eps)
        summaries[lane] = summary
        decisions[lane] = classify_lane(summary, policy)

    classifications: dict[str, int] = {}
    for decision in decisions.values():
        classifications[decision.classification] = (
            classifications.get(decision.classification, 0) + 1
        )

    ordered_actions = sorted(
        decisions.values(),
        key=lambda item: (item.priority, item.lane),
    )
    return {
        "generated_at_unix": time.time(),
        "run_root": str(run_root),
        "policy": policy,
        "summaries": {lane: asdict(summary) for lane, summary in summaries.items()},
        "decisions": {lane: asdict(decision) for lane, decision in decisions.items()},
        "classification_counts": classifications,
        "action_order": [decision.lane for decision in ordered_actions],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Bridge GRPO diagnostics",
        "",
        f"- Run root: `{report['run_root']}`",
        f"- Lanes analyzed: `{len(report['summaries'])}`",
        f"- Classification counts: `{json.dumps(report['classification_counts'], sort_keys=True)}`",
        "",
        "## Decision table",
        "",
        (
            "| lane | class | mean correct | max correct | final correct | "
            "format | adv steps | action |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for lane in sorted(report["summaries"]):
        summary = report["summaries"][lane]
        decision = report["decisions"][lane]
        final_correct = summary["final_correct_rate"]
        final_text = "n/a" if final_correct is None else f"{final_correct:.3f}"
        lines.append(
            "| {lane} | {cls} | {mean:.3f} | {maxc:.3f} | {final} | "
            "{fmt:.3f} | {adv} | {action} |".format(
                lane=lane,
                cls=decision["classification"],
                mean=summary["mean_correct_rate"],
                maxc=summary["max_correct_rate"],
                final=final_text,
                fmt=summary["mean_format_rate"],
                adv=summary["adv_signal_steps"],
                action=decision["action"],
            )
        )

    lines.extend([
        "",
        "## Lane notes",
        "",
    ])
    for lane in sorted(report["decisions"]):
        decision = report["decisions"][lane]
        summary = report["summaries"][lane]
        lines.append(f"### {lane}")
        lines.append("")
        lines.append(f"- Prompt tasks: `{', '.join(summary['prompt_tasks']) or 'n/a'}`")
        lines.append(f"- Rationale: {'; '.join(decision['rationale']) or 'n/a'}")
        if decision["warnings"]:
            lines.append(f"- Warnings: {'; '.join(decision['warnings'])}")
        lines.append("")

    lines.extend([
        "## Operational rule",
        "",
        (
            "Treat zero reward variance / zero advantage lanes as data or verifier "
            "problems before RL problems. Extending GRPO without a nonzero group "
            "signal only spends compute without creating a useful policy update."
        ),
        "",
    ])
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: Path, *, prefix: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    md_path = out_dir / f"{prefix}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="H:/elt_data/runs")
    parser.add_argument("--config", default="configs/bridge_diagnostics.yaml")
    parser.add_argument("--out-dir", default="_docs/assets/bridge_diagnostics")
    parser.add_argument("--prefix", default="bridge_diagnostics")
    parser.add_argument("--json", action="store_true", help="print JSON to stdout")
    return parser


def cli(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config) if args.config else None
    policy = load_policy(config_path)
    report = analyze_bridge_runs(Path(args.run_root), policy=policy)
    json_path, md_path = write_report(report, Path(args.out_dir), prefix=args.prefix)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")


if __name__ == "__main__":
    cli()
