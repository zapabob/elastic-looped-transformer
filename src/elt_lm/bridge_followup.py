"""Build data and manifests for the post-diagnostic bridge follow-up."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import yaml

from elt_lm.posttrain_data import render_chat_text
from elt_lm.verifiers import (
    json_tool_call_match_correctness,
    python_exec_correctness,
    exact_math_correctness,
)


DEFAULT_DATA_ROOT = Path("training_data/synthetic_v2_hard")
DEFAULT_OUTPUT_ROOT = Path("training_data/bridge_followup")
H_DATA_ROOT = Path("H:/elt_data/synthetic_v2_hard")


def _read_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def _existing_path(primary: Path, fallback: Path | None = None) -> Path:
    if primary.exists():
        return primary
    if fallback is not None and fallback.exists():
        return fallback
    return primary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _canonical_json(text: str) -> str:
    payload = json.loads(text)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _tool_prompt(prompt: str) -> str:
    marker = "Return strict JSON with keys: tool_name, arguments.\nDo not add prose or markdown fences."
    replacement = (
        "Return your result using <think>...</think><answer>...</answer>.\n"
        "Inside <answer>, put strict JSON with exactly the keys: tool_name, arguments.\n"
        "Preserve safety-critical arguments such as read_only, dry_run, request_id, and limits."
    )
    return prompt.replace(marker, replacement)


def _tool_response(reference: str) -> str:
    payload = _canonical_json(reference)
    return (
        "<think>Select the safest matching tool and preserve required arguments.</think>"
        f"<answer>{payload}</answer>"
    )


def _sft_record(row: dict[str, Any], *, lane: str, prompt: str, response: str) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.update({"bridge_followup_lane": lane, "repair_policy": "post_diagnostic"})
    return {
        "bucket": f"bridge_followup_{lane}",
        "mode": "sft",
        "source": "bridge-followup",
        "task": row.get("task", ""),
        "prompt": prompt,
        "response": response,
        "reference": row.get("reference", ""),
        "system": "",
        "text": render_chat_text(prompt, response),
        "metadata": metadata,
    }


def _preference_record(row: dict[str, Any], *, prompt: str, chosen: str) -> dict[str, Any]:
    rejected = str(row.get("bad_response", ""))
    metadata = dict(row.get("metadata") or {})
    metadata.update({
        "failure_label": row.get("failure_label"),
        "failure_reason": row.get("failure_reason"),
        "repair_policy": "tool_call_failure_contrast",
    })
    return {
        "bucket": "bridge_followup_tool_use_preference",
        "mode": "preference",
        "source": "bridge-followup",
        "task": "json_tool_call_match",
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "reference": row.get("reference", ""),
        "chosen_text": render_chat_text(prompt, chosen),
        "rejected_text": render_chat_text(prompt, rejected),
        "metadata": metadata,
    }


def _repair_existing_sft(row: dict[str, Any], *, lane: str) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    response = str(row.get("response", ""))
    if lane == "tool_use":
        prompt = _tool_prompt(prompt)
        response = _tool_response(str(row.get("reference", "")))
        task = "json_tool_call_match"
    else:
        task = str(row.get("task", ""))
    record = _sft_record(row, lane=lane, prompt=prompt, response=response)
    record["task"] = task
    return record


def build_tool_repair(root: Path, out_root: Path, *, train_limit: int, val_limit: int) -> dict[str, Any]:
    lane_root = root / "tool_use"
    h_lane_root = H_DATA_ROOT / "tool_use"
    out = out_root / "tool_use_repair"
    train_rows = _read_jsonl(lane_root / "distill_train.jsonl", limit=train_limit)
    val_rows = _read_jsonl(lane_root / "distill_val.jsonl", limit=val_limit)
    bridge_cases = _read_jsonl(
        _existing_path(
            lane_root / "benchmarks" / "synthetic_v2_bridge_tool_use_val_cases.jsonl",
            h_lane_root / "benchmarks" / "synthetic_v2_bridge_tool_use_val_cases.jsonl",
        ),
        limit=val_limit,
    )
    failure_train = _read_jsonl(lane_root / "failures_train.jsonl", limit=train_limit)
    failure_val = _read_jsonl(lane_root / "failures_val.jsonl", limit=val_limit)

    train = [_repair_existing_sft(row, lane="tool_use") for row in train_rows]
    val = [_repair_existing_sft(row, lane="tool_use") for row in val_rows]
    benchmark_rows = []
    for row in bridge_cases:
        prompt = _tool_prompt(str(row["prompt"]))
        response = _tool_response(str(row["reference"]))
        repaired = _sft_record(row, lane="tool_use", prompt=prompt, response=response)
        repaired["task"] = "json_tool_call_match"
        val.append(repaired)
        benchmark_rows.append({
            "prompt": prompt,
            "reference": row["reference"],
            "task": "json_tool_call_match",
            "bucket": "bridge_followup_tool_use_eval",
            "source": "bridge-followup",
            "metadata": row.get("metadata", {}),
        })

    preferences_train = [
        _preference_record(row, prompt=_tool_prompt(str(row["prompt"])), chosen=_tool_response(str(row["reference"])))
        for row in failure_train
    ]
    preferences_val = [
        _preference_record(row, prompt=_tool_prompt(str(row["prompt"])), chosen=_tool_response(str(row["reference"])))
        for row in failure_val
    ]

    _write_jsonl(out / "distill_train.jsonl", train)
    _write_jsonl(out / "distill_val.jsonl", val)
    _write_jsonl(out / "preference_train.jsonl", preferences_train)
    _write_jsonl(out / "preference_val.jsonl", preferences_val)
    _write_jsonl(out / "benchmarks" / "tool_repair_val_cases.jsonl", benchmark_rows)

    manifest = {
        "benchmarks": [{
            "name": "tool_repair_bridge_val",
            "kind": "jsonl",
            "task": "json_tool_call_match",
            "path": str((out / "benchmarks" / "tool_repair_val_cases.jsonl").as_posix()),
            "prompt_field": "prompt",
            "reference_field": "reference",
            "limit": val_limit,
        }],
    }
    (out / "benchmarks" / "tool_repair_val_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    chosen_scores = [
        json_tool_call_match_correctness(row["chosen"], str(row["reference"]))
        for row in preferences_train + preferences_val
    ]
    rejected_scores = [
        json_tool_call_match_correctness(row["rejected"], str(row["reference"]))
        for row in preferences_train + preferences_val
    ]
    summary = {
        "lane": "tool_use",
        "task": "json_tool_call_match",
        "train_records": len(train),
        "val_records": len(val),
        "preference_train_records": len(preferences_train),
        "preference_val_records": len(preferences_val),
        "benchmark_records": len(benchmark_rows),
        "chosen_min_score": min(chosen_scores) if chosen_scores else 0.0,
        "rejected_max_score": max(rejected_scores) if rejected_scores else 0.0,
        "failure_labels": dict(Counter(str(row.get("failure_label", "")) for row in failure_train + failure_val)),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def build_replay_subset(
    root: Path,
    out_root: Path,
    *,
    lane: str,
    task: str,
    train_limit: int,
    val_limit: int,
) -> dict[str, Any]:
    lane_root = root / lane
    out = out_root / f"{lane}_replay"
    train = [_repair_existing_sft(row, lane=lane) for row in _read_jsonl(lane_root / "distill_train.jsonl", limit=train_limit)]
    val = [_repair_existing_sft(row, lane=lane) for row in _read_jsonl(lane_root / "distill_val.jsonl", limit=val_limit)]
    _write_jsonl(out / "distill_train.jsonl", train)
    _write_jsonl(out / "distill_val.jsonl", val)

    if lane == "code":
        scores = [python_exec_correctness(str(row["response"]), str(row["reference"])) for row in train + val]
    elif lane == "math":
        scores = [exact_math_correctness(str(row["response"]), str(row["reference"])) for row in train + val]
    else:
        scores = []
    summary = {
        "lane": lane,
        "task": task,
        "train_records": len(train),
        "val_records": len(val),
        "verifier_pass_rate": (sum(score == 1.0 for score in scores) / len(scores)) if scores else 0.0,
        "action": "run replay SFT / prompt repair before another GRPO continuation",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def build_stem_eval_manifest(root: Path, out_root: Path, *, limit: int) -> dict[str, Any]:
    cases = _existing_path(
        root / "stem_reasoning" / "benchmarks" / "synthetic_v2_bridge_stem_reasoning_val_cases.jsonl",
        H_DATA_ROOT / "stem_reasoning" / "benchmarks" / "synthetic_v2_bridge_stem_reasoning_val_cases.jsonl",
    )
    out = out_root / "stem_eval"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmarks": [{
            "name": "stem_bridge_candidate_val",
            "kind": "jsonl",
            "task": "mcq_reasoning",
            "path": str(cases.as_posix()),
            "prompt_field": "prompt",
            "reference_field": "reference",
            "limit": limit,
        }],
    }
    path = out / "stem_bridge_eval_manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"manifest_path": str(path), "case_path": str(cases), "limit": limit}


def build_action_plan(out_root: Path, summaries: dict[str, Any]) -> dict[str, Any]:
    plan = {
        "stem": {
            "state": "ready_for_export_eval",
            "next": "export adapter and run bounded bridge STEM eval",
            "artifact": "stem_eval/stem_bridge_eval_manifest.yaml",
        },
        "code": {
            "state": "sparse_success",
            "next": "use code_replay SFT subset before another GRPO continuation",
            "artifact": "code_replay/distill_train.jsonl",
        },
        "math": {
            "state": "sparse_success",
            "next": "use math_replay SFT subset before another GRPO continuation",
            "artifact": "math_replay/distill_train.jsonl",
        },
        "tool": {
            "state": "blocked_no_reward_signal",
            "next": "switch repair probe to json_tool_call_match and failure-contrast SFT/preference data",
            "artifact": "tool_use_repair/distill_train.jsonl",
        },
        "summaries": summaries,
    }
    (out_root / "lane_action_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Bridge follow-up action plan",
        "",
        "| lane | state | next artifact |",
        "|---|---|---|",
    ]
    for lane in ("stem", "code", "math", "tool"):
        item = plan[lane]
        lines.append(f"| {lane} | {item['state']} | `{item['artifact']}` |")
    lines.append("")
    lines.append("Use the tool repair verifier for probe GRPO only; keep exact JSON match for final eval.")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


def build_followup(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    train_limit: int = 192,
    val_limit: int = 64,
    stem_eval_limit: int = 128,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {
        "tool": build_tool_repair(data_root, output_root, train_limit=train_limit, val_limit=val_limit),
        "code": build_replay_subset(
            data_root,
            output_root,
            lane="code",
            task="python_exec",
            train_limit=train_limit,
            val_limit=val_limit,
        ),
        "math": build_replay_subset(
            data_root,
            output_root,
            lane="math",
            task="exact_math",
            train_limit=train_limit,
            val_limit=val_limit,
        ),
        "stem": build_stem_eval_manifest(data_root, output_root, limit=stem_eval_limit),
    }
    return build_action_plan(output_root, summaries)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train-limit", type=int, default=192)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--stem-eval-limit", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    return parser


def cli(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    plan = build_followup(
        Path(args.data_root),
        Path(args.output_root),
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        stem_eval_limit=args.stem_eval_limit,
    )
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"wrote {Path(args.output_root) / 'lane_action_plan.json'}")
        print(f"wrote {Path(args.output_root) / 'README.md'}")


if __name__ == "__main__":
    cli()
