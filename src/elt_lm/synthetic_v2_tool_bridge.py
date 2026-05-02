"""Build bridge/easy-hard GRPO prompts for synthetic v2 tool-use.

The hard tool-use lane gives clean JSON format supervision, but the first hard
GRPO pass produced no positive verifier signal. This creates a smaller
easy/bridge/hard curriculum where exact JSON matches are reachable before the
hard disambiguation prompts return.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_NAME = "synthetic-v2-tool-bridge"
DEFAULT_HARD_CASES = Path(
    "H:/elt_data/synthetic_v2_hard/tool_use/benchmarks/synthetic_v2_hard_tool_use_val_cases.jsonl"
)
DEFAULT_OUTPUT = Path(
    "H:/elt_data/synthetic_v2_hard/tool_use/benchmarks/synthetic_v2_bridge_tool_use_val_cases.jsonl"
)


def _prompt(user_request: str, idx: int, difficulty: str) -> str:
    return (
        "Select the best tool call for the following user request.\n"
        "Return strict JSON with keys: tool_name, arguments.\n"
        "Do not add prose or markdown fences.\n\n"
        "User request:\n"
        f"{user_request.strip()} Synthetic v2 bridge id {idx} ({difficulty})."
    )


def _json_response(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class BridgeToolPrompt:
    prompt: str
    reference: str
    correct_response: str
    difficulty: str
    domain: str
    idx: int

    def to_record(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "reference": self.reference,
            "task": "json_match",
            "bucket": "gguf_tool_use_distill_v2_bridge",
            "source": SOURCE_NAME,
            "metadata": {
                "lane": "tool_use",
                "task_name": self.domain,
                "difficulty": self.difficulty,
                "curriculum": "bridge_easy_hard",
                "variant": f"synthetic_v2_bridge_{self.difficulty}_{self.idx}",
                "tags": [
                    "tool_use",
                    "synthetic_v2_bridge",
                    self.difficulty,
                    "json_match",
                ],
            },
        }


def _case(
    *,
    idx: int,
    difficulty: str,
    domain: str,
    request: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> BridgeToolPrompt:
    response = {"tool_name": tool_name, "arguments": arguments}
    reference = _json_response(response)
    return BridgeToolPrompt(
        prompt=_prompt(request, idx, difficulty),
        reference=reference,
        correct_response=reference,
        difficulty=difficulty,
        domain=domain,
        idx=idx,
    )


def _easy_prompt(idx: int) -> BridgeToolPrompt:
    kind = idx % 6
    request_id = f"tool-easy-{idx}"
    if kind == 0:
        return _case(
            idx=idx,
            difficulty="easy",
            domain="easy_file_search_read_only",
            request="Search project files for the word reward_std. Do not edit files.",
            tool_name="mcp.files.search",
            arguments={"query": "reward_std", "root": ".", "limit": 5, "read_only": True, "request_id": request_id},
        )
    if kind == 1:
        return _case(
            idx=idx,
            difficulty="easy",
            domain="easy_metrics_recent_window",
            request="Read recent GRPO correct_rate metrics for the code run without launching training.",
            tool_name="mcp.metrics.query",
            arguments={"run_dir": "H:/elt_data/runs/grpo_side_lora_code_synthetic_v2_hard", "metric": "correct_rate", "window": 8, "read_only": True, "request_id": request_id},
        )
    if kind == 2:
        return _case(
            idx=idx,
            difficulty="easy",
            domain="easy_plan_dry_run",
            request="Create a dry-run plan for bridge evaluation. Do not mutate the workspace.",
            tool_name="agent.plan.execute",
            arguments={"plan_id": "bridge-eval", "max_steps": 4, "dry_run": True, "request_id": request_id},
        )
    if kind == 3:
        return _case(
            idx=idx,
            difficulty="easy",
            domain="easy_gpu_status",
            request="Inspect GPU memory and utilization for the current run.",
            tool_name="mcp.metrics.gpu",
            arguments={"fields": ["memory.used", "memory.total", "utilization.gpu"], "read_only": True, "request_id": request_id},
        )
    if kind == 4:
        return _case(
            idx=idx,
            difficulty="easy",
            domain="easy_checkpoint_freshness",
            request="Check the latest checkpoint age in the math bridge run directory.",
            tool_name="mcp.files.stat",
            arguments={"path": "H:/elt_data/runs/grpo_side_lora_math_synthetic_v2_bridge/last.pt", "read_only": True, "request_id": request_id},
        )
    return _case(
        idx=idx,
        difficulty="easy",
        domain="easy_git_status",
        request="Report git status only. Do not stage, commit, or push.",
        tool_name="mcp.git.status",
        arguments={"cwd": ".", "include_untracked": True, "read_only": True, "request_id": request_id},
    )


def _bridge_prompt(idx: int) -> BridgeToolPrompt:
    kind = idx % 6
    request_id = f"tool-bridge-{idx}"
    if kind == 0:
        return _case(
            idx=idx,
            difficulty="bridge",
            domain="bridge_sparse_reward_diagnosis",
            request=(
                "Compare reward_std, correct_rate, and adv_abs_mean over the last 16 steps "
                "for the math hard run. Read metrics only and do not restart training."
            ),
            tool_name="mcp.metrics.query",
            arguments={
                "run_dir": "H:/elt_data/runs/grpo_side_lora_math_synthetic_v2_hard",
                "metrics": ["reward_std", "correct_rate", "adv_abs_mean"],
                "window": 16,
                "read_only": True,
                "request_id": request_id,
            },
        )
    if kind == 1:
        return _case(
            idx=idx,
            difficulty="bridge",
            domain="bridge_manifest_inspection",
            request=(
                "Inspect the bridge prompt manifest path for math and report whether it exists. "
                "Do not create files."
            ),
            tool_name="mcp.files.stat",
            arguments={
                "path": "H:/elt_data/synthetic_v2_hard/math/benchmarks/synthetic_v2_bridge_math_val_cases.jsonl",
                "read_only": True,
                "request_id": request_id,
            },
        )
    if kind == 2:
        return _case(
            idx=idx,
            difficulty="bridge",
            domain="bridge_safe_rerun_plan",
            request=(
                "Prepare a safe dry-run plan for switching all lanes to bridge GRPO and ILSD. "
                "Require tests before launch."
            ),
            tool_name="agent.plan.execute",
            arguments={
                "plan_id": "all-lane-bridge-ilsd",
                "max_steps": 8,
                "dry_run": True,
                "requires_tests": True,
                "request_id": request_id,
            },
        )
    if kind == 3:
        return _case(
            idx=idx,
            difficulty="bridge",
            domain="bridge_disk_guard",
            request="Check free space on H: and C: before launching the next bridge run.",
            tool_name="mcp.metrics.disk",
            arguments={"drives": ["C:", "H:"], "min_free_gb": {"C:": 5, "H:": 30}, "read_only": True, "request_id": request_id},
        )
    if kind == 4:
        return _case(
            idx=idx,
            difficulty="bridge",
            domain="bridge_eval_artifact_check",
            request=(
                "Look for cv_results.json in the current synthetic v2 eval directory. "
                "Return metadata only."
            ),
            tool_name="mcp.files.search",
            arguments={
                "root": "H:/elt_data/eval/synthetic_gb_side_lora",
                "query": "cv_results.json",
                "limit": 10,
                "read_only": True,
                "request_id": request_id,
            },
        )
    return _case(
        idx=idx,
        difficulty="bridge",
        domain="bridge_process_guard",
        request="List active ELT training or evaluation processes without terminating them.",
        tool_name="mcp.process.list",
        arguments={"match": "elt-train|elt-anytime|pipeline.py", "read_only": True, "request_id": request_id},
    )


def generate_easy_tool_bridge_prompts(count: int) -> list[BridgeToolPrompt]:
    return [_easy_prompt(i) for i in range(count)]


def generate_bridge_tool_prompts(count: int) -> list[BridgeToolPrompt]:
    return [_bridge_prompt(i) for i in range(count)]


def _read_hard_cases(path: Path, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or count <= 0:
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            metadata = dict(obj.get("metadata") or {})
            metadata["curriculum"] = "bridge_easy_hard"
            metadata["difficulty"] = "hard"
            obj["metadata"] = metadata
            rows.append(obj)
            if len(rows) >= count:
                break
    return rows


def build_tool_bridge_prompts(
    *,
    output_path: Path,
    hard_cases_path: Path = DEFAULT_HARD_CASES,
    total_cases: int = 256,
    easy_cases: int = 64,
    bridge_cases: int = 128,
) -> dict[str, Any]:
    hard_cases = max(0, total_cases - easy_cases - bridge_cases)
    prompts = [
        *[item.to_record() for item in generate_bridge_tool_prompts(bridge_cases)],
        *[item.to_record() for item in generate_easy_tool_bridge_prompts(easy_cases)],
        *_read_hard_cases(hard_cases_path, hard_cases),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    difficulty_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for row in prompts:
        metadata = row.get("metadata") or {}
        difficulty = str(metadata.get("difficulty", "unknown"))
        domain = str(metadata.get("task_name", "unknown"))
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    summary = {
        "source": SOURCE_NAME,
        "lane": "tool_use",
        "output_path": str(output_path),
        "hard_cases_path": str(hard_cases_path),
        "total_cases": len(prompts),
        "difficulty_counts": difficulty_counts,
        "domain_counts": domain_counts,
        "task": "json_match",
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic v2 tool bridge prompts.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hard-cases", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--total-cases", type=int, default=256)
    parser.add_argument("--easy-cases", type=int, default=64)
    parser.add_argument("--bridge-cases", type=int, default=128)
    args = parser.parse_args()
    summary = build_tool_bridge_prompts(
        output_path=args.out,
        hard_cases_path=args.hard_cases,
        total_cases=args.total_cases,
        easy_cases=args.easy_cases,
        bridge_cases=args.bridge_cases,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()

