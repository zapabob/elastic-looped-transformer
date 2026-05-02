"""Build bridge/easy-hard GRPO prompts for synthetic v2 code.

The hard v2 code prompts are useful for reward variance, but the first live
GRPO run showed zero positive verifier hits.  This module creates a smaller
curriculum prompt file that mixes:

* easy python_exec tasks that should produce positive rewards,
* bridge tasks that reuse the v2 domains with fewer constraints,
* a retained slice of the original hard prompts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_NAME = "synthetic-v2-code-bridge"
DEFAULT_HARD_CASES = Path(
    "H:/elt_data/synthetic_v2_hard/code/benchmarks/synthetic_v2_hard_code_val_cases.jsonl"
)
DEFAULT_OUTPUT = Path(
    "H:/elt_data/synthetic_v2_hard/code/benchmarks/synthetic_v2_bridge_code_val_cases.jsonl"
)


def _prompt(user_request: str, idx: int, difficulty: str) -> str:
    return (
        "Solve the following Python task.\n"
        "Return executable Python code only inside a fenced ```python block.\n"
        "Use Python 3.12 standard library only unless the prompt explicitly says otherwise.\n\n"
        "Task:\n"
        f"{user_request} Synthetic v2 bridge id {idx} ({difficulty})."
    )


def _code_response(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


@dataclass(frozen=True)
class BridgePrompt:
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
            "task": "python_exec",
            "bucket": "gguf_code_distill_v2_bridge",
            "source": SOURCE_NAME,
            "metadata": {
                "lane": "code",
                "task_name": self.domain,
                "difficulty": self.difficulty,
                "curriculum": "bridge_easy_hard",
                "variant": f"synthetic_v2_bridge_{self.difficulty}_{self.idx}",
                "language": "python",
                "tags": [
                    "code",
                    "synthetic_v2_bridge",
                    self.difficulty,
                    "python_exec",
                ],
            },
        }


def _easy_prompt(idx: int) -> BridgePrompt:
    kind = idx % 6
    if kind == 0:
        code = (
            "def add_cents(items: list[tuple[str, int]]) -> int:\n"
            "    total = 0\n"
            "    for _name, cents in items:\n"
            "        if cents < 0:\n"
            "            raise ValueError('negative cents')\n"
            "        total += cents\n"
            "    return total"
        )
        reference = (
            "assert add_cents([('tea', 250), ('cake', 400)]) == 650\n"
            "try:\n"
            "    add_cents([('bad', -1)])\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'negative cents'\n"
            "else:\n"
            "    raise AssertionError('expected negative cents error')"
        )
        request = "Implement add_cents(items). Items are (name, cents) pairs. Reject negative cents and return the total."
        domain = "easy_sum_with_guard"
    elif kind == 1:
        code = (
            "def clamp_values(values: list[int], low: int, high: int) -> list[int]:\n"
            "    if low > high:\n"
            "        raise ValueError('low must be <= high')\n"
            "    return [min(high, max(low, value)) for value in values]"
        )
        reference = (
            "assert clamp_values([-2, 3, 9], 0, 5) == [0, 3, 5]\n"
            "try:\n"
            "    clamp_values([1], 3, 2)\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'low must be <= high'\n"
            "else:\n"
            "    raise AssertionError('expected invalid bounds error')"
        )
        request = "Implement clamp_values(values, low, high). Validate bounds and clamp every integer into the inclusive range."
        domain = "easy_clamp_values"
    elif kind == 2:
        code = (
            "def count_labels(labels: list[str]) -> dict[str, int]:\n"
            "    counts: dict[str, int] = {}\n"
            "    for label in labels:\n"
            "        if not label:\n"
            "            raise ValueError('empty label')\n"
            "        counts[label] = counts.get(label, 0) + 1\n"
            "    return {label: counts[label] for label in sorted(counts)}"
        )
        reference = (
            "assert count_labels(['b', 'a', 'b']) == {'a': 1, 'b': 2}\n"
            "try:\n"
            "    count_labels(['ok', ''])\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'empty label'\n"
            "else:\n"
            "    raise AssertionError('expected empty label error')"
        )
        request = "Implement count_labels(labels). Reject empty labels, count occurrences, and return a dictionary sorted by label."
        domain = "easy_sorted_counts"
    elif kind == 3:
        code = (
            "def running_total(values: list[int]) -> list[int]:\n"
            "    out: list[int] = []\n"
            "    total = 0\n"
            "    for value in values:\n"
            "        total += value\n"
            "        out.append(total)\n"
            "    return out"
        )
        reference = "assert running_total([2, -1, 4]) == [2, 1, 5]\nassert running_total([]) == []"
        request = "Implement running_total(values). Return the cumulative sum after each input value."
        domain = "easy_running_total"
    elif kind == 4:
        code = (
            "def unique_preserve_order(values: list[str]) -> list[str]:\n"
            "    seen: set[str] = set()\n"
            "    out: list[str] = []\n"
            "    for value in values:\n"
            "        if value not in seen:\n"
            "            seen.add(value)\n"
            "            out.append(value)\n"
            "    return out"
        )
        reference = "assert unique_preserve_order(['b', 'a', 'b', 'c', 'a']) == ['b', 'a', 'c']"
        request = "Implement unique_preserve_order(values). Keep only the first occurrence of each string."
        domain = "easy_unique_order"
    else:
        code = (
            "def split_pass_fail(scores: dict[str, int], threshold: int) -> dict[str, list[str]]:\n"
            "    result = {'pass': [], 'fail': []}\n"
            "    for name in sorted(scores):\n"
            "        bucket = 'pass' if scores[name] >= threshold else 'fail'\n"
            "        result[bucket].append(name)\n"
            "    return result"
        )
        reference = (
            "assert split_pass_fail({'zoe': 9, 'amy': 5, 'kai': 7}, 7) == "
            "{'pass': ['kai', 'zoe'], 'fail': ['amy']}"
        )
        request = "Implement split_pass_fail(scores, threshold). Return sorted names split into pass and fail buckets."
        domain = "easy_split_buckets"
    return BridgePrompt(_prompt(request, idx, "easy"), reference, _code_response(code), "easy", domain, idx)


def _bridge_prompt(idx: int) -> BridgePrompt:
    kind = idx % 6
    if kind == 0:
        code = (
            "def reconcile_inventory(initial: dict[str, int], events: list[tuple[str, int]]) -> dict[str, int]:\n"
            "    state = dict(initial)\n"
            "    for sku, delta in events:\n"
            "        state[sku] = state.get(sku, 0) + delta\n"
            "        if state[sku] < 0:\n"
            "            raise ValueError(f'negative inventory for {sku}')\n"
            "    return {sku: state[sku] for sku in sorted(state) if state[sku] != 0}"
        )
        reference = (
            "assert reconcile_inventory({'a': 2}, [('b', 3), ('a', -1)]) == {'a': 1, 'b': 3}\n"
            "try:\n"
            "    reconcile_inventory({'a': 1}, [('a', -2)])\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'negative inventory for a'\n"
            "else:\n"
            "    raise AssertionError('expected negative inventory error')"
        )
        request = "Implement reconcile_inventory(initial, events). Apply SKU deltas in order, reject negative stock, drop zeros, and return sorted keys."
        domain = "bridge_inventory_delta_reconciliation"
    elif kind == 1:
        code = (
            "def plan_batches(tasks: list[tuple[str, int]], capacity: int) -> list[list[str]]:\n"
            "    if capacity <= 0:\n"
            "        raise ValueError('capacity must be positive')\n"
            "    batches: list[list[str]] = []\n"
            "    current: list[str] = []\n"
            "    used = 0\n"
            "    for name, cost in tasks:\n"
            "        if cost > capacity:\n"
            "            raise ValueError(f'task too large: {name}')\n"
            "        if current and used + cost > capacity:\n"
            "            batches.append(current)\n"
            "            current = []\n"
            "            used = 0\n"
            "        current.append(name)\n"
            "        used += cost\n"
            "    if current:\n"
            "        batches.append(current)\n"
            "    return batches"
        )
        reference = (
            "assert plan_batches([('a', 2), ('b', 3), ('c', 2)], 5) == [['a', 'b'], ['c']]\n"
            "try:\n"
            "    plan_batches([('huge', 6)], 5)\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'task too large: huge'\n"
            "else:\n"
            "    raise AssertionError('expected oversize task error')"
        )
        request = "Implement plan_batches(tasks, capacity). Preserve order and split into batches before capacity is exceeded."
        domain = "bridge_capacity_batch_planning"
    elif kind == 2:
        code = (
            "def count_active_users(sessions: list[dict[str, int]], days: list[int]) -> dict[int, int]:\n"
            "    result: dict[int, int] = {}\n"
            "    for day in days:\n"
            "        active = {s['user'] for s in sessions if s['start'] <= day < s['end']}\n"
            "        result[day] = len(active)\n"
            "    return result"
        )
        reference = (
            "sessions = [{'user': 1, 'start': 0, 'end': 3}, {'user': 2, 'start': 2, 'end': 4}, {'user': 1, 'start': 3, 'end': 5}]\n"
            "assert count_active_users(sessions, [2, 3, 5]) == {2: 2, 3: 2, 5: 0}"
        )
        request = "Implement count_active_users(sessions, days). Treat sessions as half-open and count unique users for each day."
        domain = "bridge_active_user_session_counts"
    elif kind == 3:
        code = (
            "def summarize_downtime(events: list[dict[str, int]], grace_minutes: int) -> list[tuple[int, int, int]]:\n"
            "    by_service: dict[int, list[tuple[int, int]]] = {}\n"
            "    for event in events:\n"
            "        by_service.setdefault(event['service'], []).append((event['start'], event['end']))\n"
            "    out: list[tuple[int, int, int]] = []\n"
            "    for service in sorted(by_service):\n"
            "        intervals = sorted(by_service[service])\n"
            "        cur_start, cur_end = intervals[0]\n"
            "        for start, end in intervals[1:]:\n"
            "            if start - cur_end <= grace_minutes:\n"
            "                cur_end = max(cur_end, end)\n"
            "            else:\n"
            "                out.append((service, cur_start, cur_end))\n"
            "                cur_start, cur_end = start, end\n"
            "        out.append((service, cur_start, cur_end))\n"
            "    return out"
        )
        reference = (
            "events = [{'service': 1, 'start': 0, 'end': 2}, {'service': 1, 'start': 3, 'end': 5}, {'service': 2, 'start': 8, 'end': 9}]\n"
            "assert summarize_downtime(events, 1) == [(1, 0, 5), (2, 8, 9)]"
        )
        request = "Implement summarize_downtime(events, grace_minutes). Group by service and merge intervals whose gap is within grace."
        domain = "bridge_merge_downtime_windows"
    elif kind == 4:
        code = (
            "def net_transfer_balances(transfers: list[tuple[str, str, int]]) -> dict[str, int]:\n"
            "    balances: dict[str, int] = {}\n"
            "    for source, target, amount in transfers:\n"
            "        balances[source] = balances.get(source, 0) - amount\n"
            "        balances[target] = balances.get(target, 0) + amount\n"
            "    return {name: balances[name] for name in sorted(balances) if balances[name] != 0}"
        )
        reference = "assert net_transfer_balances([('a', 'b', 5), ('b', 'c', 2)]) == {'a': -5, 'b': 3, 'c': 2}"
        request = "Implement net_transfer_balances(transfers). Debit sources, credit targets, drop zeros, and return sorted names."
        domain = "bridge_net_transfer_balances"
    else:
        code = (
            "def group_anomaly_windows(events: list[dict[str, int]], threshold: int, max_gap: int) -> dict[int, list[tuple[int, int]]]:\n"
            "    by_sensor: dict[int, list[tuple[int, int]]] = {}\n"
            "    for event in events:\n"
            "        if event['severity'] >= threshold:\n"
            "            by_sensor.setdefault(event['sensor'], []).append((event['start'], event['end']))\n"
            "    result: dict[int, list[tuple[int, int]]] = {}\n"
            "    for sensor in sorted(by_sensor):\n"
            "        intervals = sorted(by_sensor[sensor])\n"
            "        cur_start, cur_end = intervals[0]\n"
            "        merged: list[tuple[int, int]] = []\n"
            "        for start, end in intervals[1:]:\n"
            "            if start - cur_end <= max_gap:\n"
            "                cur_end = max(cur_end, end)\n"
            "            else:\n"
            "                merged.append((cur_start, cur_end))\n"
            "                cur_start, cur_end = start, end\n"
            "        merged.append((cur_start, cur_end))\n"
            "        result[sensor] = merged\n"
            "    return result"
        )
        reference = (
            "events = [{'sensor': 1, 'start': 0, 'end': 2, 'severity': 5}, {'sensor': 1, 'start': 3, 'end': 4, 'severity': 6}, {'sensor': 2, 'start': 0, 'end': 1, 'severity': 3}]\n"
            "assert group_anomaly_windows(events, 5, 1) == {1: [(0, 4)]}"
        )
        request = "Implement group_anomaly_windows(events, threshold, max_gap). Keep high-severity events and merge nearby windows by sensor."
        domain = "bridge_anomaly_window_grouping"
    return BridgePrompt(_prompt(request, idx, "bridge"), reference, _code_response(code), "bridge", domain, idx)


def generate_easy_code_bridge_prompts(count: int) -> list[BridgePrompt]:
    return [_easy_prompt(i) for i in range(count)]


def generate_bridge_code_prompts(count: int) -> list[BridgePrompt]:
    return [_bridge_prompt(i) for i in range(count)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hard_records(path: Path, count: int) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise RuntimeError(f"no hard prompt records in {path}")
    out: list[dict[str, Any]] = []
    for i in range(count):
        row = dict(rows[i % len(rows)])
        metadata = dict(row.get("metadata") or {})
        metadata.update({
            "difficulty": "hard",
            "curriculum": "bridge_easy_hard",
            "source_cases": str(path),
        })
        tags = list(metadata.get("tags") or [])
        for tag in ("synthetic_v2_bridge", "hard"):
            if tag not in tags:
                tags.append(tag)
        metadata["tags"] = tags
        row["metadata"] = metadata
        row["source"] = SOURCE_NAME
        row["bucket"] = "gguf_code_distill_v2_bridge"
        out.append(row)
    return out


def build_code_bridge_prompts(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    hard_cases_path: Path = DEFAULT_HARD_CASES,
    total_cases: int = 256,
    easy_cases: int | None = None,
    bridge_cases: int | None = None,
) -> dict[str, Any]:
    if total_cases <= 0:
        raise ValueError("total_cases must be positive")
    easy_n = easy_cases if easy_cases is not None else max(1, total_cases // 4)
    bridge_n = bridge_cases if bridge_cases is not None else max(1, total_cases // 2)
    hard_n = total_cases - easy_n - bridge_n
    if min(easy_n, bridge_n, hard_n) <= 0:
        raise ValueError("easy, bridge, and hard case counts must all be positive")

    easy = [_easy_prompt(i).to_record() for i in range(easy_n)]
    bridge = [_bridge_prompt(i).to_record() for i in range(bridge_n)]
    hard = _hard_records(hard_cases_path, hard_n)

    max_len = max(len(easy), len(bridge), len(hard))
    # Interleave bridge-heavy batches while keeping all requested counts exact.
    mixed: list[dict[str, Any]] = []
    for i in range(max_len):
        if i < len(bridge):
            mixed.append(bridge[i])
        if i < len(easy):
            mixed.append(easy[i])
        if i < len(hard):
            mixed.append(hard[i])
        if i + len(hard) < len(bridge):
            mixed.append(bridge[i + len(hard)])
    mixed = mixed[:total_cases]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {"easy": 0, "bridge": 0, "hard": 0}
    domains: dict[str, int] = {}
    for row in mixed:
        md = row.get("metadata") or {}
        difficulty = str(md.get("difficulty", "unknown"))
        counts[difficulty] = counts.get(difficulty, 0) + 1
        domain = str(md.get("task_name", "unknown"))
        domains[domain] = domains.get(domain, 0) + 1
    summary = {
        "source": SOURCE_NAME,
        "output_path": str(output_path),
        "hard_cases_path": str(hard_cases_path),
        "total_cases": len(mixed),
        "difficulty_counts": counts,
        "domain_counts": domains,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hard-cases", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--total-cases", type=int, default=256)
    parser.add_argument("--easy-cases", type=int, default=None)
    parser.add_argument("--bridge-cases", type=int, default=None)
    args = parser.parse_args()
    summary = build_code_bridge_prompts(
        output_path=args.output,
        hard_cases_path=args.hard_cases,
        total_cases=args.total_cases,
        easy_cases=args.easy_cases,
        bridge_cases=args.bridge_cases,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
