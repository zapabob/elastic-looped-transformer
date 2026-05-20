"""Run lm-eval-harness CV over GGUF KV-cache policies.

The stock lm-eval GGUF adapter expects an older OpenAI logprobs shape. The
installed llama.cpp server returns the newer `/completion` `top_logprobs`
payload, so this script keeps lm-eval's task/evaluator pipeline but provides a
small local LM adapter that scores one-token multiple-choice labels from the
server response.

This is an external-heldout serving-surface CV gate. GSM8K rows are converted
to numeric multiple-choice rows so the same one-token letter scoring path can
compare K/V cache policies. It does not turn the current L=3 GGUF into a
loop-aware runtime or a broad GSM8K leaderboard result.
"""

from __future__ import annotations

import argparse
import csv
import decimal
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import requests

from elt_lm.eval.statistics import friedman_permutation_test, pairwise_group_comparisons


DEFAULT_SOURCE = Path("_docs/assets/2026-05-17-l3-thetom-k-protected/external_heldout/mmlu_stem_external_heldout.jsonl")
DEFAULT_OUT_DIR = Path("_docs/assets/2026-05-20-lm-eval-gguf-kv-cv")
DEFAULT_TASK = "elt_external_heldout_letter_cv"
DEFAULT_MODEL = Path("H:/elt_data/releases/elt-lm-qwen35-side-stem-aha-ilsd-l3-Q8_0.gguf")
DEFAULT_LLAMA_SERVER = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/llama-turboquant/bin/llama-server.exe"
DEFAULT_POLICIES = [
    ("q8_0", "turbo3"),
    ("bf16", "turbo3"),
    ("q8_0", "turbo4"),
    ("bf16", "turbo4"),
    ("q8_0", "turbo8"),
    ("bf16", "turbo8"),
]
LABELS = ["A", "B", "C", "D"]


@dataclass(frozen=True)
class Policy:
    cache_k: str
    cache_v: str

    @property
    def name(self) -> str:
        return f"K={self.cache_k}_V={self.cache_v}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _numeric_from_gsm8k_reference(reference: str) -> str | None:
    import re

    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", reference)
    if not match:
        match = re.search(r"-?[\d,]+(?:\.\d+)?", reference)
    return match.group(0).replace("####", "").strip().replace(",", "") if match else None


def _format_decimal(value: decimal.Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _gsm8k_distractors(answer_text: str) -> list[str]:
    answer = decimal.Decimal(answer_text)
    candidates = [
        answer + 1,
        answer - 1,
        answer * 2,
        answer / 2 if answer != 0 else decimal.Decimal(2),
        answer + 10,
        answer - 10,
    ]
    seen = {answer_text}
    distractors: list[str] = []
    for candidate in candidates:
        text = _format_decimal(candidate)
        if text not in seen:
            seen.add(text)
            distractors.append(text)
        if len(distractors) == 3:
            break
    while len(distractors) < 3:
        fallback = _format_decimal(answer + decimal.Decimal(len(distractors) + 2))
        if fallback not in seen:
            seen.add(fallback)
            distractors.append(fallback)
    return distractors


def _build_gsm8k_mcq_row(row: dict[str, Any], *, case_id: int, fold: int) -> dict[str, Any] | None:
    answer = _numeric_from_gsm8k_reference(str(row.get("reference", "")))
    if answer is None:
        return None
    distractors = _gsm8k_distractors(answer)
    correct_idx = case_id % len(LABELS)
    values = list(distractors)
    values.insert(correct_idx, answer)
    choices_text = "\n".join(f"{label}. {value}" for label, value in zip(LABELS, values))
    prompt = (
        f"{str(row['prompt']).rstrip()}\n\n"
        "Choices:\n"
        f"{choices_text}\n\n"
        "Answer:"
    )
    return {
        "case_id": case_id,
        "fold": fold,
        "prompt": prompt,
        "choices": [f" {label}" for label in LABELS],
        "target": correct_idx,
        "reference": LABELS[correct_idx],
        "source": row.get("source", ""),
        "task_name": row.get("metadata", {}).get("task_name", "gsm8k"),
        "benchmark": "gsm8k_numeric_mcq",
        "answer": answer,
        "choice_values": values,
    }


def _iter_source_rows(sources: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for source in sources:
        for row in read_jsonl(source):
            row = dict(row)
            row["_source_path"] = str(source)
            yield row


def build_lm_eval_rows(sources: Path | list[Path], *, folds: int, max_cases: int | None = None) -> list[dict[str, Any]]:
    source_list = [sources] if isinstance(sources, Path) else list(sources)
    rows: list[dict[str, Any]] = []
    for raw_idx, row in enumerate(_iter_source_rows(source_list)):
        if max_cases is not None and len(rows) >= max_cases:
            break
        reference = str(row.get("reference", "")).strip().upper()
        case_id = len(rows)
        fold = case_id % max(1, folds)
        if reference in LABELS:
            prompt = str(row["prompt"]).rstrip()
            rows.append(
                {
                    "case_id": case_id,
                    "fold": fold,
                    "prompt": f"{prompt}\n\nAnswer:",
                    "choices": [f" {label}" for label in LABELS],
                    "target": LABELS.index(reference),
                    "reference": reference,
                    "source": row.get("source", ""),
                    "task_name": row.get("metadata", {}).get("task_name", ""),
                    "benchmark": row.get("metadata", {}).get("benchmark", "mmlu_stem"),
                    "source_row_index": raw_idx,
                }
            )
            continue
        if str(row.get("task", "")).lower() == "gsm8k":
            built = _build_gsm8k_mcq_row(row, case_id=case_id, fold=fold)
            if built is not None:
                built["source_row_index"] = raw_idx
                rows.append(built)
    if not rows:
        raise ValueError(f"no usable letter-choice rows found in {', '.join(str(path) for path in source_list)}")
    return rows


def parse_policies(text: str) -> list[Policy]:
    if not text:
        return [Policy(k, v) for k, v in DEFAULT_POLICIES]
    policies: list[Policy] = []
    for item in text.split(","):
        left, right = item.split(":", 1)
        policies.append(Policy(left.strip(), right.strip()))
    return policies


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


def find_label_logprob(top_logprobs: list[dict[str, Any]], continuation: str) -> float | None:
    wanted = {continuation, continuation.strip(), f" {continuation.strip()}"}
    for item in top_logprobs:
        token = str(item.get("token", ""))
        if token in wanted:
            return float(item["logprob"])
    return None


class LlamaServerLetterLM:
    """lm-eval LM adapter for one-token letter choices through llama-server."""

    def __init__(self, base_url: str, *, n_probs: int, request_timeout: int) -> None:
        from lm_eval.api.model import LM

        class _LM(LM):
            def __init__(self, outer: "LlamaServerLetterLM") -> None:
                super().__init__()
                self.outer = outer

            def loglikelihood(self, requests, disable_tqdm: bool = False):
                return self.outer.loglikelihood(requests)

            def generate_until(self, requests, disable_tqdm: bool = False):
                raise NotImplementedError("generation is not used for this multiple-choice CV task")

            def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
                raise NotImplementedError("rolling loglikelihood is not used for this CV task")

        self.base_url = base_url.rstrip("/")
        self.n_probs = n_probs
        self.request_timeout = request_timeout
        self.context_cache: dict[str, dict[str, Any]] = {}
        self.lm = _LM(self)

    def _top_logprobs(self, context: str) -> list[dict[str, Any]]:
        cached = self.context_cache.get(context)
        if cached is not None:
            return cached["top_logprobs"]
        payload = {
            "prompt": context,
            "n_predict": 1,
            "n_probs": self.n_probs,
            "temperature": 0,
            "cache_prompt": True,
        }
        started = time.perf_counter()
        response = requests.post(f"{self.base_url}/completion", json=payload, timeout=self.request_timeout)
        response.raise_for_status()
        data = response.json()
        probabilities = data.get("completion_probabilities") or []
        if not probabilities:
            raise RuntimeError(f"llama-server returned no completion_probabilities: {data}")
        elapsed = time.perf_counter() - started
        top_logprobs = probabilities[0].get("top_logprobs") or []
        self.context_cache[context] = {
            "top_logprobs": top_logprobs,
            "content": data.get("content", ""),
            "elapsed_sec": elapsed,
            "timings": data.get("timings", {}),
        }
        return top_logprobs

    def loglikelihood(self, requests_list) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for req in requests_list:
            context, continuation = req.args
            top_logprobs = self._top_logprobs(str(context))
            logprob = find_label_logprob(top_logprobs, str(continuation))
            if logprob is None:
                finite = [float(item["logprob"]) for item in top_logprobs if "logprob" in item]
                logprob = (min(finite) - 10.0) if finite else -100.0
            top_token = str(top_logprobs[0].get("token", "")) if top_logprobs else ""
            is_greedy = top_token in {str(continuation), str(continuation).strip(), f" {str(continuation).strip()}"}
            results.append((logprob, is_greedy))
        return results


def wait_for_server(base_url: str, process: subprocess.Popen[Any], *, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
        except Exception as exc:  # noqa: BLE001 - diagnostic surface
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"llama-server did not become healthy within {timeout_sec}s: {last_error}")


def start_server(
    *,
    llama_server: Path,
    model: Path,
    policy: Policy,
    port: int,
    ctx_size: int,
    ngl: int,
    out_dir: Path,
    startup_timeout_sec: int,
) -> tuple[subprocess.Popen[Any], str, dict[str, Any]]:
    base_url = f"http://127.0.0.1:{port}"
    stdout_path = out_dir / "raw" / f"server_{policy.name}.stdout.log"
    stderr_path = out_dir / "raw" / f"server_{policy.name}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    command = [
        str(llama_server),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(ctx_size),
        "-ngl",
        str(ngl),
        "--cache-type-k",
        policy.cache_k,
        "--cache-type-v",
        policy.cache_v,
        "--no-webui",
        "--parallel",
        "1",
        "--alias",
        policy.name,
        "--no-warmup",
    ]
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
    metadata = {
        "command": public_command(command, llama_server=llama_server, model=model),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "pid": process.pid,
        "status": "starting",
    }
    try:
        wait_for_server(base_url, process, timeout_sec=startup_timeout_sec)
    except Exception as exc:  # noqa: BLE001
        metadata["status"] = "unsupported_or_failed"
        metadata["error"] = str(exc)
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        stdout.close()
        stderr.close()
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        detail = first_matching_line(
            stderr_text,
            ["Unsupported cache type", "error while handling argument", "invalid argument"],
        )
        if detail:
            metadata["error_detail"] = detail
        return process, base_url, metadata
    metadata["status"] = "ok"
    return process, base_url, metadata


def first_matching_line(text: str, needles: list[str]) -> str | None:
    for line in text.splitlines():
        for needle in needles:
            if needle.lower() in line.lower():
                return line.strip()
    return None


def public_command(command: list[str], *, llama_server: Path, model: Path) -> list[str]:
    redacted = list(command)
    if redacted:
        redacted[0] = llama_server.name
    for idx, item in enumerate(redacted[:-1]):
        if item == "-m":
            redacted[idx + 1] = model.name
            break
    return redacted


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def lm_eval_task_config(data_path: Path, *, task_name: str) -> dict[str, Any]:
    return {
        "task": task_name,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(data_path)}},
        "test_split": "test",
        "output_type": "multiple_choice",
        "doc_to_text": "{{prompt}}",
        "doc_to_choice": "{{choices}}",
        "doc_to_target": "{{target}}",
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    }


def run_policy_eval(
    *,
    policy: Policy,
    data_rows: list[dict[str, Any]],
    data_path: Path,
    llama_server: Path,
    model: Path,
    port: int,
    ctx_size: int,
    ngl: int,
    n_probs: int,
    out_dir: Path,
    startup_timeout_sec: int,
    request_timeout_sec: int,
    task_name: str,
) -> dict[str, Any]:
    process, base_url, server = start_server(
        llama_server=llama_server,
        model=model,
        policy=policy,
        port=port,
        ctx_size=ctx_size,
        ngl=ngl,
        out_dir=out_dir,
        startup_timeout_sec=startup_timeout_sec,
    )
    if server["status"] != "ok":
        return {"policy": policy.name, "server": server, "rows": [], "lm_eval": None}
    try:
        from lm_eval import evaluator

        adapter = LlamaServerLetterLM(base_url, n_probs=n_probs, request_timeout=request_timeout_sec)
        task_config = lm_eval_task_config(data_path, task_name=task_name)
        lm_result = evaluator.simple_evaluate(
            model=adapter.lm,
            tasks=[task_config],
            log_samples=True,
            verbosity="ERROR",
            bootstrap_iters=100,
        )
        rows: list[dict[str, Any]] = []
        by_prompt = {row["prompt"]: row for row in data_rows}
        for prompt, cached in adapter.context_cache.items():
            source_row = by_prompt[prompt]
            scores = [
                find_label_logprob(cached["top_logprobs"], f" {label}")
                for label in LABELS
            ]
            finite_scores = [score if score is not None else -100.0 for score in scores]
            pred_idx = max(range(len(finite_scores)), key=lambda idx: finite_scores[idx])
            rows.append(
                {
                    "policy": policy.name,
                    "case_id": source_row["case_id"],
                    "fold": source_row["fold"],
                    "task_name": source_row["task_name"],
                    "benchmark": source_row.get("benchmark", ""),
                    "reference": source_row["reference"],
                    "prediction": LABELS[pred_idx],
                    "correct": int(pred_idx == int(source_row["target"])),
                    "score_A": finite_scores[0],
                    "score_B": finite_scores[1],
                    "score_C": finite_scores[2],
                    "score_D": finite_scores[3],
                    "server_content": cached.get("content", ""),
                    "elapsed_sec": cached.get("elapsed_sec"),
                    "prompt_per_second": cached.get("timings", {}).get("prompt_per_second"),
                }
            )
        rows.sort(key=lambda row: int(row["case_id"]))
        return {
            "policy": policy.name,
            "server": server,
            "lm_eval": lm_result,
            "rows": rows,
        }
    finally:
        stop_server(process)


def compare_policies(rows: list[dict[str, Any]], *, folds: int) -> dict[str, Any]:
    fold_scores: dict[str, list[float]] = {}
    policy_names = list(dict.fromkeys(str(row["policy"]) for row in rows))
    for policy in policy_names:
        values: list[float] = []
        for fold in range(folds):
            fold_rows = [row for row in rows if row["policy"] == policy and int(row["fold"]) == fold]
            if fold_rows:
                values.append(sum(int(row["correct"]) for row in fold_rows) / len(fold_rows))
        fold_scores[policy] = values
    min_blocks = min((len(values) for values in fold_scores.values()), default=0)
    paired = {policy: scores[:min_blocks] for policy, scores in fold_scores.items() if len(scores) >= min_blocks and min_blocks > 0}
    return {
        "fold_scores": fold_scores,
        "summaries": [
            {"policy": policy, **_summary(scores)}
            for policy, scores in fold_scores.items()
        ],
        "pairwise": [item.__dict__ for item in pairwise_group_comparisons(paired, seed=21)] if len(paired) >= 2 else [],
        "omnibus": friedman_permutation_test(paired, seed=21).__dict__ if len(paired) >= 3 else None,
    }


def compare_policies_by_group(rows: list[dict[str, Any]], *, folds: int, group_key: str) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for group in sorted({str(row.get(group_key, "")) for row in rows}):
        if not group:
            continue
        group_rows = [row for row in rows if str(row.get(group_key, "")) == group]
        groups[group] = compare_policies(group_rows, folds=folds)
    return groups


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# lm-eval-harness GGUF K/V CV report",
        "",
        f"- Task: `{report['task']}`",
        f"- Cases: `{report['n_cases']}`; folds: `{report['folds']}`",
        f"- Benchmarks: `{', '.join(report.get('benchmarks', []))}`",
        f"- Model: `{report['model']}`",
        f"- Runtime: llama-server `--ngl {report['ngl']}` with OpenAI-compatible `/completion` logprobs through lm-eval.",
        "",
        "| policy | folds | accuracy mean +/- SEM | 95% CI | status |",
        "|---|---:|---:|---:|---|",
    ]
    statuses = {item["policy"]: item["server"]["status"] for item in report["policy_reports"]}
    for row in report["comparison"]["summaries"]:
        lines.append(
            f"| `{row['policy']}` | {int(row['n'])} | {row['mean']:.4f} +/- {row['sem']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | `{statuses.get(row['policy'], 'unknown')}` |"
        )
    unsupported = [item for item in report["policy_reports"] if item["server"]["status"] != "ok"]
    for item in unsupported:
        server = item["server"]
        status = server.get("error_detail") or server.get("error") or server["status"]
        lines.append(f"| `{item['policy']}` | 0 | n/a | n/a | `{status}` |")
    lines.extend(["", "| comparison | mean delta | p | method |", "|---|---:|---:|---|"])
    for row in report["comparison"]["pairwise"]:
        lines.append(f"| `{row['left']}` - `{row['right']}` | {row['mean_delta']:.4f} | {row['p_value']:.6f} | `{row['method']}` |")
    omnibus = report["comparison"].get("omnibus")
    if omnibus:
        lines.extend([
            "",
            f"Friedman within-fold permutation p: `{omnibus['p_value']:.6f}` "
            f"(statistic `{omnibus['statistic']:.4f}`, n={omnibus['n_blocks']}).",
        ])
    by_benchmark = report.get("comparison_by_benchmark") or {}
    for benchmark, comparison in by_benchmark.items():
        lines.extend([
            "",
            f"## Benchmark slice: `{benchmark}`",
            "",
            "| policy | folds | accuracy mean +/- SEM | 95% CI |",
            "|---|---:|---:|---:|",
        ])
        for row in comparison["summaries"]:
            lines.append(
                f"| `{row['policy']}` | {int(row['n'])} | {row['mean']:.4f} +/- {row['sem']:.4f} | "
                f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
            )
        bench_omnibus = comparison.get("omnibus")
        if bench_omnibus:
            lines.append("")
            lines.append(f"Friedman p: `{bench_omnibus['p_value']:.6f}`.")
    lines.append("")
    lines.append(
        "Scope: external-heldout letter-choice CV. MMLU-STEM rows are native MCQ; "
        "GSM8K rows are numeric-MCQ transformations of GSM8K test examples. "
        "This is a logged lm-eval-harness serving-surface gate, not a broad leaderboard result."
    )
    return "\n".join(lines) + "\n"


def render_plot(report: dict[str, Any], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["comparison"]["summaries"]
    labels = [row["policy"].replace("_V=", "\nV=").replace("K=", "K=") for row in rows]
    means = [row["mean"] for row in rows]
    sems = [row["sem"] for row in rows]
    colors = ["#167f7a" if "q8_0" in row["policy"] else "#d28e2b" for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor("#f6f7f2")
    ax.set_facecolor("#ffffff")
    ax.bar(range(len(rows)), means, yerr=sems, capsize=5, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("lm-eval GGUF K/V CV: external heldout")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.5, 0.02, "Mean +/- SEM over deterministic folds; Turbo8 remains unsupported by the installed runtime.", ha="center", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    path = out_dir / "gptimage2_lm_eval_gguf_kv_cv.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sources", default="", help="comma-separated JSONL sources; overrides --source")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA_SERVER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--policies", default="")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--n-probs", type=int, default=128)
    parser.add_argument("--startup-timeout-sec", type=int, default=180)
    parser.add_argument("--request-timeout-sec", type=int, default=180)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = [Path(item.strip()) for item in args.sources.split(",") if item.strip()] if args.sources else [args.source]
    data_rows = build_lm_eval_rows(sources, folds=args.folds, max_cases=args.max_cases)
    data_path = args.out_dir / "lm_eval_external_heldout_letter_cv.jsonl"
    write_jsonl(data_path, data_rows)
    policies = parse_policies(args.policies)
    policy_reports: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for policy in policies:
        report = run_policy_eval(
            policy=policy,
            data_rows=data_rows,
            data_path=data_path,
            llama_server=args.llama_server,
            model=args.model,
            port=args.port,
            ctx_size=args.ctx_size,
            ngl=args.ngl,
            n_probs=args.n_probs,
            out_dir=args.out_dir,
            startup_timeout_sec=args.startup_timeout_sec,
            request_timeout_sec=args.request_timeout_sec,
            task_name=args.task_name,
        )
        policy_reports.append({k: v for k, v in report.items() if k != "lm_eval"})
        all_rows.extend(report["rows"])
    comparison = compare_policies(all_rows, folds=args.folds)
    full_report = {
        "task": args.task_name,
        "sources": [str(source) for source in sources],
        "data": str(data_path),
        "model": args.model.name,
        "llama_server": args.llama_server.name,
        "folds": args.folds,
        "n_cases": len(data_rows),
        "ngl": args.ngl,
        "policies": [policy.name for policy in policies],
        "benchmarks": sorted({str(row.get("benchmark", "")) for row in data_rows if row.get("benchmark")}),
        "policy_reports": policy_reports,
        "comparison": comparison,
        "comparison_by_benchmark": compare_policies_by_group(all_rows, folds=args.folds, group_key="benchmark"),
    }
    write_csv(args.out_dir / "lm_eval_gguf_kv_cv_rows.csv", all_rows)
    write_csv(args.out_dir / "lm_eval_gguf_kv_cv_summary.csv", comparison["summaries"])
    write_csv(args.out_dir / "lm_eval_gguf_kv_cv_pairwise.csv", comparison["pairwise"])
    (args.out_dir / "lm_eval_gguf_kv_cv_report.json").write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "lm_eval_gguf_kv_cv_report.md").write_text(render_markdown(full_report), encoding="utf-8")
    plot = render_plot(full_report, args.out_dir)
    print(f"wrote {args.out_dir / 'lm_eval_gguf_kv_cv_report.md'}")
    print(f"wrote {plot}")


if __name__ == "__main__":
    main()
