"""Any-time sweep for perplexity plus optional benchmark runs at each L."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from elt_lm.config import TrainConfig
from elt_lm.data import PackedTokenDataset
from elt_lm.eval.benchmarks import (
    BenchmarkResult,
    evaluate_benchmark,
    load_benchmark_manifest,
)
from elt_lm.eval.statistics import fold_accuracy_stats
from elt_lm.model import build_model
from elt_lm.telemetry import make_writer
from elt_lm.train import load_checkpoint


def load_model(ckpt: str | Path, device: torch.device) -> tuple[torch.nn.Module, TrainConfig]:
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg: TrainConfig = obj["cfg"]
    model = build_model(cfg.model)
    load_checkpoint(ckpt, model, opt=None)
    model = model.to(device=device)
    model.eval()
    return model, cfg


@torch.no_grad()
def eval_at_L(
    model: torch.nn.Module,
    dl: DataLoader,
    L: int,
    device: torch.device,
    max_batches: int,
) -> dict:
    """Return held-out perplexity and throughput metrics at the given L."""
    total_nll = 0.0
    total_tok = 0
    seen = 0
    total_wall = 0.0
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for input_ids, labels in dl:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=dtype):
            out = model(input_ids, L=L)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_wall += (time.perf_counter() - t0)
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += float(loss.item())
        total_tok += int(shift_labels.numel())
        seen += 1
        if seen >= max_batches:
            break

    nll = total_nll / max(1, total_tok)
    tps = total_tok / max(1e-9, total_wall)
    latency_ms = (total_wall / max(1, seen)) * 1000.0
    return {
        "nll": nll,
        "ppl": math.exp(nll),
        "tokens_per_sec": tps,
        "latency_ms_per_batch": latency_ms,
        "total_tokens": total_tok,
        "batches": seen,
    }


def _loop_refinement_metrics(
    result: BenchmarkResult,
    *,
    baseline: BenchmarkResult,
    previous: BenchmarkResult | None,
) -> dict[str, float | None]:
    """Measure whether deeper loops fix shallow-loop mistakes or overthink them."""
    loop_gain = result.accuracy - baseline.accuracy
    marginal_gain = None if previous is None else result.accuracy - previous.accuracy
    metrics: dict[str, float | None] = {
        "loop_gain": loop_gain,
        "marginal_gain": marginal_gain if marginal_gain is not None else loop_gain,
        "self_correction_rate": None,
        "overthinking_rate": None,
    }

    if len(result.case_correct) != len(baseline.case_correct):
        return metrics
    total = max(1, len(result.case_correct))
    self_corrections = sum(
        1 for shallow, deep in zip(baseline.case_correct, result.case_correct)
        if not shallow and deep
    )
    overthinking = sum(
        1 for shallow, deep in zip(baseline.case_correct, result.case_correct)
        if shallow and not deep
    )
    metrics["self_correction_rate"] = self_corrections / total
    metrics["overthinking_rate"] = overthinking / total
    return metrics


def _csv_value(value: float | int | str | None) -> float | int | str:
    return "" if value is None else value


def _parse_l_list(raw: str, *, cfg: TrainConfig) -> list[int]:
    if not raw.strip():
        return list(range(cfg.model.L_min, cfg.model.L_max + 1))
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < cfg.model.L_min or value > cfg.model.L_max:
            raise SystemExit(
                f"--L-list value {value} is outside trained range "
                f"[{cfg.model.L_min}, {cfg.model.L_max}]"
            )
        values.append(value)
    if not values:
        raise SystemExit("--L-list did not contain any loop values")
    return values


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)

    dl = None
    if args.val_bin:
        ds = PackedTokenDataset(args.val_bin, seq_len=args.seq_len or cfg.data.seq_len)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    benchmark_specs = (
        load_benchmark_manifest(args.benchmark_manifest)
        if args.benchmark_manifest else []
    )
    if dl is None and not benchmark_specs:
        raise SystemExit("anytime_sweep needs --val-bin, --benchmark-manifest, or both")

    tok = None
    if benchmark_specs:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_path, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

    run_dir = Path(args.run_dir) if args.run_dir else Path(args.ckpt).parent
    telemetry = make_writer(run_dir)
    try:
        L_range = _parse_l_list(getattr(args, "l_list", ""), cfg=cfg)
        cv_folds = int(getattr(args, "cv_folds", 5))
        first_L = L_range[0]
        rows: list[dict[str, float | int | str]] = []
        details: list[dict[str, object]] = []
        benchmark_history: dict[str, dict[int, BenchmarkResult]] = {}
        for L in L_range:
            approx_flops = L * cfg.model.n_unique_layers
            if dl is not None:
                stats = eval_at_L(model, dl, L, device, args.max_batches)
                print(
                    f"L={L}  NLL={stats['nll']:.4f}  PPL={stats['ppl']:.3f}  "
                    f"tok/s={stats['tokens_per_sec']:.0f}  "
                    f"batch-latency={stats['latency_ms_per_batch']:.1f}ms  "
                    f"relFLOPs={approx_flops}"
                )
                telemetry.emit(
                    "inference_sweep",
                    L=L,
                    rel_flops=approx_flops,
                    nll=stats["nll"],
                    ppl=stats["ppl"],
                    tokens_per_sec=stats["tokens_per_sec"],
                    latency_ms=stats["latency_ms_per_batch"],
                    total_tokens=stats["total_tokens"],
                    batches=stats["batches"],
                    ckpt=str(args.ckpt),
                )
                rows.append({
                    "kind": "perplexity",
                    "benchmark": "",
                    "task": "",
                    "L": L,
                    "nll": stats["nll"],
                    "ppl": stats["ppl"],
                    "score": "",
                    "tokens_per_sec": stats["tokens_per_sec"],
                    "latency_ms": stats["latency_ms_per_batch"],
                    "rel_flops": approx_flops,
                    "count": stats["batches"],
                })

            for spec in benchmark_specs:
                assert tok is not None
                result = evaluate_benchmark(
                    model=model,
                    tokenizer=tok,
                    spec=spec,
                    L=L,
                    device=device,
                    max_new_tokens=args.bench_max_new_tokens,
                    temperature=args.bench_temperature,
                    top_k=args.bench_top_k,
                    num_samples=args.bench_num_samples,
                    verifier_retries=args.bench_verifier_retries,
                )
                history = benchmark_history.setdefault(result.benchmark, {})
                baseline = result if L == first_L else history.get(first_L, result)
                previous = history.get(L - 1)
                loop_metrics = _loop_refinement_metrics(
                    result,
                    baseline=baseline,
                    previous=previous,
                )
                cv = fold_accuracy_stats(result.case_correct, folds=cv_folds)
                fmt_cv = fold_accuracy_stats(result.case_format_correct, folds=cv_folds)
                history[L] = result
                print(
                    f"L={L}  benchmark={result.benchmark}  score={result.accuracy:.4f}  "
                    f"format={result.format_rate:.4f}  "
                    f"cv={cv.mean:.4f}±{cv.sem:.4f}  "
                    f"gain={loop_metrics['loop_gain']:+.4f}  "
                    f"self-correct={_csv_value(loop_metrics['self_correction_rate'])}  "
                    f"overthink={_csv_value(loop_metrics['overthinking_rate'])}  "
                    f"latency={result.latency_ms_per_case:.1f}ms/case  "
                    f"attempts/case={result.attempts_per_case:.2f}  "
                    f"tok/s={result.tokens_per_sec:.0f}"
                )
                telemetry.emit(
                    "benchmark_eval",
                    benchmark=result.benchmark,
                    task=result.task,
                    L=L,
                    score=result.accuracy,
                    correct=result.correct,
                    total=result.total,
                    format_success_rate=result.format_rate,
                    format_correct=result.format_correct,
                    verifier_success_rate=result.accuracy,
                    verifier_correct=result.correct,
                    cv_folds=cv.fold_count,
                    cv_mean=cv.mean,
                    cv_std=cv.std,
                    cv_sem=cv.sem,
                    cv_ci95_low=cv.ci95_low,
                    cv_ci95_high=cv.ci95_high,
                    format_cv_mean=fmt_cv.mean,
                    format_cv_sem=fmt_cv.sem,
                    latency_ms=result.latency_ms_per_case,
                    tokens_per_sec=result.tokens_per_sec,
                    attempts_per_case=result.attempts_per_case,
                    loop_gain=loop_metrics["loop_gain"],
                    marginal_gain=loop_metrics["marginal_gain"],
                    self_correction_rate=loop_metrics["self_correction_rate"],
                    overthinking_rate=loop_metrics["overthinking_rate"],
                    ckpt=str(args.ckpt),
                )
                rows.append({
                    "kind": "benchmark",
                    "benchmark": result.benchmark,
                    "task": result.task,
                    "L": L,
                    "nll": "",
                    "ppl": "",
                    "score": result.accuracy,
                    "format_success_rate": result.format_rate,
                    "format_correct": result.format_correct,
                    "verifier_success_rate": result.accuracy,
                    "verifier_correct": result.correct,
                    "cv_folds": cv.fold_count,
                    "cv_mean": cv.mean,
                    "cv_std": cv.std,
                    "cv_sem": cv.sem,
                    "cv_ci95_low": cv.ci95_low,
                    "cv_ci95_high": cv.ci95_high,
                    "format_cv_mean": fmt_cv.mean,
                    "format_cv_sem": fmt_cv.sem,
                    "tokens_per_sec": result.tokens_per_sec,
                    "latency_ms": result.latency_ms_per_case,
                    "rel_flops": approx_flops,
                    "count": result.total,
                    "attempts_per_case": result.attempts_per_case,
                    "loop_gain": _csv_value(loop_metrics["loop_gain"]),
                    "marginal_gain": _csv_value(loop_metrics["marginal_gain"]),
                    "self_correction_rate": _csv_value(loop_metrics["self_correction_rate"]),
                    "overthinking_rate": _csv_value(loop_metrics["overthinking_rate"]),
                })
                details.append({
                    "benchmark": result.benchmark,
                    "task": result.task,
                    "L": L,
                    "case_correct": result.case_correct,
                    "case_format_correct": result.case_format_correct,
                    "cv": cv.__dict__,
                    "format_cv": fmt_cv.__dict__,
                    "loop_metrics": loop_metrics,
                })

        out_csv = Path(args.out_csv) if args.out_csv else None
        if out_csv:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "kind",
                        "benchmark",
                        "task",
                        "L",
                        "nll",
                        "ppl",
                        "score",
                        "format_success_rate",
                        "format_correct",
                        "verifier_success_rate",
                        "verifier_correct",
                        "cv_folds",
                        "cv_mean",
                        "cv_std",
                        "cv_sem",
                        "cv_ci95_low",
                        "cv_ci95_high",
                        "format_cv_mean",
                        "format_cv_sem",
                        "tokens_per_sec",
                        "latency_ms",
                        "rel_flops",
                        "count",
                        "attempts_per_case",
                        "loop_gain",
                        "marginal_gain",
                        "self_correction_rate",
                        "overthinking_rate",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"wrote {out_csv}")
        out_json = Path(getattr(args, "out_json", "") or "") if getattr(args, "out_json", "") else None
        if out_json:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ckpt": str(args.ckpt),
                "benchmark_manifest": str(args.benchmark_manifest or ""),
                "rows": rows,
                "details": details,
                "cv_folds": cv_folds,
                "interpretation_note": (
                    "For v0 HauhauCS stem data, high format/verifier rates mainly "
                    "indicate schema memorization on a small duplicate-heavy "
                    "validation set; do not read them as broad STEM reasoning "
                    "generalization without v1 quality-gated benchmarks."
                ),
            }
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote {out_json}")
    finally:
        telemetry.close()


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-bin", default="")
    p.add_argument("--benchmark-manifest", type=str, default="")
    p.add_argument("--seq-len", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--bench-max-new-tokens", type=int, default=128)
    p.add_argument("--bench-temperature", type=float, default=0.0)
    p.add_argument("--bench-top-k", type=int, default=1)
    p.add_argument("--bench-num-samples", type=int, default=1)
    p.add_argument("--bench-verifier-retries", type=int, default=0)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--L-list", dest="l_list", type=str, default="")
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument("--out-json", type=str, default="")
    p.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="directory to write telemetry into. Defaults to the checkpoint's parent directory.",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    cli()
