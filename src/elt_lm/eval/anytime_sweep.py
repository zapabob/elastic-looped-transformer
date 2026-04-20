"""Any-Time sweep: compute validation perplexity at each L ∈ [L_min, L_max].

This is the headline Any-Time capability of ELT/ILSD — after one training run,
quality should degrade gracefully as L shrinks, all the way from L_max to L_min.

Usage:
    uv run elt-anytime --ckpt runs/tiny_10M/last.pt --val-bin data_bin/val.bin \
                       --seq-len 512 --batch-size 4 --max-batches 50
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from elt_lm.config import TrainConfig
from elt_lm.data import PackedTokenDataset
from elt_lm.model import ELTLanguageModel
from elt_lm.telemetry import make_writer


def load_model(ckpt: str | Path, device: torch.device) -> tuple[ELTLanguageModel, TrainConfig]:
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    cfg: TrainConfig = obj["cfg"]
    model = ELTLanguageModel(cfg.model).to(device=device)
    model.load_state_dict(obj["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def eval_at_L(
    model: ELTLanguageModel,
    dl: DataLoader,
    L: int,
    device: torch.device,
    max_batches: int,
) -> dict:
    """Return a dict of metrics at the given loop count.

    Keys: nll, ppl, tokens_per_sec, latency_ms_per_batch, total_tokens, batches.
    """
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


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)

    ds = PackedTokenDataset(args.val_bin, seq_len=args.seq_len or cfg.data.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # `--run-dir` (or inferred from ckpt parent) controls where inference_sweep
    # events are emitted — the dashboard's Inference panel tails this file.
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.ckpt).parent
    telemetry = make_writer(run_dir)
    try:
        L_range = range(cfg.model.L_min, cfg.model.L_max + 1)
        rows: list[dict[str, float | int]] = []
        for L in L_range:
            stats = eval_at_L(model, dl, L, device, args.max_batches)
            approx_flops = L * cfg.model.n_unique_layers         # relative proxy
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
                "L": L, "nll": stats["nll"], "ppl": stats["ppl"],
                "tokens_per_sec": stats["tokens_per_sec"],
                "latency_ms": stats["latency_ms_per_batch"],
                "rel_flops": approx_flops,
            })

        out_csv = Path(args.out_csv) if args.out_csv else None
        if out_csv:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["L", "nll", "ppl", "tokens_per_sec",
                                "latency_ms", "rel_flops"],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"wrote {out_csv}")
    finally:
        telemetry.close()


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-bin", required=True)
    p.add_argument("--seq-len", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument("--run-dir", type=str, default="",
                   help="directory to write inference_sweep telemetry into. "
                        "Defaults to the checkpoint's parent directory.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    cli()
