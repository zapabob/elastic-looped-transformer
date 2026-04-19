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
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from elt_lm.config import TrainConfig
from elt_lm.data import PackedTokenDataset
from elt_lm.model import ELTLanguageModel


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
) -> tuple[float, float]:
    """Return (nll_per_token, perplexity) at the given loop count."""
    total_nll = 0.0
    total_tok = 0
    seen = 0
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for input_ids, labels in dl:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            out = model(input_ids, L=L)
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
    return nll, math.exp(nll)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)

    ds = PackedTokenDataset(args.val_bin, seq_len=args.seq_len or cfg.data.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    L_range = range(cfg.model.L_min, cfg.model.L_max + 1)
    rows: list[dict[str, float | int]] = []
    for L in L_range:
        nll, ppl = eval_at_L(model, dl, L, device, args.max_batches)
        approx_flops = L * cfg.model.n_unique_layers         # relative proxy
        print(f"L={L}  NLL={nll:.4f}  PPL={ppl:.3f}  relFLOPs={approx_flops}")
        rows.append({"L": L, "nll": nll, "ppl": ppl, "rel_flops": approx_flops})

    out_csv = Path(args.out_csv) if args.out_csv else None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["L", "nll", "ppl", "rel_flops"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"wrote {out_csv}")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-bin", required=True)
    p.add_argument("--seq-len", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--out-csv", type=str, default="")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    cli()
