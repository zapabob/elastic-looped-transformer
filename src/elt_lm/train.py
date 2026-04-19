"""ELT-LM training loop with ILSD (arXiv:2604.09168).

Example:
    uv run elt-train --config configs/tiny_10M.yaml

The config is a YAML file parsed into TrainConfig. See configs/ for examples.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from elt_lm.config import TrainConfig, load_train_config
from elt_lm.data import PackedTokenDataset
from elt_lm.ilsd import ILSDLossFn
from elt_lm.model import ELTLanguageModel


def get_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if cfg.lr_schedule == "constant":
        return cfg.lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    if cfg.lr_schedule == "linear":
        return cfg.lr + (cfg.min_lr - cfg.lr) * progress
    # cosine
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * coeff


def configure_optimizer(model: ELTLanguageModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay on 2D params only (biases / norms excluded)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )


def save_checkpoint(model: ELTLanguageModel, opt: torch.optim.Optimizer, cfg: TrainConfig,
                    step: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:07d}.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optim": opt.state_dict(),
        "cfg": cfg,
    }, path)
    latest = out_dir / "last.pt"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
    except OSError:
        pass
    try:
        os.link(path, latest)          # hardlink (Windows supports on NTFS)
    except OSError:
        # fallback: just copy the file (rare)
        torch.save(torch.load(path, map_location="cpu"), latest)
    print(f"  saved {path}")


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(cfg.dtype)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- model ------------------------------------------------------------
    model = ELTLanguageModel(cfg.model).to(device=device, dtype=dtype)
    model.train()
    n_total = model.num_parameters()
    n_non_emb = model.num_parameters(non_embedding=True)
    print(f"model params: {n_total/1e6:.1f}M total, {n_non_emb/1e6:.1f}M non-embedding")

    opt = configure_optimizer(model, cfg)
    loss_fn = ILSDLossFn(cfg.model, cfg.ilsd, seed=cfg.seed)

    # ---- data -------------------------------------------------------------
    train_ds = PackedTokenDataset(cfg.data.train_bin, seq_len=cfg.data.seq_len)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        num_workers=0,                  # Windows + mmap: 0 workers is most robust
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"train tokens: {train_ds.n_tokens:,} | windows: {len(train_ds):,}")

    # ---- loop -------------------------------------------------------------
    global_step = 0
    micro_step = 0
    t0 = time.time()
    accum_loss = 0.0

    while global_step < cfg.total_steps:
        for input_ids, labels in train_dl:
            input_ids = input_ids.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            out = loss_fn(model, input_ids, labels, step=global_step)
            loss = out.total / cfg.grad_accum_steps
            loss.backward()
            accum_loss += out.total.detach().float().item()

            micro_step += 1
            if micro_step % cfg.grad_accum_steps != 0:
                continue

            # update step
            lr = lr_at(global_step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

            if global_step % cfg.log_every == 0:
                tps = (global_step + 1) * cfg.effective_batch_size * cfg.data.seq_len / max(1e-9, time.time() - t0)
                avg = accum_loss / (cfg.log_every * cfg.grad_accum_steps) if global_step else accum_loss / cfg.grad_accum_steps
                print(
                    f"step {global_step:6d} | lr {lr:.2e} | "
                    f"loss {avg:.4f} | L_int {out.L_int} | λ {out.lambda_value:.3f} | "
                    f"L_GT_t {out.l_gt_teacher.item():.3f} | L_GT_s {out.l_gt_student.item():.3f} | "
                    f"L_dist {out.l_dist.item():.3f} | {tps:.0f} tok/s"
                )
                accum_loss = 0.0

            if cfg.save_every and global_step > 0 and global_step % cfg.save_every == 0:
                save_checkpoint(model, opt, cfg, global_step, run_dir)

            global_step += 1
            if global_step >= cfg.total_steps:
                break

    save_checkpoint(model, opt, cfg, global_step, run_dir)
    print(f"done in {(time.time()-t0)/60:.1f} min")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[], help="override TrainConfig fields: key=value")
    args = p.parse_args()

    cfg = load_train_config(args.config)
    for kv in args.override:
        key, _, value = kv.partition("=")
        if not key:
            continue
        # naive override — only supports top-level scalar fields on TrainConfig
        if not hasattr(cfg, key):
            print(f"[warn] unknown override field: {key}", file=sys.stderr)
            continue
        current = getattr(cfg, key)
        caster = type(current) if current is not None else str
        try:
            setattr(cfg, key, caster(value))
        except (TypeError, ValueError) as e:
            print(f"[warn] could not cast {key}={value}: {e}", file=sys.stderr)

    train(cfg)


if __name__ == "__main__":
    cli()
