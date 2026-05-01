"""ELT-LM training loop with ILSD (arXiv:2604.09168).

Example:
    uv run elt-train --config configs/tiny_10M.yaml

The config is a YAML file parsed into TrainConfig. See configs/ for examples.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from elt_lm.config import TrainConfig, load_train_config
from elt_lm.data import PackedTokenDataset
from elt_lm.ilsd import ILSDLossFn
from elt_lm.model import build_model
from elt_lm.telemetry import make_writer


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


def configure_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW variants.

    kind=adamw             : stock torch.optim.AdamW (fp32 state on GPU)
    kind=paged_adamw_8bit  : bitsandbytes PagedAdamW8bit — 8-bit state paged via
                             CUDA managed memory (spills to host RAM). Large
                             VRAM savings; acceptable convergence loss for LM.
    kind=nvme_adamw        : Phase-C; fp32 state memory-mapped on NVMe.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kind = cfg.optim.kind
    if kind == "adamw":
        return torch.optim.AdamW(
            groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
        )
    if kind == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError(
                "optim.kind=paged_adamw_8bit requires bitsandbytes. "
                "Install with: uv sync --extra offload_8bit"
            ) from e
        cls_name = "PagedAdamW8bit" if cfg.optim.paged_bits == 8 else "PagedAdamW32bit"
        cls = getattr(bnb.optim, cls_name)
        return cls(
            groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
            percentile_clipping=cfg.optim.paged_percentile_clipping,
        )
    if kind == "nvme_adamw":
        # Handled out-of-band by install_offload_into_training, which needs
        # the run_dir context that configure_optimizer doesn't have. We raise
        # here so callers that forget to special-case this kind fail loudly.
        raise RuntimeError(
            "optim.kind=nvme_adamw must be built via "
            "elt_lm.offload.hooks.install_offload_into_training(model, cfg, run_dir). "
            "The train.train() loop dispatches this automatically."
        )
    raise ValueError(f"unknown optim.kind={kind!r}")


def _update_last_hardlink(out_dir: Path, path: Path) -> None:
    """Point `out_dir/last.pt` at `path` via NTFS hardlink (with copy fallback)."""
    latest = out_dir / "last.pt"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
    except OSError:
        pass
    try:
        os.link(path, latest)
    except OSError:
        shutil.copy2(path, latest)


def _build_ckpt_state(model: nn.Module, opt: torch.optim.Optimizer,
                      cfg: TrainConfig, step: int, extra: dict | None = None) -> dict:
    model_state: dict
    if cfg.model.hf_save_adapter_only and hasattr(model, "adapter_checkpoint_state"):
        adapter_state = model.adapter_checkpoint_state()  # type: ignore[attr-defined]
        model_state = dict(adapter_state)
    else:
        model_state = {"model": model.state_dict()}
    state = {
        "step": step,
        "optim": opt.state_dict(),
        "cfg": cfg,
        "rng_state": torch.get_rng_state(),
        "wall_time": time.time(),
        **model_state,
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    if extra:
        state.update(extra)
    return state


def _atomic_torch_save(state: dict, path: Path) -> None:
    """Write a checkpoint through a temp file, then atomically replace target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        if tmp.exists():
            tmp.unlink()
        torch.save(state, tmp)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def save_checkpoint(model: nn.Module, opt: torch.optim.Optimizer, cfg: TrainConfig,
                    step: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:07d}.pt"
    _atomic_torch_save(_build_ckpt_state(model, opt, cfg, step), path)
    _update_last_hardlink(out_dir, path)
    print(f"  saved {path}")


class RollingCheckpointer:
    """Time-based rolling checkpoint: writes slot_{0..keep-1} round-robin.

    Independent from save_every step-snapshots; intended for crash recovery.
    Always updates out_dir/last.pt to point at the most recent rolling write.
    """

    def __init__(self, out_dir: Path, interval_sec: int, keep: int):
        self.out_dir = out_dir
        self.interval = max(1, int(interval_sec))
        self.keep = max(1, int(keep))
        self.last_save_t = time.time()
        self.next_slot = 0

    def maybe_save(self, model: nn.Module, opt: torch.optim.Optimizer,
                   cfg: TrainConfig, step: int, force: bool = False) -> bool:
        now = time.time()
        if not force and (now - self.last_save_t) < self.interval:
            return False
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"rolling_{self.next_slot}.pt"
        _atomic_torch_save(_build_ckpt_state(model, opt, cfg, step), path)
        _update_last_hardlink(self.out_dir, path)
        saved_slot = self.next_slot
        self.next_slot = (self.next_slot + 1) % self.keep
        self.last_save_t = now
        print(f"  rolling-ckpt slot {saved_slot} -> {path}")
        return True


def load_checkpoint(path: str | Path, model: nn.Module,
                    opt: torch.optim.Optimizer | None = None) -> int:
    """Load a checkpoint, restore RNG state, return the step index to resume from."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("adapter_only") and hasattr(model, "load_adapter_checkpoint_state"):
        model.load_adapter_checkpoint_state(state)  # type: ignore[attr-defined]
    else:
        model.load_state_dict(state["model"] if "model" in state else state)
        if hasattr(model, "remember_adapter_base_checkpoint"):
            model.remember_adapter_base_checkpoint(path)  # type: ignore[attr-defined]
    if opt is not None and "optim" in state:
        opt.load_state_dict(state["optim"])
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"].to("cpu") if hasattr(state["rng_state"], "to")
                            else state["rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in state:
        try:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        except (RuntimeError, TypeError):
            pass
    step = int(state.get("step", 0))
    print(f"  resumed from {path} (step {step})")
    return step


def train(cfg: TrainConfig, resume: str | None = None) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(cfg.dtype)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- model ------------------------------------------------------------
    model = build_model(cfg.model).to(device=device, dtype=dtype)
    model.train()
    if not hasattr(model, "num_parameters"):
        raise TypeError(f"model {type(model)!r} does not expose num_parameters()")
    n_total = model.num_parameters()  # type: ignore[call-arg]
    n_non_emb = model.num_parameters(non_embedding=True)  # type: ignore[call-arg]
    print(f"model params: {n_total/1e6:.1f}M total, {n_non_emb/1e6:.1f}M non-embedding")

    if (
        resume is None
        and cfg.model.hf_trainable_mode == "lora"
        and cfg.model.hf_adapter_base_ckpt
    ):
        base_step = load_checkpoint(cfg.model.hf_adapter_base_ckpt, model, opt=None)
        print(f"  initialized LoRA base from {cfg.model.hf_adapter_base_ckpt} (base step {base_step})")

    offload_store = None
    if cfg.optim.kind == "nvme_adamw":
        from elt_lm.offload.hooks import install_offload_into_training
        opt, offload_store = install_offload_into_training(
            model, cfg=cfg, run_dir=run_dir,
        )
    else:
        opt = configure_optimizer(model, cfg)
    loss_fn = ILSDLossFn(cfg.model, cfg.ilsd, seed=cfg.seed)

    resume_step = 0
    if resume:
        resume_step = load_checkpoint(resume, model, opt)

    rolling = RollingCheckpointer(
        run_dir,
        interval_sec=cfg.rolling_ckpt_interval_sec,
        keep=cfg.rolling_ckpt_keep,
    )

    telemetry = make_writer(run_dir)
    telemetry.emit(
        "train_config",
        model_params_total=n_total,
        model_params_non_embedding=n_non_emb,
        d_model=cfg.model.d_model,
        n_unique_layers=cfg.model.n_unique_layers,
        L_min=cfg.model.L_min,
        L_max=cfg.model.L_max,
        micro_batch_size=cfg.micro_batch_size,
        grad_accum_steps=cfg.grad_accum_steps,
        dtype=cfg.dtype,
        total_steps=cfg.total_steps,
    )

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
    global_step = resume_step
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
                    f"L_dist {out.l_dist.item():.3f} | "
                    f"L_ent {out.l_entropy.item():.3f} | "
                    f"L_curve {out.l_curve.item():.3f} | "
                    f"L_logit {out.l_logit_curve.item():.3f} | "
                    f"L_local {out.l_local.item():.3f} | {tps:.0f} tok/s"
                )
                telemetry.emit(
                    "train_step",
                    step=global_step,
                    lr=lr,
                    loss=avg,
                    l_gt_teacher=out.l_gt_teacher.item(),
                    l_gt_student=out.l_gt_student.item(),
                    l_dist=out.l_dist.item(),
                    l_entropy=out.l_entropy.item(),
                    l_curve=out.l_curve.item(),
                    l_logit_curve=out.l_logit_curve.item(),
                    l_local=out.l_local.item(),
                    L_int=out.L_int,
                    lambda_value=out.lambda_value,
                    tokens_per_sec=tps,
                )
                accum_loss = 0.0

            if cfg.save_every and global_step > 0 and global_step % cfg.save_every == 0:
                save_checkpoint(model, opt, cfg, global_step, run_dir)
                telemetry.emit("checkpoint", kind="milestone", step=global_step)

            if rolling.maybe_save(model, opt, cfg, global_step):
                telemetry.emit("checkpoint", kind="rolling", step=global_step,
                               slot=(rolling.next_slot - 1) % rolling.keep)

            global_step += 1
            if global_step >= cfg.total_steps:
                break

    save_checkpoint(model, opt, cfg, global_step, run_dir)
    telemetry.emit("checkpoint", kind="final", step=global_step)
    if offload_store is not None:
        offload_store.flush()
    telemetry.close()
    print(f"done in {(time.time()-t0)/60:.1f} min")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None,
                   help="path to checkpoint to resume from (usually runs/<name>/last.pt)")
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

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    cli()
