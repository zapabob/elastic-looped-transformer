"""1B VRAM fit smoke test.

Constructs the base_1B model, runs one synthetic forward/backward/step cycle
per optimizer backend, and reports peak VRAM. The goal is to prove:

    1. configs/base_1B.yaml + optim.kind=paged_adamw_8bit fits under 12 GB
    2. configs/base_1B.yaml + optim.kind=nvme_adamw also fits (smaller
       VRAM footprint at the cost of a slow per-step NVMe round-trip)

Usage:

    uv run python scripts/smoke_1b_vram.py --optim paged_adamw_8bit
    uv run python scripts/smoke_1b_vram.py --optim nvme_adamw --run-dir H:/elt_data/runs/1B_smoke
    uv run python scripts/smoke_1b_vram.py --optim adamw           # expected OOM

Prints peak / current VRAM, elapsed seconds, and the param counts so the log
goes directly into the phase-C follow-up docs.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from elt_lm.config import OptimConfig, load_train_config
from elt_lm.model import ELTLanguageModel
from elt_lm.train import configure_optimizer


def _fmt_gb(n: int) -> str:
    return f"{n / 1024**3:.2f} GB"


def run(optim_kind: str, run_dir: Path, seq_len: int, micro_batch: int,
        config_path: str = "configs/base_1B.yaml") -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("VRAM smoke needs CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = load_train_config(config_path)
    cfg.optim = OptimConfig(kind=optim_kind)
    # Trim for smoke: one micro-batch, short seq, no grad accum.
    cfg.micro_batch_size = micro_batch
    cfg.data.seq_len = seq_len
    cfg.run_dir = str(run_dir)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[build] model bf16 on {device}…")
    t0 = time.perf_counter()
    model = ELTLanguageModel(cfg.model).to(device=device, dtype=dtype)
    model.train()
    t_build = time.perf_counter() - t0

    n_total = model.num_parameters()
    n_non_emb = model.num_parameters(non_embedding=True)
    print(f"[build] done in {t_build:.1f}s  "
          f"params total={n_total/1e9:.3f} B  non-emb={n_non_emb/1e9:.3f} B")

    print(f"[opt ] building optimizer kind={optim_kind}…")
    t0 = time.perf_counter()
    if optim_kind == "nvme_adamw":
        from elt_lm.offload.hooks import install_offload_into_training
        run_dir.mkdir(parents=True, exist_ok=True)
        opt, store = install_offload_into_training(model, cfg=cfg, run_dir=run_dir)
    else:
        store = None
        opt = configure_optimizer(model, cfg)
    t_opt = time.perf_counter() - t0
    print(f"[opt ] done in {t_opt:.1f}s")

    after_build_mem = torch.cuda.memory_allocated()
    after_build_peak = torch.cuda.max_memory_allocated()
    print(f"[mem ] post-build   current={_fmt_gb(after_build_mem)}  "
          f"peak={_fmt_gb(after_build_peak)}")

    input_ids = torch.randint(0, cfg.model.vocab_size,
                              (cfg.micro_batch_size, cfg.data.seq_len),
                              device=device, dtype=torch.long)
    labels = input_ids.clone()

    print("[fwd ] L=1 (smoke) forward + backward + step…")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model(input_ids, L=1)
    logits = out.logits[..., :-1, :].contiguous()
    tgt = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)).float(), tgt.view(-1)
    )
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    t_step = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated()
    cur = torch.cuda.memory_allocated()
    print(f"[fwd ] step done in {t_step:.1f}s  loss={loss.item():.3f}  "
          f"peak={_fmt_gb(peak)}  current={_fmt_gb(cur)}")

    if store is not None:
        store.flush()

    return {
        "optim_kind": optim_kind,
        "params_total_b": n_total / 1e9,
        "params_non_emb_b": n_non_emb / 1e9,
        "seq_len": cfg.data.seq_len,
        "micro_batch": cfg.micro_batch_size,
        "step_seconds": t_step,
        "peak_vram_bytes": peak,
        "current_vram_bytes": cur,
        "fits_12gb": peak < 12 * 1024**3,
    }


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--optim", choices=["adamw", "paged_adamw_8bit", "nvme_adamw"],
                   default="paged_adamw_8bit")
    p.add_argument("--run-dir", default="H:/elt_data/runs/1B_smoke")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--config", default="configs/base_1B.yaml")
    args = p.parse_args()

    result = run(args.optim, Path(args.run_dir), args.seq_len, args.micro_batch,
                 config_path=args.config)
    print()
    print("=== SMOKE RESULT ===")
    for k, v in result.items():
        print(f"  {k:<22} {v}")
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    cli()
