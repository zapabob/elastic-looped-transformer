"""End-to-end smoke test: build a tiny synthetic bin, run a few training steps,
verify loss goes down and all ILSD components are finite."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from elt_lm.config import ILSDConfig, ModelConfig
from elt_lm.data import PackedTokenDataset
from elt_lm.ilsd import ILSDLossFn
from elt_lm.model import ELTLanguageModel


def _make_synthetic_bin(path: Path, n_tokens: int = 8192, period: int = 17) -> None:
    """Generate a simple periodic token stream (learnable by a small model).

    The sequence cycles through [0, 1, 2, ..., P-1] repeatedly. Cross-entropy at
    random init ≈ log(vocab); after a few steps of memorization it should drop
    well below that.
    """
    base = np.arange(period, dtype=np.uint32)
    tiles = (n_tokens + period - 1) // period
    arr = np.tile(base, tiles)[:n_tokens]
    path.write_bytes(arr.tobytes())


def test_smoke_training_reduces_loss(tmp_path: Path) -> None:
    bin_path = tmp_path / "toy.bin"
    vocab = 256
    _make_synthetic_bin(bin_path, n_tokens=4096)

    try:
        mcfg = ModelConfig(
            vocab_size=vocab,
            d_model=32,
            n_unique_layers=2,
            n_heads=4,
            head_dim=8,
            d_ff=64,
            max_seq_len=64,
            L_min=1,
            L_max=2,
            tie_word_embeddings=True,
            grad_checkpoint=False,
        )
        icfg = ILSDConfig(
            enabled=True,
            lambda_anneal_steps=30,
            warmup_steps=0,
            strict_student_below_teacher=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ELTLanguageModel(mcfg).to(device=device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = ILSDLossFn(mcfg, icfg, seed=0)

        ds = PackedTokenDataset(bin_path, seq_len=32)
        dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)

        losses = []
        step = 0
        for _ in range(8):
            for input_ids, labels in dl:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                out = loss_fn(model, input_ids, labels, step=step)
                assert torch.isfinite(out.total)
                assert torch.isfinite(out.l_gt_teacher)
                assert torch.isfinite(out.l_gt_student)
                assert torch.isfinite(out.l_dist)
                assert torch.isfinite(out.l_entropy)
                assert torch.isfinite(out.l_local)
                opt.zero_grad(set_to_none=True)
                out.total.backward()
                opt.step()
                losses.append(out.total.item())
                step += 1
                if step >= 60:
                    break
            if step >= 60:
                break

        start = sum(losses[:5]) / 5
        end = sum(losses[-5:]) / 5
        assert end < start - 1.0, \
            f"loss did not meaningfully decrease: start={start:.3f} end={end:.3f}"
    finally:
        # Release the numpy.memmap so Windows can unlink the temp file.
        del ds, dl
        gc.collect()


# Silence unused-import warnings for tools used only via tmp_path fixture.
_ = pytest
