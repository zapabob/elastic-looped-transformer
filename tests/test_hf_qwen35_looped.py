from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from elt_lm.bootstrap_qwen35_elt import bootstrap_qwen35_elt_checkpoint
from elt_lm.config import ILSDConfig, ModelConfig
from elt_lm.data import PackedTokenDataset
from elt_lm.hf_qwen35_looped import HFQwen35LoopedLM
from elt_lm.ilsd import ILSDLossFn
from elt_lm.model import build_model


def _make_tiny_qwen_dir(path: Path) -> Qwen3_5ForCausalLM:
    cfg = Qwen3_5TextConfig.from_dict({
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 64,
        "layer_types": ["full_attention", "full_attention"],
        "tie_word_embeddings": False,
    })
    model = Qwen3_5ForCausalLM(cfg).eval()
    model.save_pretrained(path)
    return model


def _make_hf_loop_cfg(model_dir: Path, *, L_max: int = 2, trainable_mode: str = "all") -> ModelConfig:
    return ModelConfig(
        backbone_kind="hf_qwen35_looped",
        hf_model_path=str(model_dir),
        source_model_id=str(model_dir),
        language_only=True,
        freeze_vision=True,
        import_lm_head=True,
        parity_dtype="fp32",
        L_min=1,
        L_max=L_max,
        loop_bootstrap_L_max=1,
        hf_trainable_mode=trainable_mode,  # type: ignore[arg-type]
        hf_trainable_top_layers=1,
        grad_checkpoint=False,
    )


def _make_synthetic_bin(path: Path, n_tokens: int = 4096, period: int = 17) -> None:
    base = np.arange(period, dtype=np.uint32)
    tiles = (n_tokens + period - 1) // period
    arr = np.tile(base, tiles)[:n_tokens]
    path.write_bytes(arr.tobytes())


def test_hf_qwen35_looped_forward_shapes_and_hidden(tmp_path: Path) -> None:
    _make_tiny_qwen_dir(tmp_path)
    cfg = _make_hf_loop_cfg(tmp_path, L_max=2)
    model = build_model(cfg)
    assert isinstance(model, HFQwen35LoopedLM)
    model.load_pretrained_from_source()

    input_ids = torch.randint(0, 128, (2, 12))
    out = model(input_ids, L=2, return_hidden_at=1, return_all_loop_hidden=True)
    assert out.logits.shape == (2, 12, 128)
    assert out.intermediate_logits is not None
    assert out.intermediate_hidden is not None
    assert out.per_loop_hidden is not None
    assert len(out.per_loop_hidden) == 2


@torch.no_grad()
def test_hf_qwen35_looped_l1_matches_source_model(tmp_path: Path) -> None:
    source = _make_tiny_qwen_dir(tmp_path)
    cfg = _make_hf_loop_cfg(tmp_path, L_max=2)
    model = build_model(cfg)
    assert isinstance(model, HFQwen35LoopedLM)
    model.load_pretrained_from_source()

    input_ids = torch.randint(0, 128, (2, 10))
    ref = source(input_ids=input_ids, use_cache=False).logits
    got = model(input_ids, L=1).logits
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), \
        f"max diff = {(ref - got).abs().max().item()}"


@torch.no_grad()
def test_hf_qwen35_looped_intermediate_logits_match_first_pass(tmp_path: Path) -> None:
    source = _make_tiny_qwen_dir(tmp_path)
    cfg = _make_hf_loop_cfg(tmp_path, L_max=2)
    model = build_model(cfg)
    assert isinstance(model, HFQwen35LoopedLM)
    model.load_pretrained_from_source()

    input_ids = torch.randint(0, 128, (2, 10))
    full = model(input_ids, L=2, return_hidden_at=1)
    ref = source(input_ids=input_ids, use_cache=False).logits

    assert full.intermediate_logits is not None
    assert torch.allclose(full.intermediate_logits, ref, atol=1e-5, rtol=1e-5), \
        f"max diff = {(full.intermediate_logits - ref).abs().max().item()}"


def test_bootstrap_qwen35_elt_checkpoint_roundtrip(tmp_path: Path) -> None:
    source = _make_tiny_qwen_dir(tmp_path / "src")
    out_path = tmp_path / "bootstrap.pt"
    ckpt_path, parity = bootstrap_qwen35_elt_checkpoint(
        hf_model_path=str(tmp_path / "src"),
        out_path=out_path,
        tokenizer_path="H:/Qwen3.5-9B-official-hf",
    )
    assert parity is None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(state["cfg"].model)
    model.load_state_dict(state["model"])
    assert isinstance(model, HFQwen35LoopedLM)

    input_ids = torch.randint(0, 128, (1, 8))
    ref = source(input_ids=input_ids, use_cache=False).logits
    got = model(input_ids, L=1).logits
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5)


def test_hf_qwen35_looped_training_smoke_runs(tmp_path: Path) -> None:
    bin_path = tmp_path / "toy.bin"
    src_dir = tmp_path / "src"
    _make_synthetic_bin(bin_path, n_tokens=2048, period=23)
    _make_tiny_qwen_dir(src_dir)

    ds = None
    dl = None
    try:
        model_cfg = _make_hf_loop_cfg(src_dir, L_max=2, trainable_mode="top_layers")
        loss_cfg = ILSDConfig(enabled=True, lambda_anneal_steps=10, warmup_steps=0, strict_student_below_teacher=True)
        model = build_model(model_cfg)
        assert isinstance(model, HFQwen35LoopedLM)
        model.load_pretrained_from_source()
        model.train()

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        loss_fn = ILSDLossFn(model_cfg, loss_cfg, seed=0)

        ds = PackedTokenDataset(bin_path, seq_len=32)
        dl = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)

        losses: list[float] = []
        step = 0
        for input_ids, labels in dl:
            out = loss_fn(model, input_ids, labels, step=step)
            assert torch.isfinite(out.total)
            opt.zero_grad(set_to_none=True)
            out.total.backward()
            opt.step()
            losses.append(out.total.item())
            step += 1
            if step >= 4:
                break

        assert losses, "expected at least one optimization step"
    finally:
        del ds, dl
        gc.collect()
