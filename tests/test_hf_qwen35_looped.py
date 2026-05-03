from __future__ import annotations

import gc
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from elt_lm.bootstrap_qwen35_elt import bootstrap_qwen35_elt_checkpoint
from elt_lm.config import ILSDConfig, ModelConfig, TrainConfig
from elt_lm.data import PackedTokenDataset
from elt_lm.export_lora_adapter import export_lora_adapter
from elt_lm.export_merged_qwen35_hf import export_merged_qwen35_hf
from elt_lm.hf_qwen35_looped import HFQwen35LoopedLM, LoRALinear, patch_fla_windows_cpu_fallback
from elt_lm.ilsd import ILSDLossFn
from elt_lm.model import build_model
from elt_lm.train import _build_ckpt_state
from elt_lm.train_grpo import load_policy_checkpoint


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


def _make_hf_loop_cfg(
    model_dir: Path,
    *,
    L_max: int = 2,
    trainable_mode: str = "all",
    lora_rank: int = 0,
) -> ModelConfig:
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
        hf_lora_rank=lora_rank,
        hf_lora_alpha=max(1, lora_rank * 2),
        hf_lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        grad_checkpoint=False,
    )


def _make_synthetic_bin(path: Path, n_tokens: int = 4096, period: int = 17) -> None:
    base = np.arange(period, dtype=np.uint32)
    tiles = (n_tokens + period - 1) // period
    arr = np.tile(base, tiles)[:n_tokens]
    path.write_bytes(arr.tobytes())


def test_patch_fla_windows_cpu_fallback(monkeypatch) -> None:
    fake_fla_utils = types.SimpleNamespace(device_torch_lib=torch.cpu)
    monkeypatch.setitem(__import__("sys").modules, "fla.utils", fake_fla_utils)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    patch_fla_windows_cpu_fallback()

    assert fake_fla_utils.device == "cuda"
    assert fake_fla_utils.device_name == "cuda"
    assert fake_fla_utils.device_torch_lib is torch.cuda
    ctx = fake_fla_utils.custom_device_ctx(None)
    assert hasattr(ctx, "__enter__")


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


@torch.no_grad()
def test_hf_qwen35_looped_lora_keeps_l1_parity_at_init(tmp_path: Path) -> None:
    source = _make_tiny_qwen_dir(tmp_path)
    cfg = _make_hf_loop_cfg(tmp_path, L_max=1, trainable_mode="lora", lora_rank=4)
    model = build_model(cfg)
    assert isinstance(model, HFQwen35LoopedLM)
    model.load_pretrained_from_source()

    lora_modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert lora_modules
    assert all(not p.requires_grad for m in lora_modules for p in m.base.parameters())
    assert all(
        (".lora_" in name) == p.requires_grad
        for name, p in model.named_parameters()
    )

    input_ids = torch.randint(0, 128, (2, 10))
    ref = source(input_ids=input_ids, use_cache=False).logits
    got = model(input_ids, L=1).logits
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), \
        f"max diff = {(ref - got).abs().max().item()}"


def test_hf_qwen35_looped_lora_loads_non_lora_bootstrap_state(tmp_path: Path) -> None:
    _make_tiny_qwen_dir(tmp_path / "src")
    ckpt_path, _ = bootstrap_qwen35_elt_checkpoint(
        hf_model_path=str(tmp_path / "src"),
        out_path=tmp_path / "bootstrap.pt",
        tokenizer_path="H:/Qwen3.5-9B-official-hf",
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    lora_cfg = _make_hf_loop_cfg(tmp_path / "src", L_max=1, trainable_mode="lora", lora_rank=4)
    model = build_model(lora_cfg)
    assert isinstance(model, HFQwen35LoopedLM)
    model.load_state_dict(state["model"])

    assert any(isinstance(m, LoRALinear) for m in model.modules())


def test_hf_qwen35_looped_lora_adapter_only_checkpoint_roundtrip(tmp_path: Path) -> None:
    _make_tiny_qwen_dir(tmp_path / "src")
    base_ckpt, _ = bootstrap_qwen35_elt_checkpoint(
        hf_model_path=str(tmp_path / "src"),
        out_path=tmp_path / "base.pt",
        tokenizer_path="H:/Qwen3.5-9B-official-hf",
    )
    cfg = _make_hf_loop_cfg(tmp_path / "src", L_max=1, trainable_mode="lora", lora_rank=4)
    cfg.hf_save_adapter_only = True
    cfg.hf_adapter_base_ckpt = str(base_ckpt)

    model = build_model(cfg)
    assert isinstance(model, HFQwen35LoopedLM)
    model.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=False)["model"])
    first_lora = next(p for name, p in model.named_parameters() if ".lora_B" in name)
    first_lora.data.fill_(0.125)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state = _build_ckpt_state(model, opt, TrainConfig(model=cfg), step=7)
    ckpt = tmp_path / "adapter_only.pt"
    torch.save(state, ckpt)

    restored = build_model(cfg)
    assert isinstance(restored, HFQwen35LoopedLM)
    restored.load_adapter_checkpoint_state(torch.load(ckpt, map_location="cpu", weights_only=False))

    restored_lora = next(p for name, p in restored.named_parameters() if ".lora_B" in name)
    assert torch.allclose(restored_lora, torch.full_like(restored_lora, 0.125))
    assert ckpt.stat().st_size < base_ckpt.stat().st_size


def test_grpo_policy_loader_accepts_side_lora_adapter_checkpoint(tmp_path: Path) -> None:
    _make_tiny_qwen_dir(tmp_path / "src")
    base_ckpt, _ = bootstrap_qwen35_elt_checkpoint(
        hf_model_path=str(tmp_path / "src"),
        out_path=tmp_path / "base.pt",
        tokenizer_path="H:/Qwen3.5-9B-official-hf",
    )
    cfg = _make_hf_loop_cfg(tmp_path / "src", L_max=1, trainable_mode="lora", lora_rank=4)
    cfg.hf_save_adapter_only = True
    cfg.hf_adapter_base_ckpt = str(base_ckpt)

    model = build_model(cfg)
    assert isinstance(model, HFQwen35LoopedLM)
    model.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=False)["model"])
    first_lora = next(p for name, p in model.named_parameters() if ".lora_B" in name)
    first_lora.data.fill_(0.25)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ckpt = tmp_path / "adapter_policy.pt"
    torch.save(_build_ckpt_state(model, opt, TrainConfig(model=cfg), step=11), ckpt)

    restored = build_model(cfg)
    assert isinstance(restored, HFQwen35LoopedLM)
    step = load_policy_checkpoint(ckpt, restored)

    restored_lora = next(p for name, p in restored.named_parameters() if ".lora_B" in name)
    assert step == 11
    assert torch.allclose(restored_lora, torch.full_like(restored_lora, 0.25))


def test_export_lora_adapter_writes_small_artifact(tmp_path: Path) -> None:
    _make_tiny_qwen_dir(tmp_path / "src")
    base_ckpt, _ = bootstrap_qwen35_elt_checkpoint(
        hf_model_path=str(tmp_path / "src"),
        out_path=tmp_path / "base.pt",
        tokenizer_path="H:/Qwen3.5-9B-official-hf",
    )
    cfg = _make_hf_loop_cfg(tmp_path / "src", L_max=1, trainable_mode="lora", lora_rank=4)
    cfg.hf_save_adapter_only = True
    cfg.hf_adapter_base_ckpt = str(base_ckpt)
    model = build_model(cfg)
    assert isinstance(model, HFQwen35LoopedLM)
    model.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=False)["model"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ckpt = tmp_path / "adapter_only.pt"
    torch.save(_build_ckpt_state(model, opt, TrainConfig(model=cfg), step=1), ckpt)

    adapter_path = export_lora_adapter(ckpt, tmp_path / "adapter")

    assert adapter_path.exists()
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "adapter" / "README.md").exists()


@torch.no_grad()
def test_export_merged_qwen35_hf_merges_lora_delta(tmp_path: Path) -> None:
    _make_tiny_qwen_dir(tmp_path / "src")
    base_ckpt, _ = bootstrap_qwen35_elt_checkpoint(
        hf_model_path=str(tmp_path / "src"),
        out_path=tmp_path / "base.pt",
        tokenizer_path="H:/Qwen3.5-9B-official-hf",
    )
    cfg = _make_hf_loop_cfg(tmp_path / "src", L_max=1, trainable_mode="lora", lora_rank=4)
    cfg.hf_save_adapter_only = True
    cfg.hf_adapter_base_ckpt = str(base_ckpt)

    looped = build_model(cfg)
    assert isinstance(looped, HFQwen35LoopedLM)
    looped.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=False)["model"])
    for name, param in looped.named_parameters():
        if ".lora_B" in name:
            param.fill_(0.125)
    opt = torch.optim.AdamW([p for p in looped.parameters() if p.requires_grad], lr=1e-3)
    adapter_ckpt = tmp_path / "adapter_only.pt"
    torch.save(_build_ckpt_state(looped, opt, TrainConfig(model=cfg), step=3), adapter_ckpt)

    manifest = export_merged_qwen35_hf(
        ckpt_path=adapter_ckpt,
        out_dir=tmp_path / "hf_merged",
        tokenizer_path=tmp_path / "src",
        repo_id="zapabob/test-merged",
        dtype_name="fp32",
        require_tokenizer=False,
    )

    exported = Qwen3_5ForCausalLM.from_pretrained(tmp_path / "hf_merged", torch_dtype=torch.float32).eval()
    input_ids = torch.randint(0, 128, (2, 10))
    ref = looped(input_ids, L=1).logits
    got = exported(input_ids=input_ids, use_cache=False).logits
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5)
    assert manifest["merged_lora_modules"] > 0
    assert manifest["tokenizer_ready"] is False
    assert (tmp_path / "hf_merged" / "elt_export_manifest.json").exists()


def test_export_merged_qwen35_hf_writes_elt_config_metadata(tmp_path: Path) -> None:
    _make_tiny_qwen_dir(tmp_path / "src")
    base_ckpt, _ = bootstrap_qwen35_elt_checkpoint(
        hf_model_path=str(tmp_path / "src"),
        out_path=tmp_path / "base.pt",
        tokenizer_path="H:/Qwen3.5-9B-official-hf",
    )
    cfg = _make_hf_loop_cfg(tmp_path / "src", L_max=3, trainable_mode="lora", lora_rank=4)
    cfg.hf_save_adapter_only = True
    cfg.hf_adapter_base_ckpt = str(base_ckpt)
    model = build_model(cfg)
    assert isinstance(model, HFQwen35LoopedLM)
    model.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=False)["model"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    adapter_ckpt = tmp_path / "adapter_only.pt"
    torch.save(_build_ckpt_state(model, opt, TrainConfig(model=cfg), step=1), adapter_ckpt)

    export_merged_qwen35_hf(
        ckpt_path=adapter_ckpt,
        out_dir=tmp_path / "hf_merged",
        tokenizer_path=tmp_path / "src",
        dtype_name="fp32",
        require_tokenizer=False,
    )

    config = (tmp_path / "hf_merged" / "config.json").read_text(encoding="utf-8")
    assert '"elt_config"' in config
    assert '"L_max": 3' in config
    tensors = load_file(tmp_path / "hf_merged" / "model.safetensors")
    assert "model.embed_tokens.weight" in tensors
    assert not any(".lora_" in key for key in tensors)


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


def test_bootstrap_qwen35_elt_checkpoint_respects_save_dtype(tmp_path: Path) -> None:
    _make_tiny_qwen_dir(tmp_path / "src")
    template = tmp_path / "template.yaml"
    template.write_text("model:\n  parity_dtype: bf16\n", encoding="utf-8")

    ckpt_path, _ = bootstrap_qwen35_elt_checkpoint(
        hf_model_path=str(tmp_path / "src"),
        out_path=tmp_path / "bootstrap_bf16.pt",
        tokenizer_path="H:/Qwen3.5-9B-official-hf",
        template_config=str(template),
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    first_tensor = next(iter(state["model"].values()))
    assert first_tensor.dtype == torch.bfloat16
    assert state["cfg"].model.parity_dtype == "bf16"


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
