"""HuggingFace wrapper round-trip: ELTConfig, ELTForCausalLM save/load, bitwise parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from elt_lm.config import ModelConfig
from elt_lm.hf.configuration_elt import ELTConfig
from elt_lm.hf.modeling_elt import ELTForCausalLM


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128, d_model=32, n_unique_layers=2, n_heads=4, head_dim=8,
        d_ff=64, max_seq_len=32, grad_checkpoint=False, L_min=1, L_max=2,
    )


def test_elt_config_roundtrip_to_model_config() -> None:
    mc = _tiny_model_config()
    hf = ELTConfig.from_model_config(mc)

    for field in ("vocab_size", "d_model", "n_unique_layers", "n_heads",
                  "n_kv_heads", "head_dim", "d_ff", "max_seq_len",
                  "rope_theta", "rms_norm_eps", "tie_word_embeddings",
                  "dropout", "L_min", "L_max", "init_std"):
        assert getattr(hf, field) == getattr(mc, field), f"{field} mismatch"

    mc2 = hf.to_model_config()
    for field in ("vocab_size", "d_model", "n_unique_layers", "d_ff", "L_min", "L_max"):
        assert getattr(mc2, field) == getattr(mc, field)


def test_elt_for_causal_lm_forward_with_labels_yields_loss_and_grads() -> None:
    cfg = ELTConfig.from_model_config(_tiny_model_config())
    model = ELTForCausalLM(cfg).train()

    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids=ids, labels=ids, L=2)

    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.logits.shape == (2, 8, cfg.vocab_size)

    out.loss.backward()
    g = model.elt.tok_embed.weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_elt_for_causal_lm_save_pretrained_roundtrip(tmp_path: Path) -> None:
    cfg = ELTConfig.from_model_config(_tiny_model_config())
    model = ELTForCausalLM(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        ref_logits = model(input_ids=ids, L=2).logits

    # Save via HF machinery (safetensors).
    model.save_pretrained(tmp_path)

    # Must exist in any HF model directory.
    assert (tmp_path / "config.json").exists()
    safet = list(tmp_path.glob("*.safetensors"))
    assert safet, "safetensors shard missing"

    # Load back and confirm bitwise parity.
    loaded_cfg = ELTConfig.from_pretrained(tmp_path)
    loaded = ELTForCausalLM.from_pretrained(tmp_path, config=loaded_cfg).eval()

    with torch.no_grad():
        new_logits = loaded(input_ids=ids, L=2).logits

    # FP32 machine-epsilon drift across independent module builds is expected;
    # anything above ~1e-6 means the tied-head / RoPE buffer reload is broken.
    assert torch.allclose(ref_logits, new_logits, atol=1e-6, rtol=1e-6), \
        "logits diverged after save/load — tied-head reload likely broken"


def test_default_L_falls_back_to_config() -> None:
    mc = _tiny_model_config()
    cfg = ELTConfig.from_model_config(mc, L_default=2)
    model = ELTForCausalLM(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out_auto = model(input_ids=ids).logits
        out_explicit = model(input_ids=ids, L=2).logits

    assert torch.allclose(out_auto, out_explicit, atol=0, rtol=0)


def test_elt_config_save_load_yaml_fields(tmp_path: Path) -> None:
    mc = _tiny_model_config()
    cfg = ELTConfig.from_model_config(mc)
    cfg.save_pretrained(tmp_path)

    loaded = ELTConfig.from_pretrained(tmp_path)
    for f in ("vocab_size", "d_model", "n_unique_layers", "L_min", "L_max", "L_default"):
        assert getattr(loaded, f) == getattr(cfg, f), f"{f} changed across save/load"


@pytest.mark.parametrize("L", [1, 2])
def test_generate_accepts_L(L: int) -> None:
    cfg = ELTConfig.from_model_config(_tiny_model_config())
    model = ELTForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    # Greedy + short generation to keep this fast; we only check it doesn't blow up.
    out = cast(Any, model).generate(ids, max_new_tokens=3, do_sample=False, L=L)
    assert out.shape[1] >= ids.shape[1]
