"""Tests for offload.hooks — LayerTimingInstrumentor + install_offload_into_training."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from elt_lm.config import ModelConfig, TrainConfig
from elt_lm.model import ELTLanguageModel
from elt_lm.offload.hooks import (
    LayerTimingInstrumentor,
    install_offload_into_training,
)
from elt_lm.offload.placement import StorageTier
from elt_lm.telemetry import TelemetryWriter


def _tiny_model() -> ELTLanguageModel:
    return ELTLanguageModel(ModelConfig(
        vocab_size=128, d_model=32, n_unique_layers=3,
        n_heads=2, head_dim=16, d_ff=64, max_seq_len=16,
        tie_word_embeddings=True, grad_checkpoint=False,
    ))


def test_layer_timing_emits_one_event_per_layer_per_loop(tmp_path: Path):
    model = _tiny_model().eval()
    metrics = tmp_path / "metrics.jsonl"
    tw = TelemetryWriter(metrics)

    ids = torch.zeros((1, 8), dtype=torch.long)
    with LayerTimingInstrumentor(model, tw):
        with torch.no_grad():
            model(ids, L=2)

    tw.close()
    events = [json.loads(l) for l in metrics.read_text(encoding="utf-8").splitlines() if l.strip()]
    layer_events = [e for e in events if e["event"] == "layer_computed"]
    # 3 unique layers × L=2 loops = 6 forward calls
    assert len(layer_events) == 3 * 2
    # layer_idx should cover 0..N-1, repeated each loop
    assert {int(e["layer_idx"]) for e in layer_events} == {0, 1, 2}
    # every event has a positive duration
    for e in layer_events:
        assert float(e["duration_us"]) > 0


def test_layer_timing_hooks_are_removed_on_exit(tmp_path: Path):
    model = _tiny_model().eval()
    tw = TelemetryWriter(tmp_path / "m.jsonl")
    with LayerTimingInstrumentor(model, tw):
        pass
    tw.close()

    # Exiting the context should have removed every hook.
    composite = model.composite
    for layer in composite.layers:
        assert not layer._forward_hooks
        assert not layer._forward_pre_hooks


def test_install_offload_into_training_returns_nvme_adamw(tmp_path: Path):
    from elt_lm.config import OffloadConfig
    model = _tiny_model()
    cfg = TrainConfig(model=ModelConfig(
        vocab_size=128, d_model=32, n_unique_layers=3, n_heads=2,
        head_dim=16, d_ff=64, max_seq_len=16, tie_word_embeddings=True,
    ))
    # Disable the 20 GB free-space guard — test tmp dirs may be on a tight drive.
    cfg.offload = OffloadConfig(min_free_gb=0.0)
    opt, store = install_offload_into_training(model, cfg=cfg, run_dir=tmp_path)
    from elt_lm.offload.optim_offload import NvmeAdamW
    from elt_lm.offload.tiered_store import TieredParameterStore
    assert isinstance(opt, NvmeAdamW)
    assert isinstance(store, TieredParameterStore)
    # Expected NVMe shards exist on disk
    shards = list((tmp_path / "offload_nvme").glob("*.f32"))
    assert shards


def test_install_offload_end_to_end_step(tmp_path: Path):
    """Backprop through the model then run an NvmeAdamW.step() — no crashes."""
    from elt_lm.config import OffloadConfig
    torch.manual_seed(0)
    model = _tiny_model().train()
    cfg = TrainConfig(model=ModelConfig(
        vocab_size=128, d_model=32, n_unique_layers=3, n_heads=2,
        head_dim=16, d_ff=64, max_seq_len=16, tie_word_embeddings=True,
        grad_checkpoint=False,
    ))
    cfg.offload = OffloadConfig(min_free_gb=0.0)
    opt, store = install_offload_into_training(model, cfg=cfg, run_dir=tmp_path)

    ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 128, (2, 8), dtype=torch.long)
    out = model(ids, L=2)
    logits = out.logits[..., :-1, :].contiguous()
    tgt = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), tgt.view(-1)
    )
    loss.backward()
    opt.step()
    store.flush()

    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_install_offload_supports_non_composite_trainable_params(tmp_path: Path):
    """HF-backed side models do not expose native `composite`, but still need
    NVMe optimizer state for trainable parameters such as lm_head/top layers.
    """
    from elt_lm.config import OffloadConfig

    class TinySideModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frozen = torch.nn.Linear(4, 4)
            self.trainable = torch.nn.Linear(4, 2)
            for p in self.frozen.parameters():
                p.requires_grad_(False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.trainable(self.frozen(x))

    model = TinySideModel()
    cfg = TrainConfig()
    cfg.offload = OffloadConfig(min_free_gb=0.0)
    opt, store = install_offload_into_training(model, cfg=cfg, run_dir=tmp_path)

    assert store.plan.param_tier["trainable.weight"] is StorageTier.RAM
    assert store.plan.param_tier["trainable.bias"] is StorageTier.RAM
    assert "frozen.weight" not in store.plan.param_tier

    x = torch.randn(3, 4)
    loss = model(x).pow(2).mean()
    loss.backward()
    opt.step()
    store.flush()

    assert list((tmp_path / "offload_nvme").glob("trainable_*__m.f32"))
