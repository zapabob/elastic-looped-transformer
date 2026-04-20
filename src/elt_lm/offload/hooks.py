"""Model-side hooks that bridge the offload package with the training loop.

Two pieces:

  - `LayerTimingInstrumentor`: PyTorch forward pre/post hooks that time each
    composite-layer forward and emit a `layer_computed` telemetry event per
    call. The Storage-tiers dashboard panel reads these events.

  - `install_offload_into_training`: wire a `TieredParameterStore` +
    `NvmeAdamW` into the training loop. Called from `train.train()` when
    `cfg.optim.kind == "nvme_adamw"`. Returns the optimizer and the store
    so the caller can flush/close them at shutdown.

Design note: because `NvmeAdamW` keeps live params on GPU and only offloads
fp32 optimizer state, we do NOT need to swap a layer's `.data` on every
forward. Instrumentation here is observability-only. The real memory savings
come from `NvmeAdamW._step_tiered` running the update on CPU with NVMe state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from elt_lm.offload.optim_offload import NvmeAdamW, build_name_lookup
from elt_lm.offload.placement import StorageTier, plan_placement
from elt_lm.offload.tiered_store import TieredParameterStore


class _TelemetryLike(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...


class LayerTimingInstrumentor:
    """Attach forward-pre/post hooks that time each composite layer.

    Usage:
        with LayerTimingInstrumentor(model, telemetry) as _:
            out = model(input_ids, L=4)
    """

    def __init__(self, model: nn.Module, telemetry: _TelemetryLike,
                 composite_attr: str = "composite",
                 store: TieredParameterStore | None = None):
        self.model = model
        self.telemetry = telemetry
        self.composite_attr = composite_attr
        self.store = store
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._t0: dict[int, float] = {}

    def _pre(self, idx: int):
        def _f(_module, _inputs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._t0[idx] = time.perf_counter()
        return _f

    def _post(self, idx: int):
        def _f(_module, _inputs, _output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = self._t0.pop(idx, None)
            if t0 is None:
                return
            dur_us = (time.perf_counter() - t0) * 1e6
            tier = "GPU"
            if self.store is not None:
                # If a store is wired in, attribute the layer's *weight* tier —
                # live param is on GPU at compute time, but its master lives
                # elsewhere. Report the master tier for the dashboard.
                name = f"{self.composite_attr}.layers.{idx}"
                # The composite has submodule params; use the first one to
                # determine tier (all of a layer's params share a tier).
                plan = self.store.plan
                weight_tier = next(
                    (plan.param_tier[n] for n in plan.param_tier
                     if n.startswith(name)),
                    StorageTier.GPU,
                )
                tier = weight_tier.value
            self.telemetry.emit(
                "layer_computed",
                layer_idx=idx,
                tier=tier,
                duration_us=dur_us,
            )
        return _f

    def __enter__(self) -> "LayerTimingInstrumentor":
        composite = getattr(self.model, self.composite_attr)
        for idx, layer in enumerate(composite.layers):
            self._handles.append(layer.register_forward_pre_hook(self._pre(idx)))
            self._handles.append(layer.register_forward_hook(self._post(idx)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def install_offload_into_training(
    model: nn.Module,
    *,
    cfg,
    run_dir: Path,
) -> tuple[NvmeAdamW, TieredParameterStore]:
    """Build the `TieredParameterStore` + `NvmeAdamW` needed for kind=nvme_adamw.

    Returns (optimizer, store). The caller is responsible for calling
    `store.flush()` periodically and on shutdown.
    """
    from elt_lm.offload.hardware_profile import probe_hardware

    nvme_root = Path(run_dir) / "offload_nvme"
    hw = probe_hardware(nvme_path=nvme_root)
    plan = plan_placement(model, hw)

    store = TieredParameterStore(model, plan, nvme_root=nvme_root)
    name_lookup = build_name_lookup(model)

    # Mirror the weight-decay-on-2D-params-only convention from train.configure_optimizer.
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    opt = NvmeAdamW(
        groups, store=store, name_lookup=name_lookup,
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
    )
    return opt, store
