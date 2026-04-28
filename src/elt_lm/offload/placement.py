"""Greedy tier placement for ELT parameters.

ELT's structure makes an LP solve unnecessary: the model has three well-defined
param groups — embeddings, N unique composite layers, and the final norm. A
fixed greedy assignment fits the hardware budget cleanly:

    T0 GPU   : token embedding, final RMSNorm, LM head (= tied embedding)
    T2 RAM   : N unique composite layers (bf16 master copies)
    T3 NVMe  : optimizer state + fp32 master weights (loaded to RAM on step)

Pinned-RAM staging (T1) and NVMe-read double-buffering live inside the
prefetcher + tiered store — they aren't parameter placements per se.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from elt_lm.offload.hardware_profile import HardwareProfile


class StorageTier(str, Enum):
    GPU = "GPU"
    PINNED = "PINNED"
    RAM = "RAM"
    NVME = "NVME"


@dataclass
class PlacementPlan:
    gpu_params: list[str] = field(default_factory=list)
    ram_params: list[str] = field(default_factory=list)
    nvme_params: list[str] = field(default_factory=list)
    # per-param name → tier (convenience inverse)
    param_tier: dict[str, StorageTier] = field(default_factory=dict)
    # summary bytes per tier (from the assigned parameters only; opt state is tracked separately)
    bytes_gpu: int = 0
    bytes_ram: int = 0
    bytes_nvme: int = 0

    def tier_of(self, name: str) -> StorageTier:
        return self.param_tier[name]


def _param_bytes_bf16(p: nn.Parameter) -> int:
    return p.numel() * 2


def _param_bytes_fp32(p: nn.Parameter) -> int:
    return p.numel() * 4


def plan_placement(model: nn.Module, hw: HardwareProfile,
                   composite_attr: str = "composite",
                   tok_embed_attr: str = "tok_embed",
                   final_norm_attr: str = "final_norm",
                   lm_head_attr: str = "lm_head") -> PlacementPlan:
    """Produce a placement plan for an ELT-style LM.

    Assigns embedding / final norm / (optional) LM head to GPU, every parameter
    inside `composite.layers[*]` to RAM, and every param's fp32 optimizer
    shadow to NVMe.
    """
    plan = PlacementPlan()

    # GPU-resident parameters ------------------------------------------------
    gpu_modules = []
    for attr in (tok_embed_attr, final_norm_attr, lm_head_attr):
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        gpu_modules.append((attr, mod))

    for prefix, mod in gpu_modules:
        for name, p in mod.named_parameters(recurse=True):
            full_name = f"{prefix}.{name}" if name else prefix
            plan.gpu_params.append(full_name)
            plan.param_tier[full_name] = StorageTier.GPU
            plan.bytes_gpu += _param_bytes_bf16(p)

    # RAM-resident trainable body -------------------------------------------
    composite = getattr(model, composite_attr, None)
    if composite is not None:
        for name, p in composite.named_parameters(recurse=True):
            full_name = f"{composite_attr}.{name}"
            plan.ram_params.append(full_name)
            plan.param_tier[full_name] = StorageTier.RAM
            plan.bytes_ram += _param_bytes_bf16(p)
    else:
        # HF-backed side branches do not expose the native `composite` module.
        # For those models, keep the model weights resident as usual but place
        # optimizer state for trainable params in the NVMe-backed path.
        for full_name, p in model.named_parameters(recurse=True):
            if not p.requires_grad:
                continue
            if full_name in plan.param_tier:
                continue
            plan.ram_params.append(full_name)
            plan.param_tier[full_name] = StorageTier.RAM
            plan.bytes_ram += _param_bytes_bf16(p)

    # NVMe: fp32 master weights + AdamW m/v ---------------------------------
    # Only non-embedding params have NVMe shadows — tok_embed/head stay fp32-on-GPU.
    for full_name, tier in plan.param_tier.items():
        if tier is StorageTier.RAM:
            # fp32 master (4x) + m (4x) + v (4x) = 12x numel bytes per RAM param
            # numel is "bytes_bf16 / 2".
            plan.nvme_params.append(full_name)
            plan.bytes_nvme += 0  # populated by TieredParameterStore on allocation

    return plan


def assert_fits(plan: PlacementPlan, hw: HardwareProfile, *,
                vram_headroom_bytes: int = 3 * 1024**3,
                ram_headroom_bytes: int = 4 * 1024**3) -> None:
    """Raise RuntimeError if the plan obviously overflows the available hardware.

    Headrooms account for activations (GPU) and OS + tokenizer (RAM).
    """
    if plan.bytes_gpu + vram_headroom_bytes > hw.gpu_vram_bytes:
        raise RuntimeError(
            f"placement needs {plan.bytes_gpu/1e9:.2f} GB on GPU + "
            f"{vram_headroom_bytes/1e9:.2f} GB headroom, "
            f"but GPU has {hw.gpu_vram_bytes/1e9:.2f} GB")
    if plan.bytes_ram + ram_headroom_bytes > hw.ram_bytes:
        raise RuntimeError(
            f"placement needs {plan.bytes_ram/1e9:.2f} GB RAM + "
            f"{ram_headroom_bytes/1e9:.2f} GB headroom, "
            f"but machine has {hw.ram_bytes/1e9:.2f} GB")
