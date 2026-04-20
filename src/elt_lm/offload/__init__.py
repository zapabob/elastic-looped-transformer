r"""Hypura-style 4-tier parameter offload for ELT on Windows + RTX 3060.

Tier hierarchy (best → worst):

    T0  GPU                 12 GB VRAM
    T1  pinned host RAM      staging buffers for async DMA
    T2  paged host RAM       bf16 layer master copies
    T3  NVMe (H:\)           fp32 optimizer state + fp32 master weights

The ELT composite block is iterated L times per forward pass, so the cost of
promoting a layer from T2/T3 → T0 is amortized across L × 2 (forward + backward)
compute passes — making offload materially cheaper here than for standard LLMs.

Public API:

    from elt_lm.offload import (
        HardwareProfile, probe_hardware,
        PlacementPlan, plan_placement,
        NvmePrefetcher,
        TieredParameterStore,
        NvmeAdamW,
    )
"""

from __future__ import annotations

from elt_lm.offload.hardware_profile import HardwareProfile, probe_hardware
from elt_lm.offload.placement import PlacementPlan, StorageTier, plan_placement
from elt_lm.offload.prefetcher import NvmePrefetcher

# Heavier imports (optim, tiered store) guarded behind explicit re-export
# to keep the plain `import elt_lm.offload` cheap.
from elt_lm.offload.optim_offload import NvmeAdamW
from elt_lm.offload.tiered_store import TieredParameterStore

__all__ = [
    "HardwareProfile",
    "probe_hardware",
    "PlacementPlan",
    "StorageTier",
    "plan_placement",
    "NvmePrefetcher",
    "TieredParameterStore",
    "NvmeAdamW",
]
