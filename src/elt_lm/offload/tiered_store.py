"""4-tier parameter store: GPU / pinned / RAM / NVMe.

Owns the master copies of each parameter across tiers and exposes `promote` /
`demote` to move a composite-layer weight shard onto the GPU for a
forward/backward pass and evict it afterwards.

Design (per the plan, §2.2):
    - tok_embed / final_norm / lm_head   → GPU (always resident, never promoted)
    - composite.layers[i].*              → bf16 master in RAM, promoted to GPU
                                           on demand using a double-buffered
                                           pinned staging tensor
    - fp32 optimizer state (m, v, master) → NVMe, memory-mapped (read/written
                                           only during optimizer.step)

The store is a separate object from the model so existing training loops can
opt into offload without changing the model's `state_dict` shape — the
in-model `nn.Parameter` values are still the source of truth at step time;
offload acts as a *cache*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import shutil

import numpy as np
import torch
from torch import Tensor, nn

from elt_lm.offload.placement import PlacementPlan, StorageTier
from elt_lm.config import OffloadConfig


@dataclass
class _NvmeShard:
    """Memory-mapped fp32 shard on NVMe (master weight OR optimizer m OR v)."""
    path: Path
    shape: tuple[int, ...]
    numel: int
    mmap: np.memmap = field(repr=False)

    @classmethod
    def create(cls, path: Path, shape: tuple[int, ...], init: str = "zeros") -> "_NvmeShard":
        path.parent.mkdir(parents=True, exist_ok=True)
        numel = int(np.prod(shape)) if shape else 0
        # Pre-allocate the file; np.memmap creates it when mode='w+'.
        arr = np.memmap(path, dtype=np.float32, mode="w+", shape=shape if shape else (1,))
        if init == "zeros":
            arr[...] = 0.0
        elif init == "from_param":
            pass   # caller fills from a tensor immediately after.
        else:
            raise ValueError(f"unknown init={init!r}")
        arr.flush()
        return cls(path=path, shape=shape, numel=numel, mmap=arr)

    @classmethod
    def open_existing(cls, path: Path, shape: tuple[int, ...]) -> "_NvmeShard":
        arr = np.memmap(path, dtype=np.float32, mode="r+", shape=shape if shape else (1,))
        return cls(path=path, shape=shape, numel=int(np.prod(shape)), mmap=arr)

    def read_tensor(self) -> Tensor:
        """Return a fp32 CPU tensor that views the mmap (zero-copy)."""
        return torch.from_numpy(np.asarray(self.mmap)).view(self.shape)

    def write_tensor(self, t: Tensor) -> None:
        """Copy a fp32 CPU tensor into the mmap + flush."""
        assert t.dtype is torch.float32, f"expected fp32, got {t.dtype}"
        assert tuple(t.shape) == self.shape, f"shape {tuple(t.shape)} != {self.shape}"
        self.mmap[...] = t.detach().cpu().numpy().reshape(self.shape)
        self.mmap.flush()


class TieredParameterStore:
    """Owns multi-tier master copies of an ELT model's parameters.

    Typical wiring inside a training loop:

        hw = probe_hardware()
        plan = plan_placement(model, hw)
        store = TieredParameterStore(model, plan, nvme_root=hw.nvme_path)

        # In CompositeBlock.forward: ...for i in range(N): store.promote(i); run layer; store.demote(i)
        # In optimizer.step: ...for p in params: state = store.open_adam_state(p_name); update; store.flush()

    This Phase-C skeleton exposes the data structures + accessors; the model
    hook integration lives in `offload/hooks.py`.
    """

    def __init__(self, model: nn.Module, plan: PlacementPlan, nvme_root: Path | str | None = None,
                 composite_attr: str = "composite", offload_config: OffloadConfig | None = None):
        self.model = model
        self.plan = plan
        # Determine the NVMe root and minimum free space.
        # When no OffloadConfig is supplied (tests, ad-hoc scripts) we skip the
        # safety margin — callers that want the guard pass an OffloadConfig
        # explicitly, which is the production path via install_offload_into_training.
        if offload_config is not None and offload_config.root is not None:
            self.nvme_root = Path(offload_config.root)
            self.min_free_gb = offload_config.min_free_gb
        else:
            if nvme_root is None:
                raise ValueError("Either nvme_root or offload_config.root must be provided")
            self.nvme_root = Path(nvme_root)
            self.min_free_gb = offload_config.min_free_gb if offload_config is not None else 0.0

        # Ensure the directory exists
        self.nvme_root.mkdir(parents=True, exist_ok=True)

        # RAM masters: bf16 view of each composite-layer param. We don't copy
        # the data — we hold a reference to the existing Parameter's tensor,
        # which already lives in host memory before `.to('cuda')` is called.
        self._ram_master_bf16: Dict[str, Tensor] = {}
        # NVMe shards: fp32 master weight + AdamW m + v per RAM-tier param.
        self._nvme_master_fp32: Dict[str, _NvmeShard] = {}
        self._nvme_adam_m: Dict[str, _NvmeShard] = {}
        self._nvme_adam_v: Dict[str, _NvmeShard] = {}

        self.composite_attr = composite_attr

        # Compute required NVMe bytes for optimizer state (3 fp32 shards per RAM-tier param)
        self.required_nvme_bytes = 0
        for full_name, p in self._iter_composite_named_params():
            if self.plan.param_tier.get(full_name) is StorageTier.RAM:
                self.required_nvme_bytes += 12 * p.numel()  # 3 × fp32

        # Check free space before allocating NVMe state
        self._check_free_space()

        self._wire_ram_masters()
        self._allocate_nvme_state()

    def _check_free_space(self) -> None:
        """Ensure the NVMe root has at least min_free_gb free, plus required_nvme_bytes."""
        total_required = self.required_nvme_bytes + int(self.min_free_gb * 1024**3)
        free_bytes = shutil.disk_usage(self.nvme_root).free
        if free_bytes < total_required:
            raise RuntimeError(
                f"Insufficient free space on NVMe offload root {self.nvme_root}. "
                f"Required: {total_required / 1024**3:.2f} GB "
                f"(for optimizer state: {self.required_nvme_bytes / 1024**3:.2f} GB "
                f"+ free margin: {self.min_free_gb} GB), "
                f"Available: {free_bytes / 1024**3:.2f} GB"
            )

    # --- initialization -----------------------------------------------------

    def _iter_composite_named_params(self):
        composite = getattr(self.model, self.composite_attr)
        for name, p in composite.named_parameters(recurse=True):
            yield f"{self.composite_attr}.{name}", p

    def _wire_ram_masters(self) -> None:
        for full_name, p in self._iter_composite_named_params():
            if full_name not in self.plan.param_tier:
                continue
            if self.plan.param_tier[full_name] is not StorageTier.RAM:
                continue
            # We hold a bf16 CPU reference; the actual training param may live
            # on GPU but we retain a CPU mirror to rehydrate from.
            self._ram_master_bf16[full_name] = p.detach().to("cpu", dtype=torch.bfloat16).clone()

    def _allocate_nvme_state(self) -> None:
        for full_name, p in self._iter_composite_named_params():
            if self.plan.param_tier.get(full_name) is not StorageTier.RAM:
                continue
            shape = tuple(p.shape)
            safe_name = full_name.replace(".", "_")
            master_path = self.nvme_root / f"{safe_name}__master.f32"
            m_path = self.nvme_root / f"{safe_name}__m.f32"
            v_path = self.nvme_root / f"{safe_name}__v.f32"

            if not master_path.exists():
                master = _NvmeShard.create(master_path, shape, init="from_param")
                master.write_tensor(p.detach().to("cpu", dtype=torch.float32))
            else:
                master = _NvmeShard.open_existing(master_path, shape)

            if not m_path.exists():
                m = _NvmeShard.create(m_path, shape, init="zeros")
            else:
                m = _NvmeShard.open_existing(m_path, shape)

            if not v_path.exists():
                v = _NvmeShard.create(v_path, shape, init="zeros")
            else:
                v = _NvmeShard.open_existing(v_path, shape)

            self._nvme_master_fp32[full_name] = master
            self._nvme_adam_m[full_name] = m
            self._nvme_adam_v[full_name] = v
            self.plan.bytes_nvme += 12 * p.numel()    # 3 × fp32

    # --- accessors ----------------------------------------------------------

    def ram_master(self, name: str) -> Tensor:
        return self._ram_master_bf16[name]

    def nvme_master(self, name: str) -> _NvmeShard:
        return self._nvme_master_fp32[name]

    def adam_state(self, name: str) -> tuple[_NvmeShard, _NvmeShard]:
        return self._nvme_adam_m[name], self._nvme_adam_v[name]

    def ram_param_names(self) -> list[str]:
        return list(self._ram_master_bf16.keys())

    def total_nvme_bytes(self) -> int:
        return self.plan.bytes_nvme

    # --- promote / demote ---------------------------------------------------

    def promote_to_gpu(self, name: str, device: torch.device,
                       stream: torch.cuda.Stream | None = None) -> Tensor:
        """Copy the RAM bf16 master of `name` to GPU, returning the GPU tensor.

        Caller owns the returned tensor; it should be freed (by going out of
        scope) after the forward/backward that needs it.
        """
        cpu_master = self._ram_master_bf16[name]
        ctx = torch.cuda.stream(stream) if stream is not None else _NullCtx()
        with ctx:
            gpu = cpu_master.to(device=device, dtype=cpu_master.dtype, non_blocking=True)
        return gpu

    # --- checkpointing ------------------------------------------------------

    def flush(self) -> None:
        """Ensure all NVMe shards hit the platter."""
        for shard_dict in (self._nvme_master_fp32, self._nvme_adam_m, self._nvme_adam_v):
            for shard in shard_dict.values():
                shard.mmap.flush()


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *exc): return None
