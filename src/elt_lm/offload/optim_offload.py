"""NvmeAdamW — AdamW with fp32 state (m, v, master weight) memory-mapped on NVMe.

The update math is stock AdamW (Loshchilov & Hutter 2019, arXiv:1711.05101)
with bias-correction:

    m_t = β1 m_{t-1} + (1-β1) g_t
    v_t = β2 v_{t-1} + (1-β2) g_t²
    m̂_t = m_t / (1 - β1^t)
    v̂_t = v_t / (1 - β2^t)
    θ_t = θ_{t-1} - lr · (m̂_t / (sqrt(v̂_t) + ε) + wd · θ_{t-1})

What's non-standard:
    - Master weight θ, first moment m, and second moment v are all kept as
      fp32 numpy memmaps on NVMe (backing store = `TieredParameterStore`).
    - On `step()`, for each parameter we: read m/v/θ from NVMe to RAM, upcast
      the incoming gradient (bf16 → fp32), run the AdamW update on CPU,
      write m/v/θ back to NVMe, and copy the updated θ to the param's device
      (downcasting to bf16 as needed so the training copy reflects the new
      weights).

Performance: the plan puts grad_accum=64, so NVMe state is touched ~once
every 64 forward/backward passes. At a modest 500 MB/s NVMe write that's
~2 s/step for a 1 B non-emb model — tolerable for the expected 10-30 s
optimizer interval on RTX 3060.

For parameters that are NOT in the tiered store (token embedding, LM head,
final norm), we fall back to a standard torch.optim.AdamW internally — so
this optimizer is safe to use for the full model without partial-ownership
juggling.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from elt_lm.offload.tiered_store import TieredParameterStore, _NvmeShard


class NvmeAdamW(Optimizer):
    """AdamW whose state lives on NVMe for the params in a `TieredParameterStore`.

    Hybrid optimizer: params that appear in `store.ram_param_names()` use the
    NVMe-backed path; everything else uses stock `torch.optim.AdamW` semantics.
    """

    def __init__(
        self,
        params,
        store: TieredParameterStore,
        name_lookup: dict[int, str],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Args:
            params:       parameter list or param-groups (same as torch.optim.AdamW)
            store:        tiered parameter store that owns NVMe shards
            name_lookup:  map id(param)→full_name, so we know which params are
                          tiered. Callers should build this from
                          `model.named_parameters()` before constructing the
                          optimizer.
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.store = store
        self._name_lookup = name_lookup

    # -----------------------------------------------------------------------

    def _nvme_shards(self, name: str) -> tuple[_NvmeShard, _NvmeShard, _NvmeShard]:
        m, v = self.store.adam_state(name)
        master = self.store.nvme_master(name)
        return m, v, master

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("NvmeAdamW does not support sparse grads")

                name = self._name_lookup.get(id(p))
                is_tiered = (name is not None
                             and name in self.store.ram_param_names())

                state = self.state[p]
                state.setdefault("step", 0)
                state["step"] += 1
                t = state["step"]
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                if is_tiered:
                    self._step_tiered(p, grad, name, lr, beta1, beta2, eps, wd,
                                      bias_correction1, bias_correction2)
                else:
                    self._step_standard(p, grad, state, lr, beta1, beta2, eps, wd,
                                        bias_correction1, bias_correction2)

        return loss

    # -----------------------------------------------------------------------

    def _step_tiered(
        self, p: Tensor, grad: Tensor, name: str,
        lr: float, beta1: float, beta2: float, eps: float, wd: float,
        bc1: float, bc2: float,
    ) -> None:
        m_shard, v_shard, master_shard = self._nvme_shards(name)

        # Pull fp32 state into RAM (zero-copy on the mmap; .clone() materializes)
        m = m_shard.read_tensor().clone()
        v = v_shard.read_tensor().clone()
        master = master_shard.read_tensor().clone()

        g32 = grad.detach().to("cpu", dtype=torch.float32).view(master.shape)

        # AdamW update in fp32 on CPU
        m.mul_(beta1).add_(g32, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
        m_hat = m / bc1
        v_hat = v / bc2
        denom = v_hat.sqrt().add_(eps)
        # decoupled weight decay
        if wd != 0.0:
            master.mul_(1.0 - lr * wd)
        master.addcdiv_(m_hat, denom, value=-lr)

        # Persist state.
        m_shard.write_tensor(m)
        v_shard.write_tensor(v)
        master_shard.write_tensor(master)

        # Reflect the update back into the live param (downcast as needed).
        new_weight = master.to(dtype=p.dtype, device=p.device)
        p.data.copy_(new_weight)

        # Also update the bf16 RAM master so future `promote_to_gpu` calls
        # see the latest weights.
        self.store._ram_master_bf16[name].copy_(master.to(torch.bfloat16))

    def _step_standard(
        self, p: Tensor, grad: Tensor, state: dict[str, Any],
        lr: float, beta1: float, beta2: float, eps: float, wd: float,
        bc1: float, bc2: float,
    ) -> None:
        if "m" not in state:
            state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        m = state["m"]
        v = state["v"]
        g = grad.to(dtype=m.dtype)
        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        m_hat = m / bc1
        v_hat = v / bc2
        denom = v_hat.sqrt().add_(eps)
        if wd != 0.0:
            p.data.mul_(1.0 - lr * wd)
        p.data.addcdiv_(m_hat, denom, value=-lr)


def build_name_lookup(model) -> dict[int, str]:
    """Handy helper — id(Parameter) → fully-qualified name."""
    return {id(p): n for n, p in model.named_parameters()}
