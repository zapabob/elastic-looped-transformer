"""HF-backed Qwen3.5 looped language model for ELT side-family experiments.

This keeps the native ELT implementation intact and adds a second backbone kind:
`hf_qwen35_looped`. The loop axis wraps a Qwen3.5 text backbone, treating one
full language-model pass as g_Theta and re-applying it L times.
"""

from __future__ import annotations

import math
import gc
import sys
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextModel

from elt_lm.config import ModelConfig
from elt_lm.model import ELTOutput


def patch_fla_windows_cpu_fallback() -> None:
    """Work around FLA's Windows Triton fallback selecting ``torch.cpu.device``.

    On Windows, ``triton-windows`` can fail to expose an active backend during
    FLA import, causing FLA to mark its backend as ``cpu`` even while tensors are
    CUDA tensors. Recent PyTorch does not expose ``torch.cpu.device``; the
    resulting context-manager lookup fails before the operation can run. Keep
    this patch narrow and runtime-only so the vendored dependency stays intact.
    """

    fla_utils = sys.modules.get("fla.utils")
    if fla_utils is None or not torch.cuda.is_available():
        return

    device_torch_lib = getattr(fla_utils, "device_torch_lib", None)
    if device_torch_lib is not getattr(torch, "cpu", None):
        return

    def _cuda_device_ctx(index: int | None):
        return torch.cuda.device(0 if index is None else index)

    fla_utils.device = "cuda"
    fla_utils.device_name = "cuda"
    fla_utils.device_torch_lib = torch.cuda
    fla_utils.custom_device_ctx = _cuda_device_ctx


def _torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


class LoRALinear(nn.Module):
    """Minimal LoRA adapter around a frozen Linear projection."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        lora = torch.matmul(self.dropout(x), self.lora_A.t())
        lora = torch.matmul(lora, self.lora_B.t())
        return out + lora * self.scaling


def _parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    return parent, child_name


def _layer_index_from_name(module_name: str) -> int | None:
    parts = module_name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def inject_lora_adapters(model: "HFQwen35LoopedLM", cfg: ModelConfig) -> int:
    """Attach LoRA adapters to selected Qwen3.5 Linear projections."""

    rank = int(cfg.hf_lora_rank)
    if rank <= 0:
        raise ValueError("hf_trainable_mode='lora' requires model.hf_lora_rank > 0")

    target_names = set(cfg.hf_lora_target_modules)
    top_layers = int(cfg.hf_lora_top_layers)
    first_trainable_layer = max(0, len(model.qwen.model.layers) - top_layers) if top_layers > 0 else 0
    replacements: list[tuple[str, nn.Linear]] = []

    for module_name, module in model.qwen.model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, LoRALinear):
            continue
        if module_name.rsplit(".", 1)[-1] not in target_names:
            continue
        layer_idx = _layer_index_from_name(module_name)
        if layer_idx is not None and layer_idx < first_trainable_layer:
            continue
        replacements.append((f"qwen.model.{module_name}", module))

    for full_name, linear in replacements:
        parent, child_name = _parent_module(model, full_name)
        setattr(parent, child_name, LoRALinear(
            linear,
            rank=rank,
            alpha=cfg.hf_lora_alpha,
            dropout=cfg.hf_lora_dropout,
        ))

    return len(replacements)


def _lora_module_names(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]


def load_qwen35_text_config(hf_model_path: str) -> Qwen3_5TextConfig:
    cfg = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    if not isinstance(text_cfg, Qwen3_5TextConfig):
        raise TypeError(
            f"expected Qwen3_5TextConfig from {hf_model_path!r}, got {type(text_cfg)!r}"
        )
    return text_cfg


def sync_model_config_from_qwen35(model_cfg: ModelConfig, text_cfg: Qwen3_5TextConfig) -> ModelConfig:
    model_cfg.vocab_size = int(text_cfg.vocab_size)
    model_cfg.d_model = int(text_cfg.hidden_size)
    model_cfg.n_unique_layers = int(text_cfg.num_hidden_layers)
    model_cfg.n_heads = int(text_cfg.num_attention_heads)
    model_cfg.n_kv_heads = int(text_cfg.num_key_value_heads)
    model_cfg.head_dim = int(text_cfg.head_dim)
    model_cfg.d_ff = int(text_cfg.intermediate_size)
    model_cfg.max_seq_len = int(text_cfg.max_position_embeddings)
    rope = getattr(text_cfg, "rope_parameters", None)
    if isinstance(rope, dict) and "rope_theta" in rope:
        model_cfg.rope_theta = float(rope["rope_theta"])
    model_cfg.tie_word_embeddings = bool(getattr(text_cfg, "tie_word_embeddings", False))
    return model_cfg


def extract_qwen35_text_components(model: nn.Module) -> tuple[Qwen3_5TextModel, nn.Linear]:
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, nn.Linear):
        raise TypeError(f"model {type(model)!r} does not expose a Linear lm_head")

    root = getattr(model, "model", None)
    if root is None:
        raise TypeError(f"model {type(model)!r} does not expose a `.model` backbone")

    if isinstance(root, Qwen3_5TextModel):
        return root, lm_head

    language_model = getattr(root, "language_model", None)
    if isinstance(language_model, Qwen3_5TextModel):
        return language_model, lm_head

    raise TypeError(f"unable to locate Qwen3.5 text backbone in {type(model)!r}")


def apply_hf_trainable_mode(model: "HFQwen35LoopedLM", cfg: ModelConfig) -> None:
    mode = cfg.hf_trainable_mode
    for p in model.parameters():
        p.requires_grad = (mode == "all")

    if mode == "frozen":
        return
    if mode == "all":
        return
    if mode == "lora":
        injected = 0 if _lora_module_names(model) else inject_lora_adapters(model, cfg)
        if injected == 0 and not _lora_module_names(model):
            raise ValueError("hf_trainable_mode='lora' did not match any target modules")
        for name, p in model.named_parameters():
            p.requires_grad = ".lora_" in name
        return

    # The loop-introduction default: keep the readout path and stabilizers trainable.
    norm_modules: list[nn.Module] = [cast(nn.Module, model.qwen.model.norm)]
    for layer in model.qwen.model.layers:
        norm_modules.append(cast(nn.Module, layer.input_layernorm))
        norm_modules.append(cast(nn.Module, layer.post_attention_layernorm))

    if mode == "norm_lm_head":
        for module in norm_modules:
            for p in module.parameters():
                p.requires_grad = True
        for p in model.qwen.lm_head.parameters():
            p.requires_grad = True
        return

    if mode == "top_layers":
        n = max(0, int(cfg.hf_trainable_top_layers))
        if n > 0:
            for layer in model.qwen.model.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        for module in norm_modules:
            for p in module.parameters():
                p.requires_grad = True
        for p in model.qwen.lm_head.parameters():
            p.requires_grad = True
        return

    raise ValueError(f"unknown hf_trainable_mode={mode!r}")


class HFQwen35LoopedLM(nn.Module):
    """Looped wrapper around a HF Qwen3.5 text backbone."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if not cfg.hf_model_path:
            raise ValueError("hf_qwen35_looped requires model.hf_model_path")

        patch_fla_windows_cpu_fallback()
        self.cfg = cfg
        self._adapter_base_checkpoint_path = cfg.hf_adapter_base_ckpt
        self.text_cfg = load_qwen35_text_config(cfg.hf_model_path)
        sync_model_config_from_qwen35(cfg, self.text_cfg)
        self.qwen = Qwen3_5ForCausalLM(self.text_cfg)
        apply_hf_trainable_mode(self, cfg)

    def load_pretrained_from_source(self, hf_model_path: str | None = None) -> None:
        source_path = hf_model_path or self.cfg.hf_model_path
        source = AutoModelForCausalLM.from_pretrained(
            source_path,
            trust_remote_code=True,
            torch_dtype=_torch_dtype_from_name(self.cfg.parity_dtype),
            low_cpu_mem_usage=True,
        )
        text_model, lm_head = extract_qwen35_text_components(source)
        if self.cfg.hf_trainable_mode == "lora":
            prefixed = {
                f"qwen.model.{key}": value
                for key, value in text_model.state_dict().items()
            }
            if self.cfg.import_lm_head:
                prefixed.update({
                    f"qwen.lm_head.{key}": value
                    for key, value in lm_head.state_dict().items()
                })
            self.load_state_dict(prefixed)
        else:
            self.qwen.model.load_state_dict(text_model.state_dict())
            if self.cfg.import_lm_head:
                self.qwen.lm_head.load_state_dict(lm_head.state_dict())
        del source, text_model, lm_head
        gc.collect()
        apply_hf_trainable_mode(self, self.cfg)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # type: ignore[override]
        lora_names = _lora_module_names(self)
        if not lora_names:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        mapped = dict(state_dict)
        for module_name in lora_names:
            for suffix in ("weight", "bias"):
                src_key = f"{module_name}.{suffix}"
                dst_key = f"{module_name}.base.{suffix}"
                if src_key in mapped and dst_key not in mapped:
                    mapped[dst_key] = mapped.pop(src_key)

        result = super().load_state_dict(mapped, strict=False, assign=assign)
        if strict:
            allowed_missing = {
                f"{module_name}.{suffix}"
                for module_name in lora_names
                for suffix in ("lora_A", "lora_B")
            }
            missing = [key for key in result.missing_keys if key not in allowed_missing]
            unexpected = list(result.unexpected_keys)
            if missing or unexpected:
                lines = []
                if missing:
                    lines.append(f"Missing key(s): {missing}")
                if unexpected:
                    lines.append(f"Unexpected key(s): {unexpected}")
                raise RuntimeError(
                    "Error(s) in loading state_dict for HFQwen35LoopedLM:\n"
                    + "\n".join(lines)
                )
        return result

    def remember_adapter_base_checkpoint(self, path: str | Path) -> None:
        self._adapter_base_checkpoint_path = str(path)

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if ".lora_" in name
        }

    def adapter_checkpoint_state(self) -> dict[str, object]:
        if self.cfg.hf_trainable_mode != "lora":
            raise RuntimeError("adapter-only checkpoint requested for a non-LoRA model")
        base = self._adapter_base_checkpoint_path or self.cfg.hf_adapter_base_ckpt
        if not base:
            raise RuntimeError(
                "adapter-only checkpoint requires a base checkpoint. Start from "
                "--resume H:/elt_data/runs/qwen35_4b_elt_bootstrap/last.pt or set "
                "model.hf_adapter_base_ckpt."
            )
        return {
            "adapter_only": True,
            "adapter_format": "elt_hf_qwen35_lora_v1",
            "base_checkpoint": base,
            "model": self.adapter_state_dict(),
        }

    def load_adapter_checkpoint_state(self, state: dict[str, object]) -> None:
        base = str(state.get("base_checkpoint") or self.cfg.hf_adapter_base_ckpt or "")
        if not base:
            raise RuntimeError("adapter checkpoint does not include base_checkpoint")
        base_state = torch.load(base, map_location="cpu", weights_only=False)
        self.load_state_dict(base_state["model"] if "model" in base_state else base_state)
        adapter_state = state.get("model")
        if not isinstance(adapter_state, dict):
            raise RuntimeError("adapter checkpoint missing model adapter state")
        missing, unexpected = super().load_state_dict(adapter_state, strict=False)
        unexpected = list(unexpected)
        if unexpected:
            raise RuntimeError(f"unexpected adapter keys: {unexpected}")
        allowed_missing = set(self.state_dict()) - set(adapter_state)
        extra_missing = [key for key in missing if key not in allowed_missing]
        if extra_missing:
            raise RuntimeError(f"missing adapter keys: {extra_missing}")
        self._adapter_base_checkpoint_path = base
        apply_hf_trainable_mode(self, self.cfg)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.qwen.model.embed_tokens

    def _project(self, hidden: Tensor) -> Tensor:
        return self.qwen.lm_head(hidden)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.qwen.model.embed_tokens.weight.numel()
            n -= self.qwen.lm_head.weight.numel()
        return n

    def forward(
        self,
        input_ids: Tensor,
        L: int,
        return_hidden_at: int | None = None,
        return_all_loop_hidden: bool = False,
        attention_mask: Tensor | None = None,
    ) -> ELTOutput:
        assert L >= 1, f"L must be >= 1, got {L}"
        if return_hidden_at is not None:
            assert 1 <= return_hidden_at <= L, (
                f"return_hidden_at={return_hidden_at} out of range [1, {L}]"
            )

        x = self.qwen.model.embed_tokens(input_ids)
        intermediate_logits: Tensor | None = None
        intermediate_hidden: Tensor | None = None
        per_loop_hidden: list[Tensor] | None = [] if return_all_loop_hidden else None

        for l_idx in range(1, L + 1):
            outputs = self.qwen.model(
                inputs_embeds=x,
                attention_mask=attention_mask,
                use_cache=False,
            )
            x = outputs.last_hidden_state
            if per_loop_hidden is not None:
                per_loop_hidden.append(x)
            if return_hidden_at is not None and l_idx == return_hidden_at and l_idx != L:
                intermediate_hidden = x
                intermediate_logits = self._project(x)

        logits = self._project(x)
        if return_hidden_at is not None and return_hidden_at == L:
            intermediate_hidden = x
            intermediate_logits = logits

        return ELTOutput(
            logits=logits,
            intermediate_logits=intermediate_logits,
            intermediate_hidden=intermediate_hidden,
            per_loop_hidden=tuple(per_loop_hidden) if per_loop_hidden is not None else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        L: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ids = input_ids[:, -self.cfg.max_seq_len :]
            attn = attention_mask[:, -self.cfg.max_seq_len :] if attention_mask is not None else None
            out = self.forward(ids, L=L, attention_mask=attn)
            logits = out.logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                vals, idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(-1, idx, vals)
                logits = mask

            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
            if attention_mask is not None:
                next_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        return input_ids
