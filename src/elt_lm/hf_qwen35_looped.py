"""HF-backed Qwen3.5 looped language model for ELT side-family experiments.

This keeps the native ELT implementation intact and adds a second backbone kind:
`hf_qwen35_looped`. The loop axis wraps a Qwen3.5 text backbone, treating one
full language-model pass as g_Theta and re-applying it L times.
"""

from __future__ import annotations

import gc
from typing import cast

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextModel

from elt_lm.config import ModelConfig
from elt_lm.model import ELTOutput


def _torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


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

        self.cfg = cfg
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
        self.qwen.model.load_state_dict(text_model.state_dict())
        if self.cfg.import_lm_head:
            self.qwen.lm_head.load_state_dict(lm_head.state_dict())
        del source, text_model, lm_head
        gc.collect()
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
