"""HuggingFace-compatible wrapper around ELTLanguageModel.

Inherits PreTrainedModel so that the standard
`AutoModelForCausalLM.from_pretrained(...)` / `model.generate(...)` / safetensors
round-trip all work. The internal `ELTLanguageModel` does the real compute; this
wrapper just exposes the HF I/O contract and a small `L` plumbing hook so the
user can pick the loop count at inference.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from elt_lm.hf.configuration_elt import ELTConfig
from elt_lm.model import ELTLanguageModel


class ELTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ELTConfig
    base_model_prefix = "elt"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerLayer"]           # type: ignore[assignment]
    _tied_weights_keys = {"lm_head.weight": "elt.tok_embed.weight"}  # type: ignore[assignment]
    _keys_to_ignore_on_load_missing = ["lm_head.weight"]
    main_input_name = "input_ids"

    def __init__(self, config: ELTConfig) -> None:
        super().__init__(config)
        self.elt = ELTLanguageModel(config.to_model_config())
        # Register a real Linear as `lm_head` and *share* its weight with the
        # token embedding. HF's tie machinery + safetensors dedupe will then
        # produce a single on-disk tensor.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.elt.tok_embed.weight
        # `elt.lm_head` is unused when we expose our own tied head.
        self.elt.lm_head = None

        self.post_init()

    # --- HF interface -----------------------------------------------------

    def get_input_embeddings(self) -> nn.Embedding:
        return self.elt.tok_embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        assert isinstance(value, nn.Embedding)
        self.elt.tok_embed = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def tie_weights(self, *args, **kwargs) -> None:  # type: ignore[override]
        # Re-establish the lm_head / tok_embed weight sharing. HF calls this on
        # every from_pretrained AFTER weights are loaded, which is exactly what
        # we need: it undoes the un-tying that happens when safetensors loads
        # into the separately-registered lm_head.weight slot.
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.elt.tok_embed.weight

    # --- forward ----------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        L: int | None = None,
        return_dict: bool | None = None,
        **_unused: Any,
    ) -> CausalLMOutputWithPast:
        L_used = int(L) if L is not None else int(getattr(self.config, "L_default", self.config.L_max))
        out = self.elt(input_ids, L=L_used)
        logits = out.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,          # type: ignore[arg-type]
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    # --- generation helper ------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values: Any = None,
        attention_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Propagate the custom `L` through model_kwargs so each generate() step
        # sees it; HF default otherwise drops unknown kwargs.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "L": kwargs.get("L", getattr(self.config, "L_default", self.config.L_max)),
        }
