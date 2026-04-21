"""HuggingFace-style config for the ELT causal LM.

Mirrors `elt_lm.config.ModelConfig` fields 1:1 plus two generation-control
fields (`L_default`, `l_at_inference`). `PretrainedConfig` lets HF `AutoConfig`
deserialize `config.json`, and is required for `PreTrainedModel` subclasses.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class ELTConfig(PretrainedConfig):
    """Config for Elastic Looped Transformer (arXiv:2604.09168) causal LM."""

    model_type = "elt_lm"
    # HF `generate()` / `GenerationMixin` reads `num_hidden_layers` for cache sizing.
    # Our "effective" depth at inference is n_unique_layers × L, but since we don't
    # expose a per-layer KV cache, advertising the unique count is the right signal.
    attribute_map = {"num_hidden_layers": "n_unique_layers",
                     "num_attention_heads": "n_heads"}

    def __init__(
        self,
        vocab_size: int = 248_320,
        d_model: int = 768,
        n_unique_layers: int = 12,
        n_heads: int = 12,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        rope_theta: float = 10_000.0,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        dropout: float = 0.0,
        L_min: int = 1,
        L_max: int = 4,
        L_default: int = 4,
        init_std: float = 0.02,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_unique_layers = n_unique_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else (d_model // n_heads)
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.dropout = dropout
        self.L_min = L_min
        self.L_max = L_max
        self.L_default = L_default
        self.init_std = init_std

        # `hidden_size` is expected by some HF utilities (e.g. generation).
        self.hidden_size = d_model
        # `max_position_embeddings` is consulted by HF generation length checks.
        self.max_position_embeddings = max_seq_len

        super().__init__(**kwargs)

    @classmethod
    def from_model_config(cls, mc: Any, L_default: int | None = None) -> "ELTConfig":
        """Build an ELTConfig from an internal `elt_lm.config.ModelConfig`."""
        return cls(
            vocab_size=mc.vocab_size,
            d_model=mc.d_model,
            n_unique_layers=mc.n_unique_layers,
            n_heads=mc.n_heads,
            n_kv_heads=mc.n_kv_heads,
            head_dim=mc.head_dim,
            d_ff=mc.d_ff,
            max_seq_len=mc.max_seq_len,
            rope_theta=mc.rope_theta,
            rms_norm_eps=mc.rms_norm_eps,
            tie_word_embeddings=mc.tie_word_embeddings,
            dropout=mc.dropout,
            L_min=mc.L_min,
            L_max=mc.L_max,
            L_default=L_default if L_default is not None else mc.L_max,
            init_std=mc.init_std,
        )

    def to_model_config(self) -> Any:
        """Inverse of `from_model_config` — rebuild the internal ModelConfig."""
        from elt_lm.config import ModelConfig
        return ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_unique_layers=self.n_unique_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=self.tie_word_embeddings,
            dropout=self.dropout,
            L_min=self.L_min,
            L_max=self.L_max,
            init_std=self.init_std,
            grad_checkpoint=False,   # inference-time default
        )
