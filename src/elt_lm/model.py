"""ELTLanguageModel — causal-LM port of arXiv:2604.09168 §4.

Forward signature (central to ILSD training):

    logits, intermediate_logits = model(input_ids, L, return_hidden_at=L_int)

We run `composite_block` L times (weight-shared). At loop step `return_hidden_at`
(1-indexed — i.e. after that many iterations) we additionally apply the final
RMSNorm and the LM head to obtain student logits. The final logits after L loops
are the teacher logits. This lets a single forward pass produce both ends of the
ILSD loss (eq. 3), saving ~2× VRAM vs. two separate forwards.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.utils.checkpoint as gckpt
from torch import Tensor, nn

from elt_lm.composite import CompositeBlock
from elt_lm.config import ModelConfig
from elt_lm.norm import RMSNorm
from elt_lm.rope import RoPECache


class ELTOutput(NamedTuple):
    logits: Tensor                       # (B, T, V) — after L loops
    intermediate_logits: Tensor | None   # (B, T, V) — after return_hidden_at loops, or None


class ELTLanguageModel(nn.Module):
    """Embed -> composite_block^L -> RMSNorm -> LM head (weight-tied)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.composite = CompositeBlock(cfg)
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        assert cfg.head_dim is not None
        self.rope = RoPECache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

        if cfg.tie_word_embeddings:
            # Re-uses token embedding matrix as the LM projection; no new params.
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    # --- helpers ---------------------------------------------------------

    def _project(self, hidden: Tensor) -> Tensor:
        """Final RMSNorm + (tied) LM head."""
        h = self.final_norm(hidden)
        if self.lm_head is None:
            return h @ self.tok_embed.weight.T
        return self.lm_head(h)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_embed.weight.numel()
            if self.lm_head is not None:
                n -= self.lm_head.weight.numel()
        return n

    # --- forward ---------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        L: int,
        return_hidden_at: int | None = None,
    ) -> ELTOutput:
        """Run the ELT forward pass.

        input_ids:        (B, T) int64
        L:                number of composite-block iterations (>= 1)
        return_hidden_at: if given, also return logits after this many iterations
                          (must satisfy 1 <= return_hidden_at <= L).
        """
        assert L >= 1, f"L must be >= 1, got {L}"
        if return_hidden_at is not None:
            assert 1 <= return_hidden_at <= L, \
                f"return_hidden_at={return_hidden_at} out of range [1, {L}]"

        _, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, \
            f"seq_len={T} exceeds max_seq_len={self.cfg.max_seq_len}"

        x: Tensor = self.tok_embed(input_ids)                   # (B, T, D)
        cos, sin = self.rope(T, device=x.device, dtype=x.dtype)

        intermediate_logits: Tensor | None = None

        for l_idx in range(1, L + 1):
            if self.cfg.grad_checkpoint and self.training:
                x = gckpt.checkpoint(self.composite, x, cos, sin, use_reentrant=False)  # type: ignore[assignment]
            else:
                x = self.composite(x, cos, sin)

            if return_hidden_at is not None and l_idx == return_hidden_at and l_idx != L:
                # Student read-off. Detach not used here — gradients flow through
                # the student projection to loops 1..l_idx. Teacher is a separate
                # continuation (loops l_idx+1..L) past this point.
                intermediate_logits = self._project(x)

        logits = self._project(x)

        if return_hidden_at is not None and return_hidden_at == L:
            # Student == teacher in this degenerate case (L_int == L_max).
            # Return the same tensor for both so downstream dist-loss term becomes 0.
            intermediate_logits = logits

        return ELTOutput(logits=logits, intermediate_logits=intermediate_logits)

    # --- generation ------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        L: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Autoregressive sampling at a user-chosen loop count L.

        Note: no KV cache — each step does a full forward over the grown sequence.
        The ELT paper's Algorithm 2 already performs a fresh L-loop per token since
        input representations change each step. KV caching across tokens could be
        added later per-layer but is intentionally omitted here for correctness
        and simplicity.
        """
        self.eval()
        for _ in range(max_new_tokens):
            ids = input_ids[:, -self.cfg.max_seq_len:]
            out = self.forward(ids, L=L)
            logits = out.logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                vals, idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(-1, idx, vals)
                logits = mask

            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=-1)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        return input_ids


# Convenience constructor ------------------------------------------------

def build_model(cfg: ModelConfig) -> ELTLanguageModel:
    model = ELTLanguageModel(cfg)
    return model
