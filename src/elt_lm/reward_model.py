"""Reward-model head and pairwise preference helpers for ELT."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from elt_lm.config import ModelConfig
from elt_lm.model import ELTLanguageModel


@dataclass
class PreferenceBatch:
    chosen_ids: Tensor
    rejected_ids: Tensor


class ELTRewardModel(nn.Module):
    """Scalar reward head on top of the dense ELT backbone."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.backbone = ELTLanguageModel(cfg)
        self.reward_head = nn.Linear(cfg.d_model, 1, bias=False)

    def encode_last(self, input_ids: Tensor, L: int) -> Tensor:
        _, T = input_ids.shape
        x = self.backbone.tok_embed(input_ids)
        cos, sin = self.backbone.rope(T, device=x.device, dtype=x.dtype)
        for _ in range(L):
            x = self.backbone.composite(x, cos, sin)
        hidden = self.backbone.final_norm(x)
        return hidden[:, -1, :]

    def forward(self, input_ids: Tensor, L: int) -> Tensor:
        pooled = self.encode_last(input_ids, L=L)
        return self.reward_head(pooled).squeeze(-1)


def load_preference_records(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "chosen_text" in row and "rejected_text" in row:
                records.append(row)
    if not records:
        raise RuntimeError(f"no preference records found in {path}")
    return records


def bradley_terry_loss(chosen_rewards: Tensor, rejected_rewards: Tensor, margin: float = 0.0) -> Tensor:
    return -F.logsigmoid(chosen_rewards - rejected_rewards - margin).mean()


@torch.no_grad()
def score_text_batch(
    model: ELTRewardModel,
    tokenizer,
    texts: list[str],
    *,
    L: int,
    device: torch.device,
    max_length: int,
) -> Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    return model(encoded, L=L)
