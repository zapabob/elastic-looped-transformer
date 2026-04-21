from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from elt_lm.config import GRPOConfig, ModelConfig, RewardModelConfig, TrainConfig
from elt_lm.model import ELTLanguageModel
from elt_lm.reward_model import ELTRewardModel
from elt_lm.train_grpo import train_grpo


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 7
    eos_token = "<eos>"
    pad_token = "<pad>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [min(6, max(1, len(text) % 6))]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return "<think>small reasoning trace</think><answer>ok</answer>"

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            rows.append(self.encode(text))
        width = max(len(row) for row in rows)
        data = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            data[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        return SimpleNamespace(input_ids=data)


def test_train_grpo_hybrid_smoke(tmp_path: Path, monkeypatch) -> None:
    cfg = TrainConfig(
        model=ModelConfig(
            vocab_size=16,
            d_model=16,
            n_unique_layers=1,
            n_heads=1,
            head_dim=16,
            d_ff=32,
            max_seq_len=16,
            tie_word_embeddings=True,
            grad_checkpoint=False,
        ),
        grpo=GRPOConfig(
            enabled=True,
            init_ckpt=str(tmp_path / "sft.pt"),
            prompts_file=str(tmp_path / "prompts.jsonl"),
            task="exact_match",
            group_size=2,
            rollout_max_new_tokens=2,
            rollout_L=1,
            reward_model_ckpt=str(tmp_path / "rm.pt"),
            reward_alpha=0.3,
            verifier_beta=0.7,
            prompt_budget=1,
        ),
        reward_model=RewardModelConfig(enabled=True, train_L=1),
        dtype="fp32",
        micro_batch_size=1,
        grad_accum_steps=1,
        total_steps=1,
        log_every=1,
        save_every=0,
        run_dir=str(tmp_path / "run"),
    )

    lm = ELTLanguageModel(cfg.model).to(dtype=torch.float32)
    rm = ELTRewardModel(cfg.model).to(dtype=torch.float32)
    torch.save({"model": lm.state_dict(), "cfg": cfg}, tmp_path / "sft.pt")
    torch.save({"model": rm.state_dict(), "cfg": cfg}, tmp_path / "rm.pt")
    with open(tmp_path / "prompts.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "say ok", "reference": "ok"}) + "\n")

    monkeypatch.setattr("elt_lm.train_grpo.load_tokenizer", lambda _path: _FakeTokenizer())
    train_grpo(cfg)

    assert (tmp_path / "run" / "last.pt").is_file()
    metrics = (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8")
    assert "reward_model_mean" in metrics
