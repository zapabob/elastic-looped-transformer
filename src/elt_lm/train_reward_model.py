"""Train a pairwise reward model on normalized preference pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from elt_lm.config import TrainConfig, load_train_config
from elt_lm.reward_model import ELTRewardModel, bradley_terry_loss, load_preference_records
from elt_lm.telemetry import make_writer


def _batched(records: list[dict], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start:start + batch_size]


def train_reward_model(cfg: TrainConfig) -> None:
    assert cfg.reward_model.enabled, "reward_model.enabled must be true"
    assert cfg.reward_model.preferences_file, "reward_model.preferences_file must be set"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records = load_preference_records(cfg.reward_model.preferences_file)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = ELTRewardModel(cfg.model).to(device)
    if cfg.reward_model.init_ckpt:
        state = torch.load(cfg.reward_model.init_ckpt, map_location="cpu", weights_only=False)
        backbone_state = state["model"] if "model" in state else state
        model.backbone.load_state_dict(backbone_state, strict=False)
    if cfg.reward_model.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    telemetry = make_writer(run_dir)
    step = 0
    try:
        for epoch in range(max(1, cfg.total_steps)):
            for batch in _batched(records, cfg.micro_batch_size):
                chosen = [row["chosen_text"] for row in batch]
                rejected = [row["rejected_text"] for row in batch]
                chosen_ids = tok(
                    chosen,
                    padding=True,
                    truncation=True,
                    max_length=cfg.data.seq_len,
                    return_tensors="pt",
                ).input_ids.to(device)
                rejected_ids = tok(
                    rejected,
                    padding=True,
                    truncation=True,
                    max_length=cfg.data.seq_len,
                    return_tensors="pt",
                ).input_ids.to(device)

                chosen_reward = model(chosen_ids, L=cfg.reward_model.train_L)
                rejected_reward = model(rejected_ids, L=cfg.reward_model.train_L)
                loss = bradley_terry_loss(
                    chosen_reward,
                    rejected_reward,
                    margin=cfg.reward_model.margin,
                )

                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

                telemetry.emit(
                    "reward_model_step",
                    step=step,
                    epoch=epoch,
                    loss=loss.item(),
                    chosen_mean=chosen_reward.mean().item(),
                    rejected_mean=rejected_reward.mean().item(),
                )
                step += 1
                if step % max(1, cfg.log_every) == 0:
                    print(
                        f"reward-step {step:6d} | "
                        f"loss {loss.item():.4f} | "
                        f"chosen {chosen_reward.mean().item():.3f} | "
                        f"rejected {rejected_reward.mean().item():.3f}"
                    )
                if step >= cfg.total_steps:
                    break
            if step >= cfg.total_steps:
                break
    finally:
        torch.save(
            {"step": step, "cfg": cfg, "model": model.state_dict()},
            run_dir / "last.pt",
        )
        telemetry.close()


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    train_reward_model(load_train_config(args.config))


if __name__ == "__main__":
    cli()
