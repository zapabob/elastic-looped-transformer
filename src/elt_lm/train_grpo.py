"""GRPO post-training loop for the ELT causal-LM."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

from elt_lm.config import TrainConfig, load_train_config
from elt_lm.grpo import GRPOOutput, group_advantage, grpo_loss
from elt_lm.model import build_model
from elt_lm.posttrain_data import render_chat_text
from elt_lm.reward_model import ELTRewardModel, score_text_batch
from elt_lm.telemetry import make_writer
from elt_lm.train import (
    RollingCheckpointer,
    configure_optimizer,
    get_dtype,
    load_checkpoint,
    lr_at,
    save_checkpoint,
)
from elt_lm.verifiers import CompositeVerifier


def load_prompts(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" in row and "reference" in row:
                out.append(row)
    if not out:
        raise RuntimeError(f"no usable prompts in {path}")
    return out


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy_checkpoint(path: str | Path, model: torch.nn.Module) -> int:
    """Load the GRPO initial policy without restoring optimizer/RNG state.

    GRPO owns its optimizer separately from SFT. The initial checkpoint may be a
    full native ELT checkpoint or a side-branch LoRA adapter-only checkpoint.
    """

    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        isinstance(state, dict)
        and state.get("adapter_only")
        and hasattr(model, "load_adapter_checkpoint_state")
    ):
        model.load_adapter_checkpoint_state(state)  # type: ignore[attr-defined]
    else:
        model.load_state_dict(
            state["model"] if isinstance(state, dict) and "model" in state else state
        )
        if hasattr(model, "remember_adapter_base_checkpoint"):
            model.remember_adapter_base_checkpoint(path)  # type: ignore[attr-defined]
    return int(state.get("step", 0)) if isinstance(state, dict) else 0


@torch.no_grad()
def rollout_group(
    model: torch.nn.Module,
    prompt_ids: Tensor,
    group_size: int,
    L: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    pad_id: int,
    eos_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    device = next(model.parameters()).device
    model.eval()
    batch = prompt_ids.unsqueeze(0).repeat(group_size, 1).to(device=device)
    prompt_len = batch.shape[1]

    eos_hit = torch.zeros(group_size, dtype=torch.bool, device=device)
    produced: list[list[int]] = [[] for _ in range(group_size)]

    for _ in range(max_new_tokens):
        ids = batch[:, -model.cfg.max_seq_len:]
        out = model(ids, L=L)
        logits = out.logits[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0:
            vals, idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, idx, vals)
            logits = mask
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
        nxt = torch.where(eos_hit, torch.full_like(nxt, pad_id), nxt)
        for i in range(group_size):
            if not eos_hit[i]:
                produced[i].append(int(nxt[i].item()))
        batch = torch.cat([batch, nxt.unsqueeze(-1)], dim=-1)
        eos_hit |= (nxt == eos_id)
        if eos_hit.all():
            break

    resp_width = max((len(p) for p in produced), default=1)
    resp = torch.full((group_size, resp_width), pad_id, dtype=torch.long, device=device)
    resp_lens = torch.zeros(group_size, dtype=torch.long, device=device)
    for i, items in enumerate(produced):
        if items:
            resp[i, : len(items)] = torch.tensor(items, dtype=torch.long, device=device)
            resp_lens[i] = len(items)

    prompt_block = prompt_ids.unsqueeze(0).repeat(group_size, 1).to(device=device)
    full = torch.cat([prompt_block, resp], dim=-1)
    resp_mask = torch.zeros_like(full, dtype=torch.long)
    for i in range(group_size):
        resp_mask[i, prompt_len : prompt_len + int(resp_lens[i])] = 1
    return full, resp_mask, resp_lens


def _load_reward_model(cfg: TrainConfig, device: torch.device, dtype: torch.dtype) -> ELTRewardModel | None:
    if not cfg.grpo.reward_model_ckpt:
        return None
    state = torch.load(cfg.grpo.reward_model_ckpt, map_location="cpu", weights_only=False)
    model = ELTRewardModel(cfg.model).to(device=device, dtype=dtype)
    model.load_state_dict(state["model"] if "model" in state else state)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def train_grpo(cfg: TrainConfig, resume: str | None = None) -> None:
    assert cfg.grpo.enabled, "grpo.enabled must be true for train_grpo"
    assert cfg.grpo.init_ckpt, "grpo.init_ckpt must point at the SFT checkpoint"
    assert cfg.grpo.prompts_file, "grpo.prompts_file must point at prompts JSONL"

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(cfg.dtype)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(cfg.data.tokenizer_path)
    prompts = load_prompts(cfg.grpo.prompts_file)
    if cfg.grpo.prompt_budget > 0:
        prompts = prompts[: cfg.grpo.prompt_budget]
    print(f"loaded {len(prompts):,} prompts from {cfg.grpo.prompts_file}")

    model = build_model(cfg.model).to(device=device, dtype=dtype)
    init_step = load_policy_checkpoint(cfg.grpo.init_ckpt, model)

    ref = copy.deepcopy(model).to(device=device, dtype=dtype)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    old = copy.deepcopy(model).to(device=device, dtype=dtype)
    for p in old.parameters():
        p.requires_grad_(False)
    old.eval()

    reward_model = _load_reward_model(cfg, device=device, dtype=dtype)

    model.train()
    if not hasattr(model, "num_parameters"):
        raise TypeError(f"model {type(model)!r} does not expose num_parameters()")
    print(
        f"policy params: {model.num_parameters()/1e6:.1f}M · "
        f"init_step={init_step} · ref/old frozen"
    )

    offload_store = None
    if cfg.optim.kind == "nvme_adamw":
        from elt_lm.offload.hooks import install_offload_into_training
        opt, offload_store = install_offload_into_training(
            model, cfg=cfg, run_dir=run_dir,
        )
    else:
        opt = configure_optimizer(model, cfg)
    verifiers: dict[str, CompositeVerifier] = {}

    def verifier_for(task: str) -> CompositeVerifier:
        if task not in verifiers:
            verifiers[task] = CompositeVerifier(task=task)
        return verifiers[task]

    resume_step = 0
    if resume:
        resume_step = load_checkpoint(resume, model, opt)
        old.load_state_dict(model.state_dict())

    rolling = RollingCheckpointer(
        run_dir,
        interval_sec=cfg.rolling_ckpt_interval_sec,
        keep=cfg.rolling_ckpt_keep,
    )

    telemetry = make_writer(run_dir)
    telemetry.emit(
        "grpo_config",
        group_size=cfg.grpo.group_size,
        rollout_L=cfg.grpo.rollout_L,
        kl_beta=cfg.grpo.kl_beta,
        clip_eps=cfg.grpo.clip_eps,
        task=cfg.grpo.task,
        reward_model=bool(reward_model),
        reward_alpha=cfg.grpo.reward_alpha,
        verifier_beta=cfg.grpo.verifier_beta,
    )

    pad_id = int(tok.pad_token_id)
    eos_id = int(tok.eos_token_id)
    t0 = time.time()

    for step in range(resume_step, cfg.total_steps):
        row = prompts[step % len(prompts)]
        prompt_text = row["prompt"]
        reference = row["reference"]
        task = str(row.get("task") or cfg.grpo.task)

        prompt_ids = tok(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.model.max_seq_len // 2,
        ).input_ids[0]

        full_ids, resp_mask, resp_lens = rollout_group(
            old,
            prompt_ids,
            group_size=cfg.grpo.group_size,
            L=cfg.grpo.rollout_L,
            max_new_tokens=cfg.grpo.rollout_max_new_tokens,
            temperature=cfg.grpo.rollout_temperature,
            top_k=cfg.grpo.rollout_top_k,
            pad_id=pad_id,
            eos_id=eos_id,
        )

        prompt_len = prompt_ids.numel()
        responses: list[str] = []
        rewards = torch.zeros(cfg.grpo.group_size, device=device)
        verifier_rewards = torch.zeros_like(rewards)
        rm_rewards = torch.zeros_like(rewards)
        r_correct = torch.zeros_like(rewards)
        r_format = torch.zeros_like(rewards)
        breakdowns = []
        for i in range(cfg.grpo.group_size):
            resp_tokens = full_ids[i, prompt_len : prompt_len + int(resp_lens[i])].tolist()
            response = tok.decode(resp_tokens, skip_special_tokens=True)
            responses.append(response)
            breakdown = verifier_for(task).reward(
                prompt=prompt_text,
                response=response,
                reference=reference,
            )
            breakdowns.append(breakdown)
            verifier_rewards[i] = breakdown.verifier_total()
            r_correct[i] = breakdown.correct
            r_format[i] = breakdown.format

        if reward_model is not None:
            rm_scores = score_text_batch(
                reward_model,
                tok,
                [render_chat_text(prompt_text, response) for response in responses],
                L=cfg.grpo.rollout_L,
                device=device,
                max_length=cfg.model.max_seq_len,
            )
            rm_rewards.copy_(rm_scores.to(device))

        for i, breakdown in enumerate(breakdowns):
            rewards[i] = breakdown.total(
                reward_model_score=rm_rewards[i].item(),
                reward_alpha=cfg.grpo.reward_alpha if reward_model is not None else 0.0,
                verifier_beta=cfg.grpo.verifier_beta,
            )

        adv = group_advantage(rewards)

        input_ids = full_ids[:, :-1].contiguous()
        target_ids = full_ids[:, 1:].contiguous()
        action_mask = resp_mask[:, 1:].contiguous()

        logits_theta = model(input_ids, L=cfg.grpo.rollout_L).logits
        with torch.no_grad():
            logits_old = old(input_ids, L=cfg.grpo.rollout_L).logits
            logits_ref = ref(input_ids, L=cfg.grpo.rollout_L).logits

        out: GRPOOutput = grpo_loss(
            logits_theta=logits_theta,
            logits_old=logits_old,
            logits_ref=logits_ref,
            actions=target_ids,
            response_mask=action_mask,
            advantages=adv,
            clip_eps=cfg.grpo.clip_eps,
            kl_beta=cfg.grpo.kl_beta,
        )

        out.loss.backward()
        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % max(1, cfg.grpo.mu_steps) == 0:
            old.load_state_dict(model.state_dict())

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"step {step:6d} | lr {lr:.2e} | "
                f"loss {out.loss.item():.4f} | pol {out.policy_loss.item():.4f} | "
                f"kl {out.kl.item():.4f} | clip {out.clip_frac.item():.3f} | "
                f"|A| {out.adv_abs_mean.item():.3f} | "
                f"R {rewards.mean().item():.3f}±{rewards.std(unbiased=False).item():.3f} | "
                f"V {verifier_rewards.mean().item():.3f} | "
                f"RM {rm_rewards.mean().item():.3f} | "
                f"corr {r_correct.mean().item():.2f} fmt {r_format.mean().item():.2f} | "
                f"{elapsed:.0f}s"
            )
            telemetry.emit(
                "grpo_step",
                step=step,
                lr=lr,
                loss=out.loss.item(),
                policy_loss=out.policy_loss.item(),
                kl=out.kl.item(),
                clip_frac=out.clip_frac.item(),
                adv_abs_mean=out.adv_abs_mean.item(),
                reward_mean=rewards.mean().item(),
                reward_std=rewards.std(unbiased=False).item(),
                verifier_reward_mean=verifier_rewards.mean().item(),
                reward_model_mean=rm_rewards.mean().item(),
                correct_rate=r_correct.mean().item(),
                format_rate=r_format.mean().item(),
                prompt_task=task,
            )

        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            save_checkpoint(model, opt, cfg, step, run_dir)
            telemetry.emit("checkpoint", kind="milestone", step=step)

        if rolling.maybe_save(model, opt, cfg, step):
            telemetry.emit(
                "checkpoint",
                kind="rolling",
                step=step,
                slot=(rolling.next_slot - 1) % rolling.keep,
            )

    save_checkpoint(model, opt, cfg, cfg.total_steps, run_dir)
    telemetry.emit("checkpoint", kind="final", step=cfg.total_steps)
    if offload_store is not None:
        offload_store.flush()
    telemetry.close()
    print(f"grpo done in {(time.time()-t0)/60:.1f} min")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()

    cfg = load_train_config(args.config)
    for kv in args.override:
        key, _, value = kv.partition("=")
        if not key or not hasattr(cfg, key):
            print(f"[warn] bad override: {kv}", file=sys.stderr)
            continue
        current = getattr(cfg, key)
        caster = type(current) if current is not None else str
        try:
            setattr(cfg, key, caster(value))
        except (TypeError, ValueError):
            pass

    train_grpo(cfg, resume=args.resume)


if __name__ == "__main__":
    cli()
