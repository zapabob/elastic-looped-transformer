"""GRPO post-training loop for the ELT causal-LM.

Pipeline per optimizer step:
  1. Sample a prompt `q` from the prompts JSONL.
  2. With π_θ_old (= frozen snapshot of current policy), generate G rollouts
     {o_1, ..., o_G} by autoregressive sampling.
  3. Score each rollout with CompositeVerifier → scalar rewards (anti-hack gate
     inside verifier: correct · format + penalties).
  4. Compute group-relative advantages Â_i = (R_i - mean) / (std + ε).
  5. Build a (G, T) batch of (prompt ∥ response) tokens + response_mask.
  6. Forward π_θ, π_θ_old, π_ref at ELT L = rollout_L → logits.
  7. Compute grpo_loss (clipped surrogate + β · KL vs π_ref).
  8. Backprop + AdamW step.

π_ref is frozen (SFT init); π_θ_old is resynced from π_θ every `mu_steps` updates
(DeepSeek μ=1 == fully on-policy, our default).

CLI:
    uv run python -m elt_lm.train_grpo --config configs/grpo_gsm8k.yaml
"""

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
from elt_lm.model import ELTLanguageModel
from elt_lm.train import configure_optimizer, get_dtype, lr_at, save_checkpoint
from elt_lm.verifiers import CompositeVerifier


# ---------------------------------------------------------------------------
# prompt source
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(path: str):
    from transformers import AutoTokenizer  # lazy — only needed for GRPO
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# rollout: (G, T) padded response tokens + mask
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_group(
    model: ELTLanguageModel,
    prompt_ids: Tensor,           # (P,) int64
    group_size: int,
    L: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    pad_id: int,
    eos_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (full_ids, response_mask, response_lens).

    full_ids:      (G, P+T_resp) padded with `pad_id`
    response_mask: (G, P+T_resp) 1 for generated tokens, 0 otherwise
    response_lens: (G,) number of non-pad response tokens per sample
    """
    device = next(model.parameters()).device
    model.eval()
    batch = prompt_ids.unsqueeze(0).repeat(group_size, 1).to(device=device)
    P = batch.shape[1]

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
        nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)        # (G,)
        # freeze any sequence that already emitted EOS
        nxt = torch.where(eos_hit, torch.full_like(nxt, pad_id), nxt)
        for i in range(group_size):
            if not eos_hit[i]:
                produced[i].append(int(nxt[i].item()))
        batch = torch.cat([batch, nxt.unsqueeze(-1)], dim=-1)
        eos_hit |= (nxt == eos_id)
        if eos_hit.all():
            break

    # pack into fixed-width response segment
    T_resp = max((len(p) for p in produced), default=1)
    resp = torch.full((group_size, T_resp), pad_id, dtype=torch.long, device=device)
    resp_lens = torch.zeros(group_size, dtype=torch.long, device=device)
    for i, p in enumerate(produced):
        if p:
            resp[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)
            resp_lens[i] = len(p)

    prompt_block = prompt_ids.unsqueeze(0).repeat(group_size, 1).to(device=device)
    full = torch.cat([prompt_block, resp], dim=-1)

    # mask is 1 only on real response positions (not pad, not prompt)
    resp_mask = torch.zeros_like(full, dtype=torch.long)
    for i in range(group_size):
        resp_mask[i, P : P + int(resp_lens[i])] = 1

    return full, resp_mask, resp_lens


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def train_grpo(cfg: TrainConfig) -> None:
    assert cfg.grpo.enabled, "grpo.enabled must be true for train_grpo"
    assert cfg.grpo.init_ckpt, "grpo.init_ckpt must point at the SFT checkpoint"
    assert cfg.grpo.prompts_file, "grpo.prompts_file must point at prompts JSONL"

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(cfg.dtype)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokenizer + prompts ---------------------------------------------
    tok = load_tokenizer(cfg.data.tokenizer_path)
    prompts = load_prompts(cfg.grpo.prompts_file)
    print(f"loaded {len(prompts):,} prompts from {cfg.grpo.prompts_file}")

    # ---- models: π_θ (trainable), π_θ_old (snapshot), π_ref (frozen) -----
    state = torch.load(cfg.grpo.init_ckpt, map_location="cpu")
    model = ELTLanguageModel(cfg.model).to(device=device, dtype=dtype)
    model.load_state_dict(state["model"] if "model" in state else state)

    # π_ref: deep copy, eval-only, no grad
    ref = copy.deepcopy(model).to(device=device, dtype=dtype)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    # π_θ_old: snapshot that gets resynced every mu_steps
    old = copy.deepcopy(model).to(device=device, dtype=dtype)
    for p in old.parameters():
        p.requires_grad_(False)
    old.eval()

    model.train()
    print(f"π_θ params: {model.num_parameters()/1e6:.1f}M — ref/old frozen")

    opt = configure_optimizer(model, cfg)
    verifier = CompositeVerifier(task=cfg.grpo.task)

    pad_id = int(tok.pad_token_id)
    eos_id = int(tok.eos_token_id)

    # ---- loop ------------------------------------------------------------
    t0 = time.time()
    for step in range(cfg.total_steps):
        row = prompts[step % len(prompts)]
        prompt_text = row["prompt"]
        reference = row["reference"]

        prompt_ids = tok(prompt_text, return_tensors="pt", truncation=True,
                         max_length=cfg.model.max_seq_len // 2).input_ids[0]

        # 1. rollouts (behavior = old)
        full_ids, resp_mask, resp_lens = rollout_group(
            old, prompt_ids,
            group_size=cfg.grpo.group_size,
            L=cfg.grpo.rollout_L,
            max_new_tokens=cfg.grpo.rollout_max_new_tokens,
            temperature=cfg.grpo.rollout_temperature,
            top_k=cfg.grpo.rollout_top_k,
            pad_id=pad_id,
            eos_id=eos_id,
        )

        # 2. decode + reward
        P = prompt_ids.numel()
        rewards = torch.zeros(cfg.grpo.group_size, device=device)
        r_correct = torch.zeros_like(rewards)
        r_format = torch.zeros_like(rewards)
        for i in range(cfg.grpo.group_size):
            resp_tokens = full_ids[i, P : P + int(resp_lens[i])].tolist()
            response = tok.decode(resp_tokens, skip_special_tokens=True)
            r = verifier.reward(
                prompt=prompt_text, response=response, reference=reference,
            )
            rewards[i] = r.total()
            r_correct[i] = r.correct
            r_format[i] = r.format

        # 3. advantage
        adv = group_advantage(rewards)

        # 4. teacher-forced logits for π_θ, π_θ_old, π_ref
        input_ids = full_ids[:, :-1].contiguous()
        target_ids = full_ids[:, 1:].contiguous()
        action_mask = resp_mask[:, 1:].contiguous()

        logits_theta = model(input_ids, L=cfg.grpo.rollout_L).logits
        with torch.no_grad():
            logits_old = old(input_ids, L=cfg.grpo.rollout_L).logits
            logits_ref = ref(input_ids, L=cfg.grpo.rollout_L).logits

        # 5. GRPO loss
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

        # 6. backprop
        (out.loss).backward()
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        # 7. resync π_θ_old
        if (step + 1) % max(1, cfg.grpo.mu_steps) == 0:
            old.load_state_dict(model.state_dict())

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"step {step:6d} | lr {lr:.2e} | "
                f"loss {out.loss.item():.4f} | pol {out.policy_loss.item():.4f} | "
                f"kl {out.kl.item():.4f} | clip {out.clip_frac.item():.3f} | "
                f"|Â| {out.adv_abs_mean.item():.3f} | "
                f"R {rewards.mean().item():.3f}±{rewards.std(unbiased=False).item():.3f} | "
                f"corr {r_correct.mean().item():.2f} fmt {r_format.mean().item():.2f} | "
                f"{elapsed:.0f}s"
            )

        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            save_checkpoint(model, opt, cfg, step, run_dir)

    save_checkpoint(model, opt, cfg, cfg.total_steps, run_dir)
    print(f"grpo done in {(time.time()-t0)/60:.1f} min")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
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

    train_grpo(cfg)


if __name__ == "__main__":
    cli()
