---
license: apache-2.0
language:
- ja
- en
library_name: transformers
tags:
- elastic-looped-transformer
- causal-lm
- reasoning
- weight-sharing
- anytime-inference
pipeline_tag: text-generation
---

# {repo_id}

**Elastic Looped Transformer (ELT) + Intra-Loop Self-Distillation (ILSD) + GRPO**,
causal-LM port of [arXiv:2604.09168](https://arxiv.org/abs/2604.09168).

## TL;DR

- **{total_m:.1f} M parameters** total (~**{nonemb_m:.1f} M** without the {vocab_k:,}-token embedding).
- **{n_unique_layers} unique Transformer layers** iterated up to **L={L_max}** times
  at inference → **{eff_depth}-layer-equivalent compute** with only {n_unique_layers}-layer memory.
- **Any-Time inference**: choose `L ∈ [{L_min}, {L_max}]` at call time — higher L = better quality, more FLOPs.
- Tokenizer: **Qwen3.5** (vocab {vocab_k:,}), supports Japanese, English, code, and math out of the box.

## Architecture

```
input_ids ──embed──► x₀ ──g_Θ──► x₁ ──g_Θ──► x₂ ── ... ──g_Θ──► x_L ──norm + lm_head──► logits
                       (composite block, N={n_unique_layers} unique layers, weights SHARED across iterations)
```

Paper equations preserved verbatim in code (`src/elt_lm/`):
- `g_Θ(x) = f_{{θ_N}} ∘ … ∘ f_{{θ_1}}(x)` (eq. 1)
- `F_{{N,L}}(x) = g_Θ^L(x)` (eq. 2)
- `L_ILSD = L_GT(teacher) + λ · L_GT(student) + (1−λ) · L_dist(student, sg(teacher))` (eq. 3)
- `L_int ~ U(L_min, L_max)` stochastic student sampling
- `λ` linear curriculum 1 → 0

## ILSD stability objective

This checkpoint family treats ELT as a loop-wise refinement system. The deepest
loop is the local teacher and an intermediate loop is the student:

```text
z_T = logits at L_T = L_max
z_S = logits at L_S ~ U(L_min, L_max)
p_T = stopgrad(softmax(z_T / tau_T))
p_S = softmax(z_S)
L_ILSD = L_GT(T) + lambda L_GT(S) + (1 - lambda) CE(p_T, p_S)
```

The stop-gradient teacher is intentional: it keeps the maximum-loop path from
being pulled around by the sampled student loop. Teacher-only temperature,
masked soft CE, entropy-floor regularization, Delta^2 entropy curvature, and
sampled Delta^2 logit curvature are used as stabilizers so larger L refines
instead of simply sharpening into collapse.

## Anytime loop evaluation

The key question is whether deeper loops repair shallow mistakes without
overthinking correct answers. `elt-anytime` benchmark telemetry includes:

```text
loop_gain(L=k)       = score(L=k) - score(L=1)
marginal_gain(L=k)   = score(L=k) - score(L=k-1)
self_correction_rate = count(L=1 wrong and L=k correct) / N
overthinking_rate    = count(L=1 correct and L=k wrong) / N
```

Track these with per-loop accuracy, entropy trajectory, latency/token,
tokens/sec, and VRAM before claiming test-time scaling.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}", trust_remote_code=True, torch_dtype=torch.bfloat16,
).eval().cuda()

prompt = "Solve: If 3x + 7 = 22, what is x?\nThink step by step."
ids = tok(prompt, return_tensors="pt").input_ids.cuda()

# Choose L at inference (higher = better, slower):
out = model.generate(ids, max_new_tokens=256, L=4, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

Any-Time sweep:

```python
for L in (1, 2, 3, 4):
    out = model.generate(ids, max_new_tokens=128, L=L)
    print(f"L={{L}}: {{tok.decode(out[0], skip_special_tokens=True)}}")
```

## Training pipeline

1. **Phase 1 — pretraining**: ~{pretrain_tokens_b:.1f} B tokens of Japanese + English Wikipedia,
   MetaMathQA, GSM8K, Magicoder, OpenMathInstruct-2, OpenWebMath, FineMath, Cosmopedia,
   OpenCodeReasoning, OpenCodeInstruct, etc. ILSD enabled after a 2000-step GT-only warmup.
2. **Phase 2 — SFT**: Chain-of-Thought instruction tuning on the same ILSD objective.
3. **Phase 3 — GRPO** (DeepSeekMath §4.1): group-relative advantage + clipped surrogate +
   unbiased KL vs. the SFT reference policy. Verifier = `correct · format` multiplicative
   gate + length/repeat penalties, closing the reward-hacking surface.

## Intended use

- Reasoning (math, coding, multi-step science / tool use) in Japanese and English.
- Experiments on Any-Time inference trade-offs.

## Limitations

- Small model (~100 M non-embedding). Not a chatbot substitute for large LLMs.
- Knowledge cutoff = training corpus cutoff (2024-2025 range).
- No guardrails beyond standard pretraining data filtering. Validate outputs before deployment.

## Citation

Underlying architecture:

```
@article{{goyal2026elt,
  title={{Elastic Looped Transformers for Visual Generation}},
  author={{Goyal et al.}},
  journal={{arXiv:2604.09168}},
  year={{2026}}
}}
```

GRPO objective:

```
@article{{shao2024deepseekmath,
  title={{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
  author={{Shao et al.}},
  journal={{arXiv:2402.03300}},
  year={{2024}}
}}
```

## License

Apache 2.0 (model weights + code). Tokenizer is inherited from Qwen3.5 — see the
upstream tokenizer repo for its terms.
