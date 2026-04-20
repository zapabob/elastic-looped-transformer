# elastic-looped-transformer

![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![python](https://img.shields.io/badge/python-3.12-blue.svg)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![tests](https://img.shields.io/badge/tests-59%2F59%20passing-brightgreen.svg)
![params](https://img.shields.io/badge/non--emb%20params-85M-informational.svg)
![effective depth](https://img.shields.io/badge/effective%20depth-48--layer%20at%20L%3D4-success.svg)

> **TL;DR** — An **85M non-embedding** causal LM that computes like a **340M** one.
> Twelve Transformer layers whose **weights are shared across L iterations** at
> inference — you pick `L ∈ [1, 4]` per request to trade quality for latency.
> Trained with **Intra-Loop Self-Distillation (ILSD)**, then **GRPO** with a
> `correct × format` verifier. Bilingual JA / EN, Qwen3.5 tokenizer (248K vocab).
> Faithful PyTorch port of **[arXiv:2604.09168](https://arxiv.org/abs/2604.09168)**
> with paper equations preserved verbatim.

---

## Why this is interesting

- **Thin-tall on a diet.** 12 unique layers × L=4 iterations ≈ **48-layer effective
  depth** on 12 layers of memory. You store a small model and compute a big one.
- **Any-time inference.** `model.generate(ids, L=2)` is fast. `L=4` thinks harder.
  Same checkpoint, user picks at call time. No retraining, no re-export.
- **Self-teaching.** ILSD (eq. 3) treats `L = L_max` as the teacher for a
  `L_int ∼ U(L_min, L_max)` student. Fixes the "short-loop output is noisy"
  failure mode without a separate teacher model.
- **GRPO post-training.** DeepSeekMath §4.1: group-relative advantage, clipped
  surrogate, unbiased KL vs. the SFT reference. Verifier is `correct · format`
  with length + repeat guards, so the policy cannot reward-hack by emitting
  boilerplate.
- **Rolling 5-minute checkpoints.** Round-robin `rolling_{0..2}.pt` + `last.pt`
  hardlink + full CPU/CUDA RNG state in the save, so a crash loses at most
  ~5 minutes of work and resumes are bit-reproducible.
- **HuggingFace Hub ready.** `trust_remote_code=True` export bundles the model
  code with the checkpoint — one directory, one `from_pretrained` call.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok   = AutoTokenizer.from_pretrained("zapabob/elt-lm-base-275m")
model = AutoModelForCausalLM.from_pretrained(
    "zapabob/elt-lm-base-275m",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).eval().cuda()

prompt = "If 3x + 7 = 22, what is x? Think step by step."
ids = tok(prompt, return_tensors="pt").input_ids.cuda()

out = model.generate(ids, max_new_tokens=256, L=4, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

Any-Time sweep — same checkpoint, different quality:

```python
for L in (1, 2, 3, 4):
    out = model.generate(ids, max_new_tokens=128, L=L, do_sample=False)
    print(f"L={L}: {tok.decode(out[0], skip_special_tokens=True)}")
```

## Architecture

```
input_ids ──embed──► x₀ ──g_Θ──► x₁ ──g_Θ──► x₂ ── ... ──g_Θ──► x_L ──norm + lm_head──► logits
                       └─── weights SHARED across every iteration ───┘
```

Paper equations preserved verbatim in code:

| eq. | where | what |
|---|---|---|
| `g_Θ(x) = f_{θ_N} ∘ … ∘ f_{θ_1}(x)` | `src/elt_lm/composite.py` | composite block (N unique layers) |
| `F_{N,L}(x) = g_Θ^L(x)` | `src/elt_lm/model.py` | L-fold iteration |
| `L_ILSD = L_GT(T) + λ L_GT(S) + (1−λ) L_dist(S, sg T)` | `src/elt_lm/losses.py` | intra-loop distillation |
| `L_int ∼ U(L_min, L_max)` | `src/elt_lm/train.py` | stochastic student L |
| `λ: 1 → 0` (linear) | `src/elt_lm/train.py` | distillation curriculum |

## Sizing (base_100M config)

| | total | non-embedding | effective compute (L=4) |
|---|---|---|---|
| base_100M | **275.7 M** | **85.0 M** | ≈ 340 M-class FLOPs |

Token embedding (248K × 768) dominates the parameter count — the model body is
only 85M. On-disk weight size: ~551 MB bf16 / ~1.1 GB fp32.

## Training pipeline

Three phases, all resumable, all driven by one orchestrator:

| stage | config | what it does |
|---|---|---|
| **Phase 1** — Pretrain | `configs/base_100M.yaml` | ILSD with warmup-then-anneal λ, bf16 + grad-ckpt + 32× accum |
| **Phase 2** — SFT | `configs/sft_cot.yaml` | CoT instruction + offline distillation from `huihui-ai/Huihui-Qwopus3.5-4B-v3-abliterated` |
| **Phase 3** — GRPO | `configs/grpo_gsm8k.yaml` | clipped surrogate + unbiased KL, `correct × format` verifier with length guards |

```bash
# Single-command end-to-end pipeline (11 stages, respects .done markers)
uv run python scripts/pipeline.py

# Register as Windows startup task — auto-resumes on every boot, deletes
# itself from Task Scheduler after the final stage completes.
powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1
```

## HuggingFace Hub export

```bash
uv run python scripts/export_to_hf.py \
  --ckpt      runs/grpo_gsm8k/last.pt \
  --out       hf_export/elt-lm-base-275m \
  --tokenizer H:/Qwen3.5-9B-official-hf \
  --repo-id   zapabob/elt-lm-base-275m \
  --push-to-hub
```

The export bundles `configuration_elt.py`, `modeling_elt.py`, `config.json`,
`model.safetensors`, tokenizer files, and a rendered `README.md`. Downstream
users need nothing beyond `pip install transformers`.

## Rolling checkpoints

Every training loop runs a `RollingCheckpointer`:

- `rolling_{0, 1, 2}.pt` round-robin every `rolling_ckpt_interval_sec` (5 min)
- `last.pt` hardlinked to the latest save — resume-friendly anchor
- `step_*.pt` milestone saves every `save_every`
- RNG state (CPU + CUDA) in each save → deterministic resume

Crash and lose at most one interval; `--resume runs/<dir>/last.pt` picks up.

## Repo layout

```
src/elt_lm/          # model, layers, losses, train loops, HF wrapper
src/elt_lm/hf/       # trust_remote_code bundle (ELTConfig, ELTForCausalLM)
src/elt_lm/eval/     # any-time L-sweep, verifiers, python-exec guard
configs/             # tiny_10M, base_100M, sft_cot, grpo_gsm8k
scripts/             # data DL / clean / tokenize / pipeline / HF export
tests/               # 59 tests, run with `uv run pytest -q`
_docs/               # implementation log (YYYY-MM-DD-<slug>-<AI>.md)
```

## Install

```bash
uv sync          # Python 3.12 + PyTorch 2.x + transformers + huggingface-hub
uv run pytest -q # 59 passing
```

## Roadmap

- [x] ELT + ILSD scaffold, paper equations faithful
- [x] GRPO post-training with verifier (DeepSeekMath §4.1)
- [x] Rolling checkpoints + deterministic resume
- [x] HuggingFace Hub export (`trust_remote_code=True`)
- [x] End-to-end pipeline with boot-time auto-resume
- [x] Offline distillation from Qwen3.5-4B
- [ ] 10B-token Phase 1 pretrain run (in progress)
- [ ] GSM8K / HumanEval / MMLU-STEM / MATH-500 L-sweep results
- [ ] `elt-lm-base-275m` pushed to HuggingFace Hub

## Citation

```
@article{goyal2026elt,
  title   = {Elastic Looped Transformers for Visual Generation},
  author  = {Goyal et al.},
  journal = {arXiv:2604.09168},
  year    = {2026}
}
@article{shao2024deepseekmath,
  title   = {DeepSeekMath: Pushing the Limits of Mathematical Reasoning
             in Open Language Models},
  author  = {Shao et al.},
  journal = {arXiv:2402.03300},
  year    = {2024}
}
```

## License

Apache 2.0 (model weights + code). Tokenizer inherits from Qwen3.5 — see the
upstream repo for its terms.

---

If you find this useful, a star helps others discover it.
