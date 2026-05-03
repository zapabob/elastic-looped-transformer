---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated
tags:
  - elastic-looped-transformer
  - ilsd
  - grpo
  - self-distillation
  - synthetic-data
---

# elastic-looped-transformer

![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![python](https://img.shields.io/badge/python-3.12-blue.svg)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![tests](https://img.shields.io/badge/tests-105%2F105%20passing-brightgreen.svg)
![scales](https://img.shields.io/badge/configs-10M%20%7C%20100M%20%7C%20300M%20%7C%201B-informational.svg)
![effective depth](https://img.shields.io/badge/effective%20depth-up%20to%20112--layer%20at%20L%3D4-success.svg)

> **TL;DR** — A causal LM whose Transformer layers are **weight-shared across L
> iterations**. Pick `L ∈ [1, 4]` per request at inference to trade quality for
> latency — from the **same checkpoint**. Trained with **Intra-Loop
> Self-Distillation** + **GRPO** with a `correct × format` verifier. Scales
> shipped from 10M to **1 B non-embedding**, runnable on a single **RTX 3060 12 GB**
> via 8-bit paged or fp32-on-NVMe optimizer state. Faithful PyTorch port of
> **[arXiv:2604.09168](https://arxiv.org/abs/2604.09168)**.

---

## What's in the box

- **ELT core** (`src/elt_lm/`) — N shared Transformer layers iterated L times at
  inference. Paper equations preserved verbatim in code.
- **ILSD** — Intra-Loop Self-Distillation (`loss = L_GT(T) + λ L_GT(S) + (1−λ) L_dist(S, sg T)`)
  with `L_int ∼ U(L_min, L_max)` student, linear λ decay from 1 → 0.
- **GRPO** — DeepSeekMath §4.1 post-training with clipped surrogate + unbiased
  KL to frozen SFT reference. Verifier is `correct · format` with length +
  repeat guards. Python-exec verifier for code tasks.
- **Memory stack for 1 B on 12 GB VRAM** — two optimizer back-ends:
  - `paged_adamw_8bit` (bitsandbytes) — **peak 7.88 GB VRAM** on the 1 B config, fast.
  - `nvme_adamw` — custom 4-tier store with fp32 optimizer state **memory-mapped
    on NVMe**; params stay on GPU, state round-trips CPU→NVMe each step.
- **Rolling 5-minute checkpoints** — round-robin `rolling_{0..keep-1}.pt` +
  `last.pt` hardlink + CPU/CUDA RNG state → bit-reproducible resume.
- **HuggingFace Hub export** — `trust_remote_code=True` bundle (model code +
  weights + tokenizer + rendered README in one directory).
- **Streamlit dashboard** — live panels for pipeline / training / storage tiers
  / hardware / inference Pareto / checkpoints, fed by a line-buffered JSONL
  telemetry writer.

## Quickstart (use a published checkpoint)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok   = AutoTokenizer.from_pretrained("zapabob/elt-lm-base-275m")
model = AutoModelForCausalLM.from_pretrained(
    "zapabob/elt-lm-base-275m",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).eval().cuda()

ids = tok("If 3x + 7 = 22, what is x? Think step by step.",
          return_tensors="pt").input_ids.cuda()

# Same checkpoint, user picks L per call.
for L in (1, 2, 3, 4):
    out = model.generate(ids, max_new_tokens=128, L=L, do_sample=False)
    print(f"L={L}: {tok.decode(out[0], skip_special_tokens=True)}")
```

## Architecture

```
input_ids ──embed──► x₀ ──g_Θ──► x₁ ──g_Θ──► x₂ ── ... ──g_Θ──► x_L ──norm + lm_head──► logits
                       └─── weights SHARED across every iteration ───┘
```

| eq. | where | what |
|---|---|---|
| `g_Θ(x) = f_{θ_N} ∘ … ∘ f_{θ_1}(x)` | `src/elt_lm/composite.py` | composite block, N unique layers |
| `F_{N,L}(x) = g_Θ^L(x)` | `src/elt_lm/model.py` | L-fold iteration |
| `L_ILSD = L_GT(T) + λ L_GT(S) + (1−λ) L_dist(S, sg T)` | `src/elt_lm/losses.py` | intra-loop distillation (paper eq. 3) |
| `L_int ∼ U(L_min, L_max)` | `src/elt_lm/train.py` | stochastic student L |
| `λ: 1 → 0` (linear) | `src/elt_lm/train.py` | distillation curriculum |

## Shipped scales

| config | d_model | N | non-emb | total | effective (L=4) | target hardware |
|---|---|---|---|---|---|---|
| `tiny_10M.yaml` | 256 | 4 | 3.5 M | ~67 M | 16-layer | CPU smoke |
| `base_100M.yaml` | 768 | 12 | **85 M** | 275 M | 48-layer | 12 GB GPU, fp32 Adam |
| `smoke_300M.yaml` | 1024 | 16 | 205 M | 460 M | 64-layer | NvmeAdamW validation |
| `base_1B.yaml` | 1792 | 28 | **1.09 B** | 1.54 B | **112-layer** | 12 GB GPU w/ PagedAdamW8bit |

Token embedding (248 K × d_model) dominates the parameter count at smaller
scales; the interesting number is **non-emb**, which is what gets iterated.

## ILSD stability objective

ELT is evaluated as a loop-wise refinement system, not only as a small dense LM.
The teacher is the deepest loop and the student is an intermediate loop:

```text
z_T = logits at L_T = L_max
z_S = logits at L_S ~ U(L_min, L_max)
p_T = stopgrad(softmax(z_T / tau_T))
p_S = softmax(z_S)
L_ILSD = L_GT(T) + lambda L_GT(S) + (1 - lambda) CE(p_T, p_S)
```

The `stopgrad` on `p_T` is intentional. It prevents the deepest loop from being
pulled around by the student during self-distillation, keeping the maximum-loop
path as the local teacher. The current stabilizer stack keeps teacher-only
temperature and masked soft CE, then adds entropy/loop-trajectory regularizers
so additional loops refine rather than collapse:

- **teacher-only temperature** smooths the teacher target without hiding the
  student's actual sharpness.
- **entropy floor** penalizes low-entropy collapse when the model becomes
  confidently wrong.
- **Delta^2 entropy curvature** penalizes abrupt entropy bends along the loop
  axis `L`, matching the ELT idea of incremental refinement.
- **sampled Delta^2 logit curvature** can be enabled after entropy metrics are
  stable, using sampled/top-k vocab slices instead of full-vocab curvature.

The important design choice is that safety and capability alignment are handled
mostly by data selection, lane verifiers, KL-constrained GRPO, and evaluation
rather than by blanket refusal behavior baked into the base model.

## Anytime loop evaluation

The key experimental question is not merely whether `L=4` scores higher than
`L=1`, but whether deeper loops correct shallow mistakes without overthinking
correct answers. `elt-anytime` now emits benchmark refinement telemetry:

```text
loop_gain(L=k)       = score(L=k) - score(L=1)
marginal_gain(L=k)   = score(L=k) - score(L=k-1)
self_correction_rate = count(L=1 wrong and L=k correct) / N
overthinking_rate    = count(L=1 correct and L=k wrong) / N
```

For each benchmark, track these alongside per-loop accuracy, entropy trajectory,
latency/token, tokens/sec, and VRAM. A healthy ELT run should increase
self-correction faster than overthinking as `L` grows.

## 1 B training on a 12 GB card

```bash
uv run elt-train --config configs/base_1B.yaml
```

With `optim.kind: paged_adamw_8bit`:

| measure | value |
|---|---|
| model params | 1.537 B total, 1.092 B non-emb |
| **peak VRAM** | **7.88 GB** |
| one-step smoke | ~5.0 s (incl. cuDNN warm-up) |

Alternative — NVMe-backed fp32 state (`optim.kind: nvme_adamw`):

```yaml
# configs/your_run.yaml
optim:
  kind: nvme_adamw
offload:
  enabled: true
  root: H:/elt_data/offload_nvme  # where to mmap fp32 state shards
  min_free_gb: 20.0               # refuse to start if less
```

Measured on `smoke_300M.yaml` × NvmeAdamW, RTX 3060:

| measure | value |
|---|---|
| params | 0.46 B total, 0.21 B non-emb |
| peak VRAM | 4.38 GB |
| step (fwd + bwd + NvmeAdamW.step) | 128.7 s |

VRAM drops further, but NVMe bandwidth becomes the bottleneck — use
`nvme_adamw` only when VRAM is the hard constraint.

## Training pipeline

Three phases, resumable, driven by one orchestrator:

| stage | config | what it does |
|---|---|---|
| **Phase 1** — Pretrain | `configs/base_100M.yaml` / `configs/base_1B.yaml` | ILSD with warmup-then-anneal λ, bf16 + grad-ckpt + grad-accum |
| **Phase 2** — SFT | `configs/sft_cot.yaml` | CoT instruction + offline distillation |
| **Phase 3** — GRPO | `configs/grpo_gsm8k.yaml` | clipped surrogate + unbiased KL, `correct × format` verifier |

```bash
# End-to-end 11-stage pipeline (respects .done markers)
uv run python scripts/pipeline.py

# Register as Windows startup task — auto-resumes on every boot, removes
# itself from Task Scheduler once the final stage is done.
powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1
```

## Training data provenance

The current repository includes a redistributable snapshot of the active
synthetic-v2-hard training/evaluation data under `training_data/synthetic_v2_hard/`.
That snapshot contains verifier-backed SFT traces, intentionally wrong contrast
traces, and held-out GRPO/bridge prompts for code, math, STEM reasoning, and
tool-use lanes.

Source and citation metadata is tracked in:

- `training_data/DATA_SOURCES.md`
- `training_data/source_citations.yaml`
- `scripts/download_hf_corpus.py`
- `scripts/corpus_manifest.yaml`

The large tokenized `*.bin` files under `H:/elt_data/*` are generated artifacts
and are not committed. For model releases, cite the exact public datasets listed
in `training_data/source_citations.yaml`, plus this repository commit for the
synthetic-v2-hard generated data. The loop/self-distillation method follows
ELT / ILSD ([arXiv:2604.09168](https://arxiv.org/abs/2604.09168)); GRPO follows
DeepSeekMath ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)).

## Dashboard

```bash
uv sync --extra dashboard
uv run streamlit run dashboard/app.py
# → http://localhost:8501
```

Panels:

- **Pipeline** — `.done` markers + tail of `pipeline.jsonl`
- **Training** — loss / lr / grad-norm / tok-per-sec, λ curve, L_int histogram
- **Storage tiers** — NVMe MB/s, prefetch hit rate, per-layer compute tier
- **Hardware** — VRAM (NVML), CPU/RAM (psutil), C:/H: free
- **Inference Pareto** — L vs. quality / latency / tok-per-sec (from `inference_sweep`)
- **Checkpoints** — rolling slot, age, disk usage

## HuggingFace Hub export

```bash
uv run python scripts/export_to_hf.py \
  --ckpt      runs/grpo_gsm8k/last.pt \
  --out       hf_export/elt-lm-base-275m \
  --tokenizer H:/Qwen3.5-9B-official-hf \
  --repo-id   zapabob/elt-lm-base-275m \
  --push-to-hub
```

Bundles `configuration_elt.py`, `modeling_elt.py`, `config.json`,
`model.safetensors`, tokenizer files, and a rendered `README.md`. Downstream
users only need `pip install transformers`.

For the Qwen3.5 side-LoRA bridge runs, export the adapter payload separately:

```bash
uv run elt-export-lora-adapter \
  --ckpt H:/elt_data/runs/grpo_side_lora_stem_synthetic_v2_bridge/last.pt \
  --out-dir H:/elt_data/adapters/qwen35_4b_side/synthetic_stem_v2_bridge_grpo_candidate
```

This now writes both local-runtime `adapter.pt` and portable
`adapter_model.safetensors` plus `adapter_config.json` and a minimal model card.
The 2026-05-03 stem bridge candidate has been exported at
`H:/elt_data/adapters/qwen35_4b_side/synthetic_stem_v2_bridge_grpo_candidate`
(`adapter_model.safetensors`, 64,987,976 bytes).

GGUF release readiness follows the current llama.cpp path: first produce a
Transformers/HF directory with `config.json`, tokenizer files, and safetensors,
then run `convert_hf_to_gguf.py`, then upload the resulting `.gguf` to a
separate `*-GGUF` model repo. The repo helper records the exact commands and
blockers:

```bash
uv run python -m elt_lm.release_readiness \
  --hf-dir hf_export/elt-lm-qwen35-side-stem-v2-bridge \
  --gguf-path H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge.gguf \
  --repo-id zapabob/elt-lm-qwen35-side-stem-v2-bridge \
  --llama-cpp-dir C:/Users/downl/Desktop/llama.cpp-zapabob \
  --out _docs/assets/2026-05-03-deepresearch-elt-llm-implementation/release_readiness_stem_bridge.json
```

Current status: adapter safetensors are ready; merged/full HF safetensors and
GGUF are not yet claimed ready because the side-LoRA bridge has not been merged
into a HF-loadable base directory for llama.cpp conversion.

## Cross-validated benchmark comparison

`elt-anytime` already emits case-level correctness and K-fold accuracy summaries
for local verifier-backed benchmark manifests. For vanilla-vs-finished model
comparison, preserve paired case/fold order and run:

```bash
uv run python -m elt_lm.eval.benchmark_comparison \
  --input reports/vanilla_vs_complete_groups.json \
  --out-json reports/vanilla_vs_complete_stats.json \
  --out-md reports/vanilla_vs_complete_stats.md
```

Input schema:

```json
{
  "benchmark": "mmlu_stem_cv",
  "groups": {
    "vanilla": [0, 1, 0, 1],
    "sft_replay": [0, 1, 1, 1],
    "complete": [1, 1, 1, 1]
  }
}
```

The report includes mean, SD, SEM, 95% CI, paired permutation p-values for every
group pair, and a Friedman within-block permutation p-value when at least three
groups are supplied. Use lm-eval-harness for broad external tasks with logged
samples, e.g. `lm-eval run --model hf --model_args pretrained=<hf_export_dir>
--tasks gsm8k,mmlu_stem,hellaswag --output_path <dir> --log_samples`, then
convert the paired sample correctness arrays into the JSON schema above.

Current measured bridge diagnostics are limited to internal synthetic-v2 bridge
verifiers, not broad lm-eval claims: stem is the only export/eval candidate
(mean correct 0.8958, final correct 1.0), code and math are sparse-success
lanes, and tool-use is blocked because reward/advantage signal remained zero.
Full vanilla-vs-complete lm-eval p-values should not be reported until both
groups have completed the same paired task set.

## Rolling checkpoints

- `rolling_{0..keep-1}.pt` round-robin every `rolling_ckpt_interval_sec` (5 min default)
- `last.pt` hardlinked to the latest save — resume anchor
- `step_*.pt` milestone saves every `save_every`
- CPU + CUDA RNG state in each save → deterministic resume

Crash loses at most one interval; `--resume runs/<dir>/last.pt` picks up.

## Repo layout

```
src/elt_lm/          model, layers, losses, train loops, HF wrapper
src/elt_lm/offload/  4-tier store, NvmeAdamW, prefetcher, placement planner
src/elt_lm/hf/       trust_remote_code bundle (ELTConfig, ELTForCausalLM)
src/elt_lm/eval/     any-time L-sweep, verifiers, python-exec guard
src/elt_lm/telemetry.py  thread-safe JSONL writer
dashboard/           Streamlit app + panels + metrics reader
configs/             tiny_10M / base_100M / smoke_300M / base_1B / sft_cot / grpo_gsm8k
scripts/             data DL / clean / tokenize / pipeline / HF export / 1B VRAM smoke
tests/               105 tests; `uv run pytest -q`
_docs/               implementation log (YYYY-MM-DD-<slug>-<AI>.md)
```

## Install

```bash
uv sync                             # core
uv sync --extra offload_8bit        # + bitsandbytes for paged_adamw_8bit
uv sync --extra dashboard           # + streamlit / plotly / pynvml / psutil
uv sync --extra dev                 # + pytest, for running the suite
uv run pytest -q                    # 105 passing
```

## Roadmap

- [x] ELT + ILSD scaffold, paper equations faithful
- [x] GRPO post-training with verifier (DeepSeekMath §4.1)
- [x] Rolling checkpoints + deterministic resume
- [x] HuggingFace Hub export (`trust_remote_code=True`)
- [x] End-to-end pipeline with boot-time auto-resume
- [x] Offline distillation from Qwen3.5-4B
- [x] `base_1B.yaml` fits 12 GB via PagedAdamW8bit (measured peak 7.88 GB)
- [x] Hypura-style 4-tier NVMe offload (`NvmeAdamW`) + placement planner
- [x] Streamlit dashboard with 6 live panels + JSONL telemetry
- [ ] 1 B Phase-1 pretrain run (in progress)
- [ ] GSM8K / HumanEval / MMLU-STEM / MATH-500 L-sweep results
- [ ] `elt-lm-base-1.5b` pushed to HuggingFace Hub

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
