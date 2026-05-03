# 2026-05-03 - quantization-lane-separation - gpt-5

## Goal

Integrate the DeepResearch update into the public ELT quantization framing:
separate GGUF weight compression, calibration/imatrix, tensor protection,
KV-cache compression, TurboQuant-style KV, and DFlash speculative decoding so
the README does not overclaim from the current `TQ4_1S` weight artifact.

## Files touched

- `README.md`
- `src/elt_lm/eval/gguf_quant_report.py`
- `tests/test_gguf_quant_report.py`
- `_docs/assets/2026-05-03-gguf-quant-cv-gptimage/gguf_quant_cv_report.json`
- `_docs/assets/2026-05-03-gguf-quant-cv-gptimage/gguf_quant_cv_report.md`
- `_docs/assets/2026-05-03-gguf-quant-cv-gptimage/gptimage_gguf_quant_cv_infographic.png`
- `_docs/2026-05-03-quantization-lane-separation-gpt-5.md`

## Key decisions

- Treat `TQ4_1S` as a compact local GGUF weight-compression artifact, not as
  evidence for Google TurboQuant KV-cache serving performance.
- Keep TurboQuant-style KV cache work in a separate runtime lane. As of
  2026-05-03, ggml-org/llama.cpp PR #21089 is open and scoped to CPU-only
  TBQ3/TBQ4 KV-cache types, with CUDA backend support listed as follow-up work.
- Keep DFlash in a speculative-decoding lane. It uses a block diffusion draft
  model plus target verification, so it should be evaluated with target
  equivalence, acceptance length/rate, and loop-depth stability rather than as
  a quantization method.
- Promote the stronger ELT claim: recurrent loop quality can be more sensitive
  to quantization noise than PPL alone suggests, and tensor-aware imatrix
  policies are the next proof point under the same GGUF size budget.
- Make the report CLI carry the same lane-boundary text in regenerated JSON and
  Markdown so future README assets inherit the safer language automatically.

## Experiment plan

30-day lane:

- Run BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_NL, IQ4_XS, TQ4_1S.
- Add `Q4_K_M + ELT imatrix` and `Q4_K_M + ELT imatrix + tensor policy`.
- Save PPL, KL, top1 match, top5 overlap, argmax flip rate, ELT exact, ELT step
  accuracy, GSM8K exact, needle score, prompt tok/s, generation tok/s, KV MiB,
  VRAM peak, and file size.

60-day lane:

- Repeat the matrix at `L=1`, `L=2`, `L=4`, and `L=8`.
- Compare tensor policies: all Q4_K_M; output + embedding protected; output +
  embedding + attn_v protected; output + embedding + attn_v + ffn_down
  protected; odd/even attn_q/attn_k mixed policy.

90-day lane:

- Promote TurboQuant KV only when CUDA-capable RTX 3060 measurements are
  possible.
- Promote DFlash only when a target/draft pair can prove greedy equivalence and
  report stable loop-depth acceptance behavior.
- Use Palu, RotateKV, KIVI, vLLM FP8 KV, AutoRound, GPTQModel, and ParoQuant as
  comparison or related-work axes unless local implementation becomes practical.

## Source checks

- Google Research TurboQuant blog, 2026-03-24:
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- llama.cpp PR #21089, checked 2026-05-03:
  https://github.com/ggml-org/llama.cpp/pull/21089
- llama.cpp CLI K/V cache types, checked 2026-05-03:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md
- llama.cpp quantize imatrix/tensor-type options, checked 2026-05-03:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- DFlash arXiv and implementation:
  https://arxiv.org/abs/2602.06036
  https://github.com/z-lab/dflash
- llama.cpp DFlash PR #22105, checked 2026-05-03:
  https://github.com/ggml-org/llama.cpp/pull/22105
- KIVI asymmetric KV-cache quantization:
  https://arxiv.org/abs/2402.02750

## Tests

- `uv run --extra dev --extra eval pytest -q tests/test_gguf_quant_report.py`
  - `6 passed`
- `uv run --extra dev --extra eval pytest -q`
  - stopped during collection because dashboard tests require `streamlit`.
- `uv run --extra dev --extra eval --extra dashboard pytest -q`
  - full suite passed; one `bitsandbytes` optimizer-config test skipped because
    `bitsandbytes` is not installed in this environment.

## Next session notes

- Do not put TurboQuant KV and DFlash in the same README table as if they were
  both quantization results. TurboQuant KV is a compression/runtime-memory lane;
  DFlash is a speculative-decoding serving lane.
- The immediate implementation target is an ELT imatrix + tensor-policy sweep,
  not making `TQ4_1S` smaller in isolation.
