# Goal

Add a side-family ELT runtime that can use Qwen3.5-4B / Huihui Qwen3.5-4B weights
without replacing the native ELT implementation. The target was `L=1` parity with
the HF source model, `L=2` loop support, bootstrap checkpoint creation, and basic
training/inference wiring.

# Files Touched

- `src/elt_lm/config.py`
- `src/elt_lm/model.py`
- `src/elt_lm/hf_qwen35_looped.py`
- `src/elt_lm/bootstrap_qwen35_elt.py`
- `src/elt_lm/train.py`
- `src/elt_lm/infer.py`
- `src/elt_lm/ilsd.py`
- `scripts/bootstrap_qwen35_elt.py`
- `scripts/export_to_hf.py`
- `configs/qwen35_4b_elt_bootstrap.yaml`
- `configs/qwen35_4b_elt_loop_intro.yaml`
- `tests/test_hf_qwen35_looped.py`
- `pyproject.toml`

# Key Decisions

- Kept native `ELTLanguageModel` intact and added `backbone_kind=hf_qwen35_looped`
  as an additive side family.
- Wrapped the HF Qwen3.5 text backbone instead of trying to transplant weights
  into the native ELT attention/FFN implementation.
- Treated one full Qwen text-model pass as the loop unit. `L=1` matches the source
  model path; `L>1` repeats the same text backbone over hidden states.
- Made `build_model()` the runtime branch point so train/infer/checkpoint load can
  share the same factory.
- Left `scripts/export_to_hf.py` native-only. HF-backed looped checkpoints are
  local-runtime artifacts in v1.
- Added a bootstrap utility that converts a HF/local Qwen3.5 model into an ELT-style
  checkpoint with `cfg.model.backbone_kind=hf_qwen35_looped`.

# Tests Added / Passing

- New: `tests/test_hf_qwen35_looped.py`
  - tiny local Qwen3.5 save/load fixture
  - `L=1` parity vs source model
  - `L=2` intermediate logits / loop smoke
  - bootstrap checkpoint roundtrip
  - ILSD training smoke
- Passing runs:
  - `uv run --no-sync pyright src/elt_lm/config.py src/elt_lm/model.py src/elt_lm/hf_qwen35_looped.py src/elt_lm/bootstrap_qwen35_elt.py src/elt_lm/train.py src/elt_lm/infer.py src/elt_lm/ilsd.py scripts/export_to_hf.py tests/test_hf_qwen35_looped.py`
  - `uv run --no-sync python -m pytest -q tests/test_hf_qwen35_looped.py`
  - `uv run --no-sync python -m pytest -q tests/test_shapes_and_params.py tests/test_loop_equivalence.py tests/test_smoke_train.py tests/test_hf_export.py tests/test_ilsd_gradient.py`
  - `uv run --no-sync python scripts/bootstrap_qwen35_elt.py --hf-model <tiny-local-qwen-dir> --out <tiny-local-qwen-dir>/bootstrap.pt --tokenizer H:/Qwen3.5-9B-official-hf`
  - config dry-load for:
    - `configs/qwen35_4b_elt_bootstrap.yaml`
    - `configs/qwen35_4b_elt_loop_intro.yaml`

# Next Session Notes

- Real Huihui `L=1` parity against `huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated`
  was not run in this session. The runtime path is ready, but the actual 4B model
  was not downloaded / loaded during verification.
- The looped HF backbone currently assumes the source config stays reachable via
  `cfg.model.hf_model_path` so the architecture can be reconstructed before loading
  checkpoint weights.
- If this branch becomes the main 4B path, the next high-value step is a real
  bootstrap on the Huihui model, then a short `L_max=2` loop-intro run using the
  new configs.
