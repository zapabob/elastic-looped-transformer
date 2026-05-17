# 2026-05-18 HF Qwen3.5 ELT Public Release - gpt-5

## Goal

Publish the Qwen3.5 ELT L3 artifact as a public Hugging Face model repository
with Hugging Face safetensors, GGUF variants, model card, and evaluation assets.

## Published repository

- Repo: `zapabobouj/qwen3.5-elt-l3`
- URL: `https://hf.co/zapabobouj/qwen3.5-elt-l3`
- Visibility: public
- Verified Hub SHA after upload: `becb58431e6f44bf9d986264b8a5faed524f8cf9`

## Files published to Hugging Face

- `model-00001-of-00007.safetensors` ... `model-00007-of-00007.safetensors`
- `model.safetensors.index.json`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `elt_export_manifest.json`
- `publish_manifest.json`
- `elt-lm-qwen35-side-stem-aha-ilsd-l3-BF16.gguf`
- `elt-lm-qwen35-side-stem-aha-ilsd-l3-Q8_0.gguf`
- `elt-lm-qwen35-side-stem-aha-ilsd-l3-TQ4_1S.gguf`
- `eval/l3_readme_accuracy_errorbars.png`
- `eval/l3_readme_stats.json`
- `eval/l3_readme_stats.md`
- `README.md`

## Local staging

- Publish bundle: `H:/elt_data/hf_publish/qwen3.5-elt-l3`
- Source HF export: `H:/elt_data/hf_exports/elt-lm-qwen35-side-stem-aha-ilsd-l3-merged`
- Source GGUF files: `H:/elt_data/releases/elt-lm-qwen35-side-stem-aha-ilsd-l3*.gguf`

The staging script uses hardlinks when possible so the large H: artifacts are
not duplicated just to create the publish folder.

## Key decisions

- Used repo id `zapabobouj/qwen3.5-elt-l3`, matching the authenticated HF user
  and the user's Qwen-3.5 ELT naming request.
- Kept the model card claim-bounded: local STEM bridge is strong, external
  heldouts are small, GSM8K is not solved, and L>=2 quality requires the
  loop-aware HF runtime.
- Published all three GGUF variants. `TQ4_1S` is described as offline
  weight-compression, not TurboQuant KV-cache serving proof.
- Added `scripts/prepare_hf_qwen35_elt_release.py` so the bundle and upload can
  be reproduced.

## Upload notes

- The global `hf` CLI failed on this PC with Python certificate verification
  errors.
- The Hugging Face app and a repo-local Python path authenticated as
  `zapabobouj`.
- Installed `truststore==0.10.4` into the local `.venv` using:
  `uv --native-tls pip install --python ./.venv/Scripts/python3.exe truststore`
- First `upload_large_folder` attempt with 4 workers stalled after partial
  progress; it was stopped and resumed using `--num-workers 2`.
- The resumed upload completed `29.0G/29.0G` and committed successfully.

## Verification

- `.\.venv\Scripts\python3.exe -m py_compile scripts\prepare_hf_qwen35_elt_release.py`
- Hub readback via `HfApi().model_info("zapabobouj/qwen3.5-elt-l3", files_metadata=True)`
- Hugging Face app readback for repo details and public model card metadata
- Verified `private=False`, repo id, commit SHA, and full file list.

## Next-session notes

- Re-running the same script is safe: `upload_large_folder` recovers from
  `H:/elt_data/hf_publish/qwen3.5-elt-l3/.cache/huggingface/upload`.
- Keep `--num-workers 2` on this machine unless the network path is known to be
  stable.
