"""Export an ELT checkpoint to a HuggingFace Hub-ready directory.

Produces everything `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
needs: config.json, model.safetensors, tokenizer files, custom modeling/config
modules, and a rendered model card (README.md).

Example:

    uv run python scripts/export_to_hf.py \\
        --ckpt runs/base_100M/last.pt \\
        --out  hf_export/elt-lm-base-275m \\
        --tokenizer H:/Qwen3.5-9B-official-hf \\
        --repo-id zapabob/elt-lm-base-275m \\
        --push-to-hub

Without --push-to-hub the script only writes the directory locally so you can
inspect it before uploading (`huggingface-cli login` then re-run with --push).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from elt_lm.hf.configuration_elt import ELTConfig
from elt_lm.hf.modeling_elt import ELTForCausalLM


HF_MODULE_DIR = Path(__file__).resolve().parent.parent / "src" / "elt_lm" / "hf"
TEMPLATE_PATH = HF_MODULE_DIR / "model_card_template.md"

# Files we copy into the export so `trust_remote_code=True` finds everything
# without needing the full `elt_lm` package at load time.
BUNDLED_MODULES = [
    "configuration_elt.py",
    "modeling_elt.py",
]

# Tokenizer files we attempt to copy from the source tokenizer directory.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",          # present on some tokenizers
    "merges.txt",          # present on some tokenizers
]


def _render_card(config: ELTConfig, total_params: int, nonemb_params: int,
                 repo_id: str) -> str:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    ctx = {
        "repo_id": repo_id or "elt-lm-base",
        "total_m": total_params / 1e6,
        "nonemb_m": nonemb_params / 1e6,
        "vocab_k": config.vocab_size,
        "n_unique_layers": config.n_unique_layers,
        "L_min": config.L_min,
        "L_max": config.L_max,
        "eff_depth": config.n_unique_layers * config.L_max,
        "pretrain_tokens_b": 10.0,    # approximate; edit as the run finishes
    }
    # Template uses `{name}` placeholders; double-brace `{{..}}` stays literal.
    return text.format(**ctx)


def _copy_modules(out_dir: Path) -> None:
    for name in BUNDLED_MODULES:
        src = HF_MODULE_DIR / name
        shutil.copy(src, out_dir / name)

    # Inside the export dir the modeling module must import from a sibling file,
    # not from `elt_lm.hf.*`. We also need a local `model.py` + its deps so that
    # `ELTLanguageModel` can be constructed without the outer `elt_lm` package.
    elt_src = HF_MODULE_DIR.parent        # src/elt_lm
    for fname in (
        "__init__.py", "config.py", "model.py", "composite.py", "layer.py",
        "attention.py", "ffn.py", "norm.py", "rope.py",
    ):
        if (elt_src / fname).exists():
            (out_dir / "elt_lm").mkdir(exist_ok=True)
            shutil.copy(elt_src / fname, out_dir / "elt_lm" / fname)


def _copy_tokenizer(src_dir: Path, out_dir: Path) -> int:
    n_copied = 0
    for name in TOKENIZER_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copy(src, out_dir / name)
            n_copied += 1
    return n_copied


def export(ckpt: Path, out: Path, tokenizer: Path, repo_id: str | None) -> ELTForCausalLM:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model" in state and "cfg" in state, \
        f"checkpoint {ckpt} missing 'model' or 'cfg' keys"

    inner_cfg = state["cfg"].model           # elt_lm.config.ModelConfig
    if getattr(inner_cfg, "backbone_kind", "native_elt") != "native_elt":
        raise NotImplementedError(
            "export_to_hf.py currently supports only native_elt checkpoints. "
            "hf_qwen35_looped checkpoints are local-runtime only in v1."
        )
    hf_cfg = ELTConfig.from_model_config(inner_cfg)
    model = ELTForCausalLM(hf_cfg)
    model.elt.load_state_dict(state["model"])
    model.eval()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)

    _copy_modules(out)

    # Register the custom classes so AutoModel + AutoConfig find them in the dump.
    hf_cfg.auto_map = {
        "AutoConfig": "configuration_elt.ELTConfig",
        "AutoModelForCausalLM": "modeling_elt.ELTForCausalLM",
    }
    hf_cfg.save_pretrained(out)              # re-save config with auto_map

    tok_copied = _copy_tokenizer(tokenizer, out)
    print(f"  tokenizer files copied: {tok_copied} (from {tokenizer})")

    total = sum(p.numel() for p in model.parameters())
    nonemb = total - model.elt.tok_embed.weight.numel()
    card = _render_card(hf_cfg, total, nonemb, repo_id or out.name)
    (out / "README.md").write_text(card, encoding="utf-8")

    print(f"  params: {total/1e6:.1f}M total, {nonemb/1e6:.1f}M non-embedding")
    print(f"  wrote HF export -> {out}")
    return model


def push(out: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi, create_repo
    create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    HfApi().upload_folder(
        folder_path=str(out),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"ELT-LM export from {out.name}",
    )
    print(f"  pushed -> https://huggingface.co/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to ELT .pt checkpoint")
    p.add_argument("--out", required=True, help="output directory for the HF bundle")
    p.add_argument("--tokenizer", required=True,
                   help="path to the tokenizer dir (e.g. H:/Qwen3.5-9B-official-hf)")
    p.add_argument("--repo-id", default=None,
                   help="HF Hub repo id, e.g. zapabob/elt-lm-base-275m")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    ckpt, out, tokenizer = Path(args.ckpt), Path(args.out), Path(args.tokenizer)
    export(ckpt, out, tokenizer, args.repo_id)

    if args.push_to_hub:
        assert args.repo_id, "--push-to-hub requires --repo-id"
        push(out, args.repo_id, args.private)


if __name__ == "__main__":
    main()
