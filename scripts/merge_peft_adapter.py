from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT/LoRA adapter into its base HF model."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--serialization",
        choices=("safetensors", "pytorch"),
        default="safetensors",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} already exists; pass --force to replace it")
        shutil.rmtree(out_dir)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        is_trainable=False,
    )
    model = model.merge_and_unload()
    if args.serialization == "pytorch":
        out_dir.mkdir(parents=True, exist_ok=True)
        model.config.save_pretrained(out_dir)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(out_dir)
        state_dict = model.state_dict()
        if getattr(model.config, "tie_word_embeddings", False):
            state_dict.pop("lm_head.weight", None)
        torch.save(state_dict, out_dir / "pytorch_model.bin")
    else:
        model.save_pretrained(
            out_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )

    tokenizer_path = args.tokenizer or args.adapter
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
