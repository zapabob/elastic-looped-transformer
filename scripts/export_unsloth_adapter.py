from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--save-method", default="merged_16bit")
    parser.add_argument("--maximum-memory-usage", type=float, default=0.65)
    args = parser.parse_args()

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(
        args.out_dir,
        tokenizer,
        save_method=args.save_method,
        maximum_memory_usage=args.maximum_memory_usage,
    )


if __name__ == "__main__":
    main()
