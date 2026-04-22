from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from elt_lm.prepare_gguf_lane_sft import (
    GGUFLanePrepSummary as GGUFDetectionPrepSummary,
    prepare_lane_sft_bundle,
)


def prepare_detection_sft_bundle(
    *,
    input_root: Path,
    output_root: Path,
    tokenizer_path: str,
) -> GGUFDetectionPrepSummary:
    return prepare_lane_sft_bundle(
        input_root=input_root,
        output_root=output_root,
        tokenizer_path=tokenizer_path,
        lane="detection",
    )


def write_detection_benchmark_cases(val_rows, output_path):  # type: ignore[no-untyped-def]
    from elt_lm.prepare_gguf_lane_sft import write_lane_benchmark_cases

    return write_lane_benchmark_cases(val_rows, output_path, lane="detection")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True, help="GGUF distill output dir with distill_train.jsonl / distill_val.jsonl")
    p.add_argument("--output-root", required=True, help="Prepared detection SFT output dir")
    p.add_argument("--tokenizer", required=True, help="HF tokenizer path")
    args = p.parse_args()
    summary = prepare_detection_sft_bundle(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
