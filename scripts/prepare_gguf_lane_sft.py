"""Prepare GGUF lane distill JSONL into train/val bins plus benchmark fixtures."""

from elt_lm.prepare_gguf_lane_sft import cli


if __name__ == "__main__":
    cli()
