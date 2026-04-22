"""Prepare GGUF detection distill JSONL into train or val bins plus benchmark fixtures."""

from elt_lm.prepare_gguf_detection_sft import cli


if __name__ == "__main__":
    cli()
