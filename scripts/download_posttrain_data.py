"""Download and normalize post-training datasets bucket-by-bucket."""

from __future__ import annotations

import argparse

from elt_lm.posttrain_data import load_posttrain_manifest, write_manifest


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="YAML manifest of post-training sources")
    p.add_argument("--output-root", default="", help="optional root prepended to relative outputs")
    args = p.parse_args()

    manifest = load_posttrain_manifest(args.manifest)
    for bucket_name, path, count in write_manifest(manifest, output_root=args.output_root):
        print(f"{bucket_name}: {count:,} rows -> {path}")


if __name__ == "__main__":
    cli()
