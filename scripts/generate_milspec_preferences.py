"""Generate deterministic MILSPEC-style synthetic preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elt_lm.synthetic_preferences import generate_synthetic_preference_pairs


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = generate_synthetic_preference_pairs(args.count, seed=args.seed)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.as_record(), ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs):,} preference pairs -> {out_path}")


if __name__ == "__main__":
    cli()
