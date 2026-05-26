"""
One-time script to stream nkandpa2/wiki-dolma, shuffle, and save
200K raw records to a local JSONL file. Run this once on a login node
before submitting the SLURM job.

Usage:
    python3 scripts/download_wiki_dolma.py \
        --output data/raw/wiki_dolma_200k.jsonl \
        --n 200000 \
        --seed 42 \
        --buffer 10000
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, date
from pathlib import Path


def _json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/wiki_dolma_200k.jsonl")
    parser.add_argument("--n", type=int, default=200_000, help="Number of records to save")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer", type=int, default=10_000, help="Shuffle buffer size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Count already-saved lines so we can resume if interrupted
    already_saved = 0
    if output.exists():
        with output.open("r", encoding="utf-8") as f:
            already_saved = sum(1 for _ in f)
        print(f"Resuming: {already_saved} records already saved, need {args.n - already_saved} more.")

    if already_saved >= args.n:
        print("Already complete.")
        sys.exit(0)

    from datasets import load_dataset

    print("Connecting to HuggingFace...")
    ds = load_dataset("nkandpa2/wiki-dolma", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer)

    written = already_saved
    with output.open("a", encoding="utf-8") as f:
        for i, record in enumerate(ds):
            if i < already_saved:
                continue
            f.write(json.dumps(record, default=_json_default) + "\n")
            written += 1
            if written % 10_000 == 0:
                print(f"  {written}/{args.n} saved...", flush=True)
            if written >= args.n:
                break

    print(f"Done. {written} records saved to {output}")


if __name__ == "__main__":
    main()
