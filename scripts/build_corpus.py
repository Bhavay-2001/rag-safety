from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import config_hash, load_config, resolve_paths
from src.corpus_builder import build_corpus
from src.utils_io import write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paragraph-chunked corpus CSV.")
    parser.add_argument("--config", required=True, help="Path to configs/base.yaml")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap for output docs.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path (default: data/corpus/wiki_filtered.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, cfg_path.parent)

    output_path = args.output or cfg.get("corpus", {}).get(
        "output_path", "data/corpus/wiki_filtered.csv"
    )
    stats = build_corpus(cfg, output_path, max_docs=args.max_docs)
    stats["config_hash"] = config_hash(cfg)

    meta_path = Path(output_path).with_suffix(".meta.json")
    write_json(meta_path, stats)
    print(f"Wrote corpus to {output_path}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
