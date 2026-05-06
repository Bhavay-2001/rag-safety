from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import config_hash, load_config, resolve_paths
from src.retriever_bm25 import build_index
from src.utils_io import write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BM25 index from corpus CSV.")
    parser.add_argument("--config", required=True, help="Path to configs/base.yaml")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap for corpus docs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, cfg_path.parent)

    corpus_path = cfg.get("corpus", {}).get("output_path", "data/corpus/wiki_filtered.csv")
    index_dir = cfg.get("retriever", {}).get("index_dir", "indexes/bm25")

    stats = build_index(corpus_path, index_dir, max_docs=args.max_docs)
    stats["corpus_path"] = corpus_path
    stats["config_hash"] = config_hash(cfg)

    meta_path = Path(index_dir) / "bm25.meta.json"
    write_json(meta_path, stats)
    print(f"Wrote BM25 index to {stats['index_path']}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
