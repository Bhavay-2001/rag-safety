from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils_io import ensure_dir, write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Natural Questions assets from ir_datasets.")
    parser.add_argument(
        "--irds-id",
        default="irds:natural-questions/train",
        help="ir_datasets identifier for the query set.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Deterministic query sample size.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--queries-out",
        default="data/qa/nq_queries.csv",
        help="Output CSV for sampled queries.",
    )
    parser.add_argument(
        "--answers-out",
        default="data/qa/nq_answers.jsonl",
        help="Output JSONL for query answers.",
    )
    parser.add_argument(
        "--docs-manifest-out",
        default="data/qa/nq_docs_manifest.json",
        help="Output JSON for docstore/corpus metadata.",
    )
    parser.add_argument(
        "--manifest-path",
        default="data/raw/SOURCE_MANIFEST.json",
        help="Path to provenance manifest JSON.",
    )
    return parser.parse_args()


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalize_answers(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        text = obj.strip()
        return [text] if text else []
    if isinstance(obj, (list, tuple)):
        out: List[str] = []
        for item in obj:
            out.extend(_normalize_answers(item))
        dedup: List[str] = []
        seen = set()
        for val in out:
            if val.lower() in seen:
                continue
            seen.add(val.lower())
            dedup.append(val)
        return dedup
    if hasattr(obj, "text"):
        return _normalize_answers(getattr(obj, "text"))
    if isinstance(obj, dict):
        if "text" in obj:
            return _normalize_answers(obj["text"])
        if "answers" in obj:
            return _normalize_answers(obj["answers"])
    return []


def _load_dataset(irds_id: str):
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError("ir_datasets is required. Install with `pip install ir-datasets`.") from exc
    return ir_datasets.load(irds_id)


def _sample_queries(dataset: Any, sample_size: int, seed: int) -> List[Any]:
    queries = list(dataset.queries_iter())
    if sample_size >= len(queries):
        return queries
    rng = random.Random(seed)
    return rng.sample(queries, sample_size)


def _write_queries(path: Path, sampled_queries: List[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "text", "source_benchmark", "source_dataset"])
        writer.writeheader()
        for q in sampled_queries:
            writer.writerow(
                {
                    "prompt_id": str(q.query_id),
                    "text": str(q.text),
                    "source_benchmark": "NQ",
                    "source_dataset": "natural_questions",
                }
            )


def _write_answers(path: Path, sampled_queries: List[Any], qrels_by_qid: Dict[str, List[str]]) -> None:
    rows: List[Dict[str, Any]] = []
    for q in sampled_queries:
        answers = _normalize_answers(getattr(q, "answers", None))
        rows.append(
            {
                "prompt_id": str(q.query_id),
                "query": str(q.text),
                "answers": answers,
                "positive_doc_ids": qrels_by_qid.get(str(q.query_id), []),
            }
        )
    write_jsonl(path, rows)


def _build_qrels_index(dataset: Any, allowed_qids: set[str]) -> Dict[str, List[str]]:
    qrels: Dict[str, List[str]] = {}
    if not dataset.has_qrels():
        return qrels
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        if qid not in allowed_qids:
            continue
        if getattr(qrel, "relevance", 0) <= 0:
            continue
        qrels.setdefault(qid, []).append(str(qrel.doc_id))
    return qrels


def _docs_manifest(dataset: Any) -> Dict[str, Any]:
    docstore = dataset.docs_store()
    return {
        "irds_id": dataset.dataset_id(),
        "has_docs": dataset.has_docs(),
        "has_queries": dataset.has_queries(),
        "has_qrels": dataset.has_qrels(),
        "docs_count": dataset.docs_count() if dataset.has_docs() else None,
        "queries_count": dataset.queries_count() if dataset.has_queries() else None,
        "qrels_count": dataset.qrels_count() if dataset.has_qrels() else None,
        "docstore_class": docstore.__class__.__name__ if docstore is not None else None,
    }


def _update_manifest(
    manifest_path: Path,
    irds_id: str,
    produced_files: List[Path],
) -> None:
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest.setdefault("generated_at_utc", _now_utc())
    manifest.setdefault("sources", [])
    manifest.setdefault("produced_files", [])
    manifest["generated_at_utc"] = _now_utc()
    manifest["sources"].append(
        {
            "source_type": "ir_datasets",
            "dataset_id": irds_id,
            "timestamp_utc": _now_utc(),
            "license_note": "Check Natural Questions license terms before redistribution.",
        }
    )
    for p in produced_files:
        manifest["produced_files"].append(
            {"path": str(p), "sha256": _sha256_file(p), "timestamp_utc": _now_utc()}
        )
    write_json(manifest_path, manifest)


def main() -> None:
    args = _parse_args()
    dataset = _load_dataset(args.irds_id)
    sampled_queries = _sample_queries(dataset, args.sample_size, args.seed)
    qids = {str(q.query_id) for q in sampled_queries}
    qrels_by_qid = _build_qrels_index(dataset, qids)

    queries_out = Path(args.queries_out)
    answers_out = Path(args.answers_out)
    docs_manifest_out = Path(args.docs_manifest_out)
    manifest_path = Path(args.manifest_path)

    _write_queries(queries_out, sampled_queries)
    _write_answers(answers_out, sampled_queries, qrels_by_qid)
    write_json(docs_manifest_out, _docs_manifest(dataset))
    _update_manifest(manifest_path, args.irds_id, [queries_out, answers_out, docs_manifest_out])

    print(f"Wrote sampled NQ queries to {queries_out} ({len(sampled_queries)} rows)")
    print(f"Wrote NQ answers to {answers_out}")
    print(f"Wrote NQ docs manifest to {docs_manifest_out}")
    print(f"Updated source manifest at {manifest_path}")


if __name__ == "__main__":
    main()
