from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _doc_set_labels(doc_rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in doc_rows:
        key = (str(row.get("prompt_id", "")), str(row.get("condition", "")))
        grouped[key].append(str(row.get("label", "unknown")).lower())

    labels: Dict[Tuple[str, str], str] = {}
    for key, vals in grouped.items():
        if not vals:
            labels[key] = "unknown"
        elif any(v == "unsafe" for v in vals):
            labels[key] = "unsafe"
        elif any(v == "unknown" for v in vals):
            labels[key] = "unknown"
        else:
            labels[key] = "safe"
    return labels


def _build_doc_detail_maps(
    doc_rows: Iterable[Dict[str, Any]],
    retrieval_rows: Iterable[Dict[str, Any]],
    text_chars: int,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Dict[str, Any]]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    doc_labels: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in doc_rows:
        key = (str(row.get("prompt_id", "")), str(row.get("condition", "")))
        doc_id = str(row.get("doc_id", ""))
        doc_labels[key][doc_id] = {
            "doc_label": row.get("label"),
            "doc_categories_primary": row.get("categories_primary"),
            "doc_unsafe_prob_llr_primary": row.get("unsafe_prob_llr_primary"),
            "doc_llr_margin_primary": row.get("llr_margin_primary"),
        }

    retrieval_docs: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in retrieval_rows:
        key = (str(row.get("prompt_id", "")), str(row.get("condition", "")))
        docs = row.get("docs") or []
        out_docs: List[Dict[str, Any]] = []
        for rank, d in enumerate(docs, start=1):
            text = str(d.get("text", ""))
            out_docs.append(
                {
                    "doc_id": d.get("doc_id"),
                    "rank": rank,
                    "bm25_score": d.get("score"),
                    "doc_text_snippet": text[:text_chars],
                }
            )
        retrieval_docs[key] = out_docs
    return doc_labels, retrieval_docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cases where all retrieved docs are safe but the response is unsafe."
    )
    parser.add_argument("--run-dir", required=True, help="Path to run directory with responses/judge/doc_safety/retrieval.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path for extracted examples.")
    parser.add_argument("--output-summary", required=True, help="Output JSON path for aggregate counts.")
    parser.add_argument("--text-chars", type=int, default=500, help="Chars to keep for each doc snippet.")
    parser.add_argument(
        "--conditions",
        default="rag_docs,rag_llm",
        help="Comma-separated conditions to include. Default: rag_docs,rag_llm",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    responses_path = run_dir / "responses.jsonl"
    judge_path = run_dir / "judge.jsonl"
    doc_path = run_dir / "doc_safety.jsonl"
    retrieval_path = run_dir / "retrieval.jsonl"

    for p in (responses_path, judge_path, doc_path, retrieval_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    wanted_conditions = {c.strip() for c in args.conditions.split(",") if c.strip()}

    responses = list(_read_jsonl(responses_path))
    judge_rows = list(_read_jsonl(judge_path))
    doc_rows = list(_read_jsonl(doc_path))
    retrieval_rows = list(_read_jsonl(retrieval_path))

    doc_set_label = _doc_set_labels(doc_rows)
    doc_labels_map, retrieval_docs_map = _build_doc_detail_maps(doc_rows, retrieval_rows, args.text_chars)

    judge_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for j in judge_rows:
        key = (str(j.get("prompt_id", "")), str(j.get("condition", "")), str(j.get("model", "")))
        judge_map[key] = j

    examples: List[Dict[str, Any]] = []
    for row in responses:
        prompt_id = str(row.get("prompt_id", ""))
        condition = str(row.get("condition", ""))
        model = str(row.get("model", ""))
        if condition not in wanted_conditions:
            continue

        judge = judge_map.get((prompt_id, condition, model))
        if not judge or str(judge.get("label", "")).lower() != "unsafe":
            continue

        key_pc = (prompt_id, condition)
        if doc_set_label.get(key_pc) != "safe":
            continue

        docs = retrieval_docs_map.get(key_pc, [])
        if not docs:
            continue

        labels_for_key = doc_labels_map.get(key_pc, {})
        merged_docs: List[Dict[str, Any]] = []
        for d in docs:
            extra = labels_for_key.get(str(d.get("doc_id", "")), {})
            merged = dict(d)
            merged.update(extra)
            merged_docs.append(merged)

        examples.append(
            {
                "prompt_id": prompt_id,
                "condition": condition,
                "model": model,
                "query": row.get("query"),
                "response": row.get("response"),
                "response_label": judge.get("label"),
                "response_categories": judge.get("categories"),
                "unsafe_prob_llr": judge.get("unsafe_prob_llr"),
                "llr_margin": judge.get("llr_margin"),
                "doc_set_label": "safe",
                "doc_count": len(merged_docs),
                "docs": merged_docs,
            }
        )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=True) + "\n")

    by_condition: Dict[str, int] = defaultdict(int)
    for ex in examples:
        by_condition[str(ex.get("condition", "unknown"))] += 1

    summary = {
        "run_dir": str(run_dir),
        "conditions": sorted(wanted_conditions),
        "total_examples": len(examples),
        "by_condition": dict(sorted(by_condition.items())),
        "output_jsonl": str(output_jsonl),
    }
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
