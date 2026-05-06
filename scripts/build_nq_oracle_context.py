from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.qa_metrics import contains_answer
from src.utils_io import ensure_dir, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed NQ oracle contexts from qrels + docstore.")
    parser.add_argument(
        "--irds-id",
        default="natural-questions/train",
        help="ir_datasets identifier for NQ.",
    )
    parser.add_argument(
        "--queries-path",
        default="data/qa/nq_queries.csv",
        help="CSV with prompt_id and query text.",
    )
    parser.add_argument(
        "--answers-path",
        default="data/qa/nq_answers.jsonl",
        help="JSONL with prompt_id and answers.",
    )
    parser.add_argument(
        "--output-path",
        default="data/qa/nq_oracle_context_top5.jsonl",
        help="Output JSONL for fixed oracle contexts.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of context docs to keep per query.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=1200,
        help="Max characters per stored context snippet.",
    )
    parser.add_argument(
        "--max-empty-ratio",
        type=float,
        default=0.25,
        help="Fail if ratio of rows with zero contexts exceeds this threshold.",
    )
    return parser.parse_args()


def _load_queries(path: str | Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            prompt_id = str(row.get("prompt_id") or idx).strip()
            text = (
                row.get("text")
                or row.get("question")
                or row.get("query")
                or row.get("prompt")
                or next((v for v in row.values() if v and str(v).strip()), "")
            )
            if not prompt_id or not text:
                continue
            rows.append({"prompt_id": prompt_id, "query": str(text)})
    return rows


def _load_answers(path: str | Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_id = str(row.get("prompt_id", "")).strip()
            if not prompt_id:
                continue
            answers = row.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]
            positive_doc_ids = row.get("positive_doc_ids") or []
            if isinstance(positive_doc_ids, str):
                positive_doc_ids = [positive_doc_ids]
            out[prompt_id] = {
                "answers": [str(a).strip() for a in answers if str(a).strip()],
                "positive_doc_ids": [str(d).strip() for d in positive_doc_ids if str(d).strip()],
            }
    return out


def _iter_qrels(dataset: Any, allowed_qids: set[str]) -> Dict[str, List[str]]:
    qrels_by_qid: Dict[str, List[str]] = {}
    if not dataset.has_qrels():
        return qrels_by_qid
    for qrel in dataset.qrels_iter():
        qid = str(getattr(qrel, "query_id", ""))
        if qid not in allowed_qids:
            continue
        if int(getattr(qrel, "relevance", 0)) <= 0:
            continue
        doc_id = str(getattr(qrel, "doc_id", "")).strip()
        if not doc_id:
            continue
        qrels_by_qid.setdefault(qid, []).append(doc_id)
    return qrels_by_qid


def _extract_doc_text(doc: Any) -> str:
    if doc is None:
        return ""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("text", "default_text", "body", "contents", "content", "document", "passage"):
            val = doc.get(key)
            if val:
                return str(val)
        return ""
    for key in ("text", "default_text", "body", "contents", "content", "document", "passage"):
        if hasattr(doc, key):
            val = getattr(doc, key)
            if val:
                return str(val)
    return ""


def _docstore_get_with_fallback(docstore: Any, doc_id: str) -> Any:
    # Some ir_datasets docstores key by int doc IDs; others by string.
    candidates: List[Any] = [doc_id]
    try:
        candidates.append(int(doc_id))
    except (TypeError, ValueError):
        pass
    for key in candidates:
        try:
            doc = docstore.get(key)
        except Exception:
            doc = None
        if doc is not None:
            return doc
    return None


def _paragraphs(text: str) -> List[str]:
    clean = str(text).strip()
    if not clean:
        return []
    blocks = [p.strip() for p in re.split(r"\n{2,}", clean) if p.strip()]
    if blocks:
        return blocks
    return [p.strip() for p in clean.split("\n") if p.strip()]


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _pick_snippet(text: str, answers: List[str], max_chars: int) -> str:
    paras = _paragraphs(text)
    if not paras:
        return ""
    for para in paras:
        if contains_answer(para, answers):
            return _clip(" ".join(para.split()), max_chars)
    return _clip(" ".join(paras[0].split()), max_chars)


def main() -> None:
    args = _parse_args()
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError("ir_datasets is required. Install with `pip install ir-datasets`.") from exc

    queries = _load_queries(args.queries_path)
    answers_map = _load_answers(args.answers_path)
    queries = [
        q
        for q in queries
        if q["prompt_id"] in answers_map and answers_map[q["prompt_id"]].get("answers")
    ]
    if not queries:
        raise ValueError("No answerable queries found after joining queries and answers.")

    dataset = ir_datasets.load(args.irds_id)
    docstore = dataset.docs_store()
    if docstore is None:
        raise ValueError(f"Dataset {args.irds_id} does not provide docs_store().")

    qids = {q["prompt_id"] for q in queries}
    qrels_by_qid = _iter_qrels(dataset, qids)

    rows: List[Dict[str, Any]] = []
    empty_context_rows = 0
    answer_present_rows = 0

    for q in queries:
        prompt_id = q["prompt_id"]
        query = q["query"]
        answers = answers_map[prompt_id]["answers"]
        doc_ids = answers_map[prompt_id].get("positive_doc_ids") or qrels_by_qid.get(prompt_id, [])

        contexts: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = _docstore_get_with_fallback(docstore, doc_id)
            text = _extract_doc_text(doc)
            if not text:
                continue
            snippet = _pick_snippet(text, answers, args.max_context_chars)
            if not snippet:
                continue
            contexts.append(
                {
                    "doc_id": doc_id,
                    "text": snippet,
                    "source": f"{args.irds_id}:qrels+docstore",
                }
            )
            if len(contexts) >= args.top_k:
                break

        answer_present = any(contains_answer(ctx["text"], answers) for ctx in contexts)
        if contexts:
            if answer_present:
                answer_present_rows += 1
        else:
            empty_context_rows += 1

        rows.append(
            {
                "prompt_id": prompt_id,
                "query": query,
                "answers": answers,
                "contexts": contexts,
                "answer_present_top_k": answer_present,
            }
        )

    if not rows:
        raise ValueError("No rows produced for oracle contexts.")

    empty_ratio = empty_context_rows / len(rows)
    if empty_ratio > args.max_empty_ratio:
        raise ValueError(
            "Too many rows have zero contexts in oracle build: "
            f"{empty_context_rows}/{len(rows)} ({empty_ratio:.2%})."
        )

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    write_jsonl(output_path, rows)

    print(f"Wrote oracle contexts to {output_path}")
    print(
        "rows={rows} contexts_empty={empty} answer_present={present} "
        "answer_present_rate={rate:.4f}".format(
            rows=len(rows),
            empty=empty_context_rows,
            present=answer_present_rows,
            rate=answer_present_rows / len(rows),
        )
    )


if __name__ == "__main__":
    main()
