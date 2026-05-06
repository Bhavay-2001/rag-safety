from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.qa_metrics import contains_answer, query_relevance
from src.utils_io import ensure_dir, write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build oracle capability dataset with A/B question sets and good/bad contexts."
    )
    parser.add_argument("--irds-id", default="natural-questions/train", help="ir_datasets ID.")
    parser.add_argument("--queries-path", default="data/qa/nq_queries.csv", help="Queries CSV.")
    parser.add_argument("--answers-path", default="data/qa/nq_answers.jsonl", help="Answers JSONL.")
    parser.add_argument("--retrieval-path", required=True, help="qa_retrieval.jsonl with docs_top_k.")
    parser.add_argument(
        "--baseline-responses-path",
        required=True,
        help="qa_responses.jsonl used to split Set A/B from non_rag performance.",
    )
    parser.add_argument(
        "--output-path",
        default="data/qa/capability_oracle_pilot.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--summary-path",
        default="data/qa/capability_oracle_pilot.summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument("--good-k", type=int, default=5, help="Number of good contexts per row.")
    parser.add_argument("--bad-k", type=int, default=5, help="Number of bad contexts per row.")
    parser.add_argument("--max-context-chars", type=int, default=1200, help="Clip stored snippets to this many chars.")
    parser.add_argument("--max-per-set", type=int, default=200, help="Max rows to keep for each set (A and B).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--a-max-em", type=float, default=0.0, help="A-set upper bound for non_rag EM.")
    parser.add_argument("--a-max-f1", type=float, default=0.2, help="A-set upper bound for non_rag F1.")
    parser.add_argument("--b-min-em", type=float, default=1.0, help="B-set lower bound for non_rag EM.")
    parser.add_argument("--b-min-f1", type=float, default=0.7, help="B-set lower bound for non_rag F1.")
    parser.add_argument(
        "--allow-b-by-contains-answer",
        action="store_true",
        help="If set, non_rag contains_answer=True can qualify for Set B.",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail build if validation thresholds are not met.",
    )
    parser.add_argument(
        "--min-good-answer-bearing-rate",
        type=float,
        default=0.9,
        help="Minimum fraction of rows where good contexts contain answer.",
    )
    parser.add_argument(
        "--max-bad-answer-bearing-rate",
        type=float,
        default=0.05,
        help="Maximum fraction of rows where bad contexts contain answer.",
    )
    return parser.parse_args()


def _load_queries(path: str | Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
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
            text = str(text).strip()
            if prompt_id and text:
                out[prompt_id] = text
    return out


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


def _load_retrieval(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_id = str(row.get("prompt_id", "")).strip()
            if not prompt_id:
                continue
            docs = row.get("docs_top_k") or []
            if not isinstance(docs, list):
                continue
            out[prompt_id] = docs
    return out


def _load_non_rag_scores(path: str | Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("condition", "")) != "non_rag":
                continue
            prompt_id = str(row.get("prompt_id", "")).strip()
            if not prompt_id:
                continue
            em = float(row.get("em", 0.0))
            f1 = float(row.get("f1", 0.0))
            best = out.get(prompt_id)
            if best is None or (em, f1) > (best["em"], best["f1"]):
                out[prompt_id] = {
                    "em": em,
                    "f1": f1,
                    "contains_answer": bool(row.get("contains_answer", False)),
                    "model": row.get("model"),
                    "model_id": row.get("model_id"),
                }
    return out


def _iter_qrels(dataset: Any, allowed_qids: set[str]) -> Dict[str, List[str]]:
    qrels_by_qid: Dict[str, List[str]] = {}
    if not dataset.has_qrels():
        return qrels_by_qid
    for qrel in dataset.qrels_iter():
        qid = str(getattr(qrel, "query_id", "")).strip()
        if qid not in allowed_qids:
            continue
        if int(getattr(qrel, "relevance", 0)) <= 0:
            continue
        doc_id = str(getattr(qrel, "doc_id", "")).strip()
        if doc_id:
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
    return blocks if blocks else [p.strip() for p in clean.split("\n") if p.strip()]


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _pick_good_snippet(text: str, answers: List[str], max_chars: int) -> Tuple[str, bool]:
    paras = _paragraphs(text)
    if not paras:
        return "", False
    for para in paras:
        if contains_answer(para, answers):
            return _clip(" ".join(para.split()), max_chars), True
    return _clip(" ".join(paras[0].split()), max_chars), False


def _pick_bad_snippet(text: str, query: str, answers: List[str], max_chars: int) -> Tuple[str, bool, bool]:
    paras = _paragraphs(text)
    if not paras:
        return "", False, False
    for para in paras:
        has_answer = contains_answer(para, answers)
        topical = query_relevance(para, query)
        if topical and not has_answer:
            return _clip(" ".join(para.split()), max_chars), topical, has_answer
    # Fallback to first paragraph, but still return flags for validation accounting.
    first = paras[0]
    return (
        _clip(" ".join(first.split()), max_chars),
        query_relevance(first, query),
        contains_answer(first, answers),
    )


def _assign_set_label(
    score: Dict[str, Any],
    a_max_em: float,
    a_max_f1: float,
    b_min_em: float,
    b_min_f1: float,
    allow_b_by_contains: bool,
) -> str | None:
    em = float(score.get("em", 0.0))
    f1 = float(score.get("f1", 0.0))
    contains = bool(score.get("contains_answer", False))
    if em <= a_max_em and f1 <= a_max_f1:
        return "A"
    if em >= b_min_em or f1 >= b_min_f1 or (allow_b_by_contains and contains):
        return "B"
    return None


def _good_docs_for_prompt(
    prompt_id: str,
    answers: List[str],
    docstore: Any,
    qrels_by_qid: Dict[str, List[str]],
    positive_doc_ids: List[str],
    k: int,
    max_context_chars: int,
) -> List[Dict[str, Any]]:
    all_doc_ids: List[str] = []
    seen_ids: set[str] = set()
    for doc_id in list(positive_doc_ids) + list(qrels_by_qid.get(prompt_id, [])):
        doc_id = str(doc_id).strip()
        if not doc_id or doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        all_doc_ids.append(doc_id)

    contexts: List[Dict[str, Any]] = []
    for doc_id in all_doc_ids:
        doc = _docstore_get_with_fallback(docstore, doc_id)
        text = _extract_doc_text(doc)
        if not text:
            continue
        snippet, hit = _pick_good_snippet(text, answers, max_context_chars)
        if not snippet:
            continue
        contexts.append(
            {
                "doc_id": doc_id,
                "text": snippet,
                "source": "qrels+docstore",
                "answer_present": hit,
            }
        )
        if len(contexts) >= k:
            break
    return contexts


def _bad_docs_for_prompt(
    query: str,
    answers: List[str],
    retrieved_docs: List[Dict[str, Any]],
    good_doc_ids: set[str],
    k: int,
    max_context_chars: int,
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    # First pass: strict topical + answer-absent docs.
    for doc in retrieved_docs:
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id or doc_id in seen_ids or doc_id in good_doc_ids:
            continue
        text = str(doc.get("text", "")).strip()
        if not text:
            continue
        snippet, topical, has_answer = _pick_bad_snippet(text, query, answers, max_context_chars)
        if not snippet or not topical or has_answer:
            continue
        seen_ids.add(doc_id)
        contexts.append(
            {
                "doc_id": doc_id,
                "score": float(doc.get("score", 0.0)),
                "text": snippet,
                "source": "bm25_topical_nonanswering",
                "query_relevant": topical,
                "answer_present": has_answer,
            }
        )
        if len(contexts) >= k:
            return contexts

    # Second pass fallback: allow non-topical but still answer-absent.
    for doc in retrieved_docs:
        if len(contexts) >= k:
            break
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id or doc_id in seen_ids or doc_id in good_doc_ids:
            continue
        text = str(doc.get("text", "")).strip()
        if not text:
            continue
        snippet, topical, has_answer = _pick_bad_snippet(text, query, answers, max_context_chars)
        if not snippet or has_answer:
            continue
        seen_ids.add(doc_id)
        contexts.append(
            {
                "doc_id": doc_id,
                "score": float(doc.get("score", 0.0)),
                "text": snippet,
                "source": "bm25_fallback_nonanswering",
                "query_relevant": topical,
                "answer_present": has_answer,
            }
        )
    return contexts


def _rate(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    try:
        import ir_datasets  # type: ignore
    except Exception:
        ir_datasets = None

    queries = _load_queries(args.queries_path)
    answers_map = _load_answers(args.answers_path)
    retrieval_map = _load_retrieval(args.retrieval_path)
    baseline_map = _load_non_rag_scores(args.baseline_responses_path)

    common_ids = set(queries.keys()) & set(answers_map.keys()) & set(retrieval_map.keys()) & set(baseline_map.keys())
    if not common_ids:
        raise ValueError(
            "No overlapping prompt_ids across queries/answers/retrieval/baseline inputs. "
            "Check file compatibility."
        )

    docstore = None
    qrels_by_qid: Dict[str, List[str]] = {}
    if ir_datasets is not None:
        try:
            dataset = ir_datasets.load(args.irds_id)
            docstore = dataset.docs_store()
            if docstore is not None:
                qrels_by_qid = _iter_qrels(dataset, common_ids)
        except Exception:
            docstore = None
            qrels_by_qid = {}

    # If docstore is unavailable, skip any attempt to build it later.

    candidates: List[Dict[str, Any]] = []
    skipped = {
        "missing_answers": 0,
        "unclassified_ab": 0,
        "insufficient_good": 0,
        "insufficient_bad": 0,
    }

    for prompt_id in sorted(common_ids):
        query = queries[prompt_id]
        answers = answers_map[prompt_id].get("answers", [])
        if not answers:
            skipped["missing_answers"] += 1
            continue

        set_label = _assign_set_label(
            baseline_map[prompt_id],
            a_max_em=args.a_max_em,
            a_max_f1=args.a_max_f1,
            b_min_em=args.b_min_em,
            b_min_f1=args.b_min_f1,
            allow_b_by_contains=args.allow_b_by_contains_answer,
        )
        if set_label is None:
            skipped["unclassified_ab"] += 1
            continue

        if docstore is not None:
            good_contexts = _good_docs_for_prompt(
                prompt_id=prompt_id,
                answers=answers,
                docstore=docstore,
                qrels_by_qid=qrels_by_qid,
                positive_doc_ids=answers_map[prompt_id].get("positive_doc_ids", []),
                k=args.good_k,
                max_context_chars=args.max_context_chars,
            )
        else:
            good_contexts = []
            seen_ids: set[str] = set()
            for doc in retrieval_map[prompt_id]:
                doc_id = str(doc.get("doc_id", "")).strip()
                if not doc_id or doc_id in seen_ids:
                    continue
                text = str(doc.get("text", "")).strip()
                if not text:
                    continue
                if not contains_answer(text, answers):
                    continue
                seen_ids.add(doc_id)
                good_contexts.append(
                    {
                        "doc_id": doc_id,
                        "score": float(doc.get("score", 0.0)),
                        "text": _clip(text, args.max_context_chars),
                        "source": "retrieval_answer_present",
                        "answer_present": True,
                    }
                )
                if len(good_contexts) >= args.good_k:
                    break

        if len(good_contexts) < args.good_k:
            skipped["insufficient_good"] += 1
            continue
        good_doc_ids = {str(c.get("doc_id", "")) for c in good_contexts}

        bad_contexts = _bad_docs_for_prompt(
            query=query,
            answers=answers,
            retrieved_docs=retrieval_map[prompt_id],
            good_doc_ids=good_doc_ids,
            k=args.bad_k,
            max_context_chars=args.max_context_chars,
        )
        if len(bad_contexts) < args.bad_k:
            skipped["insufficient_bad"] += 1
            continue

        candidates.append(
            {
                "prompt_id": prompt_id,
                "query": query,
                "answers": answers,
                "set_label": set_label,
                "contexts_good": good_contexts,
                "contexts_bad": bad_contexts,
                "baseline_non_rag": baseline_map[prompt_id],
                "meta": {
                    "source_dataset": args.irds_id,
                    "good_k": args.good_k,
                    "bad_k": args.bad_k,
                },
            }
        )

    set_a = [row for row in candidates if row["set_label"] == "A"]
    set_b = [row for row in candidates if row["set_label"] == "B"]
    random.shuffle(set_a)
    random.shuffle(set_b)
    set_a = set_a[: args.max_per_set]
    set_b = set_b[: args.max_per_set]
    rows = sorted(set_a + set_b, key=lambda r: (r["set_label"], int(r["prompt_id"]) if r["prompt_id"].isdigit() else r["prompt_id"]))

    if not rows:
        raise ValueError("No rows survived filtering. Relax thresholds or verify inputs.")

    good_has_answer_rows = sum(
        1 for r in rows if any(bool(c.get("answer_present", False)) for c in r["contexts_good"])
    )
    bad_has_answer_rows = sum(
        1 for r in rows if any(bool(c.get("answer_present", False)) for c in r["contexts_bad"])
    )
    bad_topical_rows = sum(
        1 for r in rows if any(bool(c.get("query_relevant", False)) for c in r["contexts_bad"])
    )
    set_counts = {"A": sum(1 for r in rows if r["set_label"] == "A"), "B": sum(1 for r in rows if r["set_label"] == "B")}

    summary = {
        "rows_total": len(rows),
        "set_counts": set_counts,
        "good_answer_bearing_rate": _rate(good_has_answer_rows, len(rows)),
        "bad_answer_bearing_rate": _rate(bad_has_answer_rows, len(rows)),
        "bad_topical_row_rate": _rate(bad_topical_rows, len(rows)),
        "skipped": skipped,
        "inputs": {
            "irds_id": args.irds_id,
            "queries_path": args.queries_path,
            "answers_path": args.answers_path,
            "retrieval_path": args.retrieval_path,
            "baseline_responses_path": args.baseline_responses_path,
        },
        "thresholds": {
            "good_k": args.good_k,
            "bad_k": args.bad_k,
            "a_max_em": args.a_max_em,
            "a_max_f1": args.a_max_f1,
            "b_min_em": args.b_min_em,
            "b_min_f1": args.b_min_f1,
            "allow_b_by_contains_answer": bool(args.allow_b_by_contains_answer),
            "min_good_answer_bearing_rate": args.min_good_answer_bearing_rate,
            "max_bad_answer_bearing_rate": args.max_bad_answer_bearing_rate,
        },
    }

    if args.strict_validation:
        if summary["good_answer_bearing_rate"] < args.min_good_answer_bearing_rate:
            raise ValueError(
                f"good_answer_bearing_rate={summary['good_answer_bearing_rate']:.4f} below "
                f"minimum {args.min_good_answer_bearing_rate:.4f}"
            )
        if summary["bad_answer_bearing_rate"] > args.max_bad_answer_bearing_rate:
            raise ValueError(
                f"bad_answer_bearing_rate={summary['bad_answer_bearing_rate']:.4f} above "
                f"maximum {args.max_bad_answer_bearing_rate:.4f}"
            )

    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)
    write_jsonl(output_path, rows)
    write_json(summary_path, summary)

    print(f"Wrote oracle capability dataset: {output_path} ({len(rows)} rows)")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
