from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.metrics import compute_doc_set_labels
from src.utils_io import read_jsonl, write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a manual calibration slice for primary-vs-second-pass safety disagreements."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory with judge/reponse artifacts.")
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Output JSONL path (default: <run-dir>/wildguard_calibration_100.jsonl).",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Output summary JSON path (default: <run-dir>/wildguard_calibration_100.summary.json).",
    )
    parser.add_argument("--size", type=int, default=100, help="Number of rows to export.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--primary-label",
        default="unsafe",
        choices=["unsafe", "safe", "unknown", "any"],
        help="Filter by primary label before sampling.",
    )
    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated conditions to include (default: all). Example: non_rag,rag_docs",
    )
    parser.add_argument(
        "--second-labels",
        default="",
        help="Optional comma-separated second labels to include (e.g., safe,unsafe,unknown).",
    )
    parser.add_argument(
        "--second-model",
        default="",
        help="Optional second judge model id filter (e.g., allenai/wildguard).",
    )
    parser.add_argument(
        "--include-only-disagreements",
        action="store_true",
        help="If set, keep only rows where primary_label != second_label.",
    )
    parser.add_argument("--max-response-chars", type=int, default=2500, help="Trim response text for export.")
    parser.add_argument("--max-doc-chars", type=int, default=700, help="Trim each retrieved doc snippet in export.")
    parser.add_argument("--docs-per-row", type=int, default=2, help="How many retrieved snippets to include per row.")
    return parser.parse_args()


def _trim(text: str, limit: int) -> str:
    t = str(text or "").strip()
    if limit <= 0 or len(t) <= limit:
        return t
    return t[:limit] + " ..."


def _load_map(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
        out[key] = row
    return out


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    out_jsonl = Path(args.output_jsonl) if args.output_jsonl else run_dir / "wildguard_calibration_100.jsonl"
    out_summary = (
        Path(args.output_summary) if args.output_summary else run_dir / "wildguard_calibration_100.summary.json"
    )

    responses = list(read_jsonl(run_dir / "responses.jsonl"))
    primary = list(read_jsonl(run_dir / "judge.jsonl"))
    second = list(read_jsonl(run_dir / "judge_second_pass.jsonl"))
    if second and any("second_judge_model_id" in x for x in second):
        chosen_second = args.second_model.strip()
        if not chosen_second:
            chosen_second = "allenai/wildguard"
        second = [x for x in second if str(x.get("second_judge_model_id", "")) == chosen_second]
    actionability = list(read_jsonl(run_dir / "actionability.jsonl")) if (run_dir / "actionability.jsonl").exists() else []
    retrieval_rows = list(read_jsonl(run_dir / "retrieval.jsonl")) if (run_dir / "retrieval.jsonl").exists() else []
    doc_rows = list(read_jsonl(run_dir / "doc_safety.jsonl")) if (run_dir / "doc_safety.jsonl").exists() else []

    response_map = _load_map(responses)
    primary_map = _load_map(primary)
    second_map = _load_map(second)
    actionability_map = _load_map(actionability)
    doc_set_map = compute_doc_set_labels(doc_rows) if doc_rows else {}

    retrieval_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in retrieval_rows:
        key = (str(row.get("prompt_id", "")), str(row.get("condition", "")))
        retrieval_map[key] = row.get("docs") or []

    allowed_conditions = {x.strip() for x in args.conditions.split(",") if x.strip()}
    allowed_second_labels = {x.strip() for x in args.second_labels.split(",") if x.strip()}

    candidates: List[Dict[str, Any]] = []
    for key, p in primary_map.items():
        s = second_map.get(key)
        r = response_map.get(key)
        if s is None or r is None:
            continue

        prompt_id, condition, model = key
        p_label = str(p.get("label", "unknown"))
        s_label = str(s.get("second_label", "unknown"))

        if args.primary_label != "any" and p_label != args.primary_label:
            continue
        if allowed_conditions and condition not in allowed_conditions:
            continue
        if allowed_second_labels and s_label not in allowed_second_labels:
            continue
        if args.include_only_disagreements and p_label == s_label:
            continue

        docs = retrieval_map.get((prompt_id, condition), [])
        doc_snips = []
        for d in docs[: max(args.docs_per_row, 0)]:
            doc_snips.append(
                {
                    "doc_id": str(d.get("doc_id", "")),
                    "score": d.get("score"),
                    "text": _trim(str(d.get("text", "")), args.max_doc_chars),
                }
            )

        a = actionability_map.get(key, {})
        record = {
            "prompt_id": prompt_id,
            "condition": condition,
            "model": model,
            "query": str(r.get("query") or r.get("prompt") or ""),
            "response": _trim(str(r.get("response", "")), args.max_response_chars),
            "primary_label": p_label,
            "primary_categories": p.get("categories") or [],
            "primary_parse_error": p.get("judge_parse_error"),
            "primary_parse_error_reason": p.get("judge_parse_error_reason"),
            "primary_raw": str(p.get("raw", "")),
            "second_label": s_label,
            "second_categories": s.get("second_categories") or [],
            "second_parse_error": s.get("second_parse_error"),
            "second_parse_error_reason": s.get("second_parse_error_reason"),
            "second_raw": str(s.get("second_raw", "")),
            "actionability_label": a.get("actionability_label"),
            "doc_set_label": doc_set_map.get((prompt_id, condition)),
            "safe_doc_unsafe_candidate": s.get("safe_doc_unsafe_candidate"),
            "docs_top": doc_snips,
        }
        candidates.append(record)

    rng = random.Random(args.seed)
    if len(candidates) > args.size:
        selected = rng.sample(candidates, args.size)
    else:
        selected = list(candidates)
    selected = sorted(selected, key=lambda x: (x["condition"], x["prompt_id"], x["model"]))

    write_jsonl(out_jsonl, selected)

    by_condition = Counter(x["condition"] for x in selected)
    by_primary = Counter(x["primary_label"] for x in selected)
    by_second = Counter(x["second_label"] for x in selected)
    by_actionability = Counter(str(x.get("actionability_label")) for x in selected)
    disagreements = sum(1 for x in selected if x["primary_label"] != x["second_label"])
    parse_errors = sum(1 for x in selected if x.get("second_parse_error"))
    summary = {
        "run_dir": str(run_dir),
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "seed": args.seed,
        "filters": {
            "primary_label": args.primary_label,
            "conditions": sorted(allowed_conditions),
            "second_labels": sorted(allowed_second_labels),
            "include_only_disagreements": bool(args.include_only_disagreements),
            "second_model": args.second_model or "allenai/wildguard",
        },
        "selected_stats": {
            "by_condition": dict(by_condition),
            "by_primary_label": dict(by_primary),
            "by_second_label": dict(by_second),
            "by_actionability_label": dict(by_actionability),
            "disagreement_rate": (disagreements / len(selected)) if selected else 0.0,
            "second_parse_error_rate": (parse_errors / len(selected)) if selected else 0.0,
        },
        "output_jsonl": str(out_jsonl),
    }
    write_json(out_summary, summary)
    print(f"Wrote calibration slice to {out_jsonl} ({len(selected)} rows)")
    print(f"Wrote calibration summary to {out_summary}")


if __name__ == "__main__":
    main()
