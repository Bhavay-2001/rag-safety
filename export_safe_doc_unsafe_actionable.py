from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.metrics import compute_doc_set_labels
from src.utils_io import read_jsonl, write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export safe-doc -> unsafe-actionable examples.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument(
        "--unsafe-policy",
        choices=["primary", "any", "majority", "unanimous"],
        default="majority",
        help="Unsafe vote policy for selecting true-positive candidates.",
    )
    parser.add_argument("--docs-per-row", type=int, default=3)
    parser.add_argument("--max-doc-chars", type=int, default=800)
    return parser.parse_args()


def _load_comparison_map(run_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    cmp_path = run_dir / "judge_comparison.jsonl"
    if cmp_path.exists():
        for row in read_jsonl(cmp_path):
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            labels = row.get("comparison_labels") or {}
            out[key] = {str(k): str(v) for k, v in labels.items()}
        return out

    second_path = run_dir / "judge_second_pass.jsonl"
    if second_path.exists():
        for row in read_jsonl(second_path):
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            sid = str(row.get("second_judge_model_id", "second"))
            out.setdefault(key, {})[sid] = str(row.get("second_label", "unknown"))
    return out


def _trim(text: str, limit: int) -> str:
    t = str(text or "").strip()
    if limit <= 0 or len(t) <= limit:
        return t
    return t[:limit] + " ..."


def _unsafe_by_policy(primary: str, second_labels: List[str], policy: str) -> bool:
    primary_unsafe = primary == "unsafe"
    total = 1 + len(second_labels)
    votes = int(primary_unsafe) + sum(1 for v in second_labels if v == "unsafe")
    if policy == "primary":
        return primary_unsafe
    if policy == "any":
        return votes >= 1
    if policy == "majority":
        return votes >= math.ceil(total / 2)
    return votes == total


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    out_jsonl = Path(args.output_jsonl) if args.output_jsonl else run_dir / "safe_doc_unsafe_actionable_examples.jsonl"
    out_summary = (
        Path(args.output_summary)
        if args.output_summary
        else run_dir / "safe_doc_unsafe_actionable_examples.summary.json"
    )

    responses = {(r["prompt_id"], r["condition"], r["model"]): r for r in read_jsonl(run_dir / "responses.jsonl")}
    primary = {(r["prompt_id"], r["condition"], r["model"]): r for r in read_jsonl(run_dir / "judge.jsonl")}
    actionability = {(r["prompt_id"], r["condition"], r["model"]): r for r in read_jsonl(run_dir / "actionability.jsonl")}
    doc_rows = list(read_jsonl(run_dir / "doc_safety.jsonl")) if (run_dir / "doc_safety.jsonl").exists() else []
    doc_set = compute_doc_set_labels(doc_rows)
    comparison = _load_comparison_map(run_dir)

    docs_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in read_jsonl(run_dir / "retrieval.jsonl"):
        key = (str(row.get("prompt_id", "")), str(row.get("condition", "")))
        docs_map[key] = row.get("docs") or []

    selected: List[Dict[str, Any]] = []
    for key, p in primary.items():
        prompt_id, condition, model = key
        a = actionability.get(key)
        if not a:
            continue
        if str(a.get("actionability_label")) != "actionable_unsafe":
            continue
        if doc_set.get((prompt_id, condition)) != "safe":
            continue

        sec = comparison.get(key, {})
        sec_labels = list(sec.values())
        if not _unsafe_by_policy(str(p.get("label", "unknown")), sec_labels, args.unsafe_policy):
            continue

        resp = responses.get(key)
        if not resp:
            continue

        docs = []
        for d in docs_map.get((prompt_id, condition), [])[: max(args.docs_per_row, 0)]:
            docs.append(
                {
                    "doc_id": str(d.get("doc_id", "")),
                    "score": d.get("score"),
                    "text": _trim(str(d.get("text", "")), args.max_doc_chars),
                }
            )

        selected.append(
            {
                "prompt_id": prompt_id,
                "condition": condition,
                "model": model,
                "query": str(resp.get("query") or resp.get("prompt") or ""),
                "response": str(resp.get("response", "")),
                "doc_set_label": "safe",
                "primary_label": p.get("label"),
                "primary_categories": p.get("categories") or [],
                "comparison_labels": sec,
                "judge_triplet": {
                    "primary": p.get("label"),
                    "wildguard": sec.get("allenai/wildguard"),
                    "shieldgemma": sec.get("google/shieldgemma-2b"),
                },
                "actionability_label": a.get("actionability_label"),
                "unsafe_policy": args.unsafe_policy,
                "docs_top": docs,
            }
        )

    selected = sorted(selected, key=lambda x: (x["condition"], x["prompt_id"], x["model"]))[: max(args.max_rows, 0)]

    if out_jsonl.exists():
        out_jsonl.unlink()
    write_jsonl(out_jsonl, selected)

    summary = {
        "run_dir": str(run_dir),
        "unsafe_policy": args.unsafe_policy,
        "selected_rows": len(selected),
        "by_condition": {},
        "output_jsonl": str(out_jsonl),
    }
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row["condition"]] = counts.get(row["condition"], 0) + 1
    summary["by_condition"] = counts
    write_json(out_summary, summary)

    print(f"Wrote safe-doc actionable examples to {out_jsonl} ({len(selected)} rows)")
    print(f"Wrote summary to {out_summary}")


if __name__ == "__main__":
    main()
