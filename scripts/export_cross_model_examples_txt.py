from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils_io import read_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export readable cross-model examples from evidence_pack.jsonl.")
    parser.add_argument(
        "--evidence-jsonl",
        default="reports/safety_cross_model_2026-03-25/evidence_pack.jsonl",
    )
    parser.add_argument(
        "--output-txt",
        default=str(ROOT.parent / "cross_model_examples_2026-03-25.txt"),
    )
    parser.add_argument(
        "--snapshots-root",
        default="outputs/cross_model_snapshots",
        help="Optional source for non_rag examples from sampled run outputs.",
    )
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--max-query-chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--max-response-chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--max-doc-chars", type=int, default=0, help="0 means no truncation.")
    return parser.parse_args()


def _trim(text: Any, max_chars: int) -> str:
    t = str(text or "").strip()
    if max_chars <= 0:
        return t
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + " ..."


def _is_rag(condition: str) -> bool:
    return condition in {"rag_docs", "rag_llm"}


def _pick(rows: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda r: (
            str(r.get("run_id", "")),
            str(r.get("condition", "")),
            str(r.get("prompt_id", "")),
        ),
    )
    return ordered[: max(limit, 0)]


def _model_sort_key(model: str) -> tuple[int, str]:
    low = model.lower()
    if "mistral" in low or "ministral" in low:
        return (0, model)
    if "llama" in low:
        return (1, model)
    return (2, model)


def _is_unsafe_majority(primary_label: str, comparison_labels: Dict[str, Any]) -> bool:
    second = [str(v) for v in (comparison_labels or {}).values()]
    unsafe_votes = int(primary_label == "unsafe") + sum(1 for v in second if v == "unsafe")
    total = 1 + len(second)
    return unsafe_votes >= (total + 1) // 2


def _is_safe_majority(primary_label: str, comparison_labels: Dict[str, Any]) -> bool:
    return not _is_unsafe_majority(primary_label, comparison_labels)


def _extract_docs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("docs_top", "docs", "contexts", "retrieved_docs", "retrieved_contexts"):
        docs = row.get(key)
        if isinstance(docs, list):
            out: List[Dict[str, Any]] = []
            for d in docs:
                if isinstance(d, dict):
                    out.append(d)
            if out:
                return out
    return []


def _load_snapshot_non_rag(snapshots_root: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Returns per-model pools from sampled run snapshots:
      pools[model]["non_rag_unsafe_actionable"], pools[model]["non_rag_unsafe_non_actionable"], pools[model]["non_rag_safe"]
    """
    pools: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    if not snapshots_root.exists():
        return pools

    run_dirs = []
    for run_root in snapshots_root.iterdir():
        if not run_root.is_dir():
            continue
        for run_dir in run_root.iterdir():
            if run_dir.is_dir():
                run_dirs.append((run_root.name, run_dir))

    for run_root_name, run_dir in run_dirs:
        responses_path = run_dir / "responses.sample_1000.jsonl"
        judge_path = run_dir / "judge.sample_1000.jsonl"
        actionability_path = run_dir / "actionability.sample_1000.jsonl"
        comparison_path = run_dir / "judge_comparison.sample_1000.jsonl"
        if not (responses_path.exists() and judge_path.exists() and actionability_path.exists()):
            continue

        responses: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        judge: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        actionability: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        comparison: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for row in read_jsonl(responses_path):
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            responses[key] = row
        for row in read_jsonl(judge_path):
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            judge[key] = row
        for row in read_jsonl(actionability_path):
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            actionability[key] = row
        if comparison_path.exists():
            for row in read_jsonl(comparison_path):
                key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
                comparison[key] = row

        for key, r in responses.items():
            prompt_id, condition, model = key
            if condition != "non_rag":
                continue
            p = judge.get(key)
            a = actionability.get(key)
            if not p or not a:
                continue
            comp_labels = (comparison.get(key) or {}).get("comparison_labels") or {}
            primary_label = str(p.get("label", "unknown"))
            is_unsafe = _is_unsafe_majority(primary_label, comp_labels)
            act = str(a.get("actionability_label", ""))
            if is_unsafe:
                bucket = "non_rag_unsafe_actionable" if act == "actionable_unsafe" else "non_rag_unsafe_non_actionable"
            else:
                bucket = "non_rag_safe"

            entry = {
                "run_id": run_dir.name,
                "condition": "non_rag",
                "model": model,
                "prompt_id": prompt_id,
                "query": r.get("query") or r.get("prompt") or "",
                "response": r.get("response", ""),
                "actionability_label": act,
                "primary_label": primary_label,
                "comparison_labels": comp_labels,
                "docs_top": [],
                "slice": bucket,
                "run_root": run_root_name,
            }
            pools.setdefault(model, {}).setdefault(bucket, []).append(entry)

    return pools


def _format_examples(title: str, rows: List[Dict[str, Any]], limit: int, max_q: int, max_r: int, max_d: int) -> List[str]:
    out: List[str] = [f"  {title}: {min(len(rows), limit)}/{limit}"]
    chosen = _pick(rows, limit)
    if not chosen:
        out.append("    - (none)")
        return out
    for i, row in enumerate(chosen, start=1):
        comp = row.get("comparison_labels") or {}
        comp_s = json.dumps(comp, ensure_ascii=True, sort_keys=True)
        act = str(row.get("actionability_label") or "")
        if not act:
            slice_name = str(row.get("slice", ""))
            act = "not_applicable_safe" if "ctx_safe" in slice_name else "missing_for_unsafe"
        docs = _extract_docs(row)
        out.extend(
            [
                f"    [{i}] prompt_id={row.get('prompt_id')} | condition={row.get('condition')} | run={row.get('run_id')}",
                "        actionability={act} | primary={primary} ({source}) | llamaguard={lg} | shieldgemma={sg} | wildguard={wg} | second={comp}".format(
                    act=act,
                    primary=row.get("primary_label"),
                    source=row.get("primary_label_source", "unknown"),
                    lg=row.get("llamaguard_label", "unknown"),
                    sg=row.get("shieldgemma_label", "unknown"),
                    wg=row.get("wildguard_label", "unknown"),
                    comp=comp_s,
                ),
                "        doc_safety(condition-level): safe={safe} unsafe={unsafe} unknown={unk} total={tot} | unsafe_rate={ur:.4f}".format(
                    safe=row.get("doc_condition_safe", 0),
                    unsafe=row.get("doc_condition_unsafe", 0),
                    unk=row.get("doc_condition_unknown", 0),
                    tot=row.get("doc_condition_total", 0),
                    ur=float(row.get("doc_condition_unsafe_rate", 0.0) or 0.0),
                ),
                f"        query: {_trim(row.get('query', ''), max_q)}",
                f"        response: {_trim(row.get('response', ''), max_r)}",
            ]
        )
        if not docs:
            out.append("        contexts: (none)")
        else:
            out.append(f"        contexts ({len(docs)}):")
            for j, d in enumerate(docs, start=1):
                doc_id = d.get("doc_id")
                score = d.get("score")
                head = f"          [{j}]"
                if doc_id is not None:
                    head += f" doc_id={doc_id}"
                if score is not None:
                    head += f" score={score}"
                label = d.get("label")
                if label is not None:
                    head += f" label={label}"
                out.append(head)
                out.append(f"              {_trim(d.get('text', ''), max_d)}")
    return out


def main() -> None:
    args = _parse_args()
    evidence_path = Path(args.evidence_jsonl)
    out_path = Path(args.output_txt)
    rows = list(read_jsonl(evidence_path))
    snapshot_non_rag = _load_snapshot_non_rag(Path(args.snapshots_root))

    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        model = str(row.get("model", "unknown"))
        by_model.setdefault(model, []).append(row)
    # Also include models that exist only in snapshot pools so output is not empty
    # when evidence_pack.jsonl is unavailable/empty.
    for model in snapshot_non_rag.keys():
        by_model.setdefault(model, [])

    lines: List[str] = []
    lines.append("Cross-Model Example Digest")
    lines.append(f"Source: {evidence_path}")
    lines.append(f"Examples per requested bucket: {args.per_bucket}")
    lines.append("")

    for model in sorted(by_model.keys(), key=_model_sort_key):
        model_rows = by_model[model]
        lines.append("=" * 110)
        lines.append(f"MODEL: {model}")
        lines.append("")
        lines.append("Section A - Unsafe answers by setting (actionable vs non-actionable)")
        lines.extend(
            _format_examples(
                "non_rag unsafe actionable",
                snapshot_non_rag.get(model, {}).get("non_rag_unsafe_actionable", []),
                args.per_bucket,
                args.max_query_chars,
                args.max_response_chars,
                args.max_doc_chars,
            )
        )
        lines.extend(
            _format_examples(
                "non_rag unsafe non-actionable",
                snapshot_non_rag.get(model, {}).get("non_rag_unsafe_non_actionable", []),
                args.per_bucket,
                args.max_query_chars,
                args.max_response_chars,
                args.max_doc_chars,
            )
        )
        lines.extend(
            _format_examples(
                "non_rag safe",
                snapshot_non_rag.get(model, {}).get("non_rag_safe", []),
                args.per_bucket,
                args.max_query_chars,
                args.max_response_chars,
                args.max_doc_chars,
            )
        )
        lines.extend(
            _format_examples(
                "rag unsafe actionable",
                [
                    r
                    for r in model_rows
                    if _is_rag(str(r.get("condition", ""))) and str(r.get("slice", "")).endswith("unsafe_actionable")
                ],
                args.per_bucket,
                args.max_query_chars,
                args.max_response_chars,
                args.max_doc_chars,
            )
        )
        lines.extend(
            _format_examples(
                "rag unsafe non-actionable",
                [
                    r
                    for r in model_rows
                    if _is_rag(str(r.get("condition", ""))) and str(r.get("slice", "")).endswith("unsafe_non_actionable")
                ],
                args.per_bucket,
                args.max_query_chars,
                args.max_response_chars,
                args.max_doc_chars,
            )
        )
        lines.extend(
            _format_examples(
                "rag safe (majority-safe)",
                [
                    r
                    for r in model_rows
                    if _is_rag(str(r.get("condition", "")))
                    and _is_safe_majority(str(r.get("primary_label", "unknown")), r.get("comparison_labels") or {})
                ],
                args.per_bucket,
                args.max_query_chars,
                args.max_response_chars,
                args.max_doc_chars,
            )
        )
        lines.append("")
        lines.append("Section B - Context-grounded unsafe answers")
        lines.extend(
            _format_examples(
                "safe context -> unsafe answer",
                [
                    r
                    for r in model_rows
                    if _is_rag(str(r.get("condition", ""))) and str(r.get("slice", "")).startswith("safe_ctx_unsafe")
                ],
                args.per_bucket,
                args.max_query_chars,
                args.max_response_chars,
                args.max_doc_chars,
            )
        )
        lines.extend(
            _format_examples(
                "unsafe context -> unsafe answer",
                [
                    r
                    for r in model_rows
                    if _is_rag(str(r.get("condition", ""))) and str(r.get("slice", "")).startswith("unsafe_ctx_unsafe")
                ],
                args.per_bucket,
                args.max_query_chars,
                args.max_response_chars,
                args.max_doc_chars,
            )
        )
        lines.extend(
            _format_examples(
                "unsafe context -> safe answer",
                [
                    r
                    for r in model_rows
                    if _is_rag(str(r.get("condition", ""))) and str(r.get("slice", "")) == "unsafe_ctx_safe"
                ],
                args.per_bucket,
                args.max_query_chars,
                args.max_response_chars,
                args.max_doc_chars,
            )
        )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote readable digest to {out_path}")


if __name__ == "__main__":
    main()
