from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils_io import read_jsonl


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export evidence examples into split TXT files by model and/or setting.")
    p.add_argument(
        "--evidence-jsonl",
        default="reports/safety_cross_model_2026-03-25/evidence_pack.jsonl",
        help="Path to evidence pack jsonl.",
    )
    p.add_argument(
        "--output-dir",
        default="reports/safety_cross_model_2026-03-25/samples/split_txt",
        help="Output directory for split text files.",
    )
    p.add_argument(
        "--mode",
        choices=["model", "setting", "both"],
        default="both",
        help="How to split outputs.",
    )
    p.add_argument("--per-file", type=int, default=80, help="Max examples per split file.")
    p.add_argument("--max-query-chars", type=int, default=0, help="0 means no truncation.")
    p.add_argument("--max-response-chars", type=int, default=0, help="0 means no truncation.")
    p.add_argument("--max-doc-chars", type=int, default=0, help="0 means no truncation.")
    return p.parse_args()


def _trim(text: Any, max_chars: int) -> str:
    s = str(text or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars] + " ..."


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "unknown"


def _extract_docs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("docs_top", "docs", "contexts", "retrieved_docs", "retrieved_contexts"):
        v = row.get(key)
        if isinstance(v, list):
            docs = [d for d in v if isinstance(d, dict)]
            if docs:
                return docs
    return []


def _format_rows(rows: Iterable[Dict[str, Any]], max_q: int, max_r: int, max_d: int, limit: int) -> str:
    lines: List[str] = []
    count = 0
    for r in rows:
        if count >= limit:
            break
        count += 1
        comp = r.get("comparison_labels") or {}
        act = str(r.get("actionability_label") or "")
        if not act:
            slice_name = str(r.get("slice", ""))
            act = "not_applicable_safe" if "ctx_safe" in slice_name else "missing_for_unsafe"
        lines.append("-" * 110)
        lines.append(
            f"[{count}] run={r.get('run_id')} | model={r.get('model')} | condition={r.get('condition')} | "
            f"prompt_id={r.get('prompt_id')} | slice={r.get('slice')}"
        )
        lines.append(
            f"    actionability={act} | primary={r.get('primary_label')} ({r.get('primary_label_source','unknown')}) | "
            f"llamaguard={r.get('llamaguard_label','unknown')} | shieldgemma={r.get('shieldgemma_label','unknown')} | "
            f"wildguard={r.get('wildguard_label','unknown')} | "
            f"comparison={json.dumps(comp, ensure_ascii=True, sort_keys=True)}"
        )
        lines.append(
            "    doc_safety(condition-level): safe={safe} unsafe={unsafe} unknown={unk} total={tot} | unsafe_rate={ur:.4f}".format(
                safe=r.get("doc_condition_safe", 0),
                unsafe=r.get("doc_condition_unsafe", 0),
                unk=r.get("doc_condition_unknown", 0),
                tot=r.get("doc_condition_total", 0),
                ur=float(r.get("doc_condition_unsafe_rate", 0.0) or 0.0),
            )
        )
        lines.append(f"    query: {_trim(r.get('query', ''), max_q)}")
        lines.append(f"    response: {_trim(r.get('response', ''), max_r)}")

        docs = _extract_docs(r)
        if not docs:
            lines.append("    contexts: (none)")
        else:
            lines.append(f"    contexts ({len(docs)}):")
            for j, d in enumerate(docs, start=1):
                doc_id = d.get("doc_id")
                score = d.get("score")
                head = f"      [{j}]"
                if doc_id is not None:
                    head += f" doc_id={doc_id}"
                if score is not None:
                    head += f" score={score}"
                label = d.get("label")
                if label is not None:
                    head += f" label={label}"
                lines.append(head)
                lines.append(f"           {_trim(d.get('text', ''), max_d)}")

    if count == 0:
        lines.append("(no rows)")
    return "\n".join(lines) + "\n"


def _write_split(out_path: Path, title: str, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            str(r.get("condition", "")),
            str(r.get("model", "")),
            str(r.get("slice", "")),
            str(r.get("run_id", "")),
            str(r.get("prompt_id", "")),
        ),
    )
    content = [
        "Cross-Model Split Export",
        title,
        f"source={args.evidence_jsonl}",
        f"rows={len(rows_sorted)}",
        "",
        _format_rows(rows_sorted, args.max_query_chars, args.max_response_chars, args.max_doc_chars, args.per_file),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(content), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    args = _parse_args()
    rows = list(read_jsonl(Path(args.evidence_jsonl)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"model", "both"}:
        by_model: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_model.setdefault(str(r.get("model", "unknown")), []).append(r)
        for model, model_rows in sorted(by_model.items(), key=lambda kv: kv[0].lower()):
            fname = f"by_model__{_safe_name(model)}.txt"
            _write_split(out_dir / fname, f"model={model}", model_rows, args)

    if args.mode in {"setting", "both"}:
        by_setting: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_setting.setdefault(str(r.get("condition", "unknown")), []).append(r)
        for setting, setting_rows in sorted(by_setting.items(), key=lambda kv: kv[0].lower()):
            fname = f"by_setting__{_safe_name(setting)}.txt"
            _write_split(out_dir / fname, f"setting={setting}", setting_rows, args)


if __name__ == "__main__":
    main()
