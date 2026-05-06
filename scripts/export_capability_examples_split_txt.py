from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils_io import read_jsonl


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export capability examples into split TXT files by model and/or setting.")
    p.add_argument(
        "--report-csv",
        default="reports/report_metrics_2026-04-13/report_capability_summary_2026-04-13.csv",
        help="Path to report capability summary CSV.",
    )
    p.add_argument(
        "--outputs-root",
        default=None,
        help="Outputs root containing capability run folders. Default: /scratch/$USER/rag-test/outputs if present else ./outputs.",
    )
    p.add_argument(
        "--output-dir",
        default="reports/report_metrics_2026-04-13/samples/split_txt",
        help="Output directory for split text files.",
    )
    p.add_argument(
        "--mode",
        choices=["model", "setting", "both"],
        default="both",
        help="How to split outputs.",
    )
    p.add_argument("--per-file", type=int, default=120, help="Max examples per split file.")
    p.add_argument("--max-query-chars", type=int, default=0, help="0 means no truncation.")
    p.add_argument("--max-response-chars", type=int, default=0, help="0 means no truncation.")
    p.add_argument("--max-doc-chars", type=int, default=0, help="0 means no truncation.")
    p.add_argument(
        "--contexts-per-row",
        type=int,
        default=3,
        help="How many retrieved snippets to show per example row.",
    )
    return p.parse_args()


def _default_outputs_root() -> Path:
    scratch = Path(f"/scratch/{Path.home().name}/rag-test/outputs")
    if scratch.exists():
        return scratch
    return ROOT / "outputs"


def _trim(text: Any, max_chars: int) -> str:
    s = str(text or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars] + " ..."


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "unknown"


def _read_report_pairs(report_csv: Path) -> List[Tuple[str, str]]:
    seen = set()
    pairs: List[Tuple[str, str]] = []
    with report_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_group = str(row.get("run_group", "")).strip()
            run_id = str(row.get("run_id", "")).strip()
            if not run_group or not run_id:
                continue
            key = (run_group, run_id)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def _load_context_map(path: Path) -> Dict[Tuple[str, str, str], List[str]]:
    out: Dict[Tuple[str, str, str], List[str]] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
        snippets = row.get("retrieved_context_snippets_used") or []
        if isinstance(snippets, list):
            out[key] = [str(s) for s in snippets]
        else:
            out[key] = []
    return out


def _build_rows(report_csv: Path, outputs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_group, run_id in _read_report_pairs(report_csv):
        run_dir = outputs_root / f"capability_{run_group}_report" / run_id
        responses_path = run_dir / "qa_responses.jsonl"
        contexts_path = run_dir / "qa_context_used.jsonl"
        if not responses_path.exists():
            print(f"WARN missing qa_responses.jsonl: {responses_path}")
            continue
        context_map = _load_context_map(contexts_path)
        for row in read_jsonl(responses_path):
            key = (
                str(row.get("prompt_id", "")),
                str(row.get("condition", "")),
                str(row.get("model", "")),
            )
            rows.append(
                {
                    "run_group": run_group,
                    "run_id": run_id,
                    "model": str(row.get("model", "")),
                    "condition": str(row.get("condition", "")),
                    "prompt_id": str(row.get("prompt_id", "")),
                    "query": str(row.get("query", "")),
                    "response": str(row.get("response", "")),
                    "em": row.get("em"),
                    "f1": row.get("f1"),
                    "contains_answer": row.get("contains_answer"),
                    "refusal": row.get("refusal"),
                    "faithfulness": row.get("faithfulness"),
                    "context_precision": row.get("context_precision"),
                    "hallucination_proxy": row.get("hallucination_proxy"),
                    "contexts": context_map.get(key, []),
                }
            )
    return rows


def _format_rows(
    rows: Iterable[Dict[str, Any]],
    max_q: int,
    max_r: int,
    max_d: int,
    limit: int,
    contexts_per_row: int,
) -> str:
    lines: List[str] = []
    count = 0
    for r in rows:
        if count >= limit:
            break
        count += 1
        lines.append("-" * 110)
        lines.append(
            f"[{count}] run={r.get('run_id')} | model={r.get('model')} | condition={r.get('condition')} | prompt_id={r.get('prompt_id')}"
        )
        lines.append(
            "    em={em} | f1={f1} | contains_answer={ca} | refusal={ref} | faithfulness={faith} | context_precision={cp} | hallucination_proxy={hall}".format(
                em=r.get("em"),
                f1=r.get("f1"),
                ca=r.get("contains_answer"),
                ref=r.get("refusal"),
                faith=r.get("faithfulness"),
                cp=r.get("context_precision"),
                hall=r.get("hallucination_proxy"),
            )
        )
        lines.append(f"    query: {_trim(r.get('query', ''), max_q)}")
        lines.append(f"    response: {_trim(r.get('response', ''), max_r)}")
        ctx = r.get("contexts") or []
        if not isinstance(ctx, list) or not ctx:
            lines.append("    contexts: (none)")
        else:
            lines.append(f"    contexts ({len(ctx)}):")
            for j, snippet in enumerate(ctx[: max(0, contexts_per_row)], start=1):
                lines.append(f"      [{j}] {_trim(snippet, max_d)}")

    if count == 0:
        lines.append("(no rows)")
    return "\n".join(lines) + "\n"


def _write_split(out_path: Path, title: str, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            str(r.get("condition", "")),
            str(r.get("model", "")),
            str(r.get("run_id", "")),
            str(r.get("prompt_id", "")),
        ),
    )
    content = [
        "Capability Split Export",
        title,
        f"source_report={args.report_csv}",
        f"rows={len(rows_sorted)}",
        "",
        _format_rows(
            rows_sorted,
            args.max_query_chars,
            args.max_response_chars,
            args.max_doc_chars,
            args.per_file,
            args.contexts_per_row,
        ),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(content), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    args = _parse_args()
    report_csv = Path(args.report_csv)
    outputs_root = Path(args.outputs_root) if args.outputs_root else _default_outputs_root()
    rows = _build_rows(report_csv, outputs_root)
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

    print(f"Rows exported: {len(rows)}")


if __name__ == "__main__":
    main()
