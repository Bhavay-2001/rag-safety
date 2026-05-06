from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils_io import read_jsonl, write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and summarize manual A/R/U calibration audits.")
    parser.add_argument("--input-jsonl", required=True, help="Calibration JSONL (e.g., wildguard_calibration_100.jsonl).")
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Optional path to export annotation template CSV.",
    )
    parser.add_argument(
        "--annotations-csv",
        default=None,
        help="Annotated CSV path with manual_label column (A/R/U).",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Summary JSON output path (default: <input>.manual_summary.json).",
    )
    parser.add_argument(
        "--output-joined-jsonl",
        default=None,
        help="Optional joined JSONL output with manual labels attached.",
    )
    parser.add_argument(
        "--label-field",
        default="manual_label",
        help="Column name in annotations CSV containing manual labels (default: manual_label).",
    )
    return parser.parse_args()


def _key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))


def _safe_rate(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def _norm_manual_label(v: str) -> str:
    s = str(v or "").strip().upper()
    if s in {"A", "ACTIONABLE", "ACTIONABLE_UNSAFE"}:
        return "A"
    if s in {"R", "REFUSAL", "BENIGN_REFUSAL", "NON_ACTIONABLE_REFUSAL"}:
        return "R"
    if s in {"U", "UNCLEAR", "MIXED", "UNKNOWN"}:
        return "U"
    return ""


def _export_template(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prompt_id",
        "condition",
        "model",
        "primary_label",
        "second_label",
        "actionability_label",
        "manual_label",
        "notes",
        "query",
        "response",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "prompt_id": r.get("prompt_id"),
                    "condition": r.get("condition"),
                    "model": r.get("model"),
                    "primary_label": r.get("primary_label"),
                    "second_label": r.get("second_label"),
                    "actionability_label": r.get("actionability_label"),
                    "manual_label": "",
                    "notes": "",
                    "query": str(r.get("query", "")),
                    "response": str(r.get("response", "")),
                }
            )
    print(f"Wrote annotation template CSV to {csv_path}")


def _load_annotations(path: Path, label_field: str) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            out[k] = {
                "manual_label": _norm_manual_label(str(row.get(label_field, ""))),
                "notes": str(row.get("notes", "")),
            }
    return out


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    by_manual = Counter(r.get("manual_label", "") for r in rows if r.get("manual_label"))
    by_condition_manual: Dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        m = r.get("manual_label", "")
        if not m:
            continue
        by_condition_manual[str(r.get("condition", ""))][m] += 1

    # Cross-tabs
    primary_vs_manual: Dict[str, Counter[str]] = defaultdict(Counter)
    second_vs_manual: Dict[str, Counter[str]] = defaultdict(Counter)
    auto_actionability_vs_manual: Dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        m = r.get("manual_label", "")
        if not m:
            continue
        primary_vs_manual[str(r.get("primary_label", ""))][m] += 1
        second_vs_manual[str(r.get("second_label", ""))][m] += 1
        auto_actionability_vs_manual[str(r.get("actionability_label", ""))][m] += 1

    annotated = sum(by_manual.values())
    a = by_manual.get("A", 0)
    r_cnt = by_manual.get("R", 0)
    u = by_manual.get("U", 0)
    summary = {
        "rows_total": total,
        "rows_annotated": annotated,
        "annotation_coverage": _safe_rate(annotated, total),
        "manual_label_counts": dict(by_manual),
        "manual_label_rates": {
            "A_rate": _safe_rate(a, annotated),
            "R_rate": _safe_rate(r_cnt, annotated),
            "U_rate": _safe_rate(u, annotated),
        },
        "derived_metrics": {
            "primary_unsafe_actionable_precision": _safe_rate(a, annotated),
            "primary_unsafe_refusal_false_positive_rate": _safe_rate(r_cnt, annotated),
        },
        "by_condition_manual_counts": {k: dict(v) for k, v in by_condition_manual.items()},
        "by_condition_manual_rates": {
            cond: {
                "A_rate": _safe_rate(cnts.get("A", 0), sum(cnts.values())),
                "R_rate": _safe_rate(cnts.get("R", 0), sum(cnts.values())),
                "U_rate": _safe_rate(cnts.get("U", 0), sum(cnts.values())),
            }
            for cond, cnts in by_condition_manual.items()
        },
        "primary_vs_manual": {k: dict(v) for k, v in primary_vs_manual.items()},
        "second_vs_manual": {k: dict(v) for k, v in second_vs_manual.items()},
        "auto_actionability_vs_manual": {k: dict(v) for k, v in auto_actionability_vs_manual.items()},
    }
    return summary


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_jsonl)
    rows = list(read_jsonl(input_path))
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    if args.export_csv:
        _export_template(rows, Path(args.export_csv))

    if not args.annotations_csv:
        return

    annotations = _load_annotations(Path(args.annotations_csv), args.label_field)
    joined_rows: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        ann = annotations.get(_key(row), {})
        out["manual_label"] = ann.get("manual_label", "")
        out["manual_notes"] = ann.get("notes", "")
        joined_rows.append(out)

    summary = _summarize(joined_rows)
    out_summary = (
        Path(args.output_summary)
        if args.output_summary
        else input_path.with_name(input_path.stem + ".manual_summary.json")
    )
    write_json(out_summary, summary)
    print(f"Wrote manual audit summary to {out_summary}")

    if args.output_joined_jsonl:
        out_joined = Path(args.output_joined_jsonl)
        write_jsonl(out_joined, joined_rows)
        print(f"Wrote joined annotated rows to {out_joined}")


if __name__ == "__main__":
    main()
