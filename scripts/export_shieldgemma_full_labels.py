from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils_io import read_jsonl, write_json, write_jsonl


TARGET_MODEL_ID = "google/shieldgemma-2b"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full ShieldGemma labels for report assets.")
    parser.add_argument("--outputs-root", required=True, help="Root outputs directory, e.g. /scratch/$USER/rag-test/outputs")
    parser.add_argument("--out-jsonl", required=True, help="Path to write shieldgemma_full_labels.jsonl")
    parser.add_argument("--out-coverage-json", default=None, help="Optional path to write coverage summary JSON")
    parser.add_argument("--out-summary-json", default=None, help="Optional path to write per-model-condition ShieldGemma unsafe-rate summary JSON")
    parser.add_argument("--out-summary-csv", default=None, help="Optional path to write per-model-condition ShieldGemma unsafe-rate summary CSV")
    parser.add_argument(
        "--run-roots",
        nargs="+",
        required=True,
        help="Run roots to include, e.g. safety_qwen3_4b_report safety_llama32_3b_report ...",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["non_rag", "rag_docs", "rag_llm"],
        help="Conditions that must be fully covered",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write output even if ShieldGemma coverage is incomplete",
    )
    return parser.parse_args()


def _latest_run_dir(root: Path) -> Path:
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run directories found in {root}")
    return max(subdirs, key=lambda p: p.name)


def _coerce_label(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in {"safe", "unsafe"}:
        return s
    return "unknown"


def _load_shield_labels(run_dir: Path) -> Dict[Tuple[str, str, str], str]:
    out: Dict[Tuple[str, str, str], str] = {}

    cmp_path = run_dir / "judge_comparison.jsonl"
    if cmp_path.exists():
        for row in read_jsonl(cmp_path):
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            comp = row.get("comparison_labels", {}) or {}
            if TARGET_MODEL_ID in comp:
                out[key] = _coerce_label(comp.get(TARGET_MODEL_ID))
        if out:
            return out

    second_path = run_dir / "judge_second_pass.jsonl"
    if second_path.exists():
        for row in read_jsonl(second_path):
            if str(row.get("second_judge_model_id", "")) != TARGET_MODEL_ID:
                continue
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            out[key] = _coerce_label(row.get("second_label"))
    return out


def _group_coverage(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    grouped: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"total": 0, "covered": 0, "unsafe": 0}))
    for row in records:
        slot = grouped[str(row["model"])][str(row["condition"])]
        slot["total"] += 1
        if row.get("shieldgemma_label") in {"safe", "unsafe"}:
            slot["covered"] += 1
        if row.get("shieldgemma_label") == "unsafe":
            slot["unsafe"] += 1
    return grouped


def _default_summary_path(out_jsonl: Path, suffix: str) -> Path:
    stem = out_jsonl.name
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    return out_jsonl.with_name(f"{stem}.{suffix}")


def _build_summary_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group_coverage(records)
    rows: List[Dict[str, Any]] = []
    for model in sorted(grouped.keys()):
        for condition in sorted(grouped[model].keys()):
            stats = grouped[model][condition]
            covered = int(stats["covered"])
            unsafe = int(stats["unsafe"])
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "shieldgemma_total": int(stats["total"]),
                    "shieldgemma_covered": covered,
                    "shieldgemma_unsafe": unsafe,
                    "shieldgemma_safe": covered - unsafe,
                    "shieldgemma_unsafe_rate": (unsafe / covered) if covered else None,
                }
            )
    return rows


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "condition",
            "shieldgemma_total",
            "shieldgemma_covered",
            "shieldgemma_unsafe",
            "shieldgemma_safe",
            "shieldgemma_unsafe_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            rate = out["shieldgemma_unsafe_rate"]
            out["shieldgemma_unsafe_rate"] = "" if rate is None else f"{float(rate):.6f}"
            writer.writerow(out)


def main() -> None:
    args = _parse_args()
    outputs_root = Path(args.outputs_root)
    out_jsonl = Path(args.out_jsonl)
    coverage_path = Path(args.out_coverage_json) if args.out_coverage_json else None
    summary_json_path = Path(args.out_summary_json) if args.out_summary_json else _default_summary_path(out_jsonl, "summary.json")
    summary_csv_path = Path(args.out_summary_csv) if args.out_summary_csv else _default_summary_path(out_jsonl, "summary.csv")

    exported_rows: List[Dict[str, Any]] = []
    coverage_issues: List[Dict[str, Any]] = []

    for run_root_name in args.run_roots:
        run_root = outputs_root / run_root_name
        run_dir = _latest_run_dir(run_root)

        responses_path = run_dir / "responses.jsonl"
        if not responses_path.exists():
            raise FileNotFoundError(f"Missing responses.jsonl in {run_dir}")

        responses = list(read_jsonl(responses_path))
        shield_map = _load_shield_labels(run_dir)

        run_rows: List[Dict[str, Any]] = []
        for row in responses:
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            run_rows.append(
                {
                    "run_root": run_root_name,
                    "run_id": run_dir.name,
                    "model": str(row.get("model", "")),
                    "condition": str(row.get("condition", "")),
                    "prompt_id": str(row.get("prompt_id", "")),
                    "shieldgemma_label": shield_map.get(key, "unknown"),
                }
            )

        grouped = _group_coverage(run_rows)
        for model, conds in grouped.items():
            for condition, stats in conds.items():
                if condition not in args.conditions:
                    continue
                if stats["covered"] != stats["total"]:
                    coverage_issues.append(
                        {
                            "run_root": run_root_name,
                            "run_id": run_dir.name,
                            "model": model,
                            "condition": condition,
                            "total": stats["total"],
                            "covered": stats["covered"],
                            "missing": stats["total"] - stats["covered"],
                        }
                    )

        exported_rows.extend(run_rows)

    if coverage_path:
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            coverage_path,
            {
                "target_model_id": TARGET_MODEL_ID,
                "conditions_required": args.conditions,
                "allow_partial": args.allow_partial,
                "coverage_issues": coverage_issues,
            },
        )

    if coverage_issues and not args.allow_partial:
        print("ShieldGemma coverage is incomplete. Refusing to write a full-labels file.")
        print("Re-run ShieldGemma second pass over all responses, or use --allow-partial for a diagnostic export.")
        for issue in coverage_issues[:20]:
            print(issue)
        raise SystemExit(2)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_jsonl, exported_rows)
    summary_rows = _build_summary_rows(exported_rows)
    write_json(summary_json_path, {"target_model_id": TARGET_MODEL_ID, "rows": summary_rows})
    _write_summary_csv(summary_csv_path, summary_rows)
    print(f"Wrote {len(exported_rows)} rows to {out_jsonl}")
    print(f"Wrote ShieldGemma summary JSON to {summary_json_path}")
    print(f"Wrote ShieldGemma summary CSV to {summary_csv_path}")
    if coverage_issues:
        print(f"Coverage issues: {len(coverage_issues)} model-condition groups are partial.")
    else:
        print("ShieldGemma coverage is complete for the requested conditions.")


if __name__ == "__main__":
    main()
