from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_MODELS = [
    "qwen3_4b",
    "llama32_3b",
    "phi3mini",
    "gemma_7b",
    "deepseek_r1d_qwen7b",
    "mistral7b_v03",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export report-ready safety and capability CSV summaries.")
    parser.add_argument(
        "--outputs-root",
        default="/scratch/adi7/rag-test/outputs",
        help="Root containing safety_*_report and capability_*_report directories.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/report_metrics_2026-04-13",
        help="Directory to write summary CSVs into.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Short model keys used in safety_<key>_report and capability_<key>_report.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_run_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_safety_rows(outputs_root: Path, models: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for short in models:
        run_dir = _latest_run_dir(outputs_root / f"safety_{short}_report")
        if run_dir is None:
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _read_json(metrics_path)
        for model_name, conds in (metrics.get("by_model_condition", {}) or {}).items():
            for condition, vals in conds.items():
                rows.append(
                    {
                        "run_group": short,
                        "run_id": run_dir.name,
                        "model": model_name,
                        "condition": condition,
                        "unsafe_actionable_rate": vals.get("unsafe_actionable_rate"),
                        "safe_doc_unsafe_actionable_rate": vals.get("safe_doc_unsafe_actionable_rate"),
                        "unsafe_raw_rate": vals.get("unsafe_raw_rate"),
                        "unsafe_rate": vals.get("unsafe_rate"),
                        "refusal_rate": vals.get("refusal_rate"),
                        "unsafe_actionable": vals.get("unsafe_actionable"),
                        "unsafe": vals.get("unsafe"),
                        "refusal": vals.get("refusal"),
                        "total": vals.get("total"),
                    }
                )
    return rows


def _collect_capability_rows(outputs_root: Path, models: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for short in models:
        run_dir = _latest_run_dir(outputs_root / f"capability_{short}_report")
        if run_dir is None:
            continue
        metrics_path = run_dir / "qa_metrics.json"
        retrieval_path = run_dir / "retrieval_metrics.json"
        if not metrics_path.exists():
            continue
        qa = _read_json(metrics_path)
        retrieval = _read_json(retrieval_path) if retrieval_path.exists() else {}
        retrieval_fields = {
            "mixed_answer_presence_at_k": retrieval.get("mixed_answer_presence_at_k"),
            "oracle_answer_presence_at_k": retrieval.get("oracle_answer_presence_at_k"),
            "k": retrieval.get("k"),
            "mode": retrieval.get("mode"),
        }
        for model_name, conds in (qa.get("by_model_condition", {}) or {}).items():
            for condition, vals in conds.items():
                row = {
                    "run_group": short,
                    "run_id": run_dir.name,
                    "model": model_name,
                    "condition": condition,
                    "em": vals.get("em"),
                    "f1": vals.get("f1"),
                    "contains_answer_rate": vals.get("contains_answer_rate"),
                    "faithfulness_score": vals.get("faithfulness_score"),
                    "context_precision": vals.get("context_precision"),
                    "hallucination_proxy": vals.get("hallucination_proxy"),
                    "refusal_rate": vals.get("refusal_rate"),
                    "count": vals.get("count"),
                }
                row.update(retrieval_fields)
                rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    outputs_root = Path(args.outputs_root)
    report_dir = Path(args.report_dir)

    safety_rows = _collect_safety_rows(outputs_root, args.models)
    capability_rows = _collect_capability_rows(outputs_root, args.models)

    _write_csv(
        report_dir / "report_safety_summary_2026-04-13.csv",
        safety_rows,
        [
            "run_group",
            "run_id",
            "model",
            "condition",
            "unsafe_actionable_rate",
            "safe_doc_unsafe_actionable_rate",
            "unsafe_raw_rate",
            "unsafe_rate",
            "refusal_rate",
            "unsafe_actionable",
            "unsafe",
            "refusal",
            "total",
        ],
    )
    _write_csv(
        report_dir / "report_capability_summary_2026-04-13.csv",
        capability_rows,
        [
            "run_group",
            "run_id",
            "model",
            "condition",
            "em",
            "f1",
            "contains_answer_rate",
            "faithfulness_score",
            "context_precision",
            "hallucination_proxy",
            "refusal_rate",
            "count",
            "oracle_answer_presence_at_k",
            "mixed_answer_presence_at_k",
            "k",
            "mode",
        ],
    )
    print(f"Wrote {report_dir / 'report_safety_summary_2026-04-13.csv'}")
    print(f"Wrote {report_dir / 'report_capability_summary_2026-04-13.csv'}")


if __name__ == "__main__":
    main()
