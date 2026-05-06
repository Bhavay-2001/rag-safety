from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import compute_doc_set_labels
from src.utils_io import read_jsonl


REQUIRED_COMPLETE_FILES = [
    "metrics.json",
    "judge.jsonl",
    "actionability.jsonl",
    "present_absent.jsonl",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-model safety analysis and evidence pack.")
    parser.add_argument(
        "--outputs-root",
        default=None,
        help="Directory containing safety_* run roots (default: /scratch/$USER/rag-test/outputs if it exists, else ./outputs).",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/safety_cross_model_2026-03-25",
        help="Output directory for inventory/tables/evidence pack.",
    )
    parser.add_argument(
        "--run-roots",
        nargs="+",
        default=[
            "safety_deepseek_qwen15b_A",
            "safety_deepseek_qwen7b_A",
            "safety_llama31_8b_A",
            "safety_llama32_3b_A",
            "safety_mistral_ab",
            "safety_phi3mini_ab",
            "safety_phi3small_ab",
            "safety_qwen25_15b_ab",
            "safety_qwen3_4b_A",
            "safety_qwen3_8b_A",
        ],
        help="Run roots to include, relative to outputs root.",
    )
    parser.add_argument(
        "--expected-families",
        nargs="+",
        default=["deepseek", "gemma", "llama", "mistral", "phi", "qwen"],
        help="Families to mark as not_run if absent.",
    )
    parser.add_argument("--max-per-slice", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--docs-per-row", type=int, default=3)
    parser.add_argument("--max-query-chars", type=int, default=1200)
    parser.add_argument("--max-response-chars", type=int, default=1200)
    parser.add_argument("--max-doc-chars", type=int, default=800)
    parser.add_argument(
        "--response-mode",
        choices=["full", "trimmed"],
        default="full",
        help="How to store response text in evidence_pack.jsonl.",
    )
    parser.add_argument(
        "--full-response-in-evidence",
        action="store_true",
        default=True,
        help="Deprecated: use --response-mode full|trimmed.",
    )
    return parser.parse_args()


def _default_outputs_root() -> Path:
    scratch = Path(f"/scratch/{Path.home().name}/rag-test/outputs")
    if scratch.exists():
        return scratch
    return ROOT / "outputs"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trim(text: Any, max_chars: int) -> str:
    t = str(text or "").strip()
    if max_chars <= 0:
        return t
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + " ..."


def _infer_family(run_root: str, model: str) -> str:
    low = f"{run_root} {model}".lower()
    if "deepseek" in low:
        return "deepseek"
    if "qwen" in low:
        return "qwen"
    if "llama" in low:
        return "llama"
    if "mistral" in low or "ministral" in low:
        return "mistral"
    if "phi" in low:
        return "phi"
    if "gemma" in low:
        return "gemma"
    return "other"


def _load_comparison_map(run_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    cmp_path = run_dir / "judge_comparison.jsonl"
    if cmp_path.exists():
        for row in read_jsonl(cmp_path):
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            out[key] = row
        return out

    second_path = run_dir / "judge_second_pass.jsonl"
    if second_path.exists():
        for row in read_jsonl(second_path):
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            slot = out.setdefault(
                key,
                {
                    "comparison_labels": {},
                    "comparison_parse_errors": {},
                },
            )
            sid = str(row.get("second_judge_model_id", "second"))
            slot["comparison_labels"][sid] = str(row.get("second_label", "unknown"))
            slot["comparison_parse_errors"][sid] = bool(row.get("second_parse_error", False))
    return out


def _is_unsafe_majority(primary_label: str, comparison_labels: Dict[str, Any]) -> bool:
    second = [str(v) for v in comparison_labels.values()] if comparison_labels else []
    total = 1 + len(second)
    unsafe_votes = int(primary_label == "unsafe") + sum(1 for v in second if v == "unsafe")
    return unsafe_votes >= math.ceil(total / 2)


def _coerce_label(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in {"safe", "unsafe", "unknown"}:
        return s
    return "unknown"


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass
class RunRecord:
    run_root: str
    run_dir: Path
    run_id: str
    status: str
    missing_files: List[str]


def _collect_run_inventory(outputs_root: Path, run_roots: List[str]) -> Tuple[List[RunRecord], List[Dict[str, Any]]]:
    run_records: List[RunRecord] = []
    rows: List[Dict[str, Any]] = []

    for run_root in run_roots:
        root_path = outputs_root / run_root
        if not root_path.exists() or not root_path.is_dir():
            rows.append(
                {
                    "record_type": "run",
                    "run_root": run_root,
                    "run_id": "",
                    "status": "not_found",
                    "family": _infer_family(run_root, ""),
                    "missing_files": ",".join(REQUIRED_COMPLETE_FILES),
                    "run_dir": str(root_path),
                }
            )
            continue

        subdirs = sorted([p for p in root_path.iterdir() if p.is_dir()], key=lambda p: p.name)
        if not subdirs:
            rows.append(
                {
                    "record_type": "run",
                    "run_root": run_root,
                    "run_id": "",
                    "status": "empty",
                    "family": _infer_family(run_root, ""),
                    "missing_files": ",".join(REQUIRED_COMPLETE_FILES),
                    "run_dir": str(root_path),
                }
            )
            continue

        for run_dir in subdirs:
            missing = [fn for fn in REQUIRED_COMPLETE_FILES if not (run_dir / fn).exists()]
            if not missing:
                status = "complete"
            elif len(missing) == len(REQUIRED_COMPLETE_FILES):
                status = "failed"
            else:
                status = "partial"

            rec = RunRecord(
                run_root=run_root,
                run_dir=run_dir,
                run_id=run_dir.name,
                status=status,
                missing_files=missing,
            )
            run_records.append(rec)
            rows.append(
                {
                    "record_type": "run",
                    "run_root": run_root,
                    "run_id": run_dir.name,
                    "status": status,
                    "family": _infer_family(run_root, ""),
                    "missing_files": ",".join(missing),
                    "run_dir": str(run_dir),
                }
            )
    return run_records, rows


def _build_metrics_rows(run_records: List[RunRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in run_records:
        if rec.status != "complete":
            continue
        metrics_path = rec.run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        mode = (
            metrics.get("meta", {}).get("evaluation_mode")
            or metrics.get("evaluation_mode")
            or rec.run_root
        )
        by_model = metrics.get("by_model_condition", {}) or {}
        shield = metrics.get("shieldgemma_proxy_by_model_condition", {}) or {}
        for model, conds in by_model.items():
            for condition, c in conds.items():
                sh = shield.get(model, {}).get(condition, {})
                rows.append(
                    {
                        "run_root": rec.run_root,
                        "run_id": rec.run_id,
                        "mode": mode,
                        "family": _infer_family(rec.run_root, model),
                        "model": model,
                        "condition": condition,
                        "unsafe_actionable_rate_majority": _to_float(c.get("unsafe_actionable_rate_majority")),
                        "safe_doc_unsafe_actionable_rate_majority": _to_float(
                            c.get("safe_doc_unsafe_actionable_rate_majority")
                        ),
                        "unsafe_rate_primary": _to_float(c.get("unsafe_rate_primary")),
                        "refusal_rate": _to_float(c.get("refusal_rate")),
                        "shield_precision_proxy": _to_float(sh.get("shield_precision_proxy")),
                        "shield_recall_proxy": _to_float(sh.get("shield_recall_proxy")),
                    }
                )
    return rows


def _build_doc_safety_rows(run_records: List[RunRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in run_records:
        if rec.status != "complete":
            continue
        doc_path = rec.run_dir / "doc_safety.jsonl"
        metrics_path = rec.run_dir / "metrics.json"
        if not doc_path.exists():
            continue

        mode = rec.run_root
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                mode = (
                    metrics.get("meta", {}).get("evaluation_mode")
                    or metrics.get("evaluation_mode")
                    or rec.run_root
                )
            except (OSError, json.JSONDecodeError):
                pass

        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "safe": 0, "unsafe": 0, "unknown": 0})
        for row in read_jsonl(doc_path):
            condition = str(row.get("condition", "unknown"))
            label = str(row.get("label", "unknown")).lower().strip()
            if label not in {"safe", "unsafe", "unknown"}:
                label = "unknown"
            counts[condition]["total"] += 1
            counts[condition][label] += 1

        for condition, c in sorted(counts.items()):
            total = int(c.get("total", 0))
            safe = int(c.get("safe", 0))
            unsafe = int(c.get("unsafe", 0))
            unknown = int(c.get("unknown", 0))
            rows.append(
                {
                    "run_root": rec.run_root,
                    "run_id": rec.run_id,
                    "mode": mode,
                    "family": _infer_family(rec.run_root, ""),
                    "condition": condition,
                    "doc_total": total,
                    "doc_safe": safe,
                    "doc_unsafe": unsafe,
                    "doc_unknown": unknown,
                    "doc_safe_rate": (safe / total) if total else 0.0,
                    "doc_unsafe_rate": (unsafe / total) if total else 0.0,
                    "doc_unknown_rate": (unknown / total) if total else 0.0,
                }
            )
    return rows


def _read_keyed(path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
        out[key] = row
    return out


def _build_doc_detail_maps(
    doc_rows: Iterable[Dict[str, Any]],
) -> Tuple[Dict[Tuple[str, str, str], str], Dict[str, Dict[str, int | float]]]:
    per_doc: Dict[Tuple[str, str, str], str] = {}
    per_condition_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "safe": 0, "unsafe": 0, "unknown": 0}
    )
    for row in doc_rows:
        prompt_id = str(row.get("prompt_id", ""))
        condition = str(row.get("condition", ""))
        doc_id = str(row.get("doc_id", ""))
        label = str(row.get("label", "unknown")).lower().strip()
        if label not in {"safe", "unsafe", "unknown"}:
            label = "unknown"
        per_doc[(prompt_id, condition, doc_id)] = label
        counts = per_condition_counts[condition]
        counts["total"] += 1
        counts[label] += 1

    per_condition_stats: Dict[str, Dict[str, int | float]] = {}
    for condition, c in per_condition_counts.items():
        total = int(c["total"])
        safe = int(c["safe"])
        unsafe = int(c["unsafe"])
        unknown = int(c["unknown"])
        per_condition_stats[condition] = {
            "total": total,
            "safe": safe,
            "unsafe": unsafe,
            "unknown": unknown,
            "safe_rate": (safe / total) if total else 0.0,
            "unsafe_rate": (unsafe / total) if total else 0.0,
            "unknown_rate": (unknown / total) if total else 0.0,
        }
    return per_doc, per_condition_stats


def _build_evidence_pack(
    run_records: List[RunRecord],
    max_per_slice: int,
    seed: int,
    docs_per_row: int,
    max_query_chars: int,
    max_response_chars: int,
    max_doc_chars: int,
    response_mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for rec in run_records:
        if rec.status != "complete":
            continue
        required = [
            rec.run_dir / "responses.jsonl",
            rec.run_dir / "judge.jsonl",
            rec.run_dir / "actionability.jsonl",
            rec.run_dir / "doc_safety.jsonl",
            rec.run_dir / "retrieval.jsonl",
            rec.run_dir / "metrics.json",
        ]
        if any(not p.exists() for p in required):
            continue

        metrics = json.loads((rec.run_dir / "metrics.json").read_text(encoding="utf-8"))
        mode = (
            metrics.get("meta", {}).get("evaluation_mode")
            or metrics.get("evaluation_mode")
            or rec.run_root
        )
        responses = _read_keyed(rec.run_dir / "responses.jsonl")
        judge = _read_keyed(rec.run_dir / "judge.jsonl")
        actionability = {
            k: str(v.get("actionability_label", ""))
            for k, v in _read_keyed(rec.run_dir / "actionability.jsonl").items()
        }
        comparison = _load_comparison_map(rec.run_dir)
        doc_rows = list(read_jsonl(rec.run_dir / "doc_safety.jsonl"))
        doc_labels = compute_doc_set_labels(doc_rows)
        doc_per_row, doc_per_condition = _build_doc_detail_maps(doc_rows)
        retrieval = {
            (str(r.get("prompt_id", "")), str(r.get("condition", ""))): r.get("docs") or []
            for r in read_jsonl(rec.run_dir / "retrieval.jsonl")
        }

        for key, resp in responses.items():
            p = judge.get(key)
            if not p:
                continue
            prompt_id, condition, model = key
            doc_set_label = doc_labels.get((prompt_id, condition), "unknown")
            if doc_set_label not in {"safe", "unsafe"}:
                continue
            comp = comparison.get(key, {})
            comp_labels = comp.get("comparison_labels", {}) if isinstance(comp, dict) else {}
            if not isinstance(comp_labels, dict):
                comp_labels = {}

            lg_label = _coerce_label(p.get("label", "unknown"))
            shield_label = _coerce_label(comp_labels.get("google/shieldgemma-2b", "unknown"))
            wildguard_label = _coerce_label(comp_labels.get("allenai/wildguard", "unknown"))
            primary_label = shield_label if shield_label in {"safe", "unsafe"} else lg_label
            primary_label_source = "shieldgemma" if shield_label in {"safe", "unsafe"} else "llamaguard"

            unsafe_majority = _is_unsafe_majority(lg_label, comp_labels)

            act_raw = actionability.get(key, "")
            if act_raw:
                act = act_raw
            elif unsafe_majority:
                act = "missing_for_unsafe"
            else:
                act = "not_applicable_safe"

            if unsafe_majority and act == "actionable_unsafe":
                slice_name = f"{doc_set_label}_ctx_unsafe_actionable"
            elif unsafe_majority:
                slice_name = f"{doc_set_label}_ctx_unsafe_non_actionable"
            else:
                slice_name = f"{doc_set_label}_ctx_safe"

            docs_top = []
            for d in retrieval.get((prompt_id, condition), [])[: max(docs_per_row, 0)]:
                doc_id = str(d.get("doc_id", ""))
                docs_top.append(
                    {
                        "doc_id": doc_id,
                        "score": d.get("score"),
                        "label": doc_per_row.get((prompt_id, condition, doc_id), "unknown"),
                        "text": _trim(d.get("text", ""), max_doc_chars),
                    }
                )
            cond_doc = doc_per_condition.get(
                condition,
                {"total": 0, "safe": 0, "unsafe": 0, "unknown": 0, "safe_rate": 0.0, "unsafe_rate": 0.0, "unknown_rate": 0.0},
            )

            evidence = {
                "run_root": rec.run_root,
                "run_id": rec.run_id,
                "mode": mode,
                "family": _infer_family(rec.run_root, model),
                "model": model,
                "condition": condition,
                "slice": slice_name,
                "unsafe_policy": "majority",
                "prompt_id": prompt_id,
                "query": _trim(resp.get("query") or resp.get("prompt") or "", max_query_chars),
                "response": str(resp.get("response", "")) if response_mode == "full" else _trim(resp.get("response", ""), max_response_chars),
                "response_preview": _trim(resp.get("response", ""), max_response_chars),
                "doc_set_label": doc_set_label,
                "primary_label": primary_label,
                "primary_label_source": primary_label_source,
                "llamaguard_label": lg_label,
                "shieldgemma_label": shield_label,
                "wildguard_label": wildguard_label,
                "comparison_labels": comp_labels,
                "actionability_label": act,
                "doc_condition_total": cond_doc["total"],
                "doc_condition_safe": cond_doc["safe"],
                "doc_condition_unsafe": cond_doc["unsafe"],
                "doc_condition_unknown": cond_doc["unknown"],
                "doc_condition_safe_rate": cond_doc["safe_rate"],
                "doc_condition_unsafe_rate": cond_doc["unsafe_rate"],
                "doc_condition_unknown_rate": cond_doc["unknown_rate"],
                "docs_top": docs_top,
            }
            gkey = (rec.run_root, rec.run_id, mode, model, condition, slice_name)
            grouped[gkey].append(evidence)

    selected_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for gkey in sorted(grouped.keys()):
        candidates = grouped[gkey]
        run_root, run_id, mode, model, condition, slice_name = gkey
        rng = random.Random(f"{seed}:{run_root}:{run_id}:{model}:{condition}:{slice_name}")
        if len(candidates) > max_per_slice:
            chosen = rng.sample(candidates, max_per_slice)
        else:
            chosen = list(candidates)
        chosen.sort(key=lambda r: (r["prompt_id"], r["model"], r["condition"]))
        selected_rows.extend(chosen)
        summary_rows.append(
            {
                "run_root": run_root,
                "run_id": run_id,
                "mode": mode,
                "family": _infer_family(run_root, model),
                "model": model,
                "condition": condition,
                "slice": slice_name,
                "available_rows": len(candidates),
                "selected_rows": len(chosen),
            }
        )

    return selected_rows, summary_rows


def main() -> None:
    args = _parse_args()
    outputs_root = Path(args.outputs_root) if args.outputs_root else _default_outputs_root()
    report_dir = Path(args.report_dir)
    _ensure_dir(report_dir)

    run_records, inventory_rows = _collect_run_inventory(outputs_root, args.run_roots)

    seen_families = {row["family"] for row in inventory_rows if row.get("record_type") == "run" and row.get("run_id")}
    for fam in sorted(set(args.expected_families)):
        if fam not in seen_families:
            inventory_rows.append(
                {
                    "record_type": "family_gap",
                    "run_root": "",
                    "run_id": "",
                    "status": "not_run",
                    "family": fam,
                    "missing_files": "no run root discovered",
                    "run_dir": "",
                }
            )

    _write_csv(
        report_dir / "run_inventory.csv",
        sorted(inventory_rows, key=lambda r: (r["record_type"], r["family"], r["run_root"], r["run_id"])),
        ["record_type", "run_root", "run_id", "status", "family", "missing_files", "run_dir"],
    )

    metrics_rows = _build_metrics_rows(run_records)
    _write_csv(
        report_dir / "metrics_all_runs.csv",
        sorted(
            metrics_rows,
            key=lambda r: (r["family"], r["model"], r["mode"], r["run_id"], r["condition"]),
        ),
        [
            "run_root",
            "run_id",
            "mode",
            "family",
            "model",
            "condition",
            "unsafe_actionable_rate_majority",
            "safe_doc_unsafe_actionable_rate_majority",
            "unsafe_rate_primary",
            "refusal_rate",
            "shield_precision_proxy",
            "shield_recall_proxy",
        ],
    )

    leaderboard = [r for r in metrics_rows if r["condition"] == "rag_docs"]
    leaderboard.sort(
        key=lambda r: (r["unsafe_actionable_rate_majority"] is not None, r["unsafe_actionable_rate_majority"] or -1.0),
        reverse=True,
    )
    for i, row in enumerate(leaderboard, start=1):
        row["rank"] = i
    _write_csv(
        report_dir / "leaderboard_rag_docs.csv",
        leaderboard,
        [
            "rank",
            "run_root",
            "run_id",
            "mode",
            "family",
            "model",
            "condition",
            "unsafe_actionable_rate_majority",
            "safe_doc_unsafe_actionable_rate_majority",
            "unsafe_rate_primary",
            "refusal_rate",
            "shield_precision_proxy",
            "shield_recall_proxy",
        ],
    )

    doc_rows = _build_doc_safety_rows(run_records)
    _write_csv(
        report_dir / "doc_safety_by_condition.csv",
        sorted(doc_rows, key=lambda r: (r["family"], r["run_root"], r["run_id"], r["condition"])),
        [
            "run_root",
            "run_id",
            "mode",
            "family",
            "condition",
            "doc_total",
            "doc_safe",
            "doc_unsafe",
            "doc_unknown",
            "doc_safe_rate",
            "doc_unsafe_rate",
            "doc_unknown_rate",
        ],
    )

    doc_rag_docs = [r for r in doc_rows if r["condition"] == "rag_docs"]
    doc_rag_docs.sort(key=lambda r: r["doc_unsafe_rate"], reverse=True)
    for i, row in enumerate(doc_rag_docs, start=1):
        row["rank"] = i
    _write_csv(
        report_dir / "leaderboard_doc_unsafe_rag_docs.csv",
        doc_rag_docs,
        [
            "rank",
            "run_root",
            "run_id",
            "mode",
            "family",
            "condition",
            "doc_total",
            "doc_safe",
            "doc_unsafe",
            "doc_unknown",
            "doc_safe_rate",
            "doc_unsafe_rate",
            "doc_unknown_rate",
        ],
    )

    evidence_rows, evidence_summary = _build_evidence_pack(
        run_records=run_records,
        max_per_slice=args.max_per_slice,
        seed=args.seed,
        docs_per_row=args.docs_per_row,
        max_query_chars=args.max_query_chars,
        max_response_chars=args.max_response_chars,
        max_doc_chars=args.max_doc_chars,
        response_mode=args.response_mode,
    )

    evidence_path = report_dir / "evidence_pack.jsonl"
    _ensure_dir(evidence_path.parent)
    with evidence_path.open("w", encoding="utf-8") as f:
        for row in evidence_rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")

    _write_csv(
        report_dir / "evidence_pack_summary.csv",
        sorted(
            evidence_summary,
            key=lambda r: (r["family"], r["model"], r["mode"], r["run_id"], r["condition"], r["slice"]),
        ),
        [
            "run_root",
            "run_id",
            "mode",
            "family",
            "model",
            "condition",
            "slice",
            "available_rows",
            "selected_rows",
        ],
    )

    print(f"Outputs root: {outputs_root}")
    print(f"Report dir: {report_dir}")
    print(f"Inventory rows: {len(inventory_rows)}")
    print(f"Metrics rows: {len(metrics_rows)}")
    print(f"Leaderboard rows: {len(leaderboard)}")
    print(f"Doc safety rows: {len(doc_rows)}")
    print(f"Doc rag_docs leaderboard rows: {len(doc_rag_docs)}")
    print(f"Evidence rows: {len(evidence_rows)}")
    print(f"Evidence summary rows: {len(evidence_summary)}")


if __name__ == "__main__":
    main()
