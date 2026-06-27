from __future__ import annotations

import argparse
import csv
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import apply_overrides, config_hash, load_config, resolve_paths
from src.metrics import compute_doc_set_labels, compute_metrics
from src.utils_io import read_jsonl, sha256_file, write_json


def _load_actionability(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    counts: Dict[str, int] = {}
    for row in read_jsonl(path):
        lab = str(row.get("actionability_label", "unknown"))
        counts[lab] = counts.get(lab, 0) + 1
    return counts


def _load_actionability_map(path: Path) -> Dict[tuple[str, str, str], str]:
    if not path.exists():
        return {}
    out: Dict[tuple[str, str, str], str] = {}
    for row in read_jsonl(path):
        key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
        out[key] = str(row.get("actionability_label", ""))
    return out


def _load_comparison_map(
    comparison_path: Path,
    second_path: Path,
) -> Dict[tuple[str, str, str], Dict[str, object]]:
    out: Dict[tuple[str, str, str], Dict[str, object]] = {}
    if comparison_path.exists():
        for row in read_jsonl(comparison_path):
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            out[key] = row
        return out

    if second_path.exists():
        for row in read_jsonl(second_path):
            key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
            slot = out.setdefault(
                key,
                {
                    "prompt_id": key[0],
                    "condition": key[1],
                    "model": key[2],
                    "comparison_labels": {},
                    "comparison_parse_errors": {},
                },
            )
            sid = str(row.get("second_judge_model_id", "second"))
            slot["comparison_labels"][sid] = str(row.get("second_label", "unknown"))
            slot["comparison_parse_errors"][sid] = bool(row.get("second_parse_error", False))
    return out


def _load_present_absent_jsonl(path: Path) -> Dict[Tuple[str, str, str], str]:
    if not path.exists():
        return {}
    out: Dict[Tuple[str, str, str], str] = {}
    for row in read_jsonl(path):
        key = (str(row.get("prompt_id", "")), str(row.get("condition", "")), str(row.get("model", "")))
        out[key] = str(row.get("doc_set_label", ""))
    return out


def _summarize_by_retrieval_mode(responses_path: Path, judge_path: Path) -> Dict[str, Dict[str, float | int]]:
    if not responses_path.exists() or not judge_path.exists():
        return {}
    resp = list(read_jsonl(responses_path))
    judge = list(read_jsonl(judge_path))
    jmap = {(j["prompt_id"], j["condition"], j["model"]): j for j in judge}
    counts: Dict[str, Dict[str, int]] = {}
    for r in resp:
        mode = str(r.get("retrieval_mode", "unknown"))
        slot = counts.setdefault(mode, {"total": 0, "unsafe": 0})
        slot["total"] += 1
        j = jmap.get((r["prompt_id"], r["condition"], r["model"]))
        if j and j.get("label") == "unsafe":
            slot["unsafe"] += 1
    out: Dict[str, Dict[str, float | int]] = {}
    for mode, c in counts.items():
        total = c.get("total", 0)
        unsafe = c.get("unsafe", 0)
        out[mode] = {"total": total, "unsafe": unsafe, "unsafe_rate": (unsafe / total) if total else 0.0}
    return out


def _retrieval_quality_buckets(
    retrieval_path: Path,
    responses_path: Path,
    judge_path: Path,
    num_buckets: int = 4,
) -> Dict[str, Dict[str, float | int]]:
    if not retrieval_path.exists() or not responses_path.exists() or not judge_path.exists() or num_buckets <= 1:
        return {}
    # Use top doc BM25 as a simple quality proxy.
    quality: Dict[tuple[str, str], float] = {}
    for row in read_jsonl(retrieval_path):
        docs = row.get("docs") or []
        if docs:
            quality[(row["prompt_id"], row["condition"])] = float(docs[0].get("score", 0.0))
    values = sorted(quality.values())
    if not values:
        return {}
    cuts = []
    for i in range(1, num_buckets):
        idx = int((len(values) - 1) * (i / num_buckets))
        cuts.append(values[idx])

    def _bucket(v: float) -> str:
        for i, c in enumerate(cuts):
            if v <= c:
                return f"q{i+1}"
        return f"q{num_buckets}"

    jmap = {(j["prompt_id"], j["condition"], j["model"]): j for j in read_jsonl(judge_path)}
    out_counts: Dict[str, Dict[str, int]] = {f"q{i+1}": {"total": 0, "unsafe": 0} for i in range(num_buckets)}
    for r in read_jsonl(responses_path):
        key_pc = (r["prompt_id"], r["condition"])
        if key_pc not in quality:
            continue
        bucket = _bucket(quality[key_pc])
        slot = out_counts[bucket]
        slot["total"] += 1
        j = jmap.get((r["prompt_id"], r["condition"], r["model"]))
        if j and j.get("label") == "unsafe":
            slot["unsafe"] += 1
    out: Dict[str, Dict[str, float | int]] = {}
    for b, c in out_counts.items():
        total = c["total"]
        unsafe = c["unsafe"]
        out[b] = {"total": total, "unsafe": unsafe, "unsafe_rate": (unsafe / total) if total else 0.0}
    return out


def _second_pass_summaries(
    responses_path: Path,
    judge_path: Path,
    second_path: Path,
    comparison_path: Path,
) -> Dict[str, Dict[str, Dict[str, float | int]]]:
    if not responses_path.exists() or not judge_path.exists():
        return {}
    responses = list(read_jsonl(responses_path))
    primary = {(r["prompt_id"], r["condition"], r["model"]): r for r in read_jsonl(judge_path)}
    comparison = _load_comparison_map(comparison_path, second_path)
    grouped: Dict[str, Dict[str, Dict[str, int]]] = {}
    for r in responses:
        key = (r["prompt_id"], r["condition"], r["model"])
        p = primary.get(key)
        comp = comparison.get(key, {})
        comp_labels = comp.get("comparison_labels", {}) if isinstance(comp, dict) else {}
        if not isinstance(comp_labels, dict):
            comp_labels = {}
        if not p:
            continue
        model = str(r["model"])
        condition = str(r["condition"])
        slot = grouped.setdefault(model, {}).setdefault(
            condition,
            {
                "total": 0,
                "rows_second_evaluated": 0,
                "primary_unsafe": 0,
                "second_unsafe_any": 0,
                "second_unsafe_majority": 0,
                "second_unsafe_unanimous": 0,
                "consensus_any": 0,
                "consensus_both": 0,
                "disagree": 0,
            },
        )
        slot["total"] += 1
        p_unsafe = str(p.get("label", "")) == "unsafe"
        second_labels = [str(v) for v in comp_labels.values()]
        judges_total = 1 + len(second_labels)
        unsafe_votes = int(p_unsafe) + sum(1 for v in second_labels if v == "unsafe")
        s_unsafe_any = sum(1 for v in second_labels if v == "unsafe") > 0
        s_unsafe_majority = unsafe_votes >= math.ceil(judges_total / 2)
        s_unsafe_unanimous = unsafe_votes == judges_total
        if p_unsafe:
            slot["primary_unsafe"] += 1
        if s_unsafe_any:
            slot["second_unsafe_any"] += 1
        if s_unsafe_majority:
            slot["second_unsafe_majority"] += 1
        if s_unsafe_unanimous:
            slot["second_unsafe_unanimous"] += 1
        if p_unsafe or s_unsafe_any:
            slot["consensus_any"] += 1
        if p_unsafe and s_unsafe_any:
            slot["consensus_both"] += 1
        if second_labels:
            slot["rows_second_evaluated"] += 1
            if any(str(p.get("label", "")) != lab for lab in second_labels):
                slot["disagree"] += 1

    out: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    for model, conds in grouped.items():
        out[model] = {}
        for condition, c in conds.items():
            total = c["total"] or 1
            second_total = c["rows_second_evaluated"] or 1
            out[model][condition] = {
                "total": c["total"],
                "rows_second_evaluated": c["rows_second_evaluated"],
                "unsafe_rate_primary": c["primary_unsafe"] / total,
                "unsafe_rate_consensus_any": c["consensus_any"] / total,
                "unsafe_rate_consensus_both": c["consensus_both"] / total,
                "unsafe_rate_second_any": c["second_unsafe_any"] / total,
                "unsafe_rate_second_majority": c["second_unsafe_majority"] / total,
                "unsafe_rate_second_unanimous": c["second_unsafe_unanimous"] / total,
                "disagreement_rate": c["disagree"] / second_total,
            }
    return out


def _ensemble_rates(
    responses: list[dict],
    judge_labels: list[dict],
    doc_safety: list[dict],
    comparison_map: Dict[tuple[str, str, str], Dict[str, object]],
    actionability_map: Dict[tuple[str, str, str], str],
) -> Dict[str, Dict[str, Dict[str, float | int]]]:
    primary_map = {(r["prompt_id"], r["condition"], r["model"]): r for r in judge_labels}
    doc_set = compute_doc_set_labels(doc_safety)
    grouped: Dict[tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in responses:
        key = (str(r.get("prompt_id", "")), str(r.get("condition", "")), str(r.get("model", "")))
        p = primary_map.get(key)
        if not p:
            continue
        mkey = (key[2], key[1])
        grouped[mkey]["total"] += 1

        p_unsafe = str(p.get("label", "")) == "unsafe"
        comp = comparison_map.get(key, {})
        comp_labels = comp.get("comparison_labels", {}) if isinstance(comp, dict) else {}
        if not isinstance(comp_labels, dict):
            comp_labels = {}
        second_labels = [str(v) for v in comp_labels.values()]

        judges_total = 1 + len(second_labels)
        unsafe_votes = int(p_unsafe) + sum(1 for v in second_labels if v == "unsafe")
        unsafe_any = unsafe_votes >= 1
        unsafe_majority = unsafe_votes >= math.ceil(judges_total / 2)
        unsafe_unanimous = unsafe_votes == judges_total

        if p_unsafe:
            grouped[mkey]["unsafe_primary"] += 1
        if unsafe_any:
            grouped[mkey]["unsafe_any"] += 1
        if unsafe_majority:
            grouped[mkey]["unsafe_majority"] += 1
        if unsafe_unanimous:
            grouped[mkey]["unsafe_unanimous"] += 1

        actionable = actionability_map.get(key) == "actionable_unsafe"
        if actionable and p_unsafe:
            grouped[mkey]["unsafe_actionable_primary"] += 1
        if actionable and unsafe_any:
            grouped[mkey]["unsafe_actionable_any"] += 1
        if actionable and unsafe_majority:
            grouped[mkey]["unsafe_actionable_majority"] += 1
        if actionable and unsafe_unanimous:
            grouped[mkey]["unsafe_actionable_unanimous"] += 1

        doc_key = (key[0], key[1])
        if doc_set.get(doc_key) == "safe":
            grouped[mkey]["safe_doc_total"] += 1
            if unsafe_any:
                grouped[mkey]["safe_doc_unsafe_any"] += 1
            if unsafe_majority:
                grouped[mkey]["safe_doc_unsafe_majority"] += 1
            if actionable and unsafe_any:
                grouped[mkey]["safe_doc_unsafe_actionable_any"] += 1
            if actionable and unsafe_majority:
                grouped[mkey]["safe_doc_unsafe_actionable_majority"] += 1

    out: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    for (model, condition), c in grouped.items():
        total = c.get("total", 0)
        safe_doc_total = c.get("safe_doc_total", 0)
        out.setdefault(model, {})[condition] = {
            "unsafe_rate_primary": c.get("unsafe_primary", 0) / total if total else 0.0,
            "unsafe_rate_any": c.get("unsafe_any", 0) / total if total else 0.0,
            "unsafe_rate_majority": c.get("unsafe_majority", 0) / total if total else 0.0,
            "unsafe_rate_unanimous": c.get("unsafe_unanimous", 0) / total if total else 0.0,
            "unsafe_actionable_rate_primary": c.get("unsafe_actionable_primary", 0) / total if total else 0.0,
            "unsafe_actionable_rate_any": c.get("unsafe_actionable_any", 0) / total if total else 0.0,
            "unsafe_actionable_rate_majority": c.get("unsafe_actionable_majority", 0) / total if total else 0.0,
            "unsafe_actionable_rate_unanimous": c.get("unsafe_actionable_unanimous", 0) / total if total else 0.0,
            "safe_doc_unsafe_rate_any": c.get("safe_doc_unsafe_any", 0) / safe_doc_total if safe_doc_total else 0.0,
            "safe_doc_unsafe_rate_majority": c.get("safe_doc_unsafe_majority", 0) / safe_doc_total if safe_doc_total else 0.0,
            "safe_doc_unsafe_actionable_rate_any": c.get("safe_doc_unsafe_actionable_any", 0) / safe_doc_total
            if safe_doc_total
            else 0.0,
            "safe_doc_unsafe_actionable_rate_majority": c.get("safe_doc_unsafe_actionable_majority", 0)
            / safe_doc_total
            if safe_doc_total
            else 0.0,
        }
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize metrics for a run.")
    parser.add_argument("--config", required=True, help="Path to configs/base.yaml")
    parser.add_argument("--run-dir", default=None, help="Specific run directory under outputs/runs")
    parser.add_argument("--present-absent", default=None, help="Optional present/absent CSV path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config value, e.g. --set output.root=outputs/gemma_7b",
    )
    return parser.parse_args()


def _latest_run_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_present_absent(path: str | Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["prompt_id"]] = row["doc_set_label"]
    return mapping


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, cfg_path.parent)
    cfg = apply_overrides(cfg, args.set)

    output_root = Path(cfg.get("output", {}).get("root", "outputs/runs"))
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(output_root)

    responses = list(read_jsonl(run_dir / "responses.jsonl"))
    judge_labels = list(read_jsonl(run_dir / "judge.jsonl"))
    doc_safety = list(read_jsonl(run_dir / "doc_safety.jsonl"))
    actionability_map = _load_actionability_map(run_dir / "actionability.jsonl")
    comparison_map = _load_comparison_map(run_dir / "judge_comparison.jsonl", run_dir / "judge_second_pass.jsonl")

    present_absent_population = str(cfg.get("analysis", {}).get("present_absent_population", "unsafe_only"))
    present_absent_rows_path = run_dir / "present_absent.jsonl"
    present_absent = _load_present_absent_jsonl(present_absent_rows_path) if present_absent_rows_path.exists() else None
    if not present_absent and args.present_absent:
        present_absent = _load_present_absent(args.present_absent)
    metrics = compute_metrics(
        responses,
        judge_labels,
        doc_safety,
        present_absent=present_absent,
        present_absent_population=present_absent_population,
        actionability=actionability_map if actionability_map else None,
    )
    metrics["by_retrieval_mode"] = _summarize_by_retrieval_mode(run_dir / "responses.jsonl", run_dir / "judge.jsonl")
    metrics["actionability_summaries"] = _load_actionability(run_dir / "actionability.jsonl")
    ensemble_summary = _ensemble_rates(responses, judge_labels, doc_safety, comparison_map, actionability_map)
    for model, conds in ensemble_summary.items():
        for condition, vals in conds.items():
            if model in metrics["by_model_condition"] and condition in metrics["by_model_condition"][model]:
                metrics["by_model_condition"][model][condition].update(vals)
    metrics["retrieval_quality_buckets"] = _retrieval_quality_buckets(
        run_dir / "retrieval.jsonl",
        run_dir / "responses.jsonl",
        run_dir / "judge.jsonl",
        num_buckets=int(cfg.get("analysis", {}).get("retrieval_quality_buckets", 4)),
    )
    second_summary = _second_pass_summaries(
        run_dir / "responses.jsonl",
        run_dir / "judge.jsonl",
        run_dir / "judge_second_pass.jsonl",
        run_dir / "judge_comparison.jsonl",
    )
    if second_summary:
        metrics["second_pass_by_model_condition"] = second_summary
    metrics["present_absent_population"] = present_absent_population
    metrics["present_absent_total_rows_considered"] = (
        metrics.get("present_absent_meta", {}).get("rows_considered", 0)
    )

    model_cfg_path = cfg.get("models", {}).get("config")
    model_use = cfg.get("models", {}).get("use")
    model_ids = sorted({row.get("model_id") for row in responses if row.get("model_id")})
    prompts_cfg = cfg.get("prompts", {})
    prompts_path = prompts_cfg.get("data_path") or prompts_cfg.get("path")
    templates_path = prompts_cfg.get("template_path") or prompts_cfg.get("path")
    corpus_source = cfg.get("corpus", {}).get("source")

    meta = {
        "run_id": run_dir.name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": config_hash(cfg),
        "prompts_path": prompts_path,
        "prompts_sha256": sha256_file(prompts_path) if prompts_path else None,
        "templates_path": templates_path,
        "templates_sha256": sha256_file(templates_path) if templates_path else None,
        "models_config": model_cfg_path,
        "models_config_sha256": sha256_file(model_cfg_path) if model_cfg_path else None,
        "models_use": model_use,
        "model_ids": model_ids,
        "judge_response_model": cfg.get("judge", {}).get("response_model"),
        "judge_doc_primary": cfg.get("judge", {}).get("doc_model_primary"),
        "judge_doc_secondary": cfg.get("judge", {}).get("doc_model_secondary"),
        "judge_score_mode": cfg.get("judge", {}).get("score_mode", "llr"),
        "judge_include_self_confidence": cfg.get("judge", {}).get("include_self_confidence", True),
        "judge_explanation_mode": cfg.get("judge", {}).get("explanation_mode", "off"),
        "judge_save_context_joined": cfg.get("judge", {}).get("save_context_joined", True),
        "retriever_top_k": cfg.get("retriever", {}).get("top_k"),
        "retriever_mode": cfg.get("retriever", {}).get("mode", "bm25_top5"),
        "score_calibration_status": cfg.get("analysis", {}).get("score_calibration_status", "needs_debug"),
        "judge_edge_second_pass_enabled": cfg.get("judge", {}).get("edge_second_pass_enabled", False),
        "judge_edge_second_pass_model": cfg.get("judge", {}).get("edge_second_pass_model"),
        "judge_edge_second_pass_scope": cfg.get("judge", {}).get("edge_second_pass_scope", "unsafe_or_safe_doc_unsafe"),
        "judge_comparison_enabled": cfg.get("judge", {}).get("comparison_enabled", False),
        "judge_comparison_models": cfg.get("judge", {}).get("comparison_models", []),
        "corpus_source": corpus_source,
        "corpus_source_sha256": sha256_file(corpus_source) if corpus_source else None,
        "output_root": cfg.get("output", {}).get("root"),
        "evaluation_mode": cfg.get("analysis", {}).get("evaluation_mode", "baseline_paper_aligned"),
    }
    metrics["meta"] = meta

    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, metrics)
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
