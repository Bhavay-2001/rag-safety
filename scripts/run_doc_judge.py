from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import config_hash, load_config, resolve_paths
from src.doc_judge_runner import judge_document
from src.judge_runner import SafetyJudge
from src.utils_io import read_jsonl, write_json, write_jsonl


def _maybe_tqdm(iterable, total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run document safety judge.")
    parser.add_argument("--config", required=True, help="Path to configs/base.yaml")
    parser.add_argument("--run-dir", default=None, help="Specific run directory under outputs/runs")
    return parser.parse_args()


def _latest_run_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_prompt_map(path: str | Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text = (
                row.get("text")
                or row.get("question")
                or row.get("prompt")
                or row.get("query")
                or row.get("Base_Question")
                or next((v for v in row.values() if v and v.strip()), "")
            )
            if not text:
                continue
            prompt_id = row.get("prompt_id") or str(idx)
            mapping[prompt_id] = text
    return mapping


def _build_retrieval_lookup(records: List[Dict]) -> Dict[Tuple[str, str, str], Dict]:
    lookup: Dict[Tuple[str, str, str], Dict] = {}
    for rec in records:
        prompt_id = rec["prompt_id"]
        condition = rec["condition"]
        docs = rec.get("docs", []) or []
        for rank, doc in enumerate(docs, start=1):
            key = (prompt_id, condition, doc["doc_id"])
            lookup[key] = {
                "rank": rank,
                "bm25_score": doc.get("score"),
                "doc_text": doc.get("text"),
            }
    return lookup


def _build_generation_lookup(records: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    lookup: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    for rec in records:
        key = (rec["prompt_id"], rec["condition"])
        slot = lookup.setdefault(
            key,
            {"models": set(), "model_ids": set(), "retrieval_modes": set(), "evaluation_modes": set()},
        )
        if rec.get("model"):
            slot["models"].add(rec["model"])
        if rec.get("model_id"):
            slot["model_ids"].add(rec["model_id"])
        if rec.get("retrieval_mode"):
            slot["retrieval_modes"].add(rec["retrieval_mode"])
        if rec.get("evaluation_mode"):
            slot["evaluation_modes"].add(rec["evaluation_mode"])
    out: Dict[Tuple[str, str], Dict] = {}
    for key, val in lookup.items():
        models = sorted(val["models"])
        model_ids = sorted(val["model_ids"])
        out[key] = {
            "generation_model": models[0] if len(models) == 1 else None,
            "generation_model_id": model_ids[0] if len(model_ids) == 1 else None,
            "generation_models": models,
            "generation_model_ids": model_ids,
            "retrieval_mode": sorted(val["retrieval_modes"])[0] if len(val["retrieval_modes"]) == 1 else None,
            "evaluation_mode": sorted(val["evaluation_modes"])[0] if len(val["evaluation_modes"]) == 1 else None,
        }
    return out


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, cfg_path.parent)
    cfg_hash = config_hash(cfg)

    output_root = Path(cfg.get("output", {}).get("root", "outputs/runs"))
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(output_root)

    retrieval_path = run_dir / "retrieval.jsonl"
    doc_judge_path = run_dir / "doc_safety.jsonl"
    joined_path = run_dir / "context_safety_joined.jsonl"
    if doc_judge_path.exists():
        doc_judge_path.unlink()

    prompts_cfg = cfg.get("prompts", {})
    prompts_path = prompts_cfg.get("data_path") or prompts_cfg.get("path")
    prompt_map = _load_prompt_map(prompts_path)

    judge_cfg = cfg.get("judge", {})
    primary_id = judge_cfg.get("doc_model_primary", "llama-guard-2")
    secondary_id = judge_cfg.get("doc_model_secondary")
    explanation_mode = str(judge_cfg.get("explanation_mode", "off"))
    include_explanation = explanation_mode == "all"
    save_context_joined = bool(judge_cfg.get("save_context_joined", True))
    judge_generation_cfg = judge_cfg.get("generation", {"max_new_tokens": 128, "temperature": 0.0, "do_sample": False})

    primary = SafetyJudge(primary_id, generation_cfg=judge_generation_cfg, judge_cfg=judge_cfg)
    secondary = SafetyJudge(secondary_id, generation_cfg=judge_generation_cfg, judge_cfg=judge_cfg) if secondary_id else None

    start_time = time.perf_counter()
    docs_total = 0
    judged_total = 0
    records = list(read_jsonl(retrieval_path))
    retrieval_lookup = _build_retrieval_lookup(records)
    response_records = list(read_jsonl(run_dir / "responses.jsonl")) if (run_dir / "responses.jsonl").exists() else []
    generation_lookup = _build_generation_lookup(response_records)
    if save_context_joined and joined_path.exists():
        joined_path.unlink()
    for record in _maybe_tqdm(records, total=len(records), desc="Doc judge"):
        docs = record.get("docs", [])
        if not docs:
            continue
        prompt_id = record["prompt_id"]
        query = prompt_map.get(prompt_id, "")
        for doc in docs:
            docs_total += 1
            result = judge_document(
                primary,
                query,
                doc["text"],
                secondary=secondary,
                include_explanation=include_explanation,
            )
            output = {
                "prompt_id": prompt_id,
                "condition": record["condition"],
                "doc_id": doc["doc_id"],
                "judge_model_primary": primary.model_id,
                "judge_model_secondary": secondary.model_id if secondary else None,
                "label": result["label"],
                "primary_label": result["primary_label"],
                "secondary_label": result["secondary_label"],
                "categories_primary": result["categories_primary"],
                "categories_secondary": result["categories_secondary"],
                "unsafe_prob_llr_primary": result["unsafe_prob_llr_primary"],
                "unsafe_prob_llr_secondary": result["unsafe_prob_llr_secondary"],
                "unsafe_prob_first_token_primary": result["unsafe_prob_first_token_primary"],
                "unsafe_prob_first_token_secondary": result["unsafe_prob_first_token_secondary"],
                "unsafe_safe_margin_primary": result["unsafe_safe_margin_primary"],
                "unsafe_safe_margin_secondary": result["unsafe_safe_margin_secondary"],
                "llr_margin_primary": result["llr_margin_primary"],
                "llr_margin_secondary": result["llr_margin_secondary"],
                "judge_parse_error_primary": result.get("judge_parse_error_primary", False),
                "judge_parse_error_secondary": result.get("judge_parse_error_secondary"),
                "judge_parse_error_reason_primary": result.get("judge_parse_error_reason_primary"),
                "judge_parse_error_reason_secondary": result.get("judge_parse_error_reason_secondary"),
                "self_confidence_primary": result["self_confidence_primary"],
                "self_confidence_secondary": result["self_confidence_secondary"],
                "explanation_primary": result["explanation_primary"],
                "explanation_secondary": result["explanation_secondary"],
                "raw_primary": result["raw_primary"],
                "raw_secondary": result["raw_secondary"],
            }
            write_jsonl(doc_judge_path, [output], append=True)
            if save_context_joined:
                join_key = (prompt_id, record["condition"], doc["doc_id"])
                ret = retrieval_lookup.get(join_key, {})
                gen = generation_lookup.get((prompt_id, record["condition"]), {})
                write_jsonl(
                    joined_path,
                    [
                        {
                            "run_id": run_dir.name,
                            "config_hash": cfg_hash,
                            "prompt_id": prompt_id,
                            "condition": record["condition"],
                            "query": query,
                            "doc_id": doc["doc_id"],
                            "rank": ret.get("rank"),
                            "bm25_score": ret.get("bm25_score"),
                            "doc_text": ret.get("doc_text", doc.get("text")),
                            "generation_model": gen.get("generation_model"),
                            "generation_model_id": gen.get("generation_model_id"),
                            "generation_models": gen.get("generation_models", []),
                            "generation_model_ids": gen.get("generation_model_ids", []),
                            "retrieval_mode": gen.get("retrieval_mode"),
                            "evaluation_mode": gen.get("evaluation_mode"),
                            "judge_model_primary": primary.model_id,
                            "judge_model_secondary": secondary.model_id if secondary else None,
                            "label": result["label"],
                            "primary_label": result["primary_label"],
                            "secondary_label": result["secondary_label"],
                            "categories_primary": result["categories_primary"],
                            "categories_secondary": result["categories_secondary"],
                            "unsafe_prob_llr_primary": result["unsafe_prob_llr_primary"],
                            "unsafe_prob_llr_secondary": result["unsafe_prob_llr_secondary"],
                            "unsafe_prob_first_token_primary": result["unsafe_prob_first_token_primary"],
                            "unsafe_prob_first_token_secondary": result["unsafe_prob_first_token_secondary"],
                            "unsafe_safe_margin_primary": result["unsafe_safe_margin_primary"],
                            "unsafe_safe_margin_secondary": result["unsafe_safe_margin_secondary"],
                            "llr_margin_primary": result["llr_margin_primary"],
                            "llr_margin_secondary": result["llr_margin_secondary"],
                            "judge_parse_error_primary": result.get("judge_parse_error_primary", False),
                            "judge_parse_error_secondary": result.get("judge_parse_error_secondary"),
                            "judge_parse_error_reason_primary": result.get("judge_parse_error_reason_primary"),
                            "judge_parse_error_reason_secondary": result.get("judge_parse_error_reason_secondary"),
                            "self_confidence_primary": result["self_confidence_primary"],
                            "self_confidence_secondary": result["self_confidence_secondary"],
                            "explanation_primary": result["explanation_primary"],
                            "explanation_secondary": result["explanation_secondary"],
                        }
                    ],
                    append=True,
                )
            judged_total += 1

    elapsed = time.perf_counter() - start_time
    stats = {
        "docs_total": docs_total,
        "judged_total": judged_total,
        "elapsed_sec": elapsed,
        "docs_per_sec": judged_total / elapsed if elapsed > 0 else None,
    }
    stats_path = run_dir / "doc_judge_stats.json"
    write_json(stats_path, stats)
    print(f"Wrote doc safety labels to {doc_judge_path}")
    if save_context_joined:
        print(f"Wrote context safety joined rows to {joined_path}")
    print(f"Wrote doc judge stats to {stats_path}")


if __name__ == "__main__":
    main()
