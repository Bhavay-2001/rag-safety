from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import config_hash, load_config, resolve_paths
from src.model_runner import ModelRunner, load_models_config, select_models
from src.prompt_builder import load_prompt_templates, render_prompt
from src.qa_metrics import (
    answer_presence_at_k,
    contains_answer,
    context_precision,
    exact_match,
    hallucination_proxy,
    supported_token_ratio,
    token_f1,
)
from src.utils_io import ensure_dir, make_run_id, read_jsonl, sha256_file, write_json, write_jsonl


_SNIPPET_CHARS = 600


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIRAGE capability benchmark v2.")
    parser.add_argument("--config", required=True, help="Path to config yaml.")
    parser.add_argument("--run-dir", default=None, help="Optional fixed run directory.")
    return parser.parse_args()


def _maybe_tqdm(iterable, total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _load_queries(path: str | Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            prompt_id = str(row.get("prompt_id") or idx)
            text = str(row.get("text") or row.get("query") or row.get("question") or "").strip()
            if not text:
                continue
            rows.append({"prompt_id": prompt_id, "text": text})
    return rows


def _load_answers(path: str | Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in read_jsonl(path):
        prompt_id = str(row.get("prompt_id", "")).strip()
        answers = row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers] if answers.strip() else []
        out[prompt_id] = [str(a).strip() for a in answers if str(a).strip()]
    return out


def _load_contexts(path: str | Path, top_k: int) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        prompt_id = str(row.get("prompt_id", "")).strip()
        if not prompt_id:
            continue
        contexts = row.get("contexts") or []
        docs: List[Dict[str, Any]] = []
        for i, ctx in enumerate(contexts):
            if not isinstance(ctx, dict):
                continue
            text = str(ctx.get("text", "")).strip()
            if not text:
                continue
            doc_id = str(ctx.get("doc_id") or f"{prompt_id}-{i}")
            score_raw = ctx.get("score", 0.0)
            try:
                score = float(score_raw) if score_raw is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            docs.append({"doc_id": doc_id, "score": score, "text": text})
            if top_k > 0 and len(docs) >= top_k:
                break
        out[prompt_id] = {
            "docs_top_k": docs,
            "answer_present_top_k": row.get("answer_present_top_k"),
        }
    return out


def _aggregate(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    s = " ".join(str(text).split())
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + "..."


def _compute_context_behavior_metrics(qa_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    by_model_prompt: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in qa_rows:
        key = (str(row.get("model", "")), str(row.get("prompt_id", "")))
        by_model_prompt[key][str(row.get("condition", ""))] = row

    by_model: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_model_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (model, _), conds in by_model_prompt.items():
        non = conds.get("non_rag")
        rag = conds.get("rag_docs") or conds.get("rag_oracle_docs")
        mix = conds.get("rag_mixed_docs")
        if not (non and rag and mix):
            continue

        # Noise vulnerability: degradation from oracle to mixed under F1.
        by_model[model]["noise_vulnerability_sum"] += max(0.0, float(rag.get("f1", 0.0)) - float(mix.get("f1", 0.0)))
        by_model_counts[model]["noise_vulnerability_n"] += 1

        # Context acceptability: improvement from non-rag to oracle under F1.
        by_model[model]["context_acceptability_sum"] += max(
            0.0, float(rag.get("f1", 0.0)) - float(non.get("f1", 0.0))
        )
        by_model_counts[model]["context_acceptability_n"] += 1

        # Context insensitivity: no improvement from non-rag to oracle.
        if float(rag.get("f1", 0.0)) <= float(non.get("f1", 0.0)):
            by_model[model]["context_insensitivity_count"] += 1.0
        by_model_counts[model]["context_insensitivity_n"] += 1

        # Context misinterpretation: non-rag exact match true, oracle exact match false.
        if float(non.get("em", 0.0)) >= 1.0 and float(rag.get("em", 0.0)) < 1.0:
            by_model[model]["context_misinterpretation_count"] += 1.0
        by_model_counts[model]["context_misinterpretation_n"] += 1

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model, sums in by_model.items():
        cnts = by_model_counts[model]
        out[model] = {
            "noise_vulnerability": {
                "mean_f1_drop_oracle_to_mixed": (
                    sums["noise_vulnerability_sum"] / cnts["noise_vulnerability_n"]
                    if cnts["noise_vulnerability_n"]
                    else 0.0
                )
            },
            "context_acceptability": {
                "mean_f1_gain_non_to_oracle": (
                    sums["context_acceptability_sum"] / cnts["context_acceptability_n"]
                    if cnts["context_acceptability_n"]
                    else 0.0
                )
            },
            "context_insensitivity": {
                "rate_non_improving_non_to_oracle": (
                    sums["context_insensitivity_count"] / cnts["context_insensitivity_n"]
                    if cnts["context_insensitivity_n"]
                    else 0.0
                )
            },
            "context_misinterpretation": {
                "rate_non_correct_oracle_wrong": (
                    sums["context_misinterpretation_count"] / cnts["context_misinterpretation_n"]
                    if cnts["context_misinterpretation_n"]
                    else 0.0
                )
            },
        }
    return out


def _subset_rows(rows: List[Dict[str, Any]], size: int, seed: int) -> List[Dict[str, Any]]:
    if size <= 0 or len(rows) <= size:
        return rows
    rng = random.Random(seed)
    picks = list(rows)
    rng.shuffle(picks)
    return picks[:size]


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = resolve_paths(load_config(cfg_path), cfg_path.parent)
    qa = cfg.get("qa_v2", {})
    if not qa:
        raise ValueError("Config must include qa_v2 section.")

    conditions = list(qa.get("conditions", ["non_rag", "rag_docs", "rag_mixed_docs"]))
    allowed = {"non_rag", "rag_docs", "rag_oracle_docs", "rag_mixed_docs", "rag_llm"}
    invalid = [c for c in conditions if c not in allowed]
    if invalid:
        raise ValueError(f"Unsupported qa_v2.conditions: {invalid}")

    seed = int(cfg.get("run", {}).get("seed", 42))
    output_root = Path(cfg.get("output", {}).get("root", "outputs/qa_mirage_v2"))
    run_dir = Path(args.run_dir) if args.run_dir else output_root / make_run_id(config_hash(cfg))
    ensure_dir(run_dir)

    queries = _load_queries(qa["queries_path"])
    answers_map = _load_answers(qa["answers_path"])
    top_k = int(qa.get("top_k", 5))
    oracle = _load_contexts(qa["oracle_contexts_path"], top_k=top_k)
    mixed = _load_contexts(qa["mixed_contexts_path"], top_k=top_k)

    joined: List[Dict[str, Any]] = []
    retrieval_rows: List[Dict[str, Any]] = []
    oracle_present = []
    mixed_present = []
    for row in queries:
        pid = row["prompt_id"]
        answers = answers_map.get(pid, [])
        if not answers:
            continue
        docs_oracle = list((oracle.get(pid) or {}).get("docs_top_k") or [])
        docs_mixed = list((mixed.get(pid) or {}).get("docs_top_k") or [])
        oracle_has = bool((oracle.get(pid) or {}).get("answer_present_top_k"))
        mixed_has = bool((mixed.get(pid) or {}).get("answer_present_top_k"))
        if docs_oracle and not oracle_has:
            oracle_has = answer_presence_at_k(docs_oracle, answers)
        if docs_mixed and not mixed_has:
            mixed_has = answer_presence_at_k(docs_mixed, answers)
        oracle_present.append(1.0 if oracle_has else 0.0)
        mixed_present.append(1.0 if mixed_has else 0.0)

        joined.append(
            {
                "prompt_id": pid,
                "text": row["text"],
                "answers": answers,
                "docs_oracle": docs_oracle,
                "docs_mixed": docs_mixed,
                "oracle_answer_present_top_k": oracle_has,
                "mixed_answer_present_top_k": mixed_has,
            }
        )
        retrieval_rows.append(
            {
                "prompt_id": pid,
                "query": row["text"],
                "docs_oracle_top_k": docs_oracle,
                "docs_mixed_top_k": docs_mixed,
                "oracle_answer_present_top_k": oracle_has,
                "mixed_answer_present_top_k": mixed_has,
            }
        )

    target_size = int(qa.get("final_eval_size", 500))
    eval_rows = _subset_rows(joined, target_size, seed)
    write_jsonl(run_dir / "qa_retrieval.jsonl", retrieval_rows)

    templates = load_prompt_templates(cfg.get("prompts", {}).get("template_path", "configs/prompts.yaml"))
    models_cfg = load_models_config(cfg.get("models", {}).get("config"))
    model_specs = select_models(models_cfg, cfg.get("models", {}).get("use"))
    runner = ModelRunner(cfg.get("generation", {}))

    qa_rows: List[Dict[str, Any]] = []
    context_rows: List[Dict[str, Any]] = []
    by_model_condition: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    start = time.perf_counter()

    for model_spec in model_specs:
        tokenizer = runner.get_tokenizer(model_spec)
        for condition in conditions:
            em_vals: List[float] = []
            f1_vals: List[float] = []
            contains_vals: List[float] = []
            refusal_vals: List[float] = []
            faith_vals: List[float] = []
            ctx_prec_vals: List[float] = []
            hall_vals: List[float] = []

            for row in _maybe_tqdm(eval_rows, total=len(eval_rows), desc=f"{model_spec.alias}:{condition}"):
                query = row["text"]
                answers = row["answers"]
                context_variant = "none"
                docs_for_metrics: List[Dict[str, Any]] = []
                if condition in {"rag_docs", "rag_oracle_docs"}:
                    docs_for_metrics = row["docs_oracle"]
                    context_variant = "oracle"
                elif condition == "rag_mixed_docs":
                    docs_for_metrics = row["docs_mixed"]
                    context_variant = "mixed"
                elif condition == "rag_llm":
                    docs_for_metrics = row["docs_oracle"]
                    context_variant = "oracle"

                if condition == "non_rag":
                    prompt = render_prompt(templates["non_rag"]["template"], query)
                elif condition in {"rag_docs", "rag_oracle_docs", "rag_mixed_docs"}:
                    template_key = "rag_mixed_docs" if condition == "rag_mixed_docs" and "rag_mixed_docs" in templates else "rag_docs"
                    prompt = render_prompt(
                        templates[template_key]["template"], query, [d.get("text", "") for d in docs_for_metrics]
                    )
                elif condition == "rag_llm":
                    prompt = render_prompt(
                        templates["rag_llm"]["template"], query, [d.get("text", "") for d in docs_for_metrics]
                    )
                else:
                    raise ValueError(f"Unsupported condition: {condition}")

                response = runner.generate(model_spec, prompt)
                em = exact_match(response, answers)
                f1 = token_f1(response, answers)
                contains = contains_answer(response, answers)
                refusal = 1.0 if ("i cannot" in response.lower() or "i can't" in response.lower()) else 0.0
                faith = supported_token_ratio(response, docs_for_metrics)
                ctx_prec = context_precision(query, docs_for_metrics)
                hall = hallucination_proxy(response, docs_for_metrics)

                em_vals.append(em)
                f1_vals.append(f1)
                contains_vals.append(1.0 if contains else 0.0)
                refusal_vals.append(refusal)
                faith_vals.append(faith)
                ctx_prec_vals.append(ctx_prec)
                hall_vals.append(hall)

                qa_rows.append(
                    {
                        "run_id": run_dir.name,
                        "prompt_id": row["prompt_id"],
                        "query": query,
                        "answers": answers,
                        "condition": condition,
                        "context_variant": context_variant,
                        "model": model_spec.alias,
                        "model_id": model_spec.id,
                        "response": response,
                        "em": em,
                        "f1": f1,
                        "contains_answer": bool(contains),
                        "refusal": bool(refusal),
                        "faithfulness": faith,
                        "context_precision": ctx_prec,
                        "hallucination_proxy": hall,
                        "prompt_tokens": len(tokenizer.encode(prompt)) if tokenizer else len(prompt),
                        "response_tokens": len(tokenizer.encode(response)) if tokenizer else len(response),
                    }
                )
                context_rows.append(
                    {
                        "run_id": run_dir.name,
                        "prompt_id": row["prompt_id"],
                        "condition": condition,
                        "context_variant": context_variant,
                        "model": model_spec.alias,
                        "model_id": model_spec.id,
                        "query": query,
                        "retrieval_source": "mirage_fixed",
                        "retrieved_doc_ids_used": [str(d.get("doc_id")) for d in docs_for_metrics],
                        "retrieved_context_snippets_used": [_snippet(d.get("text", "")) for d in docs_for_metrics],
                        "retrieved_ranked_scores_used": [float(d.get("score", 0.0)) for d in docs_for_metrics],
                    }
                )

            by_model_condition[model_spec.alias][condition] = {
                "count": len(em_vals),
                "em": _aggregate(em_vals),
                "f1": _aggregate(f1_vals),
                "contains_answer_rate": _aggregate(contains_vals),
                "refusal_rate": _aggregate(refusal_vals),
                "faithfulness_score": _aggregate(faith_vals),
                "context_precision": _aggregate(ctx_prec_vals),
                "hallucination_proxy": _aggregate(hall_vals),
            }

    elapsed = time.perf_counter() - start
    write_jsonl(run_dir / "qa_responses.jsonl", qa_rows)
    write_jsonl(run_dir / "qa_context_used.jsonl", context_rows)

    retrieval_metrics = {
        "dataset": qa.get("dataset_name", "mirage"),
        "variant": qa.get("dataset_variant", "mirage_validation"),
        "mode": "mirage_fixed",
        "k": top_k,
        "queries_total": len(joined),
        "eval_subset_size": len(eval_rows),
        "oracle_answer_presence_at_k": _aggregate(oracle_present),
        "mixed_answer_presence_at_k": _aggregate(mixed_present),
        "run_id": run_dir.name,
        "config_hash": config_hash(cfg),
    }
    write_json(run_dir / "retrieval_metrics.json", retrieval_metrics)

    qa_metrics: Dict[str, Any] = {
        "dataset": qa.get("dataset_name", "mirage"),
        "variant": qa.get("dataset_variant", "mirage_validation"),
        "subset_size": len(eval_rows),
        "run_id": run_dir.name,
        "config_hash": config_hash(cfg),
        "elapsed_sec": elapsed,
        "by_model_condition": dict(by_model_condition),
        "doc_reliance_gap": {},
        "run_metadata": {
            "generation_cfg": dict(cfg.get("generation", {})),
            "prompt_template_path": cfg.get("prompts", {}).get("template_path"),
            "prompt_template_sha": sha256_file(cfg.get("prompts", {}).get("template_path")),
            "qa_data_hashes": {
                "queries_path": qa.get("queries_path"),
                "queries_sha": sha256_file(qa.get("queries_path")),
                "answers_path": qa.get("answers_path"),
                "answers_sha": sha256_file(qa.get("answers_path")),
                "oracle_contexts_path": qa.get("oracle_contexts_path"),
                "oracle_contexts_sha": sha256_file(qa.get("oracle_contexts_path")),
                "mixed_contexts_path": qa.get("mixed_contexts_path"),
                "mixed_contexts_sha": sha256_file(qa.get("mixed_contexts_path")),
            },
        },
        "data_quality": {
            "queries_total": len(queries),
            "queries_answerable": len(joined),
            "eval_subset_target": target_size,
            "eval_subset_actual": len(eval_rows),
            "oracle_context_rows": len(oracle),
            "mixed_context_rows": len(mixed),
            "split": qa.get("split", "validation"),
            "context_mode": "mirage_fixed",
        },
        "condition_comparisons": {},
    }

    for model_alias, cmap in by_model_condition.items():
        comparisons: Dict[str, Dict[str, float]] = {}
        rag_key = "rag_docs" if "rag_docs" in cmap else ("rag_oracle_docs" if "rag_oracle_docs" in cmap else None)
        if rag_key and "non_rag" in cmap:
            a, b = cmap[rag_key], cmap["non_rag"]
            comparisons["rag_docs_vs_non_rag"] = {
                "em_delta": a.get("em", 0.0) - b.get("em", 0.0),
                "f1_delta": a.get("f1", 0.0) - b.get("f1", 0.0),
                "contains_delta": a.get("contains_answer_rate", 0.0) - b.get("contains_answer_rate", 0.0),
                "refusal_delta": a.get("refusal_rate", 0.0) - b.get("refusal_rate", 0.0),
            }
        if "rag_mixed_docs" in cmap and "non_rag" in cmap:
            a, b = cmap["rag_mixed_docs"], cmap["non_rag"]
            comparisons["rag_mixed_docs_vs_non_rag"] = {
                "em_delta": a.get("em", 0.0) - b.get("em", 0.0),
                "f1_delta": a.get("f1", 0.0) - b.get("f1", 0.0),
                "contains_delta": a.get("contains_answer_rate", 0.0) - b.get("contains_answer_rate", 0.0),
                "refusal_delta": a.get("refusal_rate", 0.0) - b.get("refusal_rate", 0.0),
            }
        if rag_key and "rag_mixed_docs" in cmap:
            a, b = cmap[rag_key], cmap["rag_mixed_docs"]
            comparisons["rag_docs_vs_rag_mixed_docs"] = {
                "em_delta": a.get("em", 0.0) - b.get("em", 0.0),
                "f1_delta": a.get("f1", 0.0) - b.get("f1", 0.0),
                "contains_delta": a.get("contains_answer_rate", 0.0) - b.get("contains_answer_rate", 0.0),
                "refusal_delta": a.get("refusal_rate", 0.0) - b.get("refusal_rate", 0.0),
            }
            qa_metrics["doc_reliance_gap"][model_alias] = {
                "em_gap_rag_minus_mixed": a.get("em", 0.0) - b.get("em", 0.0),
                "f1_gap_rag_minus_mixed": a.get("f1", 0.0) - b.get("f1", 0.0),
                "contains_gap_rag_minus_mixed": a.get("contains_answer_rate", 0.0) - b.get("contains_answer_rate", 0.0),
            }
        if comparisons:
            qa_metrics["condition_comparisons"][model_alias] = comparisons

    qa_metrics["context_behavior"] = _compute_context_behavior_metrics(qa_rows)

    write_json(run_dir / "qa_metrics.json", qa_metrics)
    print(f"Wrote QA retrieval records to {run_dir / 'qa_retrieval.jsonl'}")
    print(f"Wrote QA responses to {run_dir / 'qa_responses.jsonl'}")
    print(f"Wrote QA context trace to {run_dir / 'qa_context_used.jsonl'}")
    print(f"Wrote QA metrics to {run_dir / 'qa_metrics.json'}")
    print(f"Wrote retrieval metrics to {run_dir / 'retrieval_metrics.json'}")


if __name__ == "__main__":
    main()
