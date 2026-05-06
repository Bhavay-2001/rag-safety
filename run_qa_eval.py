from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import config_hash, load_config, resolve_paths
from src.metrics import is_refusal
from src.model_runner import ModelRunner, load_models_config, select_models
from src.prompt_builder import load_prompt_templates, render_prompt
from src.qa_metrics import (
    answer_presence_at_k,
    bm25_stats,
    contains_answer,
    context_precision,
    exact_match,
    first_answer_rank,
    hallucination_proxy,
    hit_at_k_from_qrels,
    retrieval_overlap,
    supported_token_ratio,
    tokenize,
    token_f1,
)
from src.retriever_bm25 import load_index, random_docs, retrieve
from src.utils_io import ensure_dir, make_run_id, read_jsonl, sha256_file, write_json, write_jsonl


_BAD_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{(query|c|sources|answer)\}(?!\})")
_SNIPPET_CHARS = 600


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NQ capability-first QA+RAG benchmark.")
    parser.add_argument("--config", required=True, help="Path to config yaml.")
    parser.add_argument("--run-dir", default=None, help="Optional fixed run directory.")
    return parser.parse_args()


def _resolve_qa_conditions(qa_cfg: Dict[str, Any]) -> List[str]:
    conditions = list(qa_cfg.get("conditions", ["non_rag", "rag_docs", "random_docs"]))
    if bool(qa_cfg.get("enable_rag_llm", False)) and "rag_llm" not in conditions:
        conditions.append("rag_llm")
    return conditions


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
            text = (
                row.get("text")
                or row.get("question")
                or row.get("query")
                or row.get("prompt")
                or next((v for v in row.values() if v and str(v).strip()), "")
            )
            if not text:
                continue
            rows.append({"prompt_id": prompt_id, "text": str(text)})
    return rows


def _load_answers(path: str | Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        prompt_id = str(row.get("prompt_id"))
        answers = row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        positive_doc_ids = row.get("positive_doc_ids") or []
        out[prompt_id] = {
            "answers": [str(a) for a in answers if str(a).strip()],
            "positive_doc_ids": [str(d) for d in positive_doc_ids],
        }
    return out


def _variant_query(query: str) -> str:
    return f"Please provide a short factual answer: {query}"


def _build_subset_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "text",
                "answers",
                "positive_doc_ids",
                "answer_present_top_k",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "prompt_id": row["prompt_id"],
                    "text": row["text"],
                    "answers": " || ".join(row.get("answers", [])),
                    "positive_doc_ids": " || ".join(row.get("positive_doc_ids", [])),
                    "answer_present_top_k": row.get("answer_present_top_k", False),
                }
            )


def _aggregate_scores(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    s = " ".join(str(text).split())
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + "..."


def _validate_prompt_templates(
    templates: Dict[str, Dict[str, Any]],
    sample_query: str,
    sample_docs: List[Dict[str, Any]],
) -> None:
    if "non_rag" not in templates or "rag_docs" not in templates:
        raise ValueError("Prompt templates must include 'non_rag' and 'rag_docs'.")
    non_template = templates["non_rag"].get("template", "")
    rag_template = templates["rag_docs"].get("template", "")
    if _BAD_PLACEHOLDER_RE.search(non_template) or _BAD_PLACEHOLDER_RE.search(rag_template):
        raise ValueError("Prompt templates contain single-brace placeholders (e.g., {query}). Use Jinja {{ query }}.")

    non_prompt = render_prompt(non_template, sample_query)
    rag_prompt = render_prompt(rag_template, sample_query, [d["text"] for d in sample_docs])
    if sample_query not in non_prompt or sample_query not in rag_prompt:
        raise ValueError("Rendered QA prompt does not include query text. Check prompt template configuration.")
    if sample_docs:
        rag_norm = " ".join(rag_prompt.split())
        doc_norm = " ".join(str(sample_docs[0]["text"]).split())
        if doc_norm[:40] and doc_norm[:40] not in rag_norm:
            raise ValueError("Rendered RAG QA prompt does not include retrieved document context.")
    if _BAD_PLACEHOLDER_RE.search(non_prompt) or _BAD_PLACEHOLDER_RE.search(rag_prompt):
        raise ValueError("Rendered QA prompt still contains unresolved placeholders.")


def _validate_answer_quality(answers_map: Dict[str, Dict[str, Any]]) -> None:
    all_answers: List[str] = []
    for payload in answers_map.values():
        all_answers.extend(str(a).strip() for a in payload.get("answers", []) if str(a).strip())
    if not all_answers:
        return

    token_lengths = [len(a.split()) for a in all_answers]
    avg_len = sum(token_lengths) / len(token_lengths)
    wh_words = {"who", "what", "where", "when", "why", "how", "which"}
    wh_ratio = sum(1 for a in all_answers if a.lower() in wh_words) / len(all_answers)
    if avg_len <= 1.2 and wh_ratio >= 0.25:
        raise ValueError(
            "QA answers look invalid (mostly single-token interrogatives). "
            "Regenerate qa.answers_path using a short-answer extraction source."
        )


def _count_nonempty_answers(answers_map: Dict[str, Dict[str, Any]]) -> int:
    return sum(1 for payload in answers_map.values() if payload.get("answers"))


def _answer_present_strict(
    doc_text: str,
    answers: List[str],
    query: str,
    min_answer_chars: int,
    min_query_overlap_tokens: int,
    min_bm25_score: float,
    bm25_score: float,
) -> bool:
    if bm25_score < min_bm25_score:
        return False
    if min_query_overlap_tokens > 0:
        query_tokens = {t for t in tokenize(query)}
        doc_tokens = set(tokenize(doc_text))
        if len(query_tokens & doc_tokens) < min_query_overlap_tokens:
            return False
    for ans in answers:
        ans_s = str(ans).strip()
        if len(ans_s) < min_answer_chars:
            continue
        if contains_answer(doc_text, [ans_s]):
            return True
    return False


def _answer_presence_and_rank(
    docs: List[Dict[str, Any]],
    answers: List[str],
    query: str,
    mode: str,
    min_answer_chars: int,
    min_query_overlap_tokens: int,
    min_bm25_score: float,
) -> Tuple[bool, int | None]:
    if mode != "strict":
        present = answer_presence_at_k(docs, answers)
        return present, first_answer_rank(docs, answers)

    for rank, doc in enumerate(docs, start=1):
        if _answer_present_strict(
            doc_text=doc.get("text", ""),
            answers=answers,
            query=query,
            min_answer_chars=min_answer_chars,
            min_query_overlap_tokens=min_query_overlap_tokens,
            min_bm25_score=min_bm25_score,
            bm25_score=float(doc.get("score", 0.0)),
        ):
            return True, rank
    return False, None


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, cfg_path.parent)
    cfg_hash = config_hash(cfg)

    qa_cfg = cfg.get("qa", {})
    if not qa_cfg:
        raise ValueError("Missing required 'qa' section in config.")

    output_root = Path(cfg.get("output", {}).get("root", "outputs/qa"))
    run_dir = Path(args.run_dir) if args.run_dir else ensure_dir(output_root / make_run_id(cfg_hash))
    ensure_dir(run_dir)
    write_json(run_dir / "config.json", cfg)

    seed = int(cfg.get("run", {}).get("seed", 42))
    rng = random.Random(seed)
    top_k = int(cfg.get("retriever", {}).get("top_k", 5))
    target_eval_size = int(qa_cfg.get("final_eval_size", 445))
    stability_sample = int(qa_cfg.get("stability_sample_size", 100))
    dataset_name = str(qa_cfg.get("dataset_name", ""))
    answer_presence_mode = str(
        qa_cfg.get("answer_presence_mode", "strict" if "natural_questions" in dataset_name else "basic")
    ).lower()
    min_answer_chars = int(qa_cfg.get("min_answer_chars", 2))
    min_query_overlap_tokens = int(qa_cfg.get("min_query_overlap_tokens", 1))
    min_bm25_score = float(qa_cfg.get("min_bm25_score", 0.0))

    queries = _load_queries(qa_cfg["queries_path"])
    answers_map = _load_answers(qa_cfg["answers_path"])
    queries_total = len(queries)
    answers_nonempty_total = _count_nonempty_answers(answers_map)
    query_ids_all = {q["prompt_id"] for q in queries}
    answer_ids_all = set(answers_map.keys())
    overlap_total = len(query_ids_all & answer_ids_all)
    _validate_answer_quality(answers_map)
    queries = [q for q in queries if q["prompt_id"] in answers_map and answers_map[q["prompt_id"]]["answers"]]
    if not queries:
        raise ValueError("No QA queries with answers found. Check qa.queries_path and qa.answers_path.")

    bm25, doc_ids, doc_texts = load_index(cfg.get("retriever", {}).get("index_dir", "indexes/bm25"))
    templates = load_prompt_templates(cfg.get("prompts", {}).get("template_path"))

    retrieval_records: List[Dict[str, Any]] = []
    score_values: List[float] = []
    answer_present_ids: List[str] = []
    mrr_values: List[float] = []
    hit_values: List[float] = []
    stability_values: List[float] = []

    for i, query_row in enumerate(_maybe_tqdm(queries, total=len(queries), desc="QA retrieval")):
        prompt_id = query_row["prompt_id"]
        query_text = query_row["text"]
        answers = answers_map[prompt_id]["answers"]
        positive_doc_ids = answers_map[prompt_id]["positive_doc_ids"]

        docs = retrieve(bm25, doc_ids, doc_texts, query_text, top_k=top_k)
        docs_random = random_docs(doc_ids, doc_texts, top_k=top_k, seed=seed + i)

        present, first_rank = _answer_presence_and_rank(
            docs=docs,
            answers=answers,
            query=query_text,
            mode=answer_presence_mode,
            min_answer_chars=min_answer_chars,
            min_query_overlap_tokens=min_query_overlap_tokens,
            min_bm25_score=min_bm25_score,
        )
        if present:
            answer_present_ids.append(prompt_id)
        mrr_values.append(1.0 / first_rank if first_rank else 0.0)
        if positive_doc_ids:
            hit_values.append(1.0 if hit_at_k_from_qrels(docs, positive_doc_ids) else 0.0)
        else:
            hit_values.append(1.0 if present else 0.0)

        score_values.extend(float(d.get("score", 0.0)) for d in docs)

        retrieval_records.append(
            {
                "prompt_id": prompt_id,
                "query": query_text,
                "answers": answers,
                "positive_doc_ids": positive_doc_ids,
                "docs_top_k": docs,
                "docs_random": docs_random,
                "answer_present_top_k": present,
                "first_answer_rank": first_rank,
            }
        )

    if not answer_present_ids:
        raise ValueError(
            "No answer-present queries in top-k retrieval under current QA filter settings. "
            "Check corpus/index and qa.answer_presence_* thresholds."
        )

    _validate_prompt_templates(templates, retrieval_records[0]["query"], retrieval_records[0]["docs_top_k"])

    eval_candidates = [r for r in retrieval_records if r["answer_present_top_k"]]
    if len(eval_candidates) > target_eval_size:
        eval_records = rng.sample(eval_candidates, target_eval_size)
    else:
        eval_records = eval_candidates

    # Retrieval stability on a deterministic subset.
    stability_records = eval_records[: min(stability_sample, len(eval_records))]
    for row in stability_records:
        docs_a = row["docs_top_k"]
        docs_b = retrieve(bm25, doc_ids, doc_texts, _variant_query(row["query"]), top_k=top_k)
        overlap = retrieval_overlap([d["doc_id"] for d in docs_a], [d["doc_id"] for d in docs_b])
        stability_values.append(overlap)

    eval_subset_path = Path(qa_cfg.get("eval_subset_path", "data/qa/nq_eval_445.csv"))
    subset_rows: List[Dict[str, Any]] = []
    for row in eval_records:
        subset_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "text": row["query"],
                "answers": row["answers"],
                "positive_doc_ids": row["positive_doc_ids"],
                "answer_present_top_k": row["answer_present_top_k"],
            }
        )
    _build_subset_csv(eval_subset_path, subset_rows)

    write_jsonl(run_dir / "qa_retrieval.jsonl", retrieval_records)

    models_cfg = load_models_config(cfg.get("models", {}).get("config"))
    model_specs = select_models(models_cfg, cfg.get("models", {}).get("use", "custom_qa"))
    runner = ModelRunner(cfg.get("generation", {}))
    conditions = _resolve_qa_conditions(qa_cfg)

    qa_rows: List[Dict[str, Any]] = []
    context_rows: List[Dict[str, Any]] = []
    by_model_condition: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    model_runtime_meta: Dict[str, Dict[str, Any]] = {}

    start = time.perf_counter()
    for model_spec in model_specs:
        tokenizer = runner.get_tokenizer(model_spec)
        model_runtime_meta[model_spec.alias] = {
            "model_id": model_spec.id,
            "provider": model_spec.provider,
            "tokenizer_class": tokenizer.__class__.__name__ if tokenizer is not None else None,
            "uses_chat_template": bool(getattr(tokenizer, "chat_template", None)) if tokenizer is not None else False,
        }
        for condition in conditions:
            em_vals: List[float] = []
            f1_vals: List[float] = []
            contains_vals: List[float] = []
            refusal_vals: List[float] = []
            faith_vals: List[float] = []
            ctx_prec_vals: List[float] = []
            hall_vals: List[float] = []

            for row in _maybe_tqdm(eval_records, total=len(eval_records), desc=f"{model_spec.alias}:{condition}"):
                query = row["query"]
                answers = row["answers"]
                if condition == "non_rag":
                    prompt = render_prompt(templates["non_rag"]["template"], query)
                    docs_for_metrics = []
                    retrieval_source = "none"
                elif condition == "rag_docs":
                    docs_for_metrics = row["docs_top_k"]
                    prompt = render_prompt(
                        templates["rag_docs"]["template"], query, [d["text"] for d in docs_for_metrics]
                    )
                    retrieval_source = "top_k"
                elif condition == "rag_llm":
                    docs_for_metrics = row["docs_top_k"]
                    prompt = render_prompt(
                        templates["rag_llm"]["template"], query, [d["text"] for d in docs_for_metrics]
                    )
                    retrieval_source = "top_k"
                elif condition == "random_docs":
                    docs_for_metrics = row["docs_random"]
                    prompt = render_prompt(
                        templates["rag_docs"]["template"], query, [d["text"] for d in docs_for_metrics]
                    )
                    retrieval_source = "random_docs"
                else:
                    raise ValueError(f"Unknown QA condition: {condition}")

                response = runner.generate(model_spec, prompt)
                em = exact_match(response, answers)
                f1 = token_f1(response, answers)
                contains = 1.0 if contains_answer(response, answers) else 0.0
                refusal = 1.0 if is_refusal(response) else 0.0
                faith = supported_token_ratio(response, docs_for_metrics) if docs_for_metrics else 0.0
                ctx_precision = context_precision(query, docs_for_metrics) if docs_for_metrics else 0.0
                hall = hallucination_proxy(response, docs_for_metrics) if docs_for_metrics else 0.0

                em_vals.append(em)
                f1_vals.append(f1)
                contains_vals.append(contains)
                refusal_vals.append(refusal)
                faith_vals.append(faith)
                ctx_prec_vals.append(ctx_precision)
                hall_vals.append(hall)

                qa_row = {
                    "prompt_id": row["prompt_id"],
                    "condition": condition,
                    "model": model_spec.alias,
                    "model_id": model_spec.id,
                    "query": query,
                    "answers": answers,
                    "prompt": prompt,
                    "retrieval_source": retrieval_source,
                    "retrieved_doc_ids_used": [str(d.get("doc_id")) for d in docs_for_metrics],
                    "retrieved_context_snippets_used": [_snippet(d.get("text", "")) for d in docs_for_metrics],
                    "retrieved_ranked_scores_used": [float(d.get("score", 0.0)) for d in docs_for_metrics],
                    "response": response,
                    "em": em,
                    "f1": f1,
                    "contains_answer": bool(contains),
                    "refusal": bool(refusal),
                    "faithfulness": faith,
                    "context_precision": ctx_precision,
                    "hallucination_proxy": hall,
                    "prompt_tokens": len(tokenizer.encode(prompt)) if tokenizer else len(prompt),
                    "response_tokens": len(tokenizer.encode(response)) if tokenizer else len(response),
                }
                qa_rows.append(qa_row)
                context_rows.append(
                    {
                        "run_id": run_dir.name,
                        "prompt_id": row["prompt_id"],
                        "condition": condition,
                        "model": model_spec.alias,
                        "model_id": model_spec.id,
                        "query": query,
                        "retrieval_source": retrieval_source,
                        "retrieved_doc_ids_used": [str(d.get("doc_id")) for d in docs_for_metrics],
                        "retrieved_context_snippets_used": [_snippet(d.get("text", "")) for d in docs_for_metrics],
                        "retrieved_ranked_scores_used": [float(d.get("score", 0.0)) for d in docs_for_metrics],
                    }
                )

            by_model_condition[model_spec.alias][condition] = {
                "count": len(em_vals),
                "em": _aggregate_scores(em_vals),
                "f1": _aggregate_scores(f1_vals),
                "contains_answer_rate": _aggregate_scores(contains_vals),
                "refusal_rate": _aggregate_scores(refusal_vals),
                "faithfulness_score": _aggregate_scores(faith_vals),
                "context_precision": _aggregate_scores(ctx_prec_vals),
                "hallucination_proxy": _aggregate_scores(hall_vals),
            }

    elapsed = time.perf_counter() - start
    write_jsonl(run_dir / "qa_responses.jsonl", qa_rows)
    write_jsonl(run_dir / "qa_context_used.jsonl", context_rows)

    bm25_mean, bm25_std = bm25_stats(score_values)
    retrieval_metrics = {
        "dataset": qa_cfg.get("dataset_name", "natural_questions"),
        "variant": qa_cfg.get("dataset_variant", "nq_open"),
        "k": top_k,
        "queries_total": len(queries),
        "eval_subset_size": len(eval_records),
        "answer_presence_at_k": len(answer_present_ids) / len(queries),
        "hit_at_k": _aggregate_scores(hit_values),
        "mrr": _aggregate_scores(mrr_values),
        "bm25_mean": bm25_mean,
        "bm25_std": bm25_std,
        "retrieval_stability": _aggregate_scores(stability_values),
        "run_id": run_dir.name,
        "config_hash": cfg_hash,
    }
    write_json(run_dir / "retrieval_metrics.json", retrieval_metrics)

    qa_metrics: Dict[str, Any] = {
        "dataset": qa_cfg.get("dataset_name", "natural_questions"),
        "variant": qa_cfg.get("dataset_variant", "nq_open"),
        "subset_size": len(eval_records),
        "run_id": run_dir.name,
        "config_hash": cfg_hash,
        "elapsed_sec": elapsed,
        "by_model_condition": dict(by_model_condition),
        "doc_reliance_gap": {},
        "run_metadata": {
            "generation_cfg": dict(cfg.get("generation", {})),
            "models": model_runtime_meta,
            "prompt_template_path": cfg.get("prompts", {}).get("template_path"),
            "prompt_template_sha": sha256_file(cfg.get("prompts", {}).get("template_path")),
            "qa_data_hashes": {
                "queries_path": qa_cfg.get("queries_path"),
                "queries_sha": sha256_file(qa_cfg.get("queries_path")),
                "answers_path": qa_cfg.get("answers_path"),
                "answers_sha": sha256_file(qa_cfg.get("answers_path")),
            },
        },
        "data_quality": {
            "queries_total": queries_total,
            "answers_nonempty": answers_nonempty_total,
            "query_answer_id_overlap": overlap_total,
            "queries_answerable": len(queries),
            "eval_subset_target": target_eval_size,
            "eval_subset_actual": len(eval_records),
            "answer_presence_mode": answer_presence_mode,
            "min_answer_chars": min_answer_chars,
            "min_query_overlap_tokens": min_query_overlap_tokens,
            "min_bm25_score": min_bm25_score,
        },
        "condition_comparisons": {},
    }

    for model_alias, condition_map in by_model_condition.items():
        comparisons: Dict[str, Dict[str, float]] = {}

        if "rag_docs" in condition_map and "random_docs" in condition_map:
            rag = condition_map["rag_docs"]
            random_cond = condition_map["random_docs"]
            qa_metrics["doc_reliance_gap"][model_alias] = {
                "em_gap_rag_minus_random": rag.get("em", 0.0) - random_cond.get("em", 0.0),
                "f1_gap_rag_minus_random": rag.get("f1", 0.0) - random_cond.get("f1", 0.0),
                "contains_gap_rag_minus_random": rag.get("contains_answer_rate", 0.0)
                - random_cond.get("contains_answer_rate", 0.0),
            }

        if "rag_docs" in condition_map and "non_rag" in condition_map:
            rag = condition_map["rag_docs"]
            non = condition_map["non_rag"]
            comparisons["rag_docs_vs_non_rag"] = {
                "em_delta": rag.get("em", 0.0) - non.get("em", 0.0),
                "f1_delta": rag.get("f1", 0.0) - non.get("f1", 0.0),
                "contains_delta": rag.get("contains_answer_rate", 0.0) - non.get("contains_answer_rate", 0.0),
                "refusal_delta": rag.get("refusal_rate", 0.0) - non.get("refusal_rate", 0.0),
            }

        if "rag_llm" in condition_map and "non_rag" in condition_map:
            rag_llm = condition_map["rag_llm"]
            non = condition_map["non_rag"]
            comparisons["rag_llm_vs_non_rag"] = {
                "em_delta": rag_llm.get("em", 0.0) - non.get("em", 0.0),
                "f1_delta": rag_llm.get("f1", 0.0) - non.get("f1", 0.0),
                "contains_delta": rag_llm.get("contains_answer_rate", 0.0) - non.get("contains_answer_rate", 0.0),
                "refusal_delta": rag_llm.get("refusal_rate", 0.0) - non.get("refusal_rate", 0.0),
            }

        if "rag_llm" in condition_map and "rag_docs" in condition_map:
            rag_llm = condition_map["rag_llm"]
            rag = condition_map["rag_docs"]
            comparisons["rag_llm_vs_rag_docs"] = {
                "em_delta": rag_llm.get("em", 0.0) - rag.get("em", 0.0),
                "f1_delta": rag_llm.get("f1", 0.0) - rag.get("f1", 0.0),
                "contains_delta": rag_llm.get("contains_answer_rate", 0.0) - rag.get("contains_answer_rate", 0.0),
                "refusal_delta": rag_llm.get("refusal_rate", 0.0) - rag.get("refusal_rate", 0.0),
            }

        if "rag_docs" in condition_map and "random_docs" in condition_map:
            rag = condition_map["rag_docs"]
            rnd = condition_map["random_docs"]
            comparisons["rag_docs_vs_random_docs"] = {
                "em_delta": rag.get("em", 0.0) - rnd.get("em", 0.0),
                "f1_delta": rag.get("f1", 0.0) - rnd.get("f1", 0.0),
                "contains_delta": rag.get("contains_answer_rate", 0.0) - rnd.get("contains_answer_rate", 0.0),
                "refusal_delta": rag.get("refusal_rate", 0.0) - rnd.get("refusal_rate", 0.0),
            }

        if comparisons:
            qa_metrics["condition_comparisons"][model_alias] = comparisons

    write_json(run_dir / "qa_metrics.json", qa_metrics)

    print(f"Wrote QA eval subset to {eval_subset_path}")
    print(f"Wrote QA retrieval records to {run_dir / 'qa_retrieval.jsonl'}")
    print(f"Wrote QA responses to {run_dir / 'qa_responses.jsonl'}")
    print(f"Wrote QA context trace to {run_dir / 'qa_context_used.jsonl'}")
    print(f"Wrote QA metrics to {run_dir / 'qa_metrics.json'}")
    print(f"Wrote retrieval metrics to {run_dir / 'retrieval_metrics.json'}")


if __name__ == "__main__":
    main()
