from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import apply_overrides, config_hash, load_config, resolve_paths
from src.model_runner import ModelRunner, load_models_config, select_models
from src.prompt_builder import load_prompt_templates, render_prompt
from src.retriever_bm25 import hard_negative_docs, load_index, random_docs, retrieve, retrieve_candidates
from src.retriever_rerank import rerank_with_embeddings
from src.utils_io import ensure_dir, make_run_id, read_jsonl, write_json, write_jsonl


PAD_UNIT = "This is neutral filler text for length matching."


def _maybe_tqdm(iterable, total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG/non-RAG evaluations.")
    parser.add_argument("--config", required=True, help="Path to configs/base.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override config value, e.g. --set models.use=custom_llama31_8b")
    return parser.parse_args()




def _load_prompts(path: str | Path, max_prompts: int | None = None) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
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
            row["text"] = text
            if "prompt_id" not in row or not row["prompt_id"]:
                row["prompt_id"] = str(idx)
            prompts.append(row)
            if max_prompts and len(prompts) >= max_prompts:
                break
    return prompts


def _pad_prompt_to_tokens(tokenizer: Any, prompt: str, target_tokens: int) -> str:
    if tokenizer is None:
        return _pad_prompt_to_chars(prompt, target_tokens)
    current = len(tokenizer.encode(prompt))
    if current >= target_tokens:
        return prompt
    pad_tokens = max(1, len(tokenizer.encode(PAD_UNIT)))
    repeats = int(math.ceil((target_tokens - current) / pad_tokens))
    return prompt + "\n" + (PAD_UNIT + "\n") * repeats


def _pad_prompt_to_chars(prompt: str, target_chars: int) -> str:
    if len(prompt) >= target_chars:
        return prompt
    repeats = int(math.ceil((target_chars - len(prompt)) / max(1, len(PAD_UNIT))))
    return prompt + "\n" + (PAD_UNIT * repeats)


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, cfg_path.parent)
    cfg = apply_overrides(cfg, args.set)

    run_cfg = cfg.get("run", {})
    seed = int(run_cfg.get("seed", 42))
    max_prompts = run_cfg.get("max_prompts")

    cfg_hash = config_hash(cfg)
    run_id = make_run_id(cfg_hash)
    output_root = Path(cfg.get("output", {}).get("root", "outputs/runs"))
    run_dir = ensure_dir(output_root / run_id)

    write_json(run_dir / "config.json", cfg)
    eval_mode = str(cfg.get("analysis", {}).get("evaluation_mode", "baseline_paper_aligned"))

    prompts_cfg = cfg.get("prompts", {})
    prompts_path = prompts_cfg.get("data_path") or prompts_cfg.get("path")
    template_path = prompts_cfg.get("template_path") or prompts_cfg.get("path")
    if not prompts_path:
        raise ValueError("prompts.data_path is required.")
    if not template_path:
        raise ValueError("prompts.template_path is required.")
    prompts = _load_prompts(prompts_path, max_prompts=max_prompts)

    templates = load_prompt_templates(template_path)
    bm25, doc_ids, doc_texts = load_index(cfg.get("retriever", {}).get("index_dir", "indexes/bm25"))
    retriever_cfg = cfg.get("retriever", {})
    top_k = int(retriever_cfg.get("top_k", 5))
    retrieval_mode = str(retriever_cfg.get("mode", "bm25_top5"))
    bm25_candidates_k = int(retriever_cfg.get("bm25_candidates_k", 50))
    rerank_model_id = str(retriever_cfg.get("rerank_model_id", "sentence-transformers/all-MiniLM-L6-v2"))
    rerank_alpha = float(retriever_cfg.get("rerank_alpha", 0.5))

    retrieval_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    retrieval_lines: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        prompt_id = prompt["prompt_id"]
        query = prompt["text"]

        docs_candidates = retrieve_candidates(
            bm25,
            doc_ids,
            doc_texts,
            query,
            top_k=bm25_candidates_k if retrieval_mode == "bm25_top50_rerank_top5" else top_k,
        )
        if retrieval_mode == "bm25_top50_rerank_top5":
            docs_rag = rerank_with_embeddings(
                query=query,
                candidates=docs_candidates,
                top_k=top_k,
                model_name=rerank_model_id,
                alpha=rerank_alpha,
            )
        else:
            docs_rag = docs_candidates[:top_k]
        docs_random = random_docs(doc_ids, doc_texts, top_k=top_k, seed=seed + idx)
        docs_hard = hard_negative_docs(bm25, doc_ids, doc_texts, query, top_k=top_k)

        retrieval_map[prompt_id] = {
            "rag_docs": docs_rag,
            "rag_llm": docs_rag,
            "random_docs": docs_random,
            "hard_negative": docs_hard,
        }

        retrieval_lines.extend(
            [
                {
                    "prompt_id": prompt_id,
                    "condition": "non_rag",
                    "retrieval_mode": retrieval_mode,
                    "docs": [],
                    "bm25_candidates": [],
                    "selected_docs": [],
                    "rerank_scores": [],
                },
                {
                    "prompt_id": prompt_id,
                    "condition": "len_match",
                    "retrieval_mode": retrieval_mode,
                    "docs": [],
                    "bm25_candidates": [],
                    "selected_docs": [],
                    "rerank_scores": [],
                },
                {
                    "prompt_id": prompt_id,
                    "condition": "rag_docs",
                    "retrieval_mode": retrieval_mode,
                    "docs": docs_rag,
                    "bm25_candidates": [
                        {"doc_id": d.get("doc_id"), "score": d.get("score")} for d in docs_candidates
                    ],
                    "selected_docs": [
                        {"doc_id": d.get("doc_id"), "score": d.get("score"), "hybrid_score": d.get("hybrid_score")}
                        for d in docs_rag
                    ],
                    "rerank_scores": [
                        {"doc_id": d.get("doc_id"), "rerank_score": d.get("rerank_score")} for d in docs_rag
                    ],
                },
                {
                    "prompt_id": prompt_id,
                    "condition": "rag_llm",
                    "retrieval_mode": retrieval_mode,
                    "docs": docs_rag,
                    "bm25_candidates": [
                        {"doc_id": d.get("doc_id"), "score": d.get("score")} for d in docs_candidates
                    ],
                    "selected_docs": [
                        {"doc_id": d.get("doc_id"), "score": d.get("score"), "hybrid_score": d.get("hybrid_score")}
                        for d in docs_rag
                    ],
                    "rerank_scores": [
                        {"doc_id": d.get("doc_id"), "rerank_score": d.get("rerank_score")} for d in docs_rag
                    ],
                },
                {
                    "prompt_id": prompt_id,
                    "condition": "random_docs",
                    "retrieval_mode": retrieval_mode,
                    "docs": docs_random,
                    "bm25_candidates": [],
                    "selected_docs": [],
                    "rerank_scores": [],
                },
                {
                    "prompt_id": prompt_id,
                    "condition": "hard_negative",
                    "retrieval_mode": retrieval_mode,
                    "docs": docs_hard,
                    "bm25_candidates": [],
                    "selected_docs": [],
                    "rerank_scores": [],
                },
            ]
        )

    write_jsonl(run_dir / "retrieval.jsonl", retrieval_lines)

    models_cfg = load_models_config(cfg.get("models", {}).get("config"))
    model_specs = select_models(models_cfg, cfg.get("models", {}).get("use", "paper_default"))
    runner = ModelRunner(cfg.get("generation", {}))

    responses_path = run_dir / "responses.jsonl"

    # Load already-completed (prompt_id, condition, model) tuples for resume
    done_keys: set = set()
    if responses_path.exists():
        for r in read_jsonl(responses_path):
            done_keys.add((r["prompt_id"], r["condition"], r["model"]))
        if done_keys:
            print(f"Resuming: {len(done_keys)} responses already written, skipping those.")

    stats: Dict[str, Any] = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "models": {},
        "total": {
            "responses": 0,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "elapsed_sec": 0.0,
        },
    }
    start_time = time.perf_counter()
    for model_spec in model_specs:
        tokenizer = runner.get_tokenizer(model_spec)
        model_key = model_spec.alias
        stats["models"][model_key] = {
            "responses": 0,
            "prompt_tokens": 0,
            "response_tokens": 0,
        }
        prompt_iter = _maybe_tqdm(
            prompts,
            total=len(prompts),
            desc=f"Model {model_spec.alias}",
        )
        for prompt in prompt_iter:
            prompt_id = prompt["prompt_id"]
            query = prompt["text"]
            docs = retrieval_map[prompt_id]["rag_docs"]
            rag_prompt = render_prompt(templates["rag_docs"]["template"], query, [d["text"] for d in docs])

            for condition in cfg.get("conditions", []):
                if (prompt_id, condition, model_spec.alias) in done_keys:
                    continue
                if condition == "non_rag":
                    final_prompt = render_prompt(templates["non_rag"]["template"], query)
                elif condition == "rag_docs":
                    final_prompt = rag_prompt
                elif condition == "rag_llm":
                    final_prompt = render_prompt(templates["rag_llm"]["template"], query, [d["text"] for d in docs])
                elif condition == "random_docs":
                    rd = retrieval_map[prompt_id]["random_docs"]
                    final_prompt = render_prompt(templates["rag_docs"]["template"], query, [d["text"] for d in rd])
                elif condition == "hard_negative":
                    hd = retrieval_map[prompt_id]["hard_negative"]
                    final_prompt = render_prompt(templates["rag_docs"]["template"], query, [d["text"] for d in hd])
                elif condition == "len_match":
                    base_prompt = render_prompt(templates["non_rag"]["template"], query)
                    target_len = len(rag_prompt) if tokenizer is None else len(tokenizer.encode(rag_prompt))
                    final_prompt = _pad_prompt_to_tokens(tokenizer, base_prompt, target_len)
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                response_text = runner.generate(model_spec, final_prompt)
                if tokenizer is not None:
                    prompt_tokens = len(tokenizer.encode(final_prompt))
                    response_tokens = len(tokenizer.encode(response_text))
                else:
                    prompt_tokens = len(final_prompt)
                    response_tokens = len(response_text)
                record = {
                    "prompt_id": prompt_id,
                    "condition": condition,
                    "model": model_spec.alias,
                    "model_id": model_spec.id,
                    "query": query,
                    "response": response_text,
                    "prompt": final_prompt,
                    "evaluation_mode": eval_mode,
                    "retrieval_mode": retrieval_mode,
                }
                write_jsonl(responses_path, [record], append=True)
                stats["models"][model_key]["responses"] += 1
                stats["models"][model_key]["prompt_tokens"] += prompt_tokens
                stats["models"][model_key]["response_tokens"] += response_tokens
                stats["total"]["responses"] += 1
                stats["total"]["prompt_tokens"] += prompt_tokens
                stats["total"]["response_tokens"] += response_tokens
        print(f"Completed model {model_spec.alias}")

    stats["total"]["elapsed_sec"] = time.perf_counter() - start_time
    write_json(run_dir / "run_stats.json", stats)

    print(f"Wrote retrievals to {run_dir / 'retrieval.jsonl'}")
    print(f"Wrote responses to {responses_path}")
    print(f"Wrote run stats to {run_dir / 'run_stats.json'}")


if __name__ == "__main__":
    main()
