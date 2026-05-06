from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths
from src.prompt_builder import load_prompt_templates, render_prompt
from src.retriever_bm25 import load_index, retrieve

_BAD_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{(query|c|sources|answer)\}(?!\})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA preflight checks for NQ pipeline.")
    parser.add_argument("--config", required=True, help="Path to QA config.")
    return parser.parse_args()


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
            if text:
                rows.append({"prompt_id": prompt_id, "text": str(text)})
    return rows


def _load_answers(path: str | Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_id = str(row.get("prompt_id"))
            answers = row.get("answers") or []
            if isinstance(answers, str):
                answers = [answers] if answers.strip() else []
            out[prompt_id] = [str(a).strip() for a in answers if str(a).strip()]
    return out


def _load_oracle_contexts(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_id = str(row.get("prompt_id", "")).strip()
            if not prompt_id:
                continue
            contexts = row.get("contexts") or row.get("docs_top_k") or []
            clean: List[Dict[str, Any]] = []
            for idx, ctx in enumerate(contexts):
                if not isinstance(ctx, dict):
                    continue
                text = str(ctx.get("text", "")).strip()
                if not text:
                    continue
                score_raw = ctx.get("score", 0.0)
                try:
                    score = float(score_raw) if score_raw is not None else 0.0
                except (TypeError, ValueError):
                    score = 0.0
                clean.append(
                    {
                        "doc_id": str(ctx.get("doc_id") or f"{prompt_id}-ctx-{idx}"),
                        "text": text,
                        "score": score,
                    }
                )
            out[prompt_id] = clean
    return out


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = resolve_paths(load_config(cfg_path), cfg_path.parent)
    qa_cfg = cfg.get("qa", {})
    if not qa_cfg:
        raise ValueError("Config missing 'qa' section.")
    context_mode = str(qa_cfg.get("context_mode", "bm25_live")).lower()
    if context_mode not in {"bm25_live", "oracle_fixed"}:
        raise ValueError("qa.context_mode must be one of: bm25_live, oracle_fixed.")

    queries = _load_queries(qa_cfg["queries_path"])
    answers = _load_answers(qa_cfg["answers_path"])
    qids = {q["prompt_id"] for q in queries}
    aids = set(answers.keys())
    nonempty = sum(1 for vals in answers.values() if vals)
    overlap = len(qids & aids)

    if nonempty == 0:
        raise ValueError("qa.answers_path has zero non-empty answers.")
    if overlap == 0:
        raise ValueError("No query/answer ID overlap detected.")

    templates = load_prompt_templates(cfg.get("prompts", {}).get("template_path"))
    non_template = templates["non_rag"]["template"]
    rag_template = templates["rag_docs"]["template"]
    if _BAD_PLACEHOLDER_RE.search(non_template) or _BAD_PLACEHOLDER_RE.search(rag_template):
        raise ValueError("Prompt templates contain single-brace placeholders; use Jinja {{ ... }}.")

    sample_query = queries[0]["text"]
    docs: List[Dict[str, Any]] = []
    if context_mode == "bm25_live":
        # retrieval prompt smoke render
        bm25, doc_ids, doc_texts = load_index(cfg.get("retriever", {}).get("index_dir", "indexes/bm25"))
        docs = retrieve(bm25, doc_ids, doc_texts, sample_query, top_k=int(cfg.get("retriever", {}).get("top_k", 5)))
    else:
        conditions = list(qa_cfg.get("conditions", []))
        if bool(qa_cfg.get("enable_rag_llm", False)) and "rag_llm" not in conditions:
            conditions.append("rag_llm")
        if "random_docs" in conditions:
            raise ValueError("qa.conditions includes random_docs, which is disabled for qa.context_mode=oracle_fixed.")
        oracle_contexts_path = qa_cfg.get("oracle_contexts_path")
        if not oracle_contexts_path:
            raise ValueError("qa.oracle_contexts_path is required when qa.context_mode=oracle_fixed.")
        context_rows = _load_oracle_contexts(oracle_contexts_path)
        overlap_rows = [q for q in queries if q["prompt_id"] in context_rows]
        if not overlap_rows:
            raise ValueError("No overlap between qa.queries_path and qa.oracle_contexts_path prompt IDs.")
        sample = next((q for q in overlap_rows if context_rows.get(q["prompt_id"])), overlap_rows[0])
        sample_query = sample["text"]
        docs = context_rows.get(sample["prompt_id"], [])
        if not docs:
            raise ValueError("oracle_contexts file has no non-empty contexts for sampled query.")

    non_prompt = render_prompt(non_template, sample_query)
    rag_prompt = render_prompt(rag_template, sample_query, [d["text"] for d in docs])
    if sample_query not in non_prompt or sample_query not in rag_prompt:
        raise ValueError("Rendered prompt does not include query text.")
    if docs and docs[0]["text"][:40] not in rag_prompt:
        raise ValueError("Rendered rag_docs prompt does not include retrieved context.")
    if _BAD_PLACEHOLDER_RE.search(non_prompt) or _BAD_PLACEHOLDER_RE.search(rag_prompt):
        raise ValueError("Rendered prompt contains unresolved placeholders.")

    print("QA preflight passed.")
    print(f"queries_total={len(queries)} answers_nonempty={nonempty} overlap={overlap}")
    print(f"sample_docs_top_k={len(docs)}")


if __name__ == "__main__":
    main()

