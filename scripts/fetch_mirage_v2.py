from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.qa_metrics import contains_answer
from src.utils_io import ensure_dir, write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and normalize MIRAGE dataset for capability v2.")
    parser.add_argument("--split", default="validation", help="HF split to load (default: validation).")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max row count after sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--output-dir", default="data/qa", help="Output directory for normalized files.")
    return parser.parse_args()


def _to_answers(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for v in value:
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("doc_chunk", "text", "passage", "content", "body", "default_text"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _extract_contexts(value: Any, source: str, top_k: int = 5) -> List[Dict[str, Any]]:
    def _collect_strings(obj: Any, out: List[str]) -> None:
        if isinstance(obj, str):
            s = obj.strip()
            if s:
                out.append(s)
            return
        if isinstance(obj, list):
            for x in obj:
                _collect_strings(x, out)
            return
        if isinstance(obj, dict):
            # Prefer known text-like keys first.
            for key in ("doc_chunk", "text", "passage", "content", "body", "default_text", "snippet"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    out.append(val.strip())
            for v in obj.values():
                _collect_strings(v, out)

    out: List[Dict[str, Any]] = []

    def _append_item(item: Any) -> None:
        if len(out) >= top_k:
            return
        text = _extract_text(item)
        if not text:
            candidates: List[str] = []
            _collect_strings(item, candidates)
            # Use the longest candidate as a best-effort chunk.
            if candidates:
                text = max(candidates, key=len)
        if not text:
            return
        doc_id = None
        score = 0.0
        if isinstance(item, dict):
            doc_id = item.get("doc_id") or item.get("id") or item.get("document_id")
            score_raw = item.get("score")
            try:
                score = float(score_raw) if score_raw is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
        out.append(
            {
                "doc_id": str(doc_id) if doc_id is not None else f"{source}-{len(out)}",
                "text": text,
                "score": score,
                "source": source,
            }
        )

    if isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                for nested in item:
                    _append_item(nested)
            elif isinstance(item, dict) and "docs" in item and isinstance(item["docs"], list):
                for nested in item["docs"]:
                    _append_item(nested)
            else:
                _append_item(item)
            if len(out) >= top_k:
                break
    elif isinstance(value, dict):
        if "docs" in value and isinstance(value["docs"], list):
            for item in value["docs"]:
                _append_item(item)
                if len(out) >= top_k:
                    break
        else:
            _append_item(value)
    else:
        _append_item(value)
    return out


def _pick_rows(rows: List[Dict[str, Any]], max_rows: int | None, seed: int) -> List[Dict[str, Any]]:
    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    keep = sorted(idx[:max_rows])
    return [rows[i] for i in keep]


def _dedupe_contexts(contexts: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for ctx in contexts:
        text = " ".join(str(ctx.get("text", "")).split())
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(ctx)
        if len(out) >= top_k:
            break
    return out


def _synthesize_mixed_contexts(
    prompt_id: str,
    answers: List[str],
    oracle_contexts: List[Dict[str, Any]],
    mixed_raw_contexts: List[Dict[str, Any]],
    distractor_pool: List[Dict[str, Any]],
    rng: random.Random,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    raw_neg = [c for c in mixed_raw_contexts if not contains_answer(c.get("text", ""), answers)]
    raw_pos = [c for c in mixed_raw_contexts if contains_answer(c.get("text", ""), answers)]
    oracle_neg = [c for c in oracle_contexts if not contains_answer(c.get("text", ""), answers)]
    oracle_pos = [c for c in oracle_contexts if contains_answer(c.get("text", ""), answers)]

    selected: List[Dict[str, Any]] = []
    selected.extend(raw_neg[:2])
    if len(selected) < 2:
        selected.extend(oracle_neg[: 2 - len(selected)])

    positives = raw_pos + oracle_pos
    # Cap answer-bearing evidence to at most one chunk.
    if positives and len(selected) < top_k and rng.random() < 0.35:
        selected.append(positives[0])

    need = top_k - len(selected)
    if need > 0:
        candidates = [
            {
                "doc_id": str(d.get("doc_id") or f"{prompt_id}-mx-d"),
                "text": str(d.get("text", "")),
                "score": float(d.get("score", 0.0)),
                "source": "mirage_mixed_synth",
            }
            for d in distractor_pool
            if str(d.get("owner_prompt_id")) != prompt_id and not contains_answer(str(d.get("text", "")), answers)
        ]
        rng.shuffle(candidates)
        selected.extend(candidates[:need])

    if len(selected) < top_k:
        backup = raw_neg + oracle_neg + raw_pos + oracle_pos
        selected.extend(backup[: top_k - len(selected)])

    return _dedupe_contexts(selected, top_k=top_k)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required. Install with `pip install datasets` (or module-compatible equivalent)."
        ) from exc

    ds = load_dataset("nlpai-lab/mirage", split=args.split)
    raw_rows = [dict(row) for row in ds]
    rows = _pick_rows(raw_rows, args.max_rows, args.seed)

    queries_path = out_dir / "mirage_v2_queries.csv"
    answers_path = out_dir / "mirage_v2_answers.jsonl"
    oracle_path = out_dir / "mirage_v2_contexts_oracle.jsonl"
    mixed_path = out_dir / "mirage_v2_contexts_mixed.jsonl"
    manifest_path = out_dir / "mirage_v2_manifest.json"

    queries_rows: List[Dict[str, str]] = []
    answers_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    mixed_rows: List[Dict[str, Any]] = []
    oracle_present = 0
    mixed_present = 0
    rng = random.Random(args.seed)
    records: List[Dict[str, Any]] = []
    distractor_pool: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        query = str(row.get("query") or row.get("question") or "").strip()
        if not query:
            continue
        source_dataset = str(row.get("source_dataset") or row.get("dataset") or row.get("source") or "mirage")
        query_id = str(row.get("query_id") or row.get("id") or i)
        prompt_id = f"mirage:{source_dataset}:{query_id}"

        answers = _to_answers(row.get("answer"))
        oracle_contexts = _extract_contexts(row.get("oracle"), source="mirage_oracle", top_k=5)
        mixed_contexts_raw = _extract_contexts(row.get("doc_pool"), source="mirage_mixed", top_k=5)

        oracle_has_answer = any(contains_answer(ctx.get("text", ""), answers) for ctx in oracle_contexts)
        mixed_has_answer = any(contains_answer(ctx.get("text", ""), answers) for ctx in mixed_contexts_raw)
        if oracle_has_answer:
            oracle_present += 1

        queries_rows.append(
            {
                "prompt_id": prompt_id,
                "text": query,
                "source_benchmark": "MIRAGE",
                "source_dataset": source_dataset,
            }
        )
        answers_rows.append(
            {
                "prompt_id": prompt_id,
                "query": query,
                "answers": answers,
                "source_dataset": source_dataset,
            }
        )
        oracle_rows.append(
            {
                "prompt_id": prompt_id,
                "query": query,
                "answers": answers,
                "contexts": oracle_contexts,
                "answer_present_top_k": oracle_has_answer,
                "context_variant": "oracle",
                "source_dataset": source_dataset,
            }
        )
        records.append(
            {
                "prompt_id": prompt_id,
                "query": query,
                "answers": answers,
                "source_dataset": source_dataset,
                "oracle_contexts": oracle_contexts,
                "mixed_contexts_raw": mixed_contexts_raw,
            }
        )
        for c in oracle_contexts + mixed_contexts_raw:
            text = str(c.get("text", "")).strip()
            if not text:
                continue
            distractor_pool.append(
                {
                    "owner_prompt_id": prompt_id,
                    "doc_id": str(c.get("doc_id", "")),
                    "text": text,
                    "score": float(c.get("score", 0.0)),
                }
            )

    for rec in records:
        mixed_contexts = _synthesize_mixed_contexts(
            prompt_id=rec["prompt_id"],
            answers=rec["answers"],
            oracle_contexts=rec["oracle_contexts"],
            mixed_raw_contexts=rec["mixed_contexts_raw"],
            distractor_pool=distractor_pool,
            rng=rng,
            top_k=5,
        )
        mixed_has_answer = any(contains_answer(ctx.get("text", ""), rec["answers"]) for ctx in mixed_contexts)
        if mixed_has_answer:
            mixed_present += 1
        mixed_rows.append(
            {
                "prompt_id": rec["prompt_id"],
                "query": rec["query"],
                "answers": rec["answers"],
                "contexts": mixed_contexts,
                "answer_present_top_k": mixed_has_answer,
                "context_variant": "mixed_synth" if not rec["mixed_contexts_raw"] else "mixed",
                "source_dataset": rec["source_dataset"],
            }
        )

    with queries_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "text", "source_benchmark", "source_dataset"])
        writer.writeheader()
        writer.writerows(queries_rows)

    write_jsonl(answers_path, answers_rows)
    write_jsonl(oracle_path, oracle_rows)
    write_jsonl(mixed_path, mixed_rows)

    manifest = {
        "dataset": "nlpai-lab/mirage",
        "split": args.split,
        "rows_total_raw": len(raw_rows),
        "rows_total_output": len(queries_rows),
        "max_rows": args.max_rows,
        "seed": args.seed,
        "paths": {
            "queries": str(queries_path),
            "answers": str(answers_path),
            "contexts_oracle": str(oracle_path),
            "contexts_mixed": str(mixed_path),
        },
        "coverage": {
            "oracle_answer_present_rate": (oracle_present / len(oracle_rows)) if oracle_rows else 0.0,
            "mixed_answer_present_rate": (mixed_present / len(mixed_rows)) if mixed_rows else 0.0,
        },
        "mixed_strategy": "synthetic_distractor_cap_answer_bearing_1",
        "mixed_raw_nonempty_rows": sum(1 for r in records if r["mixed_contexts_raw"]),
    }
    write_json(manifest_path, manifest)
    print(f"Wrote MIRAGE v2 queries to {queries_path} ({len(queries_rows)} rows)")
    print(f"Wrote MIRAGE v2 answers to {answers_path}")
    print(f"Wrote MIRAGE v2 oracle contexts to {oracle_path}")
    print(f"Wrote MIRAGE v2 mixed contexts to {mixed_path}")
    print(f"Wrote MIRAGE v2 manifest to {manifest_path}")


if __name__ == "__main__":
    main()
