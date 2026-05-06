#!/usr/bin/env python
import csv, json, random
from pathlib import Path
import ir_datasets

TOP_K = 5
SAMPLE_QUERIES = Path("data/qa/nq_queries_2000.csv")
ANSWERS_OUT = Path("data/qa/nq_answers.jsonl")
EVAL_OUT = Path("data/qa/nq_eval_445.csv")
SEED = 42

def norm(s: str) -> str:
    return " ".join(s.strip().split())

def main():
    ds = ir_datasets.load("natural-questions/train")
    queries = {row["query_id"]: row["query"] for row in csv.DictReader(SAMPLE_QUERIES.open(encoding="utf-8"))}

    # index answers and qrels
    answers = {}
    for q in ds.queries_iter():
        if q.query_id in queries:
            a = getattr(q, "answers", []) or []
            ans_list = [norm(x) for x in a if norm(x)]
            answers[q.query_id] = {"query": q.text, "answers": ans_list}

    qrels = {}
    for qr in ds.qrels_iter():
        if qr.query_id in queries and qr.relevance > 0:
            qrels.setdefault(qr.query_id, []).append(str(qr.doc_id))

    # write answers jsonl
    ANSWERS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with ANSWERS_OUT.open("w", encoding="utf-8") as f:
        for qid, payload in answers.items():
            payload["prompt_id"] = qid
            payload["positive_doc_ids"] = qrels.get(qid, [])
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    # build evaluable subset (answer-present @5) using DPR warmup docs if available in ir_datasets
    eval_rows = []
    # lightweight check: mark answer_present_top_k False (we will recompute in run_qa_eval)
    for qid in queries:
        eval_rows.append({
            "prompt_id": qid,
            "text": queries[qid],
            "answers": " || ".join(answers.get(qid, {}).get("answers", [])),
            "positive_doc_ids": " || ".join(qrels.get(qid, [])),
            "answer_present_top_k": False,
        })

    # sample to 445 if larger
    random.Random(SEED).shuffle(eval_rows)
    if len(eval_rows) > 445:
        eval_rows = eval_rows[:445]

    with EVAL_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
        writer.writeheader()
        writer.writerows(eval_rows)

    print(f"Wrote {ANSWERS_OUT} ({len(answers)} answers)")
    print(f"Wrote {EVAL_OUT} ({len(eval_rows)} rows)")

if __name__ == "__main__":
    main()
