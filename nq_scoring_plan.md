# Natural Questions QA Scoring Plan (Hybrid)

## Objective

Measure model QA capability for RAG vs non-RAG in a way that supports both:

1. Paper-faithful replication (live retrieval),
2. Retrieval-independent capability diagnosis (fixed oracle context).

Primary decision: use **oracle-fixed context** for week-1 capability/safety linkage, and keep BM25-live as an optional alignment branch.

## Evaluation Tracks

### Track A: Paper-faithful (`context_mode: bm25_live`)

- Query source: deterministic sample of 2,000 NQ queries.
- Retrieval: top-5 BM25 over indexed corpus.
- Evaluable filter: answer-present in retrieved top-5 under strict gate.
- Use when comparing to paper retrieval setup.

### Track B: Oracle capability (`context_mode: oracle_fixed`) - primary

- Query source: same NQ query pool.
- Context source: fixed per-question contexts from NQ qrels + docs store.
- Retrieval is frozen (no live retriever variance during model run).
- Use when evaluating whether the model can answer from provided context.

## Dataset / Artifacts

- `data/qa/nq_queries.csv`: sampled queries.
- `data/qa/nq_answers.jsonl`: short-answer references.
- `data/qa/nq_eval_445.csv`: evaluable subset selected by pipeline.
- `data/qa/nq_oracle_context_top5.jsonl`: fixed top-5 contexts for oracle mode.

Oracle context schema per row:

- `prompt_id`
- `query`
- `answers`
- `contexts`: list of `{doc_id, text, source}`
- `answer_present_top_k`

## Core Metrics

### QA Utility

- **EM**: exact normalized match to any reference.
- **Token F1**: max token overlap F1 to any reference.
- **Contains-answer rate**: response contains normalized answer span.
- **Refusal rate**: refusal detector over safe NQ prompts.

### Retrieval / Context Diagnostics

- Shared:
  - `answer_presence_at_k`
  - `hit_at_k` (qrels hit where available)
  - `mrr`
- BM25-live only:
  - `bm25_mean`, `bm25_std`
  - `retrieval_stability`
- Oracle-fixed only:
  - `mode: oracle_fixed`
  - `oracle_contexts_path`
  - `oracle_top_k`

### Grounding Proxies

- `faithfulness_score`
- `context_precision`
- `hallucination_proxy = 1 - faithfulness_score`

## Experimental Conditions

- `non_rag`: answer with model knowledge only.
- `rag_docs`: answer using provided docs only.
- `rag_llm` (optional): answer using docs + model knowledge.
- `random_docs`: only for BM25-live ablations; disabled in oracle-fixed mode.

## Key Config Interface

Under `qa`:

- `context_mode`: `bm25_live` | `oracle_fixed`
- `oracle_contexts_path`: required in `oracle_fixed`
- `oracle_top_k`: default 5
- `enable_rag_llm`: appends `rag_llm` if true
- strict gate controls:
  - `answer_presence_mode`
  - `min_answer_chars`
  - `min_query_overlap_tokens`
  - `min_bm25_score`

## Outputs

- `outputs/<run_id>/qa_metrics.json`
- `outputs/<run_id>/retrieval_metrics.json`
- `outputs/<run_id>/qa_responses.jsonl`
- `outputs/<run_id>/qa_context_used.jsonl`
- `outputs/<run_id>/qa_retrieval.jsonl`
- `data/qa/nq_eval_445.csv`

## Operational Notes

- For `oracle_fixed`, `qa_preflight.py` checks oracle file overlap and non-empty contexts.
- For `oracle_fixed`, config validation does not require BM25 index files.
- For `bm25_live`, BM25 index is required and retrieval diagnostics are fully computed.
