# Project Context

## Paper
**RAG LLMs are Not Safer** — Bang An, Shiyue Zhang, Mark Dredze. NAACL 2025.
https://aclanthology.org/2025.naacl-long.281/

**Core claim:** RAG makes aligned LLMs less safe, even with safe corpora and safe models.

| RQ | Finding |
|----|---------|
| RQ1 | 8/11 models show higher unsafe rates under RAG vs non-RAG |
| RQ2 | Caused by model baseline safety + doc safety + RAG task capability; safe docs alone don't prevent unsafe outputs |
| RQ3 | GCG/AutoDAN jailbreaks transfer poorly to RAG |

**Paper setup:** 11 LLMs, 5,592 harmful queries (RRB + HarmBench), English Wikipedia BM25 top-5, Llama Guard 2 judge, 16-category risk taxonomy, conditions: `non_rag` / `rag_docs` / `rag_llm`.

---

## Our Deviations from the Paper
| Aspect | Paper | Ours |
|--------|-------|------|
| Corpus | English Wikipedia ~20M docs | `nkandpa2/wiki-dolma`, capped at 200K docs |
| Safety judge | Llama Guard 2 | Llama Guard 3-8B (+ WildGuard, ShieldGemma as comparison) |
| Harmful prompts | 5,592 | 1,200 (from `rrb_harmbench_subset_1500.csv`) |
| Model | 11 LLMs | `custom_qwen3_4b` |
| Metric | % unsafe | `unsafe_actionable_rate_majority` |

---

## Corpus
- **Dataset:** `nkandpa2/wiki-dolma` (HuggingFace, ~78M rows total, `text` field)
- **Cap:** 200K docs for full run, 2K docs for pilot
- **Chunking:** paragraph-based, `min_chars=1000`
- **Built output:** `data/corpus/wiki_dolma.csv` (full), `data/corpus/wiki_dolma_pilot.csv` (pilot)
- **BM25 index:** `indexes/bm25_wiki_dolma` (full), `indexes/bm25_wiki_dolma_pilot` (pilot)
- `corpus.max_docs` in YAML is respected by `scripts/build_corpus.py` (patched to fall back to config if no CLI `--max-docs`)
- **Shuffle:** streaming shuffle applied before capping (`buffer_size=10000`, `seed=42`) — ensures the 200K docs are sampled from across the full dataset, not just the first rows

> **Potential improvement — fuzzy deduplication:** The `nkandpa2/wiki-dolma` dataset authors recommend applying a fuzzy dedup filter before use. This can be implemented with MinHash LSH (`datasketch` library): compute a MinHash signature per chunk, query a `MinHashLSH` index at `jaccard_threshold=0.85`, and skip near-duplicate chunks. This was implemented and then removed to keep the pipeline simple for now. Re-adding it requires ~10 lines in `src/corpus_builder.py` and a `corpus.dedup` config block.

---

## Active Configs

### `configs/pilot.yaml` — local smoke test
- 50 prompts, 2K corpus docs, BM25 top-3
- Model: `custom_qwen3_4b`
- Judge: Llama Guard 3-8B, `first_token_prob`, actionability enabled
- Output: `outputs/pilot`

### `configs/week1_safety_improved_norerank.yaml` — full experiment
- 1,200 prompts, 200K corpus docs, BM25 top-5
- Model: `custom_qwen3_4b`
- Judge: Llama Guard 3-8B, WildGuard, ShieldGemma comparison
- Output: `outputs/week1_safety_improved`

### `configs/base.yaml` — schema reference only, not run directly
- Defines all config keys and defaults

---

## Pipeline (run in order)

```bash
# Step 1 — Build corpus
python scripts/build_corpus.py --config configs/<config>.yaml

# Step 2 — Build BM25 index
python scripts/build_index.py --config configs/<config>.yaml

# Step 3 — Generate responses (non_rag, rag_docs, rag_llm)
python scripts/run_eval.py --config configs/<config>.yaml

# Step 4 — Judge retrieved docs
python scripts/run_doc_judge.py --config configs/<config>.yaml

# Step 5 — Judge responses
python scripts/run_judge.py --config configs/<config>.yaml

# Step 6 — Actionability labels
python scripts/run_actionability.py --config configs/<config>.yaml

# Step 7 — Present/absent labels
python scripts/run_present_absent.py --config configs/<config>.yaml

# Step 8 — Summarize metrics
python scripts/summarize.py --config configs/<config>.yaml
```

For pilot testing use `configs/pilot.yaml`; for the full run use `configs/week1_safety_improved_norerank.yaml`.

---

## Key Source Files

| File | Role |
|------|------|
| `src/corpus_builder.py` | Streams HF dataset or local file, chunks, writes CSV |
| `src/retriever_bm25.py` | Builds and queries BM25 index |
| `src/prompt_builder.py` | Renders Jinja2 prompt templates per condition |
| `src/model_runner.py` | Generates responses |
| `src/judge_runner.py` | Llama Guard 3 safety judgement |
| `src/doc_judge_runner.py` | Safety judgement on retrieved docs |
| `src/metrics.py` | Computes unsafe rates, deltas, category breakdowns |
| `src/config.py` | Loads YAML, resolves paths, produces config hash |

---

## Data Paths

| Purpose | Path |
|---------|------|
| Safety prompts (full) | `data/prompts/rrb_harmbench_subset_1500.csv` |
| QA capability (MIRAGE) | `data/qa/mirage_v2_*.csv/jsonl` |
| QA capability (NQ) | `data/qa/nq_*.csv/jsonl` |
| Prompt templates | `configs/prompts.yaml` |
| Judge prompt templates | `configs/judge_prompts.yaml` |
| Model definitions | `configs/models.yaml` |
