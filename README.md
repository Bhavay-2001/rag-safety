# safe-rag3

Scriptable end-to-end RAG capability + safety evaluation pipelines.

This repo supports:
- corpus/index build for BM25 retrieval,
- model generation across `non_rag` and RAG settings,
- safety judging (Llama Guard + optional WildGuard/ShieldGemma),
- actionability and context-grounded safety analysis,
- cross-model reporting and sample exports.

## Core scripts

- `scripts/build_corpus.py`: Build corpus CSV from source.
- `scripts/build_index.py`: Build BM25 index.
- `scripts/run_eval.py`: Main generation pipeline (`responses.jsonl`, `retrieval.jsonl`, `run_stats.json`).
- `scripts/run_doc_judge.py`: Judge retrieved docs safety.
- `scripts/run_judge.py`: Judge response safety.
- `scripts/run_judge_second_pass.py`: Run second-pass/comparison judges.
- `scripts/run_actionability.py`: Label unsafe outputs as actionable/non-actionable/unclear.
- `scripts/run_present_absent.py`: Label whether unsafe content appears in retrieved context.
- `scripts/summarize.py`: Build metrics summary.

Capability scripts:
- `scripts/run_qa_eval.py`
- `scripts/run_qa_eval_v2.py`

Reporting/export scripts:
- `scripts/build_cross_model_analysis_pack.py`
- `scripts/export_cross_model_examples_txt.py`
- `scripts/export_cross_model_examples_split_txt.py`

## Prompt sets used in current pipelines

Safety generation prompts:
- `configs/prompts.yaml`

Capability (MIRAGE) prompts:
- `configs/prompts_mirage_v2.yaml`

Safety judge prompt templates:
- `configs/judge_prompts.yaml` (optional override; defaults are embedded in `src/judge_runner.py`).

## Current pipeline setups used

### Safety pipeline (primary)
- Config: `configs/week1_safety_improved_norerank.yaml`
- Prompt dataset: `data/prompts/rrb_harmbench_subset_1500.csv`
- Dataset composition (current file):
  - rows: `1500`
  - columns: `prompt_id`, `text`, `source_benchmark`, `source_dataset`, `category`, `attack_style`, `split`
- Run-time subset policy:
  - config sets `run.max_prompts: 1200`, so week1 safety runs evaluate the first 1200 rows from this file.
- Source and construction notes:
  - this file is the project’s curated HarmBench-derived subset used for safety generation prompts.
  - canonical source file in repo: `data/prompts/rrb_harmbench_subset_1500.csv`
  - upstream benchmark sources used for construction:
    - Red Teaming Resistance Benchmark (RRB): `https://github.com/haizelabs/redteaming-resistance-benchmark`
    - HarmBench: `https://github.com/centerforaisafety/HarmBench`
- Conditions: `non_rag`, `rag_docs`, `rag_llm`
- Retriever: BM25 top-5 from `indexes/bm25_full`
- Judges:
  - primary response/doc judge: `meta-llama/Llama-Guard-3-8B`
  - comparison judges: `allenai/wildguard`, `google/shieldgemma-2b`
  - actionability model: `meta-llama/Llama-Guard-3-1B`
- Primary risk metric used in analysis: `unsafe_actionable_rate_majority`

### Capability pipeline (MIRAGE QA v2)
- Config: `configs/week1_capability_mirage_v2.yaml`
- Dataset files:
  - `data/qa/mirage_v2_queries.csv`
  - `data/qa/mirage_v2_answers.jsonl`
  - `data/qa/mirage_v2_contexts_oracle.jsonl`
  - `data/qa/mirage_v2_contexts_mixed.jsonl`
  - `data/qa/mirage_v2_manifest.json`
- Dataset/source and construction details (from manifest):
  - upstream dataset: `nlpai-lab/mirage`
  - selected split in manifest: `train`
  - raw rows available before filtering/subsetting: `7560`
  - output rows materialized for this project pack: `1000`
  - mixed-context construction: `synthetic_distractor_cap_answer_bearing_1`
  - answer presence @k (manifest coverage): oracle `0.99`, mixed `0.364`
- Conditions: `non_rag`, `rag_docs` (oracle), `rag_mixed_docs` (noisy mixed context)
- Configured eval size: `1000`
- Supporting source file documenting this pack:
  - `data/qa/mirage_v2_manifest.json`

### Capability pipeline (NQ QA)
- Configs:
  - `configs/week1_capability.yaml`
  - `configs/week1_capability_nq_oracle.yaml`
  - `configs/week1_capability_nq_broad.yaml`
- Dataset files:
  - `data/qa/nq_queries.csv`
  - `data/qa/nq_answers.jsonl`
  - `data/qa/nq_eval_445.csv`
  - `data/qa/nq_oracle_context_top5.jsonl` (oracle mode)

## Dataset paths to share

Safety question dataset:
- `/home/adi7/safe-rag3/data/prompts/rrb_harmbench_subset_1500.csv`

Capability datasets:
- MIRAGE:
  - `/home/adi7/safe-rag3/data/qa/mirage_v2_queries.csv`
  - `/home/adi7/safe-rag3/data/qa/mirage_v2_answers.jsonl`
  - `/home/adi7/safe-rag3/data/qa/mirage_v2_contexts_oracle.jsonl`
  - `/home/adi7/safe-rag3/data/qa/mirage_v2_contexts_mixed.jsonl`
- NQ:
  - `/home/adi7/safe-rag3/data/qa/nq_queries.csv`
  - `/home/adi7/safe-rag3/data/qa/nq_answers.jsonl`
  - `/home/adi7/safe-rag3/data/qa/nq_eval_445.csv`
  - `/home/adi7/safe-rag3/data/qa/nq_oracle_context_top5.jsonl`

## Output sample locations

Cross-model reports:
- `reports/safety_cross_model_2026-03-25/metrics_all_runs.csv`
- `reports/safety_cross_model_2026-03-25/leaderboard_rag_docs.csv`
- `reports/safety_cross_model_2026-03-25/run_inventory.csv`
- `reports/safety_cross_model_2026-03-25/evidence_pack_summary.csv`

Readable safety example exports:
- `reports/safety_cross_model_2026-03-25/samples/safety_pipeline_samples_full_2026-04-01.txt`
- `reports/safety_cross_model_2026-03-25/samples/split_txt/by_model__*.txt`
- `reports/safety_cross_model_2026-03-25/samples/split_txt/by_setting__*.txt`

FIR run outputs (cluster):
- `/scratch/adi7/rag-test/outputs/<run_root>/<run_id>/`

## Minimal run (local smoke)

```bash
python scripts/build_corpus.py --config configs/pilot.yaml --max-docs 2000
python scripts/build_index.py --config configs/pilot.yaml --max-docs 2000
python scripts/run_eval.py --config configs/pilot.yaml
python scripts/run_doc_judge.py --config configs/pilot.yaml
python scripts/run_judge.py --config configs/pilot.yaml
python scripts/summarize.py --config configs/pilot.yaml
```

## Notes

- Default corpus source in base config is `wikimedia/wikipedia:20240601.en`.
- Keep heavy JSONL artifacts in scratch/private storage; commit sampled/summary artifacts to GitHub.
- For sensitive output sharing, use private channels (institutional storage or private repo).
