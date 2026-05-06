# FIR Execution Status and Next Plan

## Purpose
This document tracks what has been completed, what failed, and the exact next commands for smoke tests and full runs on FIR.

## Current Status (as of latest terminal logs)

### Completed
- Local pipeline scaffolding was extended for the capstone plan:
  - QA runner and metrics (`scripts/run_qa_eval.py`, `src/qa_metrics.py`)
  - Data/provenance utilities (`scripts/fetch_rrb_harmbench.py`, `scripts/fetch_nq_irds.py`)
  - Reporting/table generation (`scripts/build_report_tables.py`)
  - Config validation (`scripts/validate_run_config.py`)
  - Week-1 configs and Slurm helpers (`configs/week1_capability.yaml`, `configs/week1_safety.yaml`, `scripts/run_week1.slurm`)
- Harmful prompt canonical files were generated from cloned repos (RRB/HarmBench provenance path).
- FIR prep passed:
  - `data/qa/nq_queries.csv`, `data/qa/nq_answers.jsonl`, `data/qa/nq_eval_445.csv` are present on FIR.
  - `python scripts/validate_run_config.py --config configs/week1_capability.yaml` passed after QA files were copied.
  - BM25 index exists on FIR: `indexes/bm25_full/bm25.pkl` (built successfully).

### Prior work already done (Honours pipeline baseline)
- End-to-end Honours-style safety runs were executed earlier on FIR (Tiny + full variants) and produced:
  - retrieval, responses, doc_safety, judge outputs
  - aggregated metrics/run_stats/doc_judge_stats/judge_stats
- This provides a working baseline and runtime behavior reference for current capstone runs.

### Open issues from latest logs
- Wikipedia source version mismatch happened during corpus build (`20240601.en` unavailable on current FIR `datasets` environment). Temporary workaround used `20231101.en`.
- QA smoke was not yet confirmed complete end-to-end with valid answer alignment:
  - `run_qa_eval.py` failed when `nq_answers.jsonl` had no usable answers (`No QA queries with answers found`).
  - An earlier interactive QA attempt also showed throughput too slow for one monolithic 445x3-condition job within 6 hours.

### Error log: 2026-02-15 NQ broad-corpus attempt
- Command sequence tried:
  - `build_corpus.py` with `configs/nq_broad_corpus.yaml`
  - `build_index.py` with `configs/week1_capability_nq_broad.yaml`
  - `qa_preflight.py` and `run_qa_eval.py`
- Failures observed:
  1. `OSError: [Errno 122] Disk quota exceeded` during corpus CSV write.
  2. `_csv.Error: field larger than field limit (131072)` while reading large corpus rows for BM25 build.
  3. `FileNotFoundError ... indexes/bm25_nq_broad/bm25.pkl` (downstream because index build failed).
- Important detail:
  - QA smoke still ran because `configs/qa_smoke_nq_rag.yaml` points to `indexes/bm25_full`, not `indexes/bm25_nq_broad`.

### Root-cause analysis for the above error
- The broad Wikipedia materialization strategy writes a very large local CSV on quota-limited storage.
- CSV-based loading in current BM25 pipeline is sensitive to very large `text` fields.
- Even when storage issue is bypassed, this approach adds retrieval variance that is not required for the primary capability question (model answer-from-context ability).

### Decision update: Hybrid QA capability tracks
- **Primary**: `oracle_fixed` context mode for capability/safety linkage.
  - Context is frozen per query using NQ qrels + docs store.
  - Eliminates full-corpus build requirement and retrieval variance.
- **Optional alignment branch**: `bm25_live` mode for paper-faithful retrieval comparisons.

## Decision: Run style for 6-hour limit
- Keep each GPU job capped to `06:00:00`.
- Do not run all QA conditions in one job.
- Shard QA by condition and smaller subset size per job.
- Use dependency chaining in Slurm.

## Immediate Next Step: Oracle-context QA smoke

```bash
cd ~/safe-rag3
source venv/bin/activate
module load gcc arrow || true
export HF_HOME=~/scratch/hf_cache

# Build fixed NQ contexts once (qrels + docs_store)
python -X utf8 scripts/build_nq_oracle_context.py \
  --irds-id natural-questions/train \
  --queries-path data/qa/nq_queries.csv \
  --answers-path data/qa/nq_answers.jsonl \
  --output-path data/qa/nq_oracle_context_top5.jsonl \
  --top-k 5

# Validate oracle config and run preflight
python scripts/validate_run_config.py --config configs/qa_smoke_nq_oracle.yaml
python scripts/qa_preflight.py --config configs/qa_smoke_nq_oracle.yaml

# Run smoke (non_rag + rag_docs)
python -X utf8 scripts/run_qa_eval.py --config configs/qa_smoke_nq_oracle.yaml
```

## Immediate Next Step: Safety stage smoke (after QA smoke is fixed)

Run this safety smoke test on FIR with small prompt limit:

```bash
cd ~/safe-rag3
source venv/bin/activate
module load gcc arrow || true
export HF_HOME=~/scratch/hf_cache

# Optional: quick safety config validation
python scripts/validate_run_config.py --config configs/week1_safety.yaml

# Small smoke run on GPU allocation
salloc --time=00:45:00 --gpus=h100:1 --cpus-per-task=4 --mem=16G --account=def-kfraser
module load gcc arrow || true
source ~/safe-rag3/venv/bin/activate
export HF_HOME=~/scratch/hf_cache

python -X utf8 scripts/run_eval.py --config configs/week1_safety.yaml
python -X utf8 scripts/run_doc_judge.py --config configs/week1_safety.yaml
python -X utf8 scripts/run_judge.py --config configs/week1_safety.yaml
python -X utf8 scripts/summarize.py --config configs/week1_safety.yaml
exit
```

If this smoke test passes, submit full safety as a 6-hour batch with dependency on the QA job.

## Recommended Execution Order (next)

1. Build and validate oracle contexts (`build_nq_oracle_context.py` + `qa_preflight.py`).
2. Oracle QA smoke (`run_qa_eval.py` with `configs/qa_smoke_nq_oracle.yaml`).
3. Submit week1 oracle QA (`configs/week1_capability_nq_oracle.yaml`).
4. Run safety smoke and full safety.
5. Optional: run paper-faithful BM25 capability branch for alignment comparison.
6. Generate report tables after QA + safety completion.

## Commands to monitor jobs/logs

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode,MaxRSS,AllocGRES
tail -f ~/scratch/rag-test/logs/qa-<name>-<jobid>.out
tail -f ~/scratch/rag-test/logs/safety-<name>-<jobid>.out
```

## Deliverables to share with professor
- Run status doc (this file).
- QA metrics + retrieval metrics from latest successful run dir.
- Safety metrics from latest successful run dir.
- Tradeoff table and diagnostic table markdown outputs in `reports/tables/`.
