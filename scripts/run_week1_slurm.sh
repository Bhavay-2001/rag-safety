#!/bin/bash
#SBATCH --job-name=rag-safety
#SBATCH --account=research
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/week1_%j.out
#SBATCH --error=logs/week1_%j.err

set -euo pipefail

CONFIG="configs/week1_safety_improved_norerank.yaml"

# ── Environment ──────────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023 gcc python arrow/23.0.1
source ~/scratch/venvs/safe-rag3/bin/activate
export HF_HOME=~/scratch/hf_cache

echo "=== Environment ready ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

cd "$SLURM_SUBMIT_DIR"
echo "Working dir: $(pwd)"
echo "Config: $CONFIG"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

# ── Step 1: Build corpus ─────────────────────────────────────────────────────
echo "=== [1/6] Building corpus ==="
python scripts/build_corpus.py --config "$CONFIG"

# ── Step 2: Build BM25 index ─────────────────────────────────────────────────
echo "=== [2/6] Building BM25 index ==="
python scripts/build_index.py --config "$CONFIG"

# ── Step 3: Generate responses ───────────────────────────────────────────────
echo "=== [3/6] Running generation (non_rag / rag_docs / rag_llm) ==="
python scripts/run_eval.py --config "$CONFIG"

# Capture the most recent run directory
RUN_DIR="$(ls -td outputs/week1_safety_improved/* | head -n 1)"
echo "Run dir: $RUN_DIR"

# ── Step 4: Judge retrieved documents ────────────────────────────────────────
echo "=== [4/6] Judging retrieved documents ==="
python scripts/run_doc_judge.py --config "$CONFIG"

# ── Step 5: Judge responses ──────────────────────────────────────────────────
echo "=== [5/6] Judging responses ==="
python scripts/run_judge.py --config "$CONFIG"

# ── Step 6: Summarize ────────────────────────────────────────────────────────
echo "=== [6/6] Summarising metrics ==="
python scripts/summarize.py --config "$CONFIG" --run-dir "$RUN_DIR" \

echo "=== Done: $(date) ==="
echo "Results in: $RUN_DIR"
