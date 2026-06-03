#!/bin/bash
#SBATCH --job-name=rag-safety
#SBATCH --account=def-kfraser_cpu      # FIX: Changed 'research' to your valid account allocation
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1                   # FIX: standard generic GPU request syntax on Slurm
#SBATCH --mem=32G                      
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/week1_%j.out
#SBATCH --error=logs/week1_%j.err

set -euo pipefail

CONFIG="configs/week1_safety_improved_norerank.yaml"

# ── Environment ──────────────────────────────────────────────────────────────
# 1. Load system dependencies first
module load gcc arrow/24.0.0 python/3.12.4

# FIX: Use /scratch for code, data processing, and large environments to avoid Home quota limits
cd /scratch/bmalh018/rag-safety/
source $HOME/my_env/bin/activate

# FIX: Do not hardcode /usr/lib paths into PYTHONPATH as it corrupts Slurm's clean modules
# export PYTHONPATH=$(pwd):${PYTHONPATH:-}

# FIX: Point HF cache to Scratch. Home (50GB limit) will crash trying to cache Llama/Gemma weights.
export HF_HOME=/scratch/bmalh018/.cache/huggingface
mkdir -p "$HF_HOME"

export PYTHONPATH=$(pwd):${PYTHONPATH:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

which python
echo "=== Environment ready ==="

# Diagnostic check
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

echo "Working dir: $(pwd)"
echo "Config: $CONFIG"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

# Ensure logs dir exists
mkdir -p logs

# ── Step 1: Build corpus ─────────────────────────────────────────────────────
echo "=== [1/6] Building corpus ==="
python3 scripts/build_corpus.py --config "$CONFIG"

# ── Step 2: Build BM25 index ─────────────────────────────────────────────────
echo "=== [2/6] Building BM25 index ==="
python3 scripts/build_index.py --config "$CONFIG"

# ── Step 3: Generate responses ───────────────────────────────────────────────
echo "=== [3/6] Running generation (non_rag / rag_docs / rag_llm) ==="
export TORCH_CUDNN_SDPA_ENABLED=0
export TORCH_DIA_SDPA_ENABLED=0
export TORCH_SDPA_ALLOW_INTRA_TEAM_SHARING=0

python3 scripts/run_eval.py --config "$CONFIG"

# Capture the most recent run directory
RUN_DIR="$(ls -td outputs/week1_safety_improved/* | head -n 1)"
echo "Run dir: $RUN_DIR"

# ── Step 4: Judge retrieved documents ────────────────────────────────────────
echo "=== [4/6] Judging retrieved documents ==="
python3 scripts/run_doc_judge.py --config "$CONFIG"

# ── Step 5: Judge responses ──────────────────────────────────────────────────
echo "=== [5/6] Judging responses ==="
python3 scripts/run_judge.py --config "$CONFIG"

# ── Step 6: Summarize ────────────────────────────────────────────────────────
echo "=== [6/6] Summarising metrics ==="
python3 scripts/summarize.py --config "$CONFIG" --run-dir "$RUN_DIR"

echo "=== Done: $(date) ==="
echo "Results in: $RUN_DIR"
