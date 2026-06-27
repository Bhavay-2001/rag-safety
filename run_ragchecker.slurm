#!/bin/bash
#SBATCH --job-name=ragchecker
#SBATCH --account=def-kfraser_cpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/ragchecker_%j.out
#SBATCH --error=logs/ragchecker_%j.err

# Usage:
#   sbatch run_ragchecker.slurm <run_dir>
#
# Example:
#   sbatch run_ragchecker.slurm outputs/week1_safety_improved/20260323T033415Z_ae1d0bff

set -euo pipefail

RUN_DIR="${1:?ERROR: pass the run directory as the first argument}"

# ── Environment ───────────────────────────────────────────────────────────────
module load gcc arrow/24.0.0 python/3.12.4

cd /scratch/bmalh018/rag-safety/
source $HOME/my_env/bin/activate

export HF_HOME=/home/bmalh018/scratch/.cache/huggingface
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.hf_token)
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN

export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "=== Environment ready ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
echo "HF_HOME  : $HF_HOME"
echo "Run dir  : $RUN_DIR"
echo "Job ID   : $SLURM_JOB_ID"
echo "Started  : $(date)"
mkdir -p logs

# ── Install ragchecker if not present ─────────────────────────────────────────
python3 -c "import ragchecker" 2>/dev/null || {
    echo "Installing ragchecker..."
    pip install ragchecker --quiet
    python3 -m spacy download en_core_web_sm --quiet
}

# ── Run RAGChecker ────────────────────────────────────────────────────────────
echo "=== Running RAGChecker ==="
python3 scripts/run_ragchecker.py \
    --config configs/ragchecker.yaml \
    --run-dir "$RUN_DIR"

echo "=== Done: $(date) ==="
