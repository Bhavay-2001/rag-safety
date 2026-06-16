#!/bin/bash
#SBATCH --job-name=gt-responses
#SBATCH --account=def-kfraser_cpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/gt_%j.out
#SBATCH --error=logs/gt_%j.err

# Usage:
#   sbatch run_gt_responses.slurm
#
#   # skip WildGuard judge (generation only)
#   sbatch run_gt_responses.slurm --skip-judge
#
#   # auto-drop harmful rows from the CSV
#   sbatch run_gt_responses.slurm --exclude-harmful

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module load gcc arrow/24.0.0 python/3.12.4

cd /scratch/bmalh018/rag-safety/
source $HOME/my_env/bin/activate

# Must match the path used in models_download.py so transformers finds
# the weights in cache without re-downloading.
export HF_HOME=/home/bmalh018/scratch/.cache/huggingface
mkdir -p "$HF_HOME"

export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Sanity checks ─────────────────────────────────────────────────────────────
which python
echo "=== Environment ready ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
echo "HF_HOME : $HF_HOME"
echo "Job ID  : $SLURM_JOB_ID"
echo "Started : $(date)"
mkdir -p logs

# ── Run GT pipeline ───────────────────────────────────────────────────────────
# Pass any extra CLI flags through (e.g. --skip-judge, --exclude-harmful)
EXTRA_ARGS="${@}"

echo "=== Running GT response generation + judge ==="
python3 scripts/generate_gt_responses.py \
    --config configs/gt_responses.yaml \
    $EXTRA_ARGS

echo "=== Done: $(date) ==="
