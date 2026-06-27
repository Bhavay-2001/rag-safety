#!/bin/bash
#SBATCH --job-name=rag-pipeline
#SBATCH --account=def-kfraser_cpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# General-purpose pipeline script.
# Uncomment the section(s) you want to run.
#
# Usage:
#   sbatch run_pipeline.slurm

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module load gcc arrow/24.0.0 python/3.12.4

cd /scratch/bmalh018/rag-safety/
source $HOME/my_env/bin/activate

export HF_HOME=/home/bmalh018/scratch/.cache/huggingface
mkdir -p "$HF_HOME"

export HUGGING_FACE_HUB_TOKEN=$(cat ~/.hf_token)
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN

export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "=== Environment ready ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
echo "HF_HOME  : $HF_HOME"
echo "Job ID   : $SLURM_JOB_ID"
echo "Started  : $(date)"
mkdir -p logs

# =============================================================================
# SECTION A — GT Response Generation + WildGuard Judge
# Produces:
#   outputs/gt/gt_safe_responses.jsonl
#   outputs/gt/gt_judge.jsonl
#   data/prompts/rrb_harmbench_with_gt.csv
# =============================================================================
# echo "=== [A] GT response generation + judge ==="
# python3 scripts/generate_gt_responses.py \
#     --config configs/gt_responses.yaml
#     # add --skip-judge      to skip WildGuard verification
#     # add --exclude-harmful to auto-drop flagged rows from the CSV

# =============================================================================
# SECTION B — RAGChecker Evaluation
# Requires: gt_safe_responses.jsonl (Section A) and a completed run dir.
# Set run.run_dir in configs/ragchecker.yaml before running.
# Produces:
#   outputs/ragchecker/ragchecker_input.json
#   outputs/ragchecker/ragchecker_results.json
#   outputs/ragchecker/ragchecker_metrics.json
# =============================================================================
echo "=== [B] RAGChecker evaluation ==="

# Install ragchecker if not present in the environment.
python3 -c "import ragchecker" 2>/dev/null || {
    echo "Installing ragchecker..."
    pip install ragchecker --quiet
    python3 -m spacy download en_core_web_sm --quiet
}

python3 scripts/run_ragchecker.py \
    --config configs/ragchecker.yaml
    # add --run-dir <path>  to override the run directory from the config
    # add --build-only      to build ragchecker_input.json only (no model call)

echo "=== Done: $(date) ==="
