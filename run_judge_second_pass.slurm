#!/bin/bash
#SBATCH --job-name=rag-judge2
#SBATCH --account=def-kfraser
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Re-run multi-judge second pass + summarize for an existing eval run.
#
# Usage:
#   sbatch run_judge_second_pass.slurm outputs/three_judge/phi4/20260625T142449Z_9fa2eae8
#   sbatch --job-name=judge2-mistral run_judge_second_pass.slurm \
#       outputs/three_judge/mistral7b/<run_id>
#
# Optional 2nd arg: config path (default below).

set -euo pipefail

RUN_DIR="${1:?Usage: sbatch run_judge_second_pass.slurm <run_dir> [config]}"
CONFIG="${2:-configs/week1_safety_improved_norerank.yaml}"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: run dir does not exist: $RUN_DIR"
    exit 1
fi
if [ ! -f "$RUN_DIR/responses.jsonl" ]; then
    echo "ERROR: missing responses.jsonl in $RUN_DIR"
    exit 1
fi
if [ ! -f "$RUN_DIR/judge.jsonl" ]; then
    echo "ERROR: missing judge.jsonl in $RUN_DIR (run primary judge first)"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
module load gcc arrow/24.0.0 python/3.12.4

cd /scratch/bmalh018/rag-safety/
source "$HOME/my_env/bin/activate"

export HOME="${HOME:-/home/bmalh018}"
export HF_HOME=/scratch/bmalh018/.cache/huggingface
mkdir -p "$HF_HOME" logs

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_SDPA_ENABLED=0
export TORCH_DIA_SDPA_ENABLED=0
export TORCH_SDPA_ALLOW_INTRA_TEAM_SHARING=0

which python
echo "=== Environment ready ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

echo "Working dir: $(pwd)"
echo "Config:      $CONFIG"
echo "Run dir:     $RUN_DIR"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Started:     $(date)"

# ── Step 1: Multi-judge second pass (WildGuard + ShieldGemma) ────────────────
echo "=== [1/2] Running judge second pass / comparison ==="
python3 scripts/run_judge_second_pass.py \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR"

# ── Step 2: Summarize (writes mean_unsafe_prob into metrics.json) ─────────────
echo "=== [2/2] Summarising metrics ==="
python3 scripts/summarize.py \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR"

echo "=== Done: $(date) ==="
echo "Wrote:"
echo "  $RUN_DIR/judge_second_pass.jsonl"
echo "  $RUN_DIR/judge_comparison.jsonl"
echo "  $RUN_DIR/metrics.json"
