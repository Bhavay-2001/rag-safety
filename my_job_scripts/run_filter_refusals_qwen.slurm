#!/bin/bash
#SBATCH --job-name=rag-refusefilt
#SBATCH --account=def-kfraser
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --exclusive
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Filter clear refusals from rag_llm responses using Qwen3-30B-A3B-Instruct-2507.
# Keeps clear answers and hedged/partial answers; drops clear refusals only.
#
# Usage:
#   sbatch my_job_scripts/run_filter_refusals_qwen.slurm
#   sbatch my_job_scripts/run_filter_refusals_qwen.slurm outputs/three_judge_softprob_answered
#   sbatch my_job_scripts/run_filter_refusals_qwen.slurm outputs/three_judge_softprob_answered outputs/three_judge_softprob
#
# Args:
#   $1 = NEW output root (default: outputs/three_judge_softprob_answered)
#   $2 = SOURCE output root (default: outputs/three_judge_softprob)
#
# Array tasks: gemma_7b, llama31_8b, mistral7b, phi4

set -euo pipefail

NEW_ROOT="${1:-outputs/three_judge_softprob_answered}"
SRC_ROOT="${2:-outputs/three_judge_softprob}"

MODELS=(gemma_7b llama31_8b mistral7b phi4)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "${#MODELS[@]}" ]; then
    echo "Array task $TASK_ID out of range (0-$(( ${#MODELS[@]} - 1 )))"
    exit 1
fi
model="${MODELS[$TASK_ID]}"

module load gcc arrow/24.0.0 python/3.12.4
cd /scratch/bmalh018/rag-safety/
source "$HOME/my_env/bin/activate"

export HOME="${HOME:-/home/bmalh018}"
export HF_HOME=/scratch/bmalh018/.cache/huggingface
mkdir -p "$HF_HOME" logs "$NEW_ROOT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_SDPA_ENABLED=0
export TORCH_DIA_SDPA_ENABLED=0
export TORCH_SDPA_ALLOW_INTRA_TEAM_SHARING=0

echo "=== Qwen3 refusal filter (rag_llm) ==="
echo "Task:     ${TASK_ID} -> ${model}"
echo "Source:   $SRC_ROOT"
echo "New root: $NEW_ROOT"
echo "Job:      ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-n/a})"
echo "Started:  $(date)"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python3 - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available on this allocation. Resubmit the array task.", file=sys.stderr)
    sys.exit(1)
PY
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== nvidia-smi (expect one free GPU) ==="
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
    USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [ -n "${USED_MIB}" ] && [ "${USED_MIB}" -gt 2048 ]; then
        echo "ERROR: GPU already has ${USED_MIB} MiB in use before this job started."
        nvidia-smi
        exit 1
    fi
fi

SRC_MODEL_DIR="${SRC_ROOT}/${model}"
if [ ! -d "$SRC_MODEL_DIR" ]; then
    echo "ERROR: missing ${SRC_MODEL_DIR}"
    exit 1
fi
SRC_RUN="$(ls -td "${SRC_MODEL_DIR}"/*/ 2>/dev/null | head -n 1 || true)"
SRC_RUN="${SRC_RUN%/}"
if [ -z "$SRC_RUN" ] || [ ! -f "${SRC_RUN}/responses.jsonl" ]; then
    echo "ERROR: no responses.jsonl under ${SRC_MODEL_DIR}"
    exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST_RUN="${NEW_ROOT}/${model}/${STAMP}"
mkdir -p "$DEST_RUN"

echo "=== [${model}] ${SRC_RUN} -> ${DEST_RUN} ==="
python3 scripts/filter_refusals_qwen.py \
    --src-run-dir "$SRC_RUN" \
    --out-run-dir "$DEST_RUN" \
    --condition rag_llm \
    --judge-model-id "Qwen/Qwen3-30B-A3B-Instruct-2507"

echo "=== Done ${model}: ${DEST_RUN} ==="
echo "Finished: $(date)"
