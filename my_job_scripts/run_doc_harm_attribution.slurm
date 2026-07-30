#!/bin/bash
#SBATCH --job-name=rag-docattr
#SBATCH --account=def-kfraser
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --exclusive
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Document harm attribution probe (same model re-generates; Llama-Guard labels).
# For each of 5 cases (harmful query + top-5 benign docs + known harmful response):
#   all_docs, non_rag, single:i (always Context 1), loo:i, ctx1:i (doc i -> Context 1)
# Adaptive multi-doc search when no single doc is sufficient:
#   combo pairs, then triples; summary flags combined_docs_causing_harm.
# NOTE: LOO is NOT position-controlled (remaining docs shift ranks). Use ctx1:/single: for that.
# Judge raw text is stored with newlines stripped.
#
# Array tasks: 0=llama31_8b, 1=phi4 (Phi-4-mini)
#
# Before submitting:
#   1. Fill placeholders in the per-model cases files (exactly 5 cases, 5 docs each):
#        my_job_scripts/doc_harm_cases_llama31_8b.json
#        my_job_scripts/doc_harm_cases_phi4.json
#   2. sbatch my_job_scripts/run_doc_harm_attribution.slurm
#   3. Optional override:
#        sbatch my_job_scripts/run_doc_harm_attribution.slurm outputs/doc_harm_attr
#
# Args:
#   $1 = output root (default: outputs/doc_harm_attr)
# Each array task loads its own cases file for that model.

set -euo pipefail

OUT_ROOT="${1:-outputs/doc_harm_attr}"

MODELS=(llama31_8b phi4)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "${#MODELS[@]}" ]; then
    echo "Array task $TASK_ID out of range (0-$(( ${#MODELS[@]} - 1 )))"
    exit 1
fi
model="${MODELS[$TASK_ID]}"
CASES="my_job_scripts/doc_harm_cases_${model}.json"

module load gcc arrow/24.0.0 python/3.12.4
cd /scratch/bmalh018/rag-safety/
source "$HOME/my_env/bin/activate"

export HOME="${HOME:-/home/bmalh018}"
export HF_HOME=/scratch/bmalh018/.cache/huggingface
mkdir -p "$HF_HOME" logs "$OUT_ROOT"

if [ -f "$HOME/.hf_token" ]; then
    export HUGGING_FACE_HUB_TOKEN
    HUGGING_FACE_HUB_TOKEN="$(cat "$HOME/.hf_token")"
    export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_SDPA_ENABLED=0
export TORCH_DIA_SDPA_ENABLED=0
export TORCH_SDPA_ALLOW_INTRA_TEAM_SHARING=0

echo "=== Doc harm attribution ==="
echo "Task:     ${TASK_ID} -> ${model}"
echo "Cases:    $CASES"
echo "Out root: $OUT_ROOT"
echo "Job:      ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-n/a})"
echo "Started:  $(date)"

if [ ! -f "$CASES" ]; then
    echo "ERROR: cases file not found: $CASES"
    exit 1
fi

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

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${OUT_ROOT}/${model}/${STAMP}"
mkdir -p "$DEST"
cp -f "$CASES" "$DEST/cases.json"

echo "=== [${model}] -> ${DEST} ==="
python3 my_job_scripts/doc_harm_attribution.py \
    --cases "$CASES" \
    --model "$model" \
    --out-dir "$DEST" \
    --max-new-tokens 256 \
    --judge-model-id "meta-llama/Llama-Guard-3-8B"

echo "=== Done ${model}: ${DEST} ==="
echo "Finished: $(date)"
