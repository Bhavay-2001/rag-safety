#!/bin/bash
#SBATCH --job-name=rag-pipeline
#SBATCH --account=def-kfraser_cpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# RAGChecker evaluation pipeline.
#
# Usage:
#   sbatch --job-name=ragchecker-<run_name> run_pipeline.slurm \
#       <run_name> <run_dir> [ragchecker_model_name]
#
# Model name (arg 3 or configs/ragchecker.yaml): alias from configs/models.yaml
#   (e.g. Llama-3-8B-Instruct) or full hub id (e.g. hf/Qwen/Qwen3-30B-A3B-Instruct-2507).
#
# Outputs go to:
#   outputs/ragchecker/<run_name>/

set -euo pipefail

RUN_NAME="${1:-}"
RUN_DIR="${2:-}"
RAGCHECKER_MODEL="${3:-}"

if [ -z "$RUN_NAME" ] || [ -z "$RUN_DIR" ]; then
    echo "ERROR: run_name and run_dir are required."
    echo "Usage: sbatch [--job-name=ragchecker-<name>] run_pipeline.slurm <run_name> <run_dir> [ragchecker_model_name]"
    exit 1
fi

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

RAGCHECKER_OUTPUT_DIR="outputs/ragchecker/${RUN_NAME}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "=== Environment ready ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
echo "HF_HOME       : $HF_HOME"
echo "Job ID        : $SLURM_JOB_ID"
echo "Run name      : $RUN_NAME"
echo "Run dir       : $RUN_DIR"
echo "RAG model     : ${RAGCHECKER_MODEL:-<from configs/ragchecker.yaml>}"
echo "RAG output dir: $RAGCHECKER_OUTPUT_DIR"
echo "Started       : $(date)"
mkdir -p logs "$RAGCHECKER_OUTPUT_DIR"

# =============================================================================
# SECTION A — GT Response Generation + WildGuard Judge (optional)
# =============================================================================
# echo "=== [A] GT response generation + judge ==="
# python3 scripts/generate_gt_responses.py --config configs/gt_responses.yaml

# =============================================================================
# SECTION B — RAGChecker Evaluation (vLLM + local HF cache)
# =============================================================================
echo "=== [B] RAGChecker evaluation ==="

python3 -c "import ragchecker" 2>/dev/null || {
    echo "Installing ragchecker..."
    pip install ragchecker --quiet
    python3 -m spacy download en_core_web_sm --quiet
}

python3 -c "import vllm" 2>/dev/null || {
    echo "Installing vllm..."
    pip install vllm --quiet
}

MODEL_ALIAS="${RAGCHECKER_MODEL:-Llama-3.1-8B-Instruct}"
VLLM_HUB_ID=$(python3 -c "
import sys
sys.path.insert(0, '.')
from src.model_names import resolve_model_spec
print(resolve_model_spec('${MODEL_ALIAS}', 'configs/models.yaml').id)
")

VLLM_PORT=8000
VLLM_API_KEY=sk-local
VLLM_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
VLLM_LOG="logs/vllm_${SLURM_JOB_ID}.log"

echo "Starting vLLM: ${VLLM_HUB_ID} on port ${VLLM_PORT}"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_HUB_ID" \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --port "$VLLM_PORT" \
    --api-key "$VLLM_API_KEY" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (pid $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for vLLM..."
for _ in $(seq 1 90); do
    if curl -sf "${VLLM_API_BASE}/models" -H "Authorization: Bearer ${VLLM_API_KEY}" >/dev/null; then
        echo "vLLM ready."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited early. Log:"
        tail -50 "$VLLM_LOG"
        exit 1
    fi
    sleep 10
done

export OPENAI_API_KEY="$VLLM_API_KEY"
export VLLM_API_BASE="$VLLM_API_BASE"
export RAGCHECKER_EXTRACTOR_API_BASE="$VLLM_API_BASE"
export RAGCHECKER_CHECKER_API_BASE="$VLLM_API_BASE"

RAG_ARGS=(
    --run-name "$RUN_NAME"
    --output-dir "$RAGCHECKER_OUTPUT_DIR"
)
if [ -n "$RAGCHECKER_MODEL" ]; then
    RAG_ARGS+=(--extractor-name "$RAGCHECKER_MODEL" --checker-name "$RAGCHECKER_MODEL")
fi

python3 scripts/run_ragchecker.py \
    --config configs/ragchecker.yaml \
    --run-dir "$RUN_DIR" \
    "${RAG_ARGS[@]}"

echo "=== Done: $(date) ==="
echo "Results in: $RAGCHECKER_OUTPUT_DIR"
