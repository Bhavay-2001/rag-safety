#!/bin/bash
#SBATCH --job-name=rag-safety
#SBATCH --account=def-kfraser_cpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage (single model — default):
#   sbatch run_external_dataset.sh
#
# Usage (specific model + output dir):
#   sbatch --job-name=rag-llama31  run_external_dataset.sh custom_llama31_8b  outputs/llama31_8b
#   sbatch --job-name=rag-mistral  run_external_dataset.sh custom_mistral7b_v03 outputs/mistral7b
#   sbatch --job-name=rag-qwen3   run_external_dataset.sh custom_qwen3_4b    outputs/qwen3_4b

set -euo pipefail

# Model key and output dir — use CLI args or fall back to config defaults
MODEL_KEY="${1:-}"
OUTPUT_DIR="${2:-}"

# Build --set flags only when overrides are provided
SET_ARGS=""
if [ -n "$MODEL_KEY" ]; then
    SET_ARGS="$SET_ARGS --set models.use=${MODEL_KEY}"
fi
if [ -n "$OUTPUT_DIR" ]; then
    SET_ARGS="$SET_ARGS --set output.root=${OUTPUT_DIR}"
fi

CONFIG="configs/week1_safety_improved_norerank.yaml"

# ── Environment ───────────────────────────────────────────────────────────────
module load gcc arrow/24.0.0 python/3.12.4

cd /scratch/bmalh018/rag-safety/
source $HOME/my_env/bin/activate

export HF_HOME=/scratch/bmalh018/.cache/huggingface
mkdir -p "$HF_HOME"

export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True

which python
echo "=== Environment ready ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

echo "Working dir: $(pwd)"
echo "Config:      $CONFIG"
echo "Model key:   ${MODEL_KEY:-<from config>}"
echo "Output dir:  ${OUTPUT_DIR:-<from config>}"
echo "Job ID:      $SLURM_JOB_ID"
echo "Started:     $(date)"

mkdir -p logs

# ── Step 1: Build corpus (skip if already exists) ─────────────────────────────
CORPUS="data/corpus/wiki_dolma.csv"
if [ -f "$CORPUS" ]; then
    echo "=== [1/6] Corpus already exists, skipping ==="
else
    echo "=== [1/6] Building corpus ==="
    python3 scripts/build_corpus.py --config "$CONFIG"
fi

# ── Step 2: Build BM25 index (skip if already exists) ─────────────────────────
INDEX="indexes/bm25_wiki_dolma/bm25.pkl"
if [ -f "$INDEX" ]; then
    echo "=== [2/6] Index already exists, skipping ==="
else
    echo "=== [2/6] Building BM25 index ==="
    python3 scripts/build_index.py --config "$CONFIG"
fi

# ── Step 3: Generate responses ────────────────────────────────────────────────
echo "=== [3/6] Running generation (non_rag / rag_docs / rag_llm) ==="
export TORCH_CUDNN_SDPA_ENABLED=0
export TORCH_DIA_SDPA_ENABLED=0
export TORCH_SDPA_ALLOW_INTRA_TEAM_SHARING=0

python3 scripts/run_eval.py --config "$CONFIG" $SET_ARGS

# Capture the most recent run directory
OUT_ROOT="${OUTPUT_DIR:-outputs/week1_safety_improved}"
RUN_DIR="$(ls -td ${OUT_ROOT}/* | head -n 1)"
echo "Run dir: $RUN_DIR"

# ── Step 4: Judge retrieved documents (skip if already done) ──────────────────
if [ -f "$RUN_DIR/doc_safety.jsonl" ]; then
    echo "=== [4/6] doc_safety.jsonl already exists, skipping ==="
else
    echo "=== [4/6] Judging retrieved documents ==="
    python3 scripts/run_doc_judge.py --config "$CONFIG" --run-dir "$RUN_DIR" $SET_ARGS
fi

# ── Step 5: Judge responses ───────────────────────────────────────────────────
echo "=== [5/6] Judging responses ==="
python3 scripts/run_judge.py --config "$CONFIG" --run-dir "$RUN_DIR" $SET_ARGS

# ── Step 6: Summarize ─────────────────────────────────────────────────────────
echo "=== [6/6] Summarising metrics ==="
python3 scripts/summarize.py --config "$CONFIG" --run-dir "$RUN_DIR" $SET_ARGS

echo "=== Done: $(date) ==="
echo "Results in: $RUN_DIR"
