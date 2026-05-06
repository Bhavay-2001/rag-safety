#!/bin/bash
# Slurm job: full pipeline with Qwen3-4B and Llama-Guard-3-8B.
# Assumes repo path ~/safe-rag3 and venv path ~/safe-rag3/venv.

#SBATCH --job-name=rag-full-3b
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --account=def-kfraser
#SBATCH --output=~/scratch/rag-test/logs/%x-%j.out

set -euo pipefail

export HF_HOME=~/scratch/hf_cache

cd ~/safe-rag3
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo "No virtual environment found at venv/ or .venv/"
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load gcc arrow >/dev/null 2>&1 || true
fi

echo "STEP build_index $(date)"
python scripts/validate_run_config.py --config configs/full_3b.yaml
python scripts/build_index.py --config configs/full_3b.yaml --max-docs 20000

echo "STEP run_eval $(date)"
python scripts/run_eval.py --config configs/full_3b.yaml

echo "STEP run_doc_judge $(date)"
python scripts/run_doc_judge.py --config configs/full_3b.yaml

echo "STEP run_judge $(date)"
python scripts/run_judge.py --config configs/full_3b.yaml

LATEST_RUN="$(ls -td outputs/full_3b/* outputs/runs/* 2>/dev/null | head -n 1)"
echo "STEP second_pass_judge $(date)"
python scripts/run_judge_second_pass.py --config configs/full_3b.yaml --run-dir "$LATEST_RUN"
echo "STEP actionability $(date)"
python scripts/run_actionability.py --run-dir "$LATEST_RUN" --config configs/full_3b.yaml
echo "STEP present_absent $(date)"
python scripts/run_present_absent.py --run-dir "$LATEST_RUN"

echo "STEP summarize $(date)"
python scripts/summarize.py --config configs/full_3b.yaml --run-dir "$LATEST_RUN" --present-absent "$LATEST_RUN/present_absent.csv"

echo "DONE $(date)"
