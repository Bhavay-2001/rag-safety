#!/bin/bash
# Week-1 orchestration: capability-first QA run + broad safety baseline.

#SBATCH --job-name=rag-week1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=h100:1
#SBATCH --account=def-kfraser
#SBATCH --output=~/scratch/rag-test/logs/%x-%j.out

set -euo pipefail

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

export HF_HOME=~/scratch/hf_cache

echo "STEP validate $(date)"
python scripts/validate_run_config.py --config configs/week1_capability.yaml
python scripts/validate_run_config.py --config configs/week1_safety.yaml

echo "STEP QA capability $(date)"
python scripts/run_qa_eval.py --config configs/week1_capability.yaml

echo "STEP Safety eval $(date)"
python scripts/run_eval.py --config configs/week1_safety.yaml

echo "STEP Doc judge $(date)"
python scripts/run_doc_judge.py --config configs/week1_safety.yaml

echo "STEP Response judge $(date)"
python scripts/run_judge.py --config configs/week1_safety.yaml

SAFETY_OUTPUT_ROOT="$(python -c "import yaml;print((yaml.safe_load(open('configs/week1_safety.yaml','r',encoding='utf-8')) or {}).get('output',{}).get('root','outputs/week1_safety'))")"
LATEST_SAFETY_RUN="$(ls -td "$SAFETY_OUTPUT_ROOT"/* | head -n 1)"
echo "STEP Second pass judge $(date)"
python scripts/run_judge_second_pass.py --config configs/week1_safety.yaml --run-dir "$LATEST_SAFETY_RUN"
echo "STEP Actionability $(date)"
python scripts/run_actionability.py --run-dir "$LATEST_SAFETY_RUN" --config configs/week1_safety.yaml
echo "STEP PresentAbsent $(date)"
python scripts/run_present_absent.py --run-dir "$LATEST_SAFETY_RUN"

echo "STEP Summarize $(date)"
python scripts/summarize.py --config configs/week1_safety.yaml --run-dir "$LATEST_SAFETY_RUN" --present-absent "$LATEST_SAFETY_RUN/present_absent.csv"

LATEST_QA_RUN="$(ls -td outputs/qa_week1/* | head -n 1)"

echo "STEP Report tables $(date)"
python scripts/build_report_tables.py \
  --qa-run-dir "$LATEST_QA_RUN" \
  --safety-run-dir "$LATEST_SAFETY_RUN"

echo "DONE $(date)"
echo "QA run: $LATEST_QA_RUN"
echo "Safety run: $LATEST_SAFETY_RUN"
