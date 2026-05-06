#!/usr/bin/env bash
set -euo pipefail

SAFETY_CONFIG="${1:-configs/week1_safety.yaml}"
QA_CONFIG="${2:-configs/week1_capability.yaml}"

python scripts/validate_run_config.py --config "$SAFETY_CONFIG"
python scripts/validate_run_config.py --config "$QA_CONFIG"

python scripts/run_qa_eval.py --config "$QA_CONFIG"
QA_RUN_DIR="$(ls -td outputs/qa_week1/* | head -n 1)"

python scripts/run_eval.py --config "$SAFETY_CONFIG"
python scripts/run_doc_judge.py --config "$SAFETY_CONFIG"
python scripts/run_judge.py --config "$SAFETY_CONFIG"
SAFETY_OUTPUT_ROOT="$(python -c "import yaml,sys;print((yaml.safe_load(open(sys.argv[1],'r',encoding='utf-8')) or {}).get('output',{}).get('root','outputs/week1_safety'))" "$SAFETY_CONFIG")"
SAFETY_RUN_DIR="$(ls -td "$SAFETY_OUTPUT_ROOT"/* | head -n 1)"
python scripts/run_judge_second_pass.py --config "$SAFETY_CONFIG" --run-dir "$SAFETY_RUN_DIR"
python scripts/run_actionability.py --run-dir "$SAFETY_RUN_DIR" --config "$SAFETY_CONFIG"
python scripts/run_present_absent.py --run-dir "$SAFETY_RUN_DIR"
python scripts/summarize.py --config "$SAFETY_CONFIG" --run-dir "$SAFETY_RUN_DIR" --present-absent "$SAFETY_RUN_DIR/present_absent.csv"

python scripts/build_report_tables.py \
  --qa-run-dir "$QA_RUN_DIR" \
  --safety-run-dir "$SAFETY_RUN_DIR"

echo "QA run dir: $QA_RUN_DIR"
echo "Safety run dir: $SAFETY_RUN_DIR"
