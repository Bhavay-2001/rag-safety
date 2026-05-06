#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/tiny.yaml}"
MAX_DOCS="${2:-2000}"

python scripts/build_corpus.py --config "$CONFIG_PATH" --max-docs "$MAX_DOCS"
python scripts/build_index.py --config "$CONFIG_PATH" --max-docs "$MAX_DOCS"
python scripts/run_eval.py --config "$CONFIG_PATH"
python scripts/run_doc_judge.py --config "$CONFIG_PATH"
python scripts/run_judge.py --config "$CONFIG_PATH"
RUN_DIR="$(ls -td outputs/runs/* | head -n 1)"
python scripts/run_judge_second_pass.py --config "$CONFIG_PATH" --run-dir "$RUN_DIR"
python scripts/run_actionability.py --run-dir "$RUN_DIR" --config "$CONFIG_PATH"
python scripts/run_present_absent.py --run-dir "$RUN_DIR"
python scripts/summarize.py --config "$CONFIG_PATH" --run-dir "$RUN_DIR" --present-absent "$RUN_DIR/present_absent.csv"
