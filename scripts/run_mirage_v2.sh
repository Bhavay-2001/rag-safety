#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/qa_smoke_mirage_v2.yaml}"
SPLIT="${2:-validation}"
MAX_ROWS="${3:-500}"
OUTPUT_DIR="${4:-data/qa}"

python -X utf8 scripts/fetch_mirage_v2.py --split "$SPLIT" --max-rows "$MAX_ROWS" --output-dir "$OUTPUT_DIR"
python scripts/validate_run_config.py --config "$CONFIG_PATH"
python scripts/qa_preflight_v2.py --config "$CONFIG_PATH"
python -X utf8 scripts/run_qa_eval_v2.py --config "$CONFIG_PATH"
