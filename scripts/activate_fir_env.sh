#!/usr/bin/env bash
set -euo pipefail
module --force purge
module load StdEnv/2023 gcc python arrow/23.0.1
source ~/scratch/venvs/safe-rag3/bin/activate
export HF_HOME=~/scratch/hf_cache
echo "FIR env ready"
python - <<'PY'
import pyarrow, datasets
print("pyarrow", pyarrow.__version__, "| datasets", datasets.__version__)
PY
