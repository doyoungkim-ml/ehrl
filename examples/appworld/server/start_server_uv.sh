#!/bin/bash
set -euo pipefail
# Clear ROCm/HIP visibility variables inherited from login shells or Ray.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLLM_DIR=${RLLM_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
PORT=${1:-9000}
DATA_DIR=${2:-${APPWORLD_DATA_DIR:-${RLLM_DIR}/data/appworld}}
SERVER_VENV=${3:-${APPWORLD_SERVER_VENV:-${RLLM_DIR}/.venv_appworld_server}}
RUN_ROOT=${4:-${APPWORLD_RUN_ROOT:-$DATA_DIR}}
SERVER_PYTHON="${APPWORLD_SERVER_PYTHON:-${SERVER_VENV}/bin/python}"

if [[ ! -x "$SERVER_PYTHON" ]]; then
    echo "ERROR: AppWorld server python not found: $SERVER_PYTHON"
    exit 1
fi

if [[ ! -d "$DATA_DIR/data" ]]; then
    echo "ERROR: AppWorld data directory not found: $DATA_DIR/data"
    exit 1
fi

if [[ "$RUN_ROOT" != "$DATA_DIR" ]]; then
    mkdir -p "$RUN_ROOT" "$RUN_ROOT/experiments" "$RUN_ROOT/experiments/outputs" "$RUN_ROOT/.tmp"

    if [[ ! -e "$RUN_ROOT/data" && ! -L "$RUN_ROOT/data" ]]; then
        ln -s "$DATA_DIR/data" "$RUN_ROOT/data"
    fi

    if [[ -d "$DATA_DIR/experiments/configs" && ! -e "$RUN_ROOT/experiments/configs" && ! -L "$RUN_ROOT/experiments/configs" ]]; then
        ln -s "$DATA_DIR/experiments/configs" "$RUN_ROOT/experiments/configs"
    fi
fi

APPWORLD_DATA_DIR="$DATA_DIR" APPWORLD_RUN_ROOT="$RUN_ROOT" exec "$SERVER_PYTHON" - <<PY
import os
from appworld import update_root
from appworld.serve.environment import run

run_root = os.environ['APPWORLD_RUN_ROOT']
print(f"AppWorld server root: {run_root}", flush=True)
update_root(run_root)
run(port=${PORT})
PY
