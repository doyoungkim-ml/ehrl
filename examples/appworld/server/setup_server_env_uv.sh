#!/bin/bash
set -euo pipefail
# AppWorld server environment and data. The AppWorld launchers run this once
# when either is missing; run it directly to prepare ahead of time.
# Clear ROCm/HIP visibility variables inherited from login shells.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLLM_DIR=${RLLM_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
SERVER_VENV=${APPWORLD_SERVER_VENV:-${RLLM_DIR}/.venv_appworld_server}
DATA_DIR=${APPWORLD_DATA_DIR:-${RLLM_DIR}/data/appworld}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
APPWORLD_VERSION=${APPWORLD_VERSION:-0.1.3.post1}
RECREATE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-dir) SERVER_VENV="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        --appworld-version) APPWORLD_VERSION="$2"; shift 2 ;;
        --recreate) RECREATE=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--venv-dir DIR] [--data-dir DIR] [--python VERSION] [--appworld-version V] [--recreate]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user --quiet uv
fi

export UV_CACHE_DIR=${UV_CACHE_DIR:-${RLLM_DIR}/.cache/uv}
mkdir -p "$UV_CACHE_DIR"
UV_CMD=(uv --no-config)

if [[ "$RECREATE" -eq 1 && -d "$SERVER_VENV" ]]; then
    rm -rf "$SERVER_VENV"
fi

if [[ ! -x "$SERVER_VENV/bin/python" ]]; then
    "${UV_CMD[@]}" venv --python "$PYTHON_VERSION" "$SERVER_VENV"
fi
SERVER_PYTHON="$SERVER_VENV/bin/python"

"${UV_CMD[@]}" pip install --python "$SERVER_PYTHON" --upgrade "appworld==${APPWORLD_VERSION}"
mkdir -p "$DATA_DIR"
APPWORLD_DATA_DIR="$DATA_DIR" "$SERVER_PYTHON" - <<'PY'
import os
from appworld import update_root
from appworld.download import download_data
from appworld.install import install_package

root = os.environ['APPWORLD_DATA_DIR']
update_root(root)
install_package()
download_data()
print('AppWorld server environment ready at', root)
PY

# Harden AppWorld's request-log serialization against non-JSON agent args (e.g. a
# placeholder Ellipsis from `apis.x.y(...)`), which otherwise 500s /execute and
# stalls rollouts in a retry storm. Idempotent; safe to re-run on every setup.
"$SERVER_PYTHON" "$RLLM_DIR/examples/appworld/server/apply_requester_jsonsafe_patch.py"

# Export task ids next to the current repo example for the training scripts.
APPWORLD_ROOT="$DATA_DIR" "$SERVER_PYTHON" "$RLLM_DIR/examples/appworld/export_task_ids.py"
