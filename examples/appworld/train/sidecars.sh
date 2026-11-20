#!/usr/bin/env bash
# AppWorld owns one mutable task world per server. The shared runtime owns Ray/tmp.
# Called after experiment settings and scripts/training_runtime.sh are loaded.
export APPWORLD_DATA_DIR=${APPWORLD_DATA_DIR:-$RLLM_DIR/data/appworld}
export APPWORLD_SERVER_VENV=${APPWORLD_SERVER_VENV:-$RLLM_DIR/.venv_appworld_server}
export APPWORLD_PYTHON=${APPWORLD_PYTHON:-$APPWORLD_SERVER_VENV/bin/python}
export APPWORLD_SERVER_PYTHON=$APPWORLD_PYTHON
export APPWORLD_TASK_IDS_DIR=${APPWORLD_TASK_IDS_DIR:-$RLLM_JOB_DIR/task_ids}
export APPWORLD_RUN_ROOT=${APPWORLD_RUN_ROOT:-$RLLM_JOB_TMP/appworld}
export APPWORLD_SERVER_LOG_DIR=${APPWORLD_SERVER_LOG_DIR:-$RLLM_LOG_DIR/appworld-$RLLM_JOB_ID}
: "${APPWORLD_SERVER_START_TIMEOUT:=180}"
[[ -x "$APPWORLD_PYTHON" ]] || { echo "Missing AppWorld server Python: $APPWORLD_PYTHON" >&2; exit 1; }
[[ -d "$APPWORLD_DATA_DIR/data" ]] || { echo "Missing AppWorld data: $APPWORLD_DATA_DIR/data" >&2; exit 1; }

APPWORLD_PIDS=()
appworld_cleanup() {
    local status=$?
    trap - EXIT
    set +e
    local pid attempt alive
    for pid in "${APPWORLD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # A server can catch SIGTERM and never exit; bound the wait so the job ends.
    for (( attempt=0; attempt<${APPWORLD_SERVER_STOP_TIMEOUT:-30}*10; attempt++ )); do
        alive=0
        for pid in "${APPWORLD_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        (( alive )) || break
        sleep 0.1
    done
    for pid in "${APPWORLD_PIDS[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    # Pass the original exit status to the shared cleanup, which preserves Ray logs.
    rllm_training_cleanup "$status"
}
trap appworld_cleanup EXIT

# Probe a job-specific block while holding every socket in the candidate block.
BASE_PORT=$("$PYTHON_BIN" - "${BASE_PORT:-}" "$N_SERVERS" "$RLLM_JOB_ID" <<'PYPORT'
import os
import socket
import sys

requested, count, job = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if not 1 <= count <= 40000:
    raise SystemExit("N_SERVERS must be between 1 and 40000")
blocks = 40000 // count
start = int(job) if job.isdecimal() else os.getpid()
candidates = [int(requested)] if requested else (20000 + ((start + i) % blocks) * count for i in range(blocks))
for base in candidates:
    if not 1024 <= base <= 65536 - count:
        raise SystemExit("Invalid BASE_PORT / N_SERVERS range")
    sockets = []
    try:
        for port in range(base, base + count):
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("0.0.0.0", port))
    except OSError:
        continue
    else:
        print(base)
        break
    finally:
        for sock in sockets:
            sock.close()
else:
    raise SystemExit("No free AppWorld port block; check BASE_PORT and other jobs")
PYPORT
)
export BASE_PORT
export APPWORLD_URL=${APPWORLD_URL:-http://localhost:$BASE_PORT}
mkdir -p "$APPWORLD_RUN_ROOT" "$APPWORLD_SERVER_LOG_DIR" "$APPWORLD_TASK_IDS_DIR"
for (( i=0; i<N_SERVERS; i++ )); do
    port=$(( BASE_PORT + i ))
    bash "$RLLM_DIR/examples/appworld/server/start_server_uv.sh" "$port" "$APPWORLD_DATA_DIR" \
        "$APPWORLD_SERVER_VENV" "$APPWORLD_RUN_ROOT/server_${i}_port${port}" \
        >"$APPWORLD_SERVER_LOG_DIR/port${port}.log" 2>&1 &
    APPWORLD_PIDS+=("$!")
done
"$APPWORLD_PYTHON" - "$BASE_PORT" "$N_SERVERS" "$APPWORLD_SERVER_START_TIMEOUT" "${APPWORLD_PIDS[@]}" <<'PYREADY'
import os
import sys
import time
import urllib.request

base, count, timeout = map(int, sys.argv[1:4])
pids = list(map(int, sys.argv[4:]))
deadline = time.monotonic() + timeout
ready = set()
while time.monotonic() < deadline:
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise SystemExit(f"AppWorld server {pid} exited; inspect APPWORLD_SERVER_LOG_DIR")
    for port in range(base, base + count):
        if port in ready:
            continue
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2):
                ready.add(port)
        except OSError:
            pass
    if len(ready) == count:
        print(f"[appworld] {count} servers ready on ports {base}..{base + count - 1}")
        break
    time.sleep(2)
else:
    raise SystemExit(f"Only {len(ready)}/{count} AppWorld servers ready; inspect APPWORLD_SERVER_LOG_DIR")
PYREADY
for pid in "${APPWORLD_PIDS[@]}"; do
    kill -0 "$pid" || { echo "AppWorld server exited during readiness checks" >&2; exit 1; }
done
if [[ ! -f "$APPWORLD_TASK_IDS_DIR/appworld_${APPWORLD_TRAIN_SPLIT}_task_ids.json" || ! -f "$APPWORLD_TASK_IDS_DIR/appworld_${APPWORLD_TEST_SPLIT}_task_ids.json" ]]; then
    APPWORLD_ROOT="$APPWORLD_DATA_DIR" "$APPWORLD_PYTHON" "$RLLM_DIR/examples/appworld/export_task_ids.py"
fi
PREPARE_ARGS=(--data-dir "$APPWORLD_TASK_IDS_DIR" --train-split "$APPWORLD_TRAIN_SPLIT" --test-split "$APPWORLD_TEST_SPLIT" --max-turns "$APPWORLD_MAX_TURNS")
if [[ "$TRAINING_METHOD" == ace_rl ]]; then
    [[ -z "${MAX_TRAIN_TASKS:-}" ]] || PREPARE_ARGS+=(--train-size "$MAX_TRAIN_TASKS")
    [[ -z "${MAX_TEST_TASKS:-}" ]] || PREPARE_ARGS+=(--test-size "$MAX_TEST_TASKS")
fi
"$PYTHON_BIN" "$RLLM_DIR/examples/appworld/prepare_appworld_data.py" "${PREPARE_ARGS[@]}"
