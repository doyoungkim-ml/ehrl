#!/usr/bin/env bash
# Sourced by the AppWorld launchers. Select the server environment and data,
# creating them once (under a lock) when they are missing unless
# RLLM_AUTO_SETUP=0. Explicit
# APPWORLD_DATA_DIR / APPWORLD_SERVER_VENV exports win.
if [[ -z ${APPWORLD_DATA_DIR:-} ]]; then
    for appworld_data in "$RLLM_DIR/data/appworld" "$(dirname "$RLLM_DIR")/appworld_data" "/scratch/${USER:-$(id -un)}/appworld_data"; do
        if [[ -d "$appworld_data/data" ]]; then
            APPWORLD_DATA_DIR=$appworld_data
            break
        fi
    done
fi
export APPWORLD_DATA_DIR=${APPWORLD_DATA_DIR:-$RLLM_DIR/data/appworld}
export APPWORLD_SERVER_VENV=${APPWORLD_SERVER_VENV:-$RLLM_DIR/.venv_appworld_server}

appworld_runtime_ready() {
    [[ -x "$APPWORLD_SERVER_VENV/bin/python" && -d "$APPWORLD_DATA_DIR/data" && ! -e "$APPWORLD_SERVER_VENV.setup-incomplete" ]]
}

if [[ ${RLLM_DRY_RUN:-0} != 1 && ${RLLM_AUTO_SETUP:-1} == 1 && " $* " != *" --dry-run "* ]] && ! appworld_runtime_ready; then
    mkdir -p "$RLLM_DIR/.cache/locks"
    exec {appworld_lock_fd}>"$RLLM_DIR/.cache/locks/appworld.lock"
    echo "[setup] waiting for the AppWorld setup lock" >&2
    flock "$appworld_lock_fd"
    if ! appworld_runtime_ready; then
        echo "[setup] preparing AppWorld in $APPWORLD_SERVER_VENV and $APPWORLD_DATA_DIR; this runs once" >&2
        : > "$APPWORLD_SERVER_VENV.setup-incomplete"
        bash "$RLLM_DIR/examples/appworld/server/setup_server_env_uv.sh" \
            --venv-dir "$APPWORLD_SERVER_VENV" --data-dir "$APPWORLD_DATA_DIR" >&2
        rm -f "$APPWORLD_SERVER_VENV.setup-incomplete"
    fi
    exec {appworld_lock_fd}>&-
fi
