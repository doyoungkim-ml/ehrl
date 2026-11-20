#!/usr/bin/env bash
# Benchmark runtime; method presets own evaluation settings and arguments.
source "$RLLM_DIR/scripts/evaluation_server.sh"

launch() {
    [[ $# == 0 ]] || { echo "Set options through environment variables, or use evaluate.py directly" >&2; return 2; }
    cd "$RLLM_DIR"
    MODEL_SOURCE=${MODEL_SOURCE:-hub}
    case "$MODEL_SOURCE" in
        auto|hub|local|checkpoint) ;;
        *) echo "Mastermind requires a local OpenAI-compatible completions server; use MODEL_PATH for weights" >&2; return 2 ;;
    esac
    [[ "$TASK_OFFSET" =~ ^[0-9]+$ ]] || { echo "TASK_OFFSET must be nonnegative" >&2; return 2; }
    [[ -z "$TASK_LIMIT" || "$TASK_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "TASK_LIMIT must be positive or empty" >&2; return 2; }
    eval_setup_python
    if [[ ${RLLM_DRY_RUN:-0} != 1 ]]; then
        eval_check_resume_dir
        mkdir -p "$RUN_DIR"
        eval_lock_run
        eval_check_resume_dir
    fi
    eval_merge_checkpoint
    REQUESTED_WANDB_RUN_ID=${WANDB_RUN_ID:-}
    eval_prepare_tracking
    eval_select_server_url
    COMMAND=("$PYTHON_BIN" -m "$EVAL_MODULE" "${BASE_URL_FLAG:---base-url}" "$BASE_URL" "${EVAL_ARGS[@]}" "$@")
    [[ ${RESUME:-1} == 1 ]] || COMMAND+=(--no-resume)
    if [[ ${RLLM_DRY_RUN:-0} != 1 ]]; then
        eval_record_config
        if [[ -f "$RUN_DIR/evaluation_complete.json" ]]; then
            eval_complete_run || return $?
            eval_unlock_run
            return 0
        fi
    fi
    eval_start_server
    COMMAND=("$PYTHON_BIN" -m "$EVAL_MODULE" "${BASE_URL_FLAG:---base-url}" "$BASE_URL" "${EVAL_ARGS[@]}" "$@")
    [[ ${RESUME:-1} == 1 ]] || COMMAND+=(--no-resume)
    if [[ ${RLLM_DRY_RUN:-0} == 1 ]]; then
        eval_print_command "${COMMAND[@]}"
        return
    fi
    "${COMMAND[@]}" || return $?
    eval_complete_run || return $?
    eval_stop_server
    eval_unlock_run
}
