#!/usr/bin/env bash
# Shared model resolution, container environment, and evaluation arguments.
source "$RLLM_DIR/scripts/evaluation_server.sh"

configure_eval() {
    if [[ -n ${STEP:-} && -n ${STEPS:-} && "$STEPS" != "$STEP" ]]; then
        echo "STEP and STEPS disagree" >&2; return 2
    fi
    REQUESTED_WANDB_RUN_ID=${WANDB_RUN_ID:-}
    EVAL_TRAINING_RUN_DIR=${RUN_DIR:-}
    REQUESTED_MODEL_PATH=${MODEL_PATH:-}
    MODEL_SOURCE=${MODEL_SOURCE:-auto}
    if [[ -n "$EVAL_TRAINING_RUN_DIR" || -n ${CKPT_BASE:-} ]]; then
        STEPS=${STEPS:-${STEP:-}}
        [[ -n "$STEPS" ]] || { echo "Set STEP to select a checkpoint" >&2; return 2; }
    else
        STEPS=${STEPS:-${STEP:-0}}
    fi
    MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
    N_PARALLEL_AGENTS=${N_PARALLEL_AGENTS:-8}
    TASK_CONCURRENCY=${TASK_CONCURRENCY:-1}
    LOCAL_EVAL_PROTOCOL=${LOCAL_EVAL_PROTOCOL:-sequential_adaptation_v1}
    case "$LOCAL_EVAL_PROTOCOL" in
        sequential_adaptation_v1)
            case "${FREEZE_SOLVED:-False}" in True|true|1) echo "Sequential evaluation keeps every policy visit; FREEZE_SOLVED must be False" >&2; return 2 ;; esac
            if (( G_REF > 1 || G_CUR > 1 )); then
                echo "Sequential evaluation supports one reflection and one curator delta; branching requires LOCAL_EVAL_PROTOCOL=legacy_search_v1" >&2; return 2
            fi
            ;;
        legacy_search_v1) ;;
        *) echo "Unknown LOCAL_EVAL_PROTOCOL=$LOCAL_EVAL_PROTOCOL" >&2; return 2 ;;
    esac
    MAX_PLAYBOOK_LENGTH=${MAX_PLAYBOOK_LENGTH:-2048}
    ACE_TEMPERATURE=${ACE_TEMPERATURE:-0.7}
    SELECTION_MODE=${SELECTION_MODE:-best_rejection}
    SPLIT=${SPLIT:-validation}
    TASK_OFFSET=${TASK_OFFSET:-0}
    # No subset manifest is shipped. Empty means dataset order, then offset/limit.
    TASK_LIST=${TASK_LIST:-}
    DATA_DIR=${DATA_DIR:-}
    SEED=${SEED:-42}
    REFLECTOR_OBS_MAX_CHARS=${REFLECTOR_OBS_MAX_CHARS:-300}
    REFLECTOR_TEST_SAMPLE_N=${REFLECTOR_TEST_SAMPLE_N:-0}
    # Actions and 1500-char replies; the 300-char conversation form shows only file headers.
    REFLECTOR_SIMPLIFIED_HISTORY=${REFLECTOR_SIMPLIFIED_HISTORY:-true}
    REFLECTOR_ONLY=${REFLECTOR_ONLY:-false}
    DISABLE_THINKING=${DISABLE_THINKING:-False}
    ACCUMULATE_THINKING=${ACCUMULATE_THINKING:-False}
    PROMPTS_DIR=${PROMPTS_DIR:-$RLLM_DIR/examples/mini_swe_bench/train/ace_rl/prompts}
    BASE_URL=${BASE_URL:-http://127.0.0.1:30000/v1}
    [[ "$TASK_OFFSET" =~ ^[0-9]+$ ]] || { echo "TASK_OFFSET must be nonnegative" >&2; return 2; }
    [[ -z "$MAX_TASKS" || "$MAX_TASKS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_TASKS must be positive or empty" >&2; return 2; }
    [[ "$TASK_CONCURRENCY" =~ ^[1-9][0-9]*$ && "$N_PARALLEL_AGENTS" =~ ^[1-9][0-9]*$ ]] || { echo "TASK_CONCURRENCY and N_PARALLEL_AGENTS must be positive" >&2; return 2; }
    if (( TASK_CONCURRENCY * ${G_ACT:-1} > N_PARALLEL_AGENTS )); then
        echo "TASK_CONCURRENCY * G_ACT must not exceed N_PARALLEL_AGENTS" >&2; return 2
    fi
    for STEP in $STEPS; do
        [[ "$STEP" =~ ^[0-9]+$ ]] || { echo "STEPS must contain nonnegative integers" >&2; return 2; }
    done
}

start_container_runtime() {
    [[ ${RLLM_DRY_RUN:-0} != 1 ]] || return 0
    if [[ -n ${MINISW_PREFLIGHT_MANIFEST:-} ]]; then
        [[ -n ${MINISW_SCRATCH_ROOT:-} ]] || { echo "Runtime preflight requires explicit MINISW_SCRATCH_ROOT" >&2; return 2; }
        # Slurm/site startup may replace TMPDIR. MiniSWEnv uses tempfile, so
        # select the registered scratch before starting any Python child.
        export TMPDIR=$MINISW_SCRATCH_ROOT
    fi
    command -v apptainer >/dev/null || command -v singularity >/dev/null || { echo "Apptainer or Singularity is required" >&2; return 1; }
    "$PYTHON_BIN" -c 'from swegym.harness.grading import get_eval_report; from swebench.harness.grading import get_eval_report'
    local cache_root=${XDG_CACHE_HOME:-$HOME/.cache}/rllm
    export APPTAINER_SIF_DIR=${APPTAINER_SIF_DIR:-$cache_root/sif}
    export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$cache_root/apptainer}
    # Clean only directories created by this invocation, never supplied paths.
    trap eval_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    local temporary_parent=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
    if [[ -z ${APPTAINER_TMPDIR:-} ]]; then
        APPTAINER_TMPDIR=$(mktemp -d "$temporary_parent/rllm-apptainer-${SLURM_JOB_ID:-$$}-XXXXXX")
        EVAL_OWNED_TEMP_DIRS+=("$APPTAINER_TMPDIR")
    fi
    if [[ -z ${MINISW_SCRATCH_ROOT:-} ]]; then
        MINISW_SCRATCH_ROOT=$(mktemp -d "$temporary_parent/rllm-miniswe-${SLURM_JOB_ID:-$$}-XXXXXX")
        EVAL_OWNED_TEMP_DIRS+=("$MINISW_SCRATCH_ROOT")
    fi
    export APPTAINER_TMPDIR MINISW_SCRATCH_ROOT
    mkdir -p "$APPTAINER_SIF_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$MINISW_SCRATCH_ROOT"
    if [[ -n ${MINISW_PREFLIGHT_MANIFEST:-} ]]; then
        "$PYTHON_BIN" -m examples.mini_swe_bench.eval.helpers.verify_apptainer_sif \
            --runtime-preflight "$MINISW_PREFLIGHT_MANIFEST" \
            --preflight-receipt "${MINISW_PREFLIGHT_RECEIPT:-${RUN_DIR%/}.runtime-preflight.json}" || return $?
    fi
}

build_eval_command() {
    COMMAND=("$PYTHON_BIN" -m examples.mini_swe_bench.eval.ace.evaluate
        --model_name "$MODEL_NAME" --base_url "$BASE_URL" --seed "$SEED" --wandb_step "$STEP"
        --local_eval_protocol "$LOCAL_EVAL_PROTOCOL" --local_eval_epochs "$LOCAL_EVAL_EPOCHS" --epochs 1 --train_batch_size 1
        --split "$SPLIT" --n_parallel_agents "$N_PARALLEL_AGENTS"
        --max_turns "$MAX_TURNS" --max_prompt_length "$MAX_PROMPT_LENGTH"
        --max_response_length "$MAX_RESPONSE_LENGTH" --max_model_length "$MAX_MODEL_LEN"
        --g_act "$G_ACT" --g_ref "$G_REF" --g_cur "$G_CUR"
        --selection_mode "$SELECTION_MODE" --ace_temperature "$ACE_TEMPERATURE"
        --max_playbook_length "$MAX_PLAYBOOK_LENGTH"
        --reflector_obs_max_chars "$REFLECTOR_OBS_MAX_CHARS"
        --reflector_test_sample_n "$REFLECTOR_TEST_SAMPLE_N"
        --prompts_dir "$PROMPTS_DIR" --base_system_prompt_filename "$BASE_SYSTEM_PROMPT_FILENAME"
        --playbook_snapshot_dir "$RUN_DIR" --ace_log_dir "$RUN_DIR/ace_logs"
        --metrics_file "$RUN_DIR/metrics_step${STEP}.json" --task_offset "$TASK_OFFSET")
    # Serial runs keep the historical command so recorded configurations still match.
    [[ "$TASK_CONCURRENCY" == 1 ]] || COMMAND+=(--task_concurrency "$TASK_CONCURRENCY")
    [[ ${HANDOFF_PROTOCOL:-legacy} == legacy ]] || COMMAND+=(--handoff_protocol "$HANDOFF_PROTOCOL")
    if [[ ${ADAPTATION_MODE:-apply} != apply || ${PROVIDER_SEED_SCHEME:-legacy} != legacy ]]; then
        COMMAND+=(--adaptation_mode "${ADAPTATION_MODE:-apply}")
        if [[ ${PROVIDER_SEED_SCHEME:-legacy} != legacy ]]; then
            COMMAND+=(--provider_seed_scheme "$PROVIDER_SEED_SCHEME" --provider_seed_namespace "${PROVIDER_SEED_NAMESPACE:-}")
        fi
    fi
    [[ -z "$TASK_LIST" ]] || COMMAND+=(--task_list "$TASK_LIST")
    [[ -z "$MAX_TASKS" ]] || COMMAND+=(--max_tasks "$MAX_TASKS")
    [[ -z "$DATA_DIR" ]] || COMMAND+=(--data_dir "$DATA_DIR")
    [[ ${DISABLE_THINKING,,} != true && "$DISABLE_THINKING" != 1 ]] || COMMAND+=(--disable_thinking)
    [[ ${ACCUMULATE_THINKING,,} != true && "$ACCUMULATE_THINKING" != 1 ]] || COMMAND+=(--accumulate_thinking)
    [[ ${REFLECTOR_SIMPLIFIED_HISTORY,,} != true && "$REFLECTOR_SIMPLIFIED_HISTORY" != 1 ]] || COMMAND+=(--reflector_simplified_history)
    [[ ${REFLECTOR_ONLY,,} != true && "$REFLECTOR_ONLY" != 1 ]] || COMMAND+=(--reflector_only)
    [[ -z ${API_KEY_FILE:-} ]] || COMMAND+=(--api_key_file "$API_KEY_FILE")
    [[ -z ${PROJECT_ID:-} ]] || COMMAND+=(--project_id "$PROJECT_ID")
    [[ -z ${ACE_MODEL:-} ]] || COMMAND+=(--ace_model "$ACE_MODEL")
    [[ -z ${ACE_BASE_URL:-} ]] || COMMAND+=(--ace_base_url "$ACE_BASE_URL")
    [[ -z ${ACE_KEY_FILE:-} ]] || COMMAND+=(--ace_key_file "$ACE_KEY_FILE")
    [[ -z ${ACE_PROJECT_ID:-} ]] || COMMAND+=(--ace_project_id "$ACE_PROJECT_ID")
    if [[ -n ${WANDB_PROJECT:-} ]]; then
        COMMAND+=(--wandb_project "$WANDB_PROJECT" --wandb_run_name "$(basename "$RUN_DIR")")
        [[ -z ${WANDB_RUN_ID:-} ]] || COMMAND+=(--wandb_run_id "$WANDB_RUN_ID")
    fi
    [[ ${RESUME:-1} == 1 ]] || COMMAND+=(--no-resume)
    COMMAND+=("$@")
}

launch() {
    [[ $# == 0 ]] || { echo "Set options through environment variables, or use evaluate.py directly" >&2; return 2; }
    cd "$RLLM_DIR"
    configure_eval
    if [[ ${RLLM_DRY_RUN:-0} != 1 ]]; then
        local prompt_path=$BASE_SYSTEM_PROMPT_FILENAME
        [[ "$prompt_path" == /* ]] || prompt_path="$PROMPTS_DIR/$prompt_path"
        [[ -f "$prompt_path" ]] || { echo "Missing prompt: $prompt_path" >&2; return 1; }
        [[ -z "$TASK_LIST" || -f "$TASK_LIST" ]] || { echo "Missing TASK_LIST: $TASK_LIST" >&2; return 1; }
    fi
    local original_model_name=$MODEL_NAME original_model_source=$MODEL_SOURCE
    for STEP in $STEPS; do
        MODEL_NAME=$original_model_name
        MODEL_PATH=$REQUESTED_MODEL_PATH
        MODEL_SOURCE=$original_model_source
        RUN_DIR=$EVAL_TRAINING_RUN_DIR
        eval_resolve_paths
        [[ "$MODEL_SOURCE" == hub || "$MODEL_SOURCE" == api ]] || MODEL_NAME=$MODEL_PATH
        export MODEL_NAME
        eval_setup_python
        if [[ ${RLLM_DRY_RUN:-0} != 1 ]]; then
            eval_check_resume_dir
            mkdir -p "$RUN_DIR"
            eval_lock_run
            eval_check_resume_dir
            export MINISW_EVAL_LOG_DIR="$RUN_DIR/grading"
        fi
        eval_merge_checkpoint
        start_container_runtime
        eval_prepare_tracking
        build_eval_command "$@"
        if [[ ${RLLM_DRY_RUN:-0} != 1 ]]; then
            eval_record_config
            if [[ -f "$RUN_DIR/evaluation_complete.json" ]]; then
                eval_complete_run || return $?
                eval_unlock_run
                continue
            fi
        fi
        eval_start_server
        build_eval_command "$@"
        if [[ ${RLLM_DRY_RUN:-0} == 1 ]]; then
            eval_print_command "${COMMAND[@]}"
        else
            "${COMMAND[@]}" || return $?
            eval_complete_run || return $?
            eval_stop_server
            eval_unlock_run
        fi
    done
}
