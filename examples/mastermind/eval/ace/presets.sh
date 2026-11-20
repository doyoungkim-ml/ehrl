#!/usr/bin/env bash
# ACE keeps the existing thinking default and 8192/32768/40960 token budgets.
# visits8: eight measured policy visits with seven interleaved ACE updates, eight tasks at a time.
PRESET=${PRESET:-visits8}
[[ "$PRESET" == visits8 ]] || { echo "ACE evaluation supports only PRESET=visits8 (got $PRESET)" >&2; exit 2; }
: "${LOCAL_EVAL_EPOCHS:=8}" "${TASK_CONCURRENCY:=8}" "${PARALLEL:=8}"

MODEL_VARIANT=${MODEL_VARIANT:-thinking}
case "$MODEL_VARIANT" in
    thinking)    DISABLE_THINKING_FLAG="--no-disable_thinking" ;;
    nonthinking) DISABLE_THINKING_FLAG="--disable_thinking" ;;
    *) echo "MODEL_VARIANT must be thinking or nonthinking" >&2; exit 2 ;;
esac

export MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}
if [[ -n ${STEP:-} && -n ${STEPS:-} && "$STEPS" != "$STEP" ]]; then
    echo "STEP and STEPS disagree" >&2; exit 2
fi
export STEP=${STEP:-${STEPS:-}}
if [[ -n ${RUN_DIR:-} && -z "$STEP" ]]; then
    echo "RUN_DIR requires an explicit STEP" >&2; exit 2
fi
STEP=${STEP:-0}

# Raised from the training budget (10240) because 19% of vanilla thinking games ended in
# TRUNCATION: completions peak at ~9.5k tokens and the growing observations share the same
# budget. 32768/40960 is the _32k shape used by the mbpp and alfworld evals, and 40960 is
# Qwen3-0.6B's max_position_embeddings.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}

# Generator sampling follows the training rollout settings.
export TEMPERATURE=${TEMPERATURE:-0.7}
export TOP_P=${TOP_P:-1.0}
export SEED=${SEED:-42}

export MAX_TURNS=${MAX_TURNS:-5}
export LOCAL_EVAL_EPOCHS
export LOCAL_EVAL_PROTOCOL=${LOCAL_EVAL_PROTOCOL:-sequential_adaptation_v1}
export G_ACT=${G_ACT:-1}
export G_REF=${G_REF:-1}            # eval convention 1/1 (training used 4/4)
export G_CUR=${G_CUR:-1}
export ACE_TEMPERATURE=${ACE_TEMPERATURE:-0.7}
export MAX_PLAYBOOK_LENGTH=${MAX_PLAYBOOK_LENGTH:-2048}
export REFLECTOR_MAX_PROMPT_LENGTH=${REFLECTOR_MAX_PROMPT_LENGTH:-30720}
export REFLECTOR_MAX_RESPONSE_LENGTH=${REFLECTOR_MAX_RESPONSE_LENGTH:-2048}
export CURATOR_MAX_PROMPT_LENGTH=${CURATOR_MAX_PROMPT_LENGTH:-8192}
export CURATOR_MAX_RESPONSE_LENGTH=${CURATOR_MAX_RESPONSE_LENGTH:-4096}
# Role system prompts as set by training (train/ace_rl/train.py ACE_DEFAULTS).
export REFLECTOR_SYSTEM_PROMPT=${REFLECTOR_SYSTEM_PROMPT:-"You analyze Mastermind guesses and feedback. Respond with JSON only."}
export CURATOR_SYSTEM_PROMPT=${CURATOR_SYSTEM_PROMPT:-"You curate concise Mastermind reasoning hints. Respond with JSON only."}

export TASK_OFFSET=${TASK_OFFSET:-0}
export TASK_LIMIT=${TASK_LIMIT:-}    # empty = all 100 registered test tasks
export PARALLEL=${PARALLEL:-8}
export TASK_CONCURRENCY=${TASK_CONCURRENCY:-1}
case "$LOCAL_EVAL_PROTOCOL" in
    sequential_adaptation_v1)
        case "${FREEZE_SOLVED:-False}" in True|true|1) echo "Sequential evaluation keeps every policy visit; FREEZE_SOLVED must be False" >&2; exit 2 ;; esac
        if (( G_REF > 1 || G_CUR > 1 )); then
            echo "Sequential evaluation supports one reflection and one curator delta; branching requires LOCAL_EVAL_PROTOCOL=legacy_search_v1" >&2; exit 2
        fi
        ;;
    legacy_search_v1) ;;
    *) echo "Unknown LOCAL_EVAL_PROTOCOL=$LOCAL_EVAL_PROTOCOL" >&2; exit 2 ;;
esac
[[ "$TASK_CONCURRENCY" =~ ^[1-9][0-9]*$ && "$PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "TASK_CONCURRENCY and PARALLEL must be positive" >&2; exit 2; }
if (( TASK_CONCURRENCY * ${G_ACT:-1} > PARALLEL )); then
    echo "TASK_CONCURRENCY * G_ACT must not exceed PARALLEL" >&2; exit 2
fi

export RUN_TAG=${RUN_TAG:-${MODEL_NAME##*/}}
source "$RLLM_DIR/scripts/evaluation_server.sh"
eval_resolve_paths
[[ "$MODEL_SOURCE" == hub || "$MODEL_SOURCE" == api ]] || MODEL_NAME=$MODEL_PATH
export TOKENIZER=${TOKENIZER:-$MODEL_PATH}
export TEST_SPLIT_SIZE=${TEST_SPLIT_SIZE:-100}

EVAL_MODULE=examples.mastermind.eval.ace.evaluate
# The two evaluators expose different argument spellings.
BASE_URL_FLAG=--base_url
EVAL_ARGS=(
    --prompts_dir "${PROMPTS_DIR:-$RLLM_DIR/examples/mastermind/train/ace_rl/prompts}"
    --model_name "$MODEL_NAME"
    --tokenizer "$TOKENIZER"
    --n_parallel_agents "$PARALLEL"
    --max_turns "$MAX_TURNS"
    --local_eval_protocol "$LOCAL_EVAL_PROTOCOL"
    --local_eval_epochs "$LOCAL_EVAL_EPOCHS"
    --g_act "$G_ACT"
    --g_ref "$G_REF" --g_cur "$G_CUR"
    --ace_temperature "$ACE_TEMPERATURE"
    --max_playbook_length "$MAX_PLAYBOOK_LENGTH"
    --reflector_max_prompt_length "$REFLECTOR_MAX_PROMPT_LENGTH"
    --reflector_max_response_length "$REFLECTOR_MAX_RESPONSE_LENGTH"
    --curator_max_prompt_length "$CURATOR_MAX_PROMPT_LENGTH"
    --curator_max_response_length "$CURATOR_MAX_RESPONSE_LENGTH"
    --reflector_system_prompt "$REFLECTOR_SYSTEM_PROMPT"
    --curator_system_prompt "$CURATOR_SYSTEM_PROMPT"
    --max_model_length "$MAX_MODEL_LEN"
    --gen_max_response_length "$MAX_RESPONSE_LENGTH"
    --gen_max_prompt_length "$MAX_PROMPT_LENGTH"
    --gen_temperature "$TEMPERATURE"
    --gen_top_p "$TOP_P"
    --seed "$SEED"
    --wandb_step "$STEP"
    --split test
    --task_offset "$TASK_OFFSET"
    --metrics_file "$RUN_DIR/metrics_step${STEP}.json"
    --ace_log_dir "$RUN_DIR/ace_logs"
    --playbook_snapshot_dir "$RUN_DIR/playbooks"
    $DISABLE_THINKING_FLAG
)
[ -n "$TASK_LIMIT" ] && EVAL_ARGS+=(--max_tasks "$TASK_LIMIT")
# Serial runs keep the historical command so recorded configurations still match.
[[ "$TASK_CONCURRENCY" == 1 ]] || EVAL_ARGS+=(--task_concurrency "$TASK_CONCURRENCY")
# Opt-in handoff controls; defaults keep the historical command.
[[ ${HANDOFF_PROTOCOL:-legacy} == legacy ]] || EVAL_ARGS+=(--handoff_protocol "$HANDOFF_PROTOCOL")
if [[ ${ADAPTATION_MODE:-apply} != apply || ${PROVIDER_SEED_SCHEME:-legacy} != legacy ]]; then
    EVAL_ARGS+=(--adaptation_mode "${ADAPTATION_MODE:-apply}")
    if [[ ${PROVIDER_SEED_SCHEME:-legacy} != legacy ]]; then
        EVAL_ARGS+=(--provider_seed_scheme "$PROVIDER_SEED_SCHEME" --provider_seed_namespace "${PROVIDER_SEED_NAMESPACE:-}")
    fi
fi
# Optional external ACE roles with the local Qwen generator.
[ -n "${ACE_MODEL:-}" ] && EVAL_ARGS+=(--ace_model "$ACE_MODEL")
[ -n "${ACE_KEY_FILE:-}" ] && EVAL_ARGS+=(--ace_key_file "$ACE_KEY_FILE")
[ -n "${ACE_THINKING_BUDGET:-}" ] && EVAL_ARGS+=(--ace_thinking_budget "$ACE_THINKING_BUDGET")
[ -n "${ACE_BASE_URL:-}" ] && EVAL_ARGS+=(--ace_base_url "$ACE_BASE_URL")
[ -n "${ACE_PROJECT_ID:-}" ] && EVAL_ARGS+=(--ace_project_id "$ACE_PROJECT_ID")
[ -n "${WANDB_PROJECT:-}" ] && EVAL_ARGS+=(--wandb_project "$WANDB_PROJECT" --wandb_run_name "$(basename "$RUN_DIR")")
export EVAL_ARGS RUN_DIR
