#!/usr/bin/env bash
# Vanilla keeps the existing nonthinking default and 8192/10240/32768 budgets.
# pass8: eight independent attempts per task.
PRESET=${PRESET:-pass8}
[[ "$PRESET" == pass8 ]] || { echo "Vanilla evaluation supports only PRESET=pass8 (got $PRESET)" >&2; exit 2; }
: "${G_ACT:=8}" "${PARALLEL:=8}"

MODEL_VARIANT=${MODEL_VARIANT:-nonthinking}
case "$MODEL_VARIANT" in
    nonthinking|thinking) ;;
    *) echo "MODEL_VARIANT must be nonthinking or thinking" >&2; exit 2 ;;
esac

# Set MODEL_PATH to a merged Hugging Face checkpoint to evaluate trained weights.
export MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}
if [[ -n ${STEP:-} && -n ${STEPS:-} && "$STEPS" != "$STEP" ]]; then
    echo "STEP and STEPS disagree" >&2; exit 2
fi
export STEP=${STEP:-${STEPS:-}}
if [[ -n ${RUN_DIR:-} && -z "$STEP" ]]; then
    echo "RUN_DIR requires an explicit STEP" >&2; exit 2
fi
STEP=${STEP:-0}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}      # ace_rl data.max_prompt_length
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-10240} # data.max_response_length (whole trajectory)
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}             # ace_rl rollout.max_model_len

# Generator sampling follows training rollout settings; top_p is explicit.
export TEMPERATURE=${TEMPERATURE:-0.7}
export TOP_P=${TOP_P:-1.0}
export SEED=${SEED:-42}
# Overrides the dataset rows' max_turns so ACE and vanilla share one turn budget.
export MAX_TURNS=${MAX_TURNS:-5}

export TASK_OFFSET=${TASK_OFFSET:-0}
export TASK_LIMIT=${TASK_LIMIT:-}            # empty = all 100 registered test tasks
if [[ -n ${TASK_CONCURRENCY:-} && ${TASK_CONCURRENCY} != 1 ]]; then
    echo "Mastermind vanilla batches complete pass@k groups with PARALLEL; TASK_CONCURRENCY must be 1" >&2
    exit 2
fi
export PARALLEL=${PARALLEL:-8}
export G_ACT=${G_ACT:-1}
export LOCAL_EVAL_EPOCHS=${LOCAL_EVAL_EPOCHS:-1}
[[ "$G_ACT" =~ ^[1-9][0-9]*$ && "$PARALLEL" =~ ^[1-9][0-9]*$ && "$PARALLEL" -ge "$G_ACT" ]] || {
    echo "G_ACT must be positive and PARALLEL must be at least G_ACT" >&2; exit 2
}
[[ "$LOCAL_EVAL_EPOCHS" == 1 ]] || {
    echo "Vanilla Mastermind uses LOCAL_EVAL_EPOCHS=1; set G_ACT for pass@k" >&2; exit 2
}

export RUN_TAG=${RUN_TAG:-${MODEL_NAME##*/}}
source "$RLLM_DIR/scripts/evaluation_server.sh"
eval_resolve_paths
[[ "$MODEL_SOURCE" == hub || "$MODEL_SOURCE" == api ]] || MODEL_NAME=$MODEL_PATH
export TOKENIZER=${TOKENIZER:-$MODEL_PATH}

EVAL_MODULE=examples.mastermind.eval.evaluate
EVAL_ARGS=(
    --model "$MODEL_NAME"
    --tokenizer "$TOKENIZER"
    --model-variant "$MODEL_VARIANT"
    --max-prompt-length "$MAX_PROMPT_LENGTH"
    --max-response-length "$MAX_RESPONSE_LENGTH"
    --max-model-length "$MAX_MODEL_LEN"
    --max-turns "$MAX_TURNS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --seed "$SEED"
    --offset "$TASK_OFFSET"
    --parallel "$PARALLEL"
    --g-act "$G_ACT"
    --run-tag "$RUN_TAG"
    --step "$STEP"
    --run-dir "$RUN_DIR"
    --save-trajectories
)
[ -n "$TASK_LIMIT" ] && EVAL_ARGS+=(--limit "$TASK_LIMIT")
export TEST_SPLIT_SIZE=${TEST_SPLIT_SIZE:-100}
export EVAL_ARGS RUN_DIR
