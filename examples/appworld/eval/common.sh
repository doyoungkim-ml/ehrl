#!/usr/bin/env bash
# The opt-in protocol supplies training input/sampling defaults. Explicit
# numeric overrides remain visible in the planned evaluator command.
case "${EVAL_PROTOCOL:-}" in
    "")
        case "${TRAINING_INPUT_PARITY:-False}" in
            True|true|1) EVAL_PROTOCOL=training_input_parity ;;
            False|false|0) EVAL_PROTOCOL=legacy ;;
            *) echo "TRAINING_INPUT_PARITY must be True/False or 1/0" >&2; exit 2 ;;
        esac
        ;;
    legacy)
        case "${TRAINING_INPUT_PARITY:-False}" in
            False|false|0) ;;
            *) echo "EVAL_PROTOCOL=legacy conflicts with TRAINING_INPUT_PARITY" >&2; exit 2 ;;
        esac
        ;;
    training_input_parity)
        case "${TRAINING_INPUT_PARITY:-True}" in
            True|true|1) ;;
            *) echo "EVAL_PROTOCOL=training_input_parity requires TRAINING_INPUT_PARITY=True" >&2; exit 2 ;;
        esac
        ;;
    *) echo "Unknown EVAL_PROTOCOL: $EVAL_PROTOCOL" >&2; exit 2 ;;
esac
if [[ "$EVAL_PROTOCOL" == training_input_parity ]]; then
    TRAINING_INPUT_PARITY=True
    : "${MAX_MODEL_LEN:=32768}" "${REFLECTOR_MAX_PROMPT_LENGTH:=30720}" "${REFLECTOR_MAX_RESPONSE_LENGTH:=2048}"
    : "${REFLECTOR_MAX_HISTORY_TOKENS:=20000}" "${CURATOR_MAX_PROMPT_LENGTH:=8192}" "${CURATOR_MAX_RESPONSE_LENGTH:=4096}"
    : "${GEN_TEMPERATURE:=0.7}" "${GEN_TOP_P:=1.0}" "${ACE_TEMPERATURE:=0.7}" "${TRAJECTORY_HARD_TIMEOUT:=7200}"
else
    TRAINING_INPUT_PARITY=False
fi
: "${MODEL_NAME:=Qwen/Qwen3-4B}"
case "${MODEL_NAME,,}" in
    gemini*|gpt-*|claude-*|o1-*|o3-*) : "${MODEL_VARIANT:=nonthinking}" ;;
    *) : "${MODEL_VARIANT:=thinking}" ;;
esac
case "$MODEL_VARIANT" in
    thinking) : "${DISABLE_THINKING:=False}" "${ACCUMULATE_THINKING:=True}" ;;
    nonthinking) : "${DISABLE_THINKING:=True}" "${ACCUMULATE_THINKING:=False}" ;;
    *) echo "Unknown MODEL_VARIANT: $MODEL_VARIANT" >&2; exit 2 ;;
esac
: "${ACCUMULATE_REASONING:=$ACCUMULATE_THINKING}"
: "${MAX_TURNS:=10}" "${N_PARALLEL_AGENTS:=4}" "${MAX_MODEL_LEN:=40960}"
: "${MAX_RESPONSE_LENGTH:=32768}" "${MAX_PROMPT_LENGTH:=8192}" "${TEST_SPLIT:=dev}"
: "${REFLECTOR_MAX_PROMPT_LENGTH:=20480}" "${REFLECTOR_MAX_HISTORY_TOKENS:=20000}"
: "${REFLECTOR_SYSTEM_PROMPT:=You are an expert AppWorld trajectory analyst. Respond with JSON only.}"
: "${CURATOR_SYSTEM_PROMPT:=You are a master curator of AppWorld task playbooks. Respond with JSON only.}"
source "$RLLM_DIR/scripts/evaluation_defaults.sh"
