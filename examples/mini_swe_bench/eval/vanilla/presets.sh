#!/usr/bin/env bash
# pass8: eight independent attempts per task, one task at a time.
PRESET=${PRESET:-pass8}
[[ "$PRESET" == pass8 ]] || { echo "Vanilla evaluation supports only PRESET=pass8 (got $PRESET)" >&2; exit 2; }
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3.5-4B}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
# Dedicated evaluation prompt repairs the unavailable template in both old vanilla launchers.
BASE_SYSTEM_PROMPT_FILENAME=${BASE_SYSTEM_PROMPT_FILENAME:-$RLLM_DIR/examples/mini_swe_bench/eval/vanilla/prompts/base_system_prompt.txt}
MAX_TURNS=${MAX_TURNS:-10}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_TASKS=${MAX_TASKS-100}
LOCAL_EVAL_EPOCHS=${LOCAL_EVAL_EPOCHS:-1}
G_ACT=${G_ACT:-8}
TASK_CONCURRENCY=${TASK_CONCURRENCY:-1}
N_PARALLEL_AGENTS=${N_PARALLEL_AGENTS:-8}
G_REF=0
G_CUR=0
