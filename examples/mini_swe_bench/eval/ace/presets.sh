#!/usr/bin/env bash
# Generator sampling is 0.6/0.95 in mini_swe_bench_ace_utils.py.
# visits8: eight measured policy visits with seven interleaved ACE updates, eight tasks at a time.
PRESET=${PRESET:-visits8}
[[ "$PRESET" == visits8 ]] || { echo "ACE evaluation supports only PRESET=visits8 (got $PRESET)" >&2; exit 2; }
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3.5-4B}
MAX_TURNS=${MAX_TURNS:-10}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_TASKS=${MAX_TASKS-100}
BASE_SYSTEM_PROMPT_FILENAME=${BASE_SYSTEM_PROMPT_FILENAME:-base_system_prompt_tools.txt}
LOCAL_EVAL_EPOCHS=${LOCAL_EVAL_EPOCHS:-8}
TASK_CONCURRENCY=${TASK_CONCURRENCY:-8}
N_PARALLEL_AGENTS=${N_PARALLEL_AGENTS:-8}
G_ACT=${G_ACT:-1}
G_REF=${G_REF:-1}
G_CUR=${G_CUR:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40000}
