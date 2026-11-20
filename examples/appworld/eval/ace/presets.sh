#!/usr/bin/env bash
# visits8: eight measured policy visits with seven interleaved ACE updates, eight tasks at a time.
PRESET=${PRESET:-visits8}
[[ "$PRESET" == visits8 ]] || { echo "ACE evaluation supports only PRESET=visits8 (got $PRESET)" >&2; exit 2; }
: "${G_REF:=1}" "${G_CUR:=1}"
: "${LOCAL_EVAL_EPOCHS:=8}" "${TASK_CONCURRENCY:=8}" "${N_PARALLEL_AGENTS:=8}"
: "${LOCAL_EVAL_PROTOCOL:=sequential_adaptation_v1}"
if [[ "$LOCAL_EVAL_PROTOCOL" == legacy_search_v1 ]]; then
    : "${FINAL_POLICY_EVAL:=True}"
else
    : "${FINAL_POLICY_EVAL:=False}"
fi
