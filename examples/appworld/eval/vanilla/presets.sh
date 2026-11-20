#!/usr/bin/env bash
# pass8: eight independent attempts per task, one task at a time.
PRESET=${PRESET:-pass8}
[[ "$PRESET" == pass8 ]] || { echo "Vanilla evaluation supports only PRESET=pass8 (got $PRESET)" >&2; exit 2; }
: "${G_ACT:=8}" "${TASK_CONCURRENCY:=1}" "${N_PARALLEL_AGENTS:=8}"
: "${LOCAL_EVAL_EPOCHS:=1}"
G_REF=0
G_CUR=0
