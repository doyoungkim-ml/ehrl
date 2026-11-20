#!/usr/bin/env bash
#SBATCH --job-name=appworld-eval-ace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|rtx6000"
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/rllm-flashinfer-${SLURM_JOB_ID:-$$}}
# A training RUN_DIR supplies its model and thinking mode unless they are set explicitly.
run_defaults=$(python3 "$RLLM_DIR/scripts/evaluation_runtime.py" appworld ace --run-defaults)
eval "$run_defaults"
source "$RLLM_DIR/examples/appworld/server/runtime.sh"
export EVAL_METHOD=ace
if [[ -n "${MODEL_VARIANT:-}" || ( -n "${DISABLE_THINKING:-}" && -n "${ACCUMULATE_THINKING:-}" ) ]]; then
    export EVAL_REASONING_CONFIGURED=1
else
    export EVAL_REASONING_CONFIGURED=0
fi
set -a
source "$RLLM_DIR/examples/appworld/eval/ace/presets.sh"
source "$RLLM_DIR/examples/appworld/eval/common.sh"
set +a
# set -a also exported the functions above; child shells (e.g. MiniSWE containers) must not import them.
export -fn $(compgen -A function)
# Standalone setup exports to examples/appworld/data. Reuse an existing
# evaluation export when that directory is absent in an older prepared checkout.
if [[ -z ${APPWORLD_TASK_IDS_DIR:-} ]]; then
    for task_ids_dir in "$RLLM_DIR/examples/appworld/data" "$RLLM_DIR/examples/appworld/eval/support/task-ids"; do
        for task_ids_file in "appworld_${TEST_SPLIT}_task_ids.json" "${TEST_SPLIT}_task_ids.json" "${TEST_SPLIT}.json"; do
            if [[ -f "$task_ids_dir/$task_ids_file" ]]; then
                export APPWORLD_TASK_IDS_DIR=$task_ids_dir
                break 2
            fi
        done
    done
fi
exec "${LAUNCHER_PYTHON:-python3}" "$RLLM_DIR/scripts/evaluation_runtime.py" appworld ace "$@"
