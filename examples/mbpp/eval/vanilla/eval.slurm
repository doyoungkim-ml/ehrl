#!/usr/bin/env bash
#SBATCH --job-name=mbpp-eval-vanilla
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
run_defaults=$(python3 "$RLLM_DIR/scripts/evaluation_runtime.py" mbpp vanilla --run-defaults)
eval "$run_defaults"
export EVAL_METHOD=vanilla
set -a
source "$RLLM_DIR/examples/mbpp/eval/vanilla/presets.sh"
source "$RLLM_DIR/examples/mbpp/eval/common.sh"
set +a
# set -a also exported the functions above; child shells (e.g. MiniSWE containers) must not import them.
export -fn $(compgen -A function)
exec "${LAUNCHER_PYTHON:-python3}" "$RLLM_DIR/scripts/evaluation_runtime.py" mbpp vanilla "$@"
