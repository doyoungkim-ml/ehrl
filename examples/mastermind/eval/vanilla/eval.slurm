#!/usr/bin/env bash
#SBATCH --job-name=mastermind-eval-vanilla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|rtx6000
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/rllm-flashinfer-${SLURM_JOB_ID:-$$}}
# A training RUN_DIR supplies its model and thinking mode unless they are set explicitly.
run_defaults=$(python3 "$RLLM_DIR/scripts/evaluation_runtime.py" mastermind vanilla --run-defaults)
eval "$run_defaults"
export BENCHMARK=mastermind EVAL_MODE=vanilla
set -a
source "$RLLM_DIR/examples/mastermind/eval/vanilla/presets.sh"
source "$RLLM_DIR/examples/mastermind/eval/common.sh"
set +a
# set -a also exported the functions above; child shells (e.g. MiniSWE containers) must not import them.
export -fn $(compgen -A function)
if [[ ${1:-} == --dry-run ]]; then export RLLM_DRY_RUN=1; shift; fi
launch "$@"
