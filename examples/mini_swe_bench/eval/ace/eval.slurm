#!/usr/bin/env bash
#SBATCH --job-name=miniswe-eval-ace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|rtx6000"
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/rllm-flashinfer-${SLURM_JOB_ID:-$$}}
# A training RUN_DIR supplies its model and thinking mode unless they are set explicitly.
run_defaults=$(python3 "$RLLM_DIR/scripts/evaluation_runtime.py" mini_swe_bench ace --run-defaults)
eval "$run_defaults"
# Use the paths prepared by this checkout's data and image setup commands.
if [[ -z ${MINISW_DATA_DIR:-} ]]; then
    MINISW_DATA_DIR=$RLLM_DIR/rllm/data/datasets/swegym
    if [[ -f "$RLLM_DIR/examples/mini_swe_bench/data/prepared/train.parquet" && -f "$RLLM_DIR/examples/mini_swe_bench/data/prepared/validation.parquet" ]]; then
        MINISW_DATA_DIR=$RLLM_DIR/examples/mini_swe_bench/data/prepared
    fi
    export MINISW_DATA_DIR
fi
export APPTAINER_SIF_DIR=${APPTAINER_SIF_DIR:-$RLLM_DIR/.cache/mini_swe_bench/sif}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$RLLM_DIR/.cache/mini_swe_bench/apptainer}
export BENCHMARK=mini_swe_bench EVAL_MODE=ace
set -a
source "$RLLM_DIR/examples/mini_swe_bench/eval/ace/presets.sh"
source "$RLLM_DIR/examples/mini_swe_bench/eval/common.sh"
set +a
# set -a also exported the functions above; child shells (e.g. MiniSWE containers) must not import them.
export -fn $(compgen -A function)
if [[ ${1:-} == --dry-run ]]; then export RLLM_DRY_RUN=1; shift; fi
launch "$@"
