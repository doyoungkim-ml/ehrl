#!/usr/bin/env bash
#SBATCH --job-name=miniswe-ace-rl
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=48:00:00

set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
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
export BENCHMARK=mini_swe_bench TRAINING_METHOD=ace_rl
source "$RLLM_DIR/examples/mini_swe_bench/train/ace_rl/presets.sh"
source "$RLLM_DIR/examples/mini_swe_bench/train/common.sh"
start_runtime
common_args
method_args
launch "$@"
