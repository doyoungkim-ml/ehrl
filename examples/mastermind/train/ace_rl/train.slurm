#!/usr/bin/env bash
#SBATCH --job-name=mastermind-ace-rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
export BENCHMARK=mastermind TRAINING_METHOD=ace_rl
export TRAIN_MODULE=examples.mastermind.train.ace_rl.train
source "$RLLM_DIR/examples/mastermind/train/ace_rl/presets.sh"
source "$RLLM_DIR/examples/mastermind/train/common.sh"
start_runtime
common_args
method_args
launch "$@"
