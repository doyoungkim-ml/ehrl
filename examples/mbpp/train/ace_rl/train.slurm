#!/usr/bin/env bash
#SBATCH --job-name=mbpp-ace-rl
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
# Supply the cluster account/partition and supported GPU constraint at submission.
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
export BENCHMARK=mbpp TRAINING_METHOD=ace_rl
export TRAIN_MODULE=examples.mbpp.train.ace_rl.train
source "$RLLM_DIR/examples/mbpp/train/ace_rl/presets.sh"
source "$RLLM_DIR/examples/mbpp/train/common.sh"
start_runtime
common_args
method_args
launch "$@"
