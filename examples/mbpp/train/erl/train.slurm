#!/usr/bin/env bash
#SBATCH --job-name=mbpp-erl
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2

# Account/partition/QoS are site inputs; submit with ordinary priority.
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
export BENCHMARK=mbpp TRAINING_METHOD=erl
export TRAIN_MODULE=examples.mbpp.train.erl.train
source "$RLLM_DIR/examples/mbpp/train/erl/presets.sh"
source "$RLLM_DIR/examples/mbpp/train/common.sh"
erl_set_identity
start_runtime
common_args
method_args
erl_add_args args
launch "$@"
