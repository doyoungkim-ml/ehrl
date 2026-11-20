#!/usr/bin/env bash
#SBATCH --job-name=mbpp-delayed-joint-ppo
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
export BENCHMARK=mbpp TRAINING_METHOD=delayed_joint_ppo
source "$RLLM_DIR/examples/mbpp/train/delayed_joint_ppo/presets.sh"
source "$RLLM_DIR/examples/mbpp/train/common.sh"
ppo_set_identity
start_runtime
common_args
method_args
ppo_add_args args
TRAIN_MODULE=examples.mbpp.train.delayed_joint_ppo.train
launch "$@"
