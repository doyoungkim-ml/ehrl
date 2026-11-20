#!/usr/bin/env bash
#SBATCH --job-name=appworld-erl
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200

# Account/partition/QoS are site inputs; submit with ordinary priority.
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$RLLM_DIR/examples/appworld/server/runtime.sh"
export BENCHMARK=appworld TRAINING_METHOD=erl
export TRAIN_MODULE=examples.appworld.train.erl.train
source "$RLLM_DIR/examples/appworld/train/erl/presets.sh"
source "$RLLM_DIR/examples/appworld/train/common.sh"
erl_set_identity
export MAX_TRAIN_TASKS=${MAX_TRAIN_TASKS:-} TRAIN_TASK_OFFSET=${TRAIN_TASK_OFFSET:-0}
export MAX_VAL_TASKS=${MAX_VAL_TASKS:-${MAX_TEST_TASKS:-}}
appworld_start_runtime
appworld_common_args
HYDRA_ARGS+=(
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config=$VLLM_ATTENTION_CONFIG"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config=$VLLM_COMPILATION_CONFIG"
)
erl_add_args HYDRA_ARGS
appworld_launch "$TRAIN_MODULE" "${HYDRA_ARGS[@]}" "$@"
