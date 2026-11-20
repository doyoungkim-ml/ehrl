#!/usr/bin/env bash
#SBATCH --job-name=appworld-ppo
#SBATCH --output=appworld-ppo-%j.out
#SBATCH --error=appworld-ppo-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200

# Submit from the repository root; account/partition/QoS are site inputs.
# MODEL_VARIANT=nonthinking|thinking|instruct|thinking_2507; see ../README.md.
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$RLLM_DIR/examples/appworld/server/runtime.sh"
BENCHMARK=appworld
TRAINING_METHOD=ppo
source "$RLLM_DIR/examples/appworld/train/ppo/presets.sh"
source "$RLLM_DIR/examples/appworld/train/common.sh"
ppo_set_identity
export MAX_TRAIN_TASKS=${MAX_TRAIN_TASKS:-} TRAIN_TASK_OFFSET=${TRAIN_TASK_OFFSET:-0}
export MAX_VAL_TASKS=${MAX_VAL_TASKS:-${MAX_TEST_TASKS:-}}
appworld_start_runtime
appworld_common_args
HYDRA_ARGS+=(
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config=$VLLM_ATTENTION_CONFIG"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config=$VLLM_COMPILATION_CONFIG"
    "+trainer.trajectory_log_dir=$TRAJECTORY_LOG_DIR"
    "+trainer.trajectory_log_format=${TRAJECTORY_LOG_FORMAT:-enriched}"
    "+trainer.log_trajectories_structured=${LOG_TRAJECTORIES_STRUCTURED:-False}"
)
ppo_add_args HYDRA_ARGS
appworld_launch examples.appworld.train.ppo.train "${HYDRA_ARGS[@]}" "$@"
