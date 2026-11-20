#!/usr/bin/env bash
#SBATCH --job-name=appworld-grpo
#SBATCH --output=appworld-grpo-%j.out
#SBATCH --error=appworld-grpo-%j.err
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
TRAINING_METHOD=grpo
source "$RLLM_DIR/examples/appworld/train/grpo/presets.sh"
source "$RLLM_DIR/examples/appworld/train/common.sh"
export RUN_NAME=${RUN_NAME:-appworld_${PRESET}${ACTOR_PROMPT_TAG}_${MODEL_TAG}_${MODEL_VARIANT}${RUN_STAMP:-}}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$RUN_NAME}
export APPWORLD_EXPERIMENT_NAME=${APPWORLD_EXPERIMENT_NAME:-$EXPERIMENT_NAME}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_NAME} WANDB_NAME=${WANDB_NAME:-$RUN_NAME}
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
[[ "$GRPO_ACTOR_PROMPT" == agent ]] || HYDRA_ARGS+=("+rllm.actor_prompt=$GRPO_ACTOR_PROMPT")
if [[ "$PRESET" == ppo ]]; then
    HYDRA_ARGS+=(
        "algorithm.gamma=$GAE_GAMMA" "algorithm.lam=$GAE_LAMBDA"
        "critic.strategy=fsdp2" "critic.model.path=$MODEL_NAME" "critic.model.trust_remote_code=True"
        "critic.optim.lr=$CRITIC_LR" "critic.loss_agg_mode=$CRITIC_LOSS_AGG_MODE"
        "critic.ppo_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU"
        "+critic.ppo_infer_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU"
        "critic.model.enable_gradient_checkpointing=$ACTOR_ENABLE_GRADIENT_CHECKPOINTING"
        "critic.model.enable_activation_offload=$ACTOR_ENABLE_ACTIVATION_OFFLOAD"
        "critic.fsdp.param_offload=True" "critic.fsdp.optimizer_offload=True"
        "critic.fsdp.forward_prefetch=$ACTOR_FSDP_FORWARD_PREFETCH"
        "critic.checkpoint.save_contents=$CRITIC_SAVE_CONTENTS"
    )
    if (( KEEP_LAST_N > 0 )); then
        HYDRA_ARGS+=("trainer.max_actor_ckpt_to_keep=$KEEP_LAST_N" "trainer.max_critic_ckpt_to_keep=$KEEP_LAST_N")
    fi
fi
appworld_launch examples.appworld.train.grpo.train "${HYDRA_ARGS[@]}" "$@"
