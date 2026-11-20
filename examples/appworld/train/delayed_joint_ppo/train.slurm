#!/usr/bin/env bash
#SBATCH --job-name=appworld-delayed-joint-ppo
#SBATCH --output=appworld-delayed-joint-ppo-%j.out
#SBATCH --error=appworld-delayed-joint-ppo-%j.err
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
TRAINING_METHOD=delayed_joint_ppo
source "$RLLM_DIR/examples/appworld/train/delayed_joint_ppo/presets.sh"
source "$RLLM_DIR/examples/appworld/train/common.sh"
ppo_set_identity
PROMPTS_DIR=${PROMPTS_DIR:-$RLLM_DIR/examples/appworld/train/ace_rl/prompts}
INITIAL_PLAYBOOK=${INITIAL_PLAYBOOK:-$RLLM_DIR/examples/appworld/train/ace_rl/playbooks/initial_playbook.txt}
CANDIDATE_EVAL_MAX_PARALLEL_AGENTS=${CANDIDATE_EVAL_MAX_PARALLEL_AGENTS:-$N_PARALLEL_AGENTS}
# Keep the existing AppWorld ACE evaluation chunk size independent of the
# single downstream actor sample used by joint PPO training.
CANDIDATE_EVAL_BATCH_SIZE=${CANDIDATE_EVAL_BATCH_SIZE:-4}
appworld_start_runtime
appworld_common_args
HYDRA_ARGS+=(
    "+rllm.ace.g_ref=$REFLECTOR_ROLLOUTS"
    "+rllm.ace.g_cur=$CURATOR_ROLLOUTS"
    "+rllm.ace.guidebook_aggregation_level=$GUIDEBOOK_AGGREGATION_LEVEL"
    "+rllm.ace.singlemode=$SINGLEMODE"
    "+rllm.ace.singlemode_curator_one_group=$SINGLEMODE_CURATOR_ONE_GROUP"
    "+rllm.ace.reflection_source_selection=$REFLECTION_SOURCE_SELECTION"
    "+rllm.ace.policy_train_source=$POLICY_TRAIN_SOURCE"
    "+rllm.ace.selection_mode=$SELECTION_MODE"
    "+rllm.ace.ace_temperature=0.7"
    "+rllm.ace.training_mode=$TRAINING_MODE"
    "+rllm.ace.policy_max_prompt_length=8192"
    "+rllm.ace.policy_max_response_length=32768"
    "+rllm.ace.reflector_max_prompt_length=$REFLECTOR_MAX_PROMPT_LENGTH"
    "+rllm.ace.reflector_max_response_length=2048"
    "+rllm.ace.curator_max_prompt_length=8192"
    "+rllm.ace.curator_max_response_length=4096"
    "+rllm.ace.max_playbook_length=2048"
    "+rllm.ace.max_turns=$APPWORLD_MAX_TURNS"
    "+rllm.ace.flush_playbook_every_n_steps=$FLUSH_PLAYBOOK_EVERY_N_STEPS"
    "+rllm.ace.local_eval_epochs=$LOCAL_EVAL_EPOCHS"
    "+rllm.ace.local_task_artifact_root=$TASK_ARTIFACT_ROOT"
    "+rllm.ace.local_playbook_log_dir=null"
    "+rllm.ace.playbook_snapshot_dir=null"
    "+rllm.ace.candidate_eval_batch_size=$CANDIDATE_EVAL_BATCH_SIZE"
    "+rllm.ace.candidate_eval_rollouts=$CANDIDATE_EVAL_ROLLOUTS"
    "+rllm.ace.candidate_eval_max_parallel_agents=$CANDIDATE_EVAL_MAX_PARALLEL_AGENTS"
    "+rllm.ace.prompts_dir=$PROMPTS_DIR"
    "+rllm.ace.initial_playbook=$INITIAL_PLAYBOOK"
    "+rllm.ace.reflector_adapter_class=examples.appworld.train.ace_rl.appworld_adapter.AppWorldReflectorAdapter"
    "+rllm.ace.reflector_system_prompt=$APPWORLD_REFLECTOR_SYSTEM_PROMPT"
    "+rllm.ace.curator_system_prompt=$APPWORLD_CURATOR_SYSTEM_PROMPT"
)
if [[ -n "${REFLECTOR_MAX_HISTORY_TOKENS:-}" ]]; then
    HYDRA_ARGS+=("+rllm.ace.reflector_max_history_tokens=$REFLECTOR_MAX_HISTORY_TOKENS")
fi

ppo_add_args HYDRA_ARGS
appworld_launch examples.appworld.train.delayed_joint_ppo.train "${HYDRA_ARGS[@]}" "$@"
