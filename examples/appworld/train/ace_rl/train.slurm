#!/usr/bin/env bash
#SBATCH --job-name=appworld-ace-rl
#SBATCH --output=appworld-ace-rl-%j.out
#SBATCH --error=appworld-ace-rl-%j.err
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
TRAINING_METHOD=ace_rl
source "$RLLM_DIR/examples/appworld/train/ace_rl/presets.sh"
source "$RLLM_DIR/examples/appworld/train/common.sh"
if [[ "$PRESET" == oneeval_rollout8 ]]; then
    DEFAULT_STAMP=singlemode1curgroup_rollout${POLICY_ROLLOUTS}44_seed_oneeval_best_rejection_update_2gpu
    [[ "$REFLECTION_SOURCE_SELECTION" == min_reward ]] || DEFAULT_STAMP+=_random_reflection_source
    DEFAULT_STAMP+=$AUXILIARY_UPDATE_TAG
    [[ "$MODEL_VARIANT" != nonthinking ]] || DEFAULT_STAMP+=_nothink
    RUN_STAMP=${RUN_STAMP:-$DEFAULT_STAMP}
    RUN_NAME=${RUN_NAME:-appworld_local_ace_rl_singlemode_${MODEL_TAG}${RUN_STAMP}}
elif [[ "$PRESET" == delayed_oneeval_rollout8 ]]; then
    DELAYED_NAME=appworld_local_ace_rl_delayed_joint_${MODEL_TAG}_rollout${POLICY_ROLLOUTS}_aligned11_seed_oneeval_best_rejection_update_2gpu
    [[ "$MODEL_VARIANT" != nonthinking ]] || DELAYED_NAME+=_nothink
    RUN_NAME=${RUN_NAME:-$DELAYED_NAME}
else
    RUN_NAME=${RUN_NAME:-appworld_local_ace_rl_${PRESET}_${MODEL_TAG}_${MODEL_VARIANT}${RUN_STAMP:-}}
fi
export RUN_NAME
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$RUN_NAME}
export APPWORLD_EXPERIMENT_NAME=${APPWORLD_EXPERIMENT_NAME:-$EXPERIMENT_NAME}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_NAME} WANDB_NAME=${WANDB_NAME:-$RUN_NAME}
PROMPTS_DIR=${PROMPTS_DIR:-$RLLM_DIR/examples/appworld/train/ace_rl/prompts}
INITIAL_PLAYBOOK=${INITIAL_PLAYBOOK:-$RLLM_DIR/examples/appworld/train/ace_rl/playbooks/initial_playbook.txt}
CANDIDATE_EVAL_MAX_PARALLEL_AGENTS=${CANDIDATE_EVAL_MAX_PARALLEL_AGENTS:-$N_PARALLEL_AGENTS}
CANDIDATE_EVAL_ROLLOUTS_PER_CANDIDATE=$CANDIDATE_EVAL_ROLLOUTS
if (( CANDIDATE_EVAL_ROLLOUTS_PER_CANDIDATE < 1 )); then
    CANDIDATE_EVAL_ROLLOUTS_PER_CANDIDATE=$POLICY_ROLLOUTS
fi
if [[ -z "${CANDIDATE_EVAL_BATCH_SIZE:-}" ]]; then
    CANDIDATE_EVAL_BATCH_SIZE=$(( CANDIDATE_EVAL_MAX_PARALLEL_AGENTS / CANDIDATE_EVAL_ROLLOUTS_PER_CANDIDATE ))
    (( CANDIDATE_EVAL_BATCH_SIZE >= 1 )) || CANDIDATE_EVAL_BATCH_SIZE=1
fi
appworld_start_runtime
appworld_common_args
HYDRA_ARGS+=(
    "+rllm.ace.g_ref=$REFLECTOR_ROLLOUTS"
    "+rllm.ace.g_cur=$CURATOR_ROLLOUTS"
    "+rllm.ace.guidebook_aggregation_level=$GUIDEBOOK_AGGREGATION_LEVEL"
    "+rllm.ace.singlemode=$SINGLEMODE"
    "+rllm.ace.singlemode_curator_one_group=$SINGLEMODE_CURATOR_ONE_GROUP"
    "+rllm.ace.reflection_source_selection=$REFLECTION_SOURCE_SELECTION"
    "+rllm.ace.auxiliary_update_gate=$AUXILIARY_UPDATE_GATE"
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
if [[ "$TRAINING_MODE" == iterative ]]; then
    HYDRA_ARGS+=("+rllm.ace.iterative_steps_per_role=$ITERATIVE_STEPS_PER_ROLE" "+rllm.ace.iterative_schedule=$ITERATIVE_SCHEDULE")
fi
appworld_launch examples.appworld.train.ace_rl.train "${HYDRA_ARGS[@]}" "$@"
