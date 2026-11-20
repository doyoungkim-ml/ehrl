#!/usr/bin/env bash
# Scientific defaults and accepted overrides for this benchmark/method.
# Main presets use eight policy rollouts per task; auxiliary group sizes are separate.
PRESET=${PRESET:-oneeval_rollout8}
case "$PRESET" in
    oneeval_rollout8) PRESET_TRAINING_MODE=standard; PRESET_POLICY_ROLLOUTS=8 ;;
    delayed_oneeval_rollout8) PRESET_TRAINING_MODE=delayed_joint; PRESET_POLICY_ROLLOUTS=8 ;;
    # GRPO budget: 4 seed rollouts plus at most 4 candidate-scoring rollouts per
    # task (2 reflections, one 3-sample curator group), all trained as actor rows.
    shared_rollout8) PRESET_TRAINING_MODE=standard; PRESET_POLICY_ROLLOUTS=4 ;;
    *) echo "Unknown ace_rl PRESET: $PRESET (expected oneeval_rollout8, delayed_oneeval_rollout8 or shared_rollout8)" >&2; exit 2 ;;
esac
ACE_TRAINING_MODE=${ACE_TRAINING_MODE:-$PRESET_TRAINING_MODE}
[[ "$ACE_TRAINING_MODE" == "$PRESET_TRAINING_MODE" ]] || { echo "ACE_TRAINING_MODE=$ACE_TRAINING_MODE conflicts with PRESET=$PRESET" >&2; exit 2; }
if [[ "$PRESET_TRAINING_MODE" == delayed_joint ]]; then
    # One aligned reflector/curator pair per policy rollout instead of independent sampling groups.
    REFLECTOR_ROLLOUTS=${REFLECTOR_ROLLOUTS:-1}
    CURATOR_ROLLOUTS=${CURATOR_ROLLOUTS:-1}
fi
if [[ "$PRESET" == shared_rollout8 ]]; then
    REFLECTOR_ROLLOUTS=${REFLECTOR_ROLLOUTS:-2}
    CURATOR_ROLLOUTS=${CURATOR_ROLLOUTS:-3}
    POLICY_TRAIN_SOURCE=${POLICY_TRAIN_SOURCE:-seed_and_candidates}
    CANDIDATE_EVAL_ROLLOUTS=${CANDIDATE_EVAL_ROLLOUTS:-1}
    # VeRL counts mini-batches in prompts x rollout.n rows; 16 x 4 keeps
    # GRPO's 64-row actor mini-batch and optimizer steps per row.
    ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-16}
fi
# This changes gradient admission only; delayed ACE keeps its existing objective.
if [[ "$PRESET_TRAINING_MODE" == standard ]]; then
    AUXILIARY_UPDATE_GATE=${AUXILIARY_UPDATE_GATE:-strict_improvement}
else
    AUXILIARY_UPDATE_GATE=${AUXILIARY_UPDATE_GATE:-none}
    [[ "$AUXILIARY_UPDATE_GATE" == none ]] || { echo "Delayed ACE requires AUXILIARY_UPDATE_GATE=none" >&2; exit 2; }
fi
case "$AUXILIARY_UPDATE_GATE" in
    none) AUXILIARY_UPDATE_TAG= ;;
    strict_improvement) AUXILIARY_UPDATE_TAG=_aux_strict_improvement ;;
    *) echo "AUXILIARY_UPDATE_GATE must be none or strict_improvement" >&2; exit 2 ;;
esac
MODEL_VARIANT=${MODEL_VARIANT:-nonthinking}
case "$MODEL_VARIANT" in
    nonthinking) DISABLE_THINKING=True; ACCUMULATE_THINKING=False ;;
    thinking) DISABLE_THINKING=False; ACCUMULATE_THINKING=True ;;
    *) echo "Qwen3-0.6B supports MODEL_VARIANT=nonthinking or thinking" >&2; exit 1 ;;
esac
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}
DEFAULT_MAX_TURNS=5
[[ "$PRESET_TRAINING_MODE" != standard ]] || DEFAULT_MAX_TURNS=20
MAX_TURNS=${MASTERMIND_MAX_TURNS:-$DEFAULT_MAX_TURNS}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-10240}
BASE_VAL_BATCH=128
POLICY_ROLLOUTS=${POLICY_ROLLOUTS:-$PRESET_POLICY_ROLLOUTS}
if [[ "$TRAINING_METHOD" == ace_rl && "$POLICY_ROLLOUTS" != "$PRESET_POLICY_ROLLOUTS" ]]; then
    echo "POLICY_ROLLOUTS=$POLICY_ROLLOUTS conflicts with PRESET=$PRESET; a different group size is an ablation (positional Hydra override and its own RUN_DIR)" >&2
    exit 2
fi
if [[ "$PRESET_TRAINING_MODE" == standard ]]; then
    REFLECTION_SOURCE_SELECTION=${REFLECTION_SOURCE_SELECTION:-random}
else
    REFLECTION_SOURCE_SELECTION=${REFLECTION_SOURCE_SELECTION:-min_reward}
fi
case "$REFLECTION_SOURCE_SELECTION" in
    min_reward) REFLECTION_SOURCE_TAG= ;;
    random) REFLECTION_SOURCE_TAG=_random_reflection_source ;;
    *) echo "REFLECTION_SOURCE_SELECTION must be min_reward or random" >&2; exit 2 ;;
esac
if [[ "$PRESET_TRAINING_MODE" == delayed_joint ]]; then
    METHOD_TAG=local_ace_rl_delayed_joint_rollout${POLICY_ROLLOUTS}_aligned11_seed_oneeval_best_rejection_update
elif [[ "$PRESET" == shared_rollout8 ]]; then
    METHOD_TAG=local_ace_rl_singlemode_one_curator_group_oneeval_rollout${POLICY_ROLLOUTS}${REFLECTOR_ROLLOUTS}${CURATOR_ROLLOUTS}_${POLICY_TRAIN_SOURCE}${REFLECTION_SOURCE_TAG}${AUXILIARY_UPDATE_TAG}
else
    METHOD_TAG=local_ace_rl_singlemode_one_curator_group_oneeval_rollout${POLICY_ROLLOUTS}${REFLECTION_SOURCE_TAG}${AUXILIARY_UPDATE_TAG}
fi
DEFAULT_BATCH=8
DEFAULT_PROMPT=8192
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
export HOST_CPUS=${HOST_CPUS:-${SLURM_CPUS_PER_TASK:-32}}
export PROJECT_NAME=${PROJECT_NAME:-$BENCHMARK}
MODEL_TAG=${MODEL_NAME##*/}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${BENCHMARK}_${METHOD_TAG}_${MODEL_TAG}_${MODEL_VARIANT}_${N_GPUS_PER_NODE}gpu${RUN_STAMP:-}}
export WANDB_PROJECT=${WANDB_PROJECT:-$PROJECT_NAME}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$EXPERIMENT_NAME}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$DEFAULT_BATCH}
DEFAULT_MINI_BATCH=$((DEFAULT_BATCH > 32 ? 32 : DEFAULT_BATCH))
DEFAULT_VAL_BATCH=16
ACE_EFFECTIVE_SOURCE=${POLICY_TRAIN_SOURCE:-seed}
ACE_EFFECTIVE_POLICY_ROLLOUTS=$POLICY_ROLLOUTS
ACE_EFFECTIVE_EVAL_ROLLOUTS=${CANDIDATE_EVAL_ROLLOUTS:-}
for override in "$@"; do
    key=${override%%=*}; key=${key#+}; key=${key#+}
    value=${override#*=}
    # Hydra permits quoted scalar values inside an override argument.
    value=${value#\"}; value=${value%\"}; value=${value#\'}; value=${value%\'}
    case "$key" in
        rllm.ace.policy_train_source) ACE_EFFECTIVE_SOURCE=$value ;;
        actor_rollout_ref.rollout.n) ACE_EFFECTIVE_POLICY_ROLLOUTS=$value ;;
        rllm.ace.candidate_eval_rollouts) ACE_EFFECTIVE_EVAL_ROLLOUTS=$value ;;
    esac
done
case "${ACE_EFFECTIVE_SOURCE,,}" in
    seed) DEFAULT_CANDIDATE_EVAL_ROLLOUTS=1 ;;
    selected_candidate|both) DEFAULT_CANDIDATE_EVAL_ROLLOUTS=0 ;;
    # Candidate-scoring rollouts join the seed group, so any evaluation count is complete.
    seed_and_candidates) DEFAULT_CANDIDATE_EVAL_ROLLOUTS=1 ;;
    *) echo "POLICY_TRAIN_SOURCE must be seed, selected_candidate, both, or seed_and_candidates" >&2; exit 1 ;;
esac
if [[ "${ACE_EFFECTIVE_SOURCE,,}" == seed_and_candidates && ( "$TRAINING_METHOD" != ace_rl || "$PRESET_TRAINING_MODE" != standard ) ]]; then
    echo "POLICY_TRAIN_SOURCE=seed_and_candidates requires standard ACE-RL training" >&2; exit 1
fi
CANDIDATE_EVAL_ROLLOUTS=${CANDIDATE_EVAL_ROLLOUTS:-$DEFAULT_CANDIDATE_EVAL_ROLLOUTS}
ACE_EFFECTIVE_EVAL_ROLLOUTS=${ACE_EFFECTIVE_EVAL_ROLLOUTS:-$DEFAULT_CANDIDATE_EVAL_ROLLOUTS}
if [[ ! "$ACE_EFFECTIVE_POLICY_ROLLOUTS" =~ ^[0-9]+$ || ! "$ACE_EFFECTIVE_EVAL_ROLLOUTS" =~ ^[0-9]+$ ]]; then
    echo "Policy rollouts must be a positive integer and candidate evaluation rollouts a nonnegative integer" >&2
    exit 1
fi
if (( 10#$ACE_EFFECTIVE_POLICY_ROLLOUTS < 1 )); then
    echo "Policy rollouts must be positive" >&2; exit 1
fi
if [[ "$TRAINING_METHOD" != delayed_joint_ppo && "${ACE_EFFECTIVE_SOURCE,,}" != seed && "${ACE_EFFECTIVE_SOURCE,,}" != seed_and_candidates ]] && (( 10#$ACE_EFFECTIVE_EVAL_ROLLOUTS > 0 && 10#$ACE_EFFECTIVE_EVAL_ROLLOUTS < 10#$ACE_EFFECTIVE_POLICY_ROLLOUTS )); then
    echo "POLICY_TRAIN_SOURCE=$ACE_EFFECTIVE_SOURCE requires CANDIDATE_EVAL_ROLLOUTS=0 (all) or at least $ACE_EFFECTIVE_POLICY_ROLLOUTS; got $ACE_EFFECTIVE_EVAL_ROLLOUTS" >&2
    exit 1
fi
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$DEFAULT_PROMPT}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
# Auxiliary prompts need room for a full trajectory and a completion.
ROLLOUT_MAX_MODEL_LEN=${ACE_MAX_MODEL_LEN:-32768}
N_PARALLEL_AGENTS=${N_PARALLEL_AGENTS:-24}
if (( N_PARALLEL_AGENTS > HOST_CPUS - 8 )); then N_PARALLEL_AGENTS=$((HOST_CPUS > 8 ? HOST_CPUS - 8 : 1)); fi
REJECTION_SAMPLE_ENABLE=${REJECTION_SAMPLE_ENABLE:-True}
REJECTION_SAMPLE_MULTIPLIER=${REJECTION_SAMPLE_MULTIPLIER:-2}
if [[ "${REJECTION_SAMPLE_ENABLE,,}" == false ]]; then REJECTION_SAMPLE_MULTIPLIER=1; fi

method_args() {
    args+=(
        "+rllm.ace.g_ref=${REFLECTOR_ROLLOUTS:-4}" "+rllm.ace.g_cur=${CURATOR_ROLLOUTS:-4}"
        +rllm.ace.guidebook_aggregation_level=single "+rllm.ace.singlemode=${SINGLEMODE:-True}"
        "+rllm.ace.singlemode_curator_one_group=${SINGLEMODE_CURATOR_ONE_GROUP:-True}"
        "+rllm.ace.reflection_source_selection=$REFLECTION_SOURCE_SELECTION"
        "+rllm.ace.auxiliary_update_gate=$AUXILIARY_UPDATE_GATE"
        "+rllm.ace.policy_train_source=${POLICY_TRAIN_SOURCE:-seed}"
        "+rllm.ace.selection_mode=${SELECTION_MODE:-best_rejection_update}"
        +rllm.ace.ace_temperature=0.7 "+rllm.ace.training_mode=${ACE_TRAINING_MODE:-standard}"
        "+rllm.ace.policy_max_prompt_length=$MAX_PROMPT_LENGTH" "+rllm.ace.policy_max_response_length=$MAX_RESPONSE_LENGTH"
        "+rllm.ace.reflector_max_prompt_length=${REFLECTOR_MAX_PROMPT_LENGTH:-30720}" +rllm.ace.reflector_max_response_length=2048
        +rllm.ace.curator_max_prompt_length=8192 +rllm.ace.curator_max_response_length=4096 +rllm.ace.max_playbook_length=2048
        "+rllm.ace.max_turns=$MAX_TURNS" "+rllm.ace.flush_playbook_every_n_steps=${FLUSH_PLAYBOOK_EVERY_N_STEPS:-3}"
        "+rllm.ace.local_eval_epochs=${LOCAL_EVAL_EPOCHS:-3}"
        "+rllm.ace.local_task_artifact_root=$TASK_ARTIFACT_ROOT"
        +rllm.ace.local_playbook_log_dir=null +rllm.ace.playbook_snapshot_dir=null
        "+rllm.ace.candidate_eval_batch_size=${CANDIDATE_EVAL_BATCH_SIZE:-$N_PARALLEL_AGENTS}"
        "+rllm.ace.candidate_eval_rollouts=$CANDIDATE_EVAL_ROLLOUTS"
        "+rllm.ace.candidate_eval_max_parallel_agents=${CANDIDATE_EVAL_MAX_PARALLEL_AGENTS:-$N_PARALLEL_AGENTS}"
    )
}
