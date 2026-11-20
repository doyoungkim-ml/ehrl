#!/usr/bin/env bash
# Each preset keeps a distinct controller/sampling ablation. Environment overrides win.
# Main presets use eight policy rollouts per task; auxiliary group sizes are separate.
PRESET=${PRESET:-oneeval_rollout8}
case "$PRESET" in
    oneeval_rollout8|delayed_oneeval_rollout8)
        PRESET_POLICY_ROLLOUTS=8
        : "${MODEL_VARIANT:=nonthinking}" "${POLICY_ROLLOUTS:=$PRESET_POLICY_ROLLOUTS}"
        if [[ "$TRAINING_METHOD" == ace_rl && "$POLICY_ROLLOUTS" != "$PRESET_POLICY_ROLLOUTS" ]]; then
            echo "POLICY_ROLLOUTS=$POLICY_ROLLOUTS conflicts with PRESET=$PRESET; a different group size is an ablation (positional Hydra override and its own RUN_DIR)" >&2
            exit 2
        fi
        : "${SINGLEMODE:=True}" "${SINGLEMODE_CURATOR_ONE_GROUP:=True}"
        : "${GUIDEBOOK_AGGREGATION_LEVEL:=single}" "${POLICY_TRAIN_SOURCE:=seed}"
        : "${CANDIDATE_EVAL_ROLLOUTS:=1}" "${SELECTION_MODE:=best_rejection_update}"
        : "${TRAJECTORY_HARD_TIMEOUT:=1200}" "${REFLECTOR_MAX_PROMPT_LENGTH:=30720}"
        : "${REFLECTOR_MAX_HISTORY_TOKENS:=20000}"
        if [[ "$PRESET" == delayed_oneeval_rollout8 ]]; then
            : "${TRAINING_MODE:=delayed_joint}" "${REFLECTOR_ROLLOUTS:=1}" "${CURATOR_ROLLOUTS:=1}"
            [[ "$TRAINING_MODE" == delayed_joint ]] || { echo "delayed_oneeval_rollout8 requires TRAINING_MODE=delayed_joint" >&2; exit 2; }
        else
            [[ "${TRAINING_MODE:-standard}" == standard ]] || { echo "Use PRESET=delayed_oneeval_rollout8 for aligned delayed-joint training" >&2; exit 2; }
        fi
        ;;
    shared_rollout8)
        # GRPO budget: 4 seed rollouts plus at most 4 candidate-scoring rollouts
        # per task (one reflection source, 2 reflections, one 3-sample curator
        # group). All of them train the actor in one GRPO group.
        : "${MODEL_VARIANT:=nonthinking}" "${POLICY_ROLLOUTS:=4}" "${REFLECTOR_ROLLOUTS:=2}" "${CURATOR_ROLLOUTS:=3}"
        : "${SINGLEMODE:=True}" "${SINGLEMODE_CURATOR_ONE_GROUP:=True}"
        : "${GUIDEBOOK_AGGREGATION_LEVEL:=single}" "${POLICY_TRAIN_SOURCE:=seed_and_candidates}"
        : "${CANDIDATE_EVAL_ROLLOUTS:=1}" "${SELECTION_MODE:=best_rejection_update}"
        : "${TRAJECTORY_HARD_TIMEOUT:=1200}" "${REFLECTOR_MAX_PROMPT_LENGTH:=30720}"
        : "${REFLECTOR_MAX_HISTORY_TOKENS:=20000}"
        # VeRL counts mini-batches in prompts x rollout.n rows; 16 x 4 keeps
        # GRPO's 64-row actor mini-batch and optimizer steps per row.
        : "${ACTOR_PPO_MINI_BATCH_SIZE:=16}"
        [[ "${TRAINING_MODE:-standard}" == standard ]] || { echo "shared_rollout8 requires TRAINING_MODE=standard" >&2; exit 2; }
        ;;
    standard)
        : "${POLICY_ROLLOUTS:=8}" "${POLICY_TRAIN_SOURCE:=seed}"
        : "${ROLLOUT_FREE_CACHE_ENGINE:=False}" "${ROLLOUT_ENFORCE_EAGER:=True}"
        ;;
    singlemode|one_curator_group|one_curator_group_no_rejection|seed_oneeval|seed_oneeval_no_rejection)
        : "${POLICY_ROLLOUTS:=8}" "${SINGLEMODE:=True}" "${GUIDEBOOK_AGGREGATION_LEVEL:=single}"
        if [[ "$PRESET" != singlemode ]]; then
            : "${SINGLEMODE_CURATOR_ONE_GROUP:=True}" "${TRAJECTORY_HARD_TIMEOUT:=1200}"
        fi
        case "$PRESET" in
            seed_oneeval*)
                : "${POLICY_TRAIN_SOURCE:=seed}" "${CANDIDATE_EVAL_ROLLOUTS:=1}"
                : "${REFLECTOR_MAX_PROMPT_LENGTH:=30720}" "${REFLECTOR_MAX_HISTORY_TOKENS:=20000}"
                ;;
            one_curator_group)
                if [[ "${MODEL_VARIANT:-thinking}" == instruct ]]; then
                    : "${ROLLOUT_ENFORCE_EAGER:=True}"
                else
                    : "${REFLECTOR_MAX_PROMPT_LENGTH:=30720}" "${REFLECTOR_MAX_HISTORY_TOKENS:=20000}"
                fi
                ;;
        esac
        if [[ "$PRESET" == *_no_rejection ]]; then
            : "${REJECTION_SAMPLE_ENABLE:=False}" "${REJECTION_SAMPLE_MULTIPLIER:=1}"
        fi
        ;;
    iterative_singlemode)
        : "${TRAINING_MODE:=iterative}" "${POLICY_ROLLOUTS:=8}" "${POLICY_TRAIN_SOURCE:=seed}"
        : "${SINGLEMODE:=True}" "${GUIDEBOOK_AGGREGATION_LEVEL:=single}"
        : "${ROLLOUT_FREE_CACHE_ENGINE:=False}" "${ROLLOUT_ENFORCE_EAGER:=True}"
        : "${ITERATIVE_STEPS_PER_ROLE:=10}" "${ITERATIVE_SCHEDULE:=[policy,reflector,curator]}"
        ;;
    delayed_joint)
        : "${TRAINING_MODE:=delayed_joint}" "${POLICY_ROLLOUTS:=8}" "${REFLECTOR_ROLLOUTS:=1}" "${CURATOR_ROLLOUTS:=1}"
        ;;
    *) echo "Unknown ACE-RL PRESET: $PRESET (see train/README.md)" >&2; exit 1 ;;
esac
: "${MODEL_VARIANT:=thinking}" "${TRAINING_MODE:=standard}"
: "${VLLM_ATTENTION_BACKEND:=FLASHINFER}" "${TRAJECTORY_HARD_TIMEOUT:=2400}"
: "${REFLECTOR_ROLLOUTS:=4}" "${CURATOR_ROLLOUTS:=4}"
: "${GUIDEBOOK_AGGREGATION_LEVEL:=task}" "${SINGLEMODE:=False}" "${SINGLEMODE_CURATOR_ONE_GROUP:=False}"
: "${POLICY_TRAIN_SOURCE:=selected_candidate}" "${CANDIDATE_EVAL_ROLLOUTS:=0}" "${SELECTION_MODE:=best_rejection}"
: "${REFLECTOR_MAX_PROMPT_LENGTH:=32768}" "${FLUSH_PLAYBOOK_EVERY_N_STEPS:=3}" "${LOCAL_EVAL_EPOCHS:=3}"
if [[ "$TRAINING_MODE" == standard ]]; then
    : "${AUXILIARY_UPDATE_GATE:=strict_improvement}" "${APPWORLD_MAX_TURNS:=20}"
else
    : "${AUXILIARY_UPDATE_GATE:=none}"
    [[ "$AUXILIARY_UPDATE_GATE" == none ]] || { echo "Nonstandard ACE requires AUXILIARY_UPDATE_GATE=none" >&2; exit 2; }
fi
case "$AUXILIARY_UPDATE_GATE" in
    none) AUXILIARY_UPDATE_TAG= ;;
    strict_improvement) AUXILIARY_UPDATE_TAG=_aux_strict_improvement ;;
    *) echo "AUXILIARY_UPDATE_GATE must be none or strict_improvement" >&2; exit 2 ;;
esac
if [[ "$TRAINING_MODE" == standard ]]; then
    : "${REFLECTION_SOURCE_SELECTION:=random}"
else
    : "${REFLECTION_SOURCE_SELECTION:=min_reward}"
fi
case "$REFLECTION_SOURCE_SELECTION" in
    min_reward|random) ;;
    *) echo "REFLECTION_SOURCE_SELECTION must be min_reward or random" >&2; exit 2 ;;
esac
: "${APPWORLD_REFLECTOR_SYSTEM_PROMPT:=You are an expert AppWorld trajectory analyst. Respond with JSON only.}"
: "${APPWORLD_CURATOR_SYSTEM_PROMPT:=You are a master curator of AppWorld task playbooks. Respond with JSON only.}"
export VLLM_ATTENTION_BACKEND
