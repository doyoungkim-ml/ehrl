#!/usr/bin/env bash
# Model and scientific settings for this method.
if [[ -z "${PRESET:-}" && "${TRAINING_MODE:-standard}" == delayed_joint ]]; then
    PRESET=delayed_oneeval_rollout8
fi
# Main presets use eight policy rollouts per task; auxiliary group sizes are separate.
PRESET=${PRESET:-oneeval_rollout8}
case "$PRESET" in
    oneeval_rollout8) PRESET_TRAINING_MODE=standard; PRESET_POLICY_ROLLOUTS=8 ;;
    delayed_oneeval_rollout8) PRESET_TRAINING_MODE=delayed_joint; PRESET_POLICY_ROLLOUTS=8 ;;
    # GRPO budget: 4 seed rollouts plus at most 4 candidate-scoring rollouts per
    # task (2 reflections, one 3-sample curator group), all trained as actor rows.
    shared_rollout8) PRESET_TRAINING_MODE=standard; PRESET_POLICY_ROLLOUTS=4 ;;
    *) echo "Unknown ACE-RL PRESET: $PRESET (oneeval_rollout8, delayed_oneeval_rollout8, shared_rollout8)" >&2; exit 2 ;;
esac
TRAINING_MODE=${TRAINING_MODE:-$PRESET_TRAINING_MODE}
case "$TRAINING_MODE" in
    standard|delayed_joint) ;;
    *) echo "TRAINING_MODE must be standard or delayed_joint" >&2; exit 2 ;;
esac
[[ "$TRAINING_MODE" == "$PRESET_TRAINING_MODE" ]] || { echo "TRAINING_MODE conflicts with PRESET=$PRESET" >&2; exit 2; }
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
    nonthinking) DEFAULT_MODEL=Qwen/Qwen3-4B; DISABLE_THINKING=True; ACCUMULATE_REASONING=False ;;
    thinking) DEFAULT_MODEL=Qwen/Qwen3-4B; DISABLE_THINKING=False; ACCUMULATE_REASONING=True ;;
    instruct) DEFAULT_MODEL=Qwen/Qwen3-4B-Instruct-2507; DISABLE_THINKING=True; ACCUMULATE_REASONING=False ;;
    thinking_2507) DEFAULT_MODEL=Qwen/Qwen3-4B-Thinking-2507; DISABLE_THINKING=False; ACCUMULATE_REASONING=True ;;
    *) echo "MODEL_VARIANT must be nonthinking, thinking, instruct, or thinking_2507" >&2; exit 2 ;;
esac
export MODEL_NAME=${MODEL_NAME:-$DEFAULT_MODEL}
export MINISW_REWARD_MODE=${MINISW_REWARD_MODE:-shaped}
export MAX_VAL_TASKS=${MAX_VAL_TASKS-100}
export TRAIN_TASK_OFFSET=${TRAIN_TASK_OFFSET:-0}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_SEQ_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
DEFAULT_MAX_STEPS=10
[[ "$PRESET_TRAINING_MODE" != standard ]] || DEFAULT_MAX_STEPS=20
MAX_STEPS=${MINISW_MAX_STEPS:-$DEFAULT_MAX_STEPS}
PARALLEL_TASKS=${MINISW_N_PARALLEL_TASKS:-8}
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
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
# VeRL counts mini-batches in prompts x rollout.n rows; keep GRPO's 64 rows.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$((64 / POLICY_ROLLOUTS))}
PROJECT_NAME=${PROJECT_NAME:-mini_swe_bench}
TRAINING_MODE=${TRAINING_MODE:-standard}
case "$TRAINING_MODE" in
    standard)
        REFLECTOR_ROLLOUTS=4
        CURATOR_ROLLOUTS=4
        POLICY_TRAIN_SOURCE=seed
        if [[ "$PRESET" == shared_rollout8 ]]; then
            REFLECTOR_ROLLOUTS=2
            CURATOR_ROLLOUTS=3
            POLICY_TRAIN_SOURCE=seed_and_candidates
        fi
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-mini_swe_bench_harness_${MODEL_NAME##*/}_${MODEL_VARIANT}_${MINISW_REWARD_MODE}_tasks${MAX_TRAIN_TASKS:-all}_offset${TRAIN_TASK_OFFSET}_2gpu_singlemode1curgroup_rollout${POLICY_ROLLOUTS}${REFLECTOR_ROLLOUTS}${CURATOR_ROLLOUTS}_${POLICY_TRAIN_SOURCE}_oneeval_best_rejection_update${REFLECTION_SOURCE_TAG}${AUXILIARY_UPDATE_TAG}}
        ;;
    delayed_joint)
        # One aligned reflector/curator pair per policy rollout, not independent sampling groups.
        REFLECTOR_ROLLOUTS=1
        CURATOR_ROLLOUTS=1
        # Delayed-joint PPO inherits this mode and trains complete chains (both).
        POLICY_TRAIN_SOURCE=${POLICY_TRAIN_SOURCE:-seed}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-mini_swe_bench_delayed_joint_${MODEL_NAME##*/}_${MODEL_VARIANT}_${MINISW_REWARD_MODE}_tasks${MAX_TRAIN_TASKS:-all}_offset${TRAIN_TASK_OFFSET}_2gpu_rollout${POLICY_ROLLOUTS}_aligned11_seed_oneeval_best_rejection_update}
        ;;
    *) echo "TRAINING_MODE must be standard or delayed_joint" >&2; exit 2 ;;
esac

method_args() {
    MODULE=examples.mini_swe_bench.train.ace_rl.train
    ARGS+=(
        "rllm.workflow.use_workflow=False"
        "rllm.agent.max_steps=$MAX_STEPS"
        "rllm.agent.trajectory_hard_timeout=${WORKFLOW_TIMEOUT:-7200}"
        "+rllm.agent.engine_args.n_parallel_agents=$PARALLEL_TASKS"
        "+rllm.env.env_args.max_steps=$MAX_STEPS"
        "+rllm.env.env_args.step_timeout=${MINISW_STEP_TIMEOUT:-120}"
        "+rllm.env.env_args.reward_timeout=${MINISW_REWARD_TIMEOUT:-600}"
        "+rllm.env.env_args.reward_mode=$MINISW_REWARD_MODE"
        "+rllm.ace.g_ref=$REFLECTOR_ROLLOUTS" "+rllm.ace.g_cur=$CURATOR_ROLLOUTS"
        "+rllm.ace.singlemode=True" "+rllm.ace.singlemode_curator_one_group=True"
        "+rllm.ace.reflection_source_selection=$REFLECTION_SOURCE_SELECTION"
        "+rllm.ace.auxiliary_update_gate=$AUXILIARY_UPDATE_GATE"
        "+rllm.ace.guidebook_aggregation_level=single"
        "+rllm.ace.policy_train_source=$POLICY_TRAIN_SOURCE"
        "+rllm.ace.selection_mode=best_rejection_update"
        "+rllm.ace.training_mode=$TRAINING_MODE" "+rllm.ace.ace_temperature=0.7"
        "+rllm.ace.policy_max_prompt_length=$MAX_PROMPT_LENGTH"
        "+rllm.ace.policy_max_response_length=$MAX_RESPONSE_LENGTH"
        "+rllm.ace.reflector_max_prompt_length=30720"
        "+rllm.ace.reflector_max_response_length=2048"
        "+rllm.ace.reflector_max_history_tokens=20000"
        "+rllm.ace.curator_max_prompt_length=8192"
        "+rllm.ace.curator_max_response_length=4096"
        "+rllm.ace.max_playbook_length=2048" "+rllm.ace.max_turns=$MAX_STEPS"
        "+rllm.ace.flush_playbook_every_n_steps=${FLUSH_PLAYBOOK_EVERY_N_STEPS:-3}" "+rllm.ace.local_eval_epochs=3"
        "+rllm.ace.candidate_eval_batch_size=$TRAIN_BATCH_SIZE"
        "+rllm.ace.candidate_eval_rollouts=1"
        "+rllm.ace.candidate_eval_max_parallel_agents=$PARALLEL_TASKS"
        "+rllm.ace.local_task_artifact_root=$TASK_ARTIFACT_ROOT"
        "+rllm.ace.local_playbook_log_dir=null" "+rllm.ace.playbook_snapshot_dir=null"
    )
}
