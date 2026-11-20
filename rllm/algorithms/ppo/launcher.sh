#!/usr/bin/env bash
# Method-owned PPO additions; existing benchmark launchers never source this file.

# delayed_joint_ppo presets are named batch<N>_rollout8: N tasks per step and
# eight complete ACE chains per task. Actor/critic minibatches scale with
# N * rollout.n, preserving four minibatches for the four chain segments.
# The rollout suffix isolates these runs from earlier single-chain checkpoints.
# Call before sourcing the parent presets.
ppo_select_joint_batch() {
    local batch=$1
    PPO_PRESET=${PRESET:-batch${batch}_rollout8}
    [[ "$PPO_PRESET" == "batch${batch}_rollout8" ]] || { echo "Unknown delayed_joint_ppo PRESET: $PPO_PRESET (expected batch${batch}_rollout8)" >&2; exit 2; }
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$batch}
    [[ "$TRAIN_BATCH_SIZE" == "$batch" ]] || { echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE conflicts with PRESET=$PPO_PRESET" >&2; exit 2; }
    : "${ACTOR_PPO_MINI_BATCH_SIZE:=$batch}" "${PPO_MINI_BATCH_SIZE:=$batch}"
}

ppo_defaults() {
    PPO_USER_EXPERIMENT_NAME=${EXPERIMENT_NAME:-}
    PPO_USER_RUN_NAME=${RUN_NAME:-}
    PPO_USER_WANDB_NAME=${WANDB_NAME:-}
    PPO_USER_WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
    : "${GAE_GAMMA:=1.0}" "${GAE_LAMBDA:=0.95}"
    : "${CRITIC_WARMUP:=10}" "${CRITIC_LR:=1e-5}" "${CRITIC_LOSS_AGG_MODE:=token-mean}"
    : "${CRITIC_SAVE_CONTENTS:=['model','optimizer','extra']}" "${KEEP_LAST_N:=0}"
    ADV_ESTIMATOR=gae
    if [[ "$TRAINING_METHOD" == ppo ]]; then
        : "${REJECTION_SAMPLE_ENABLE:=False}" "${REJECTION_SAMPLE_MULTIPLIER:=1}"
    else
        : "${POLICY_ROLLOUTS:=8}" "${REFLECTOR_ROLLOUTS:=1}" "${CURATOR_ROLLOUTS:=1}"
        [[ "$POLICY_ROLLOUTS" == 8 ]] || { echo "POLICY_ROLLOUTS=$POLICY_ROLLOUTS conflicts with PRESET=$PPO_PRESET; use a positional Hydra override and its own RUN_DIR for an ablation" >&2; exit 2; }
        : "${REJECTION_SAMPLE_ENABLE:=False}" "${REJECTION_SAMPLE_MULTIPLIER:=1}"
        : "${GUIDEBOOK_AGGREGATION_LEVEL:=task}" "${SINGLEMODE:=False}" "${SINGLEMODE_CURATOR_ONE_GROUP:=False}"
        : "${POLICY_TRAIN_SOURCE:=both}" "${CANDIDATE_EVAL_ROLLOUTS:=1}"
        : "${SELECTION_MODE:=best}"
        # Parent presets validate their own delayed-joint mode while supplying
        # benchmark assets/runtime. Restore this method's mode afterwards.
        PPO_JOINT_TRAINING_MODE=${TRAINING_MODE:-${ACE_TRAINING_MODE:-joint_ppo}}
        PPO_JOINT_ACE_TRAINING_MODE=${ACE_TRAINING_MODE:-$PPO_JOINT_TRAINING_MODE}
        PPO_JOINT_REFLECTOR_ROLLOUTS=$REFLECTOR_ROLLOUTS
        PPO_JOINT_CURATOR_ROLLOUTS=$CURATOR_ROLLOUTS
        PPO_JOINT_REJECTION_MULTIPLIER=$REJECTION_SAMPLE_MULTIPLIER
        TRAINING_MODE=delayed_joint
        ACE_TRAINING_MODE=delayed_joint
    fi
}

ppo_finish_joint_defaults() {
    [[ "$PPO_JOINT_TRAINING_MODE" == "$PPO_JOINT_ACE_TRAINING_MODE" ]] || {
        echo "TRAINING_MODE and ACE_TRAINING_MODE must agree for delayed_joint_ppo" >&2; exit 2;
    }
    TRAINING_MODE=$PPO_JOINT_TRAINING_MODE
    ACE_TRAINING_MODE=$PPO_JOINT_ACE_TRAINING_MODE
    REFLECTOR_ROLLOUTS=$PPO_JOINT_REFLECTOR_ROLLOUTS
    CURATOR_ROLLOUTS=$PPO_JOINT_CURATOR_ROLLOUTS
    REJECTION_SAMPLE_MULTIPLIER=$PPO_JOINT_REJECTION_MULTIPLIER
}

ppo_set_identity() {
    local identity="${BENCHMARK}_${TRAINING_METHOD}"
    [[ "$PRESET" == "$TRAINING_METHOD" ]] || identity+="_$PRESET"
    identity+="_${MODEL_NAME##*/}_${MODEL_VARIANT}_${N_GPUS_PER_NODE:-2}gpu"
    if [[ "$BENCHMARK" == mini_swe_bench ]]; then
        identity+="_${MINISW_REWARD_MODE}_tasks${MAX_TRAIN_TASKS:-all}_offset${TRAIN_TASK_OFFSET}"
    fi
    identity+="${RUN_STAMP:-}"
    export RUN_NAME=${PPO_USER_RUN_NAME:-$identity}
    export EXPERIMENT_NAME=${PPO_USER_EXPERIMENT_NAME:-$RUN_NAME}
    export WANDB_NAME=${PPO_USER_WANDB_NAME:-$EXPERIMENT_NAME}
    export WANDB_RUN_NAME=${PPO_USER_WANDB_RUN_NAME:-$EXPERIMENT_NAME}
    export APPWORLD_EXPERIMENT_NAME=${APPWORLD_EXPERIMENT_NAME:-$EXPERIMENT_NAME}
}

# Replace a generated override in place, preserving its Hydra add prefix.
# Positional user overrides are appended later by the unchanged launch functions.
ppo_set_arg() {
    local -n ppo_argv=$1
    local wanted=$2 value=$3 item key index normalized
    normalized=${wanted#+}; normalized=${normalized#+}
    for index in "${!ppo_argv[@]}"; do
        item=${ppo_argv[index]}
        key=${item%%=*}; key=${key#+}; key=${key#+}
        if [[ "$key" == "$normalized" ]]; then
            ppo_argv[index]="${item%%=*}=$value"
            return
        fi
    done
    ppo_argv+=("$wanted=$value")
}

ppo_add_args() {
    local array_name=$1
    local -n ppo_settings=$array_name
    local actor_budget=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-${PPO_TOKEN_BUDGET:-${MAX_SEQ_LEN:-32768}}}
    # GPU execution settings belong to the benchmark's existing runtime recipe.
    ppo_set_arg "$array_name" algorithm.adv_estimator gae
    ppo_set_arg "$array_name" trainer.critic_warmup "$CRITIC_WARMUP"
    # Checkpoint evaluation is submitted separately: PPO uses vanilla and
    # delayed joint PPO uses ACE, through the benchmark's eval.slurm on L40S.
    ppo_set_arg "$array_name" trainer.val_before_train False
    ppo_set_arg "$array_name" trainer.test_freq -1
    ppo_set_arg "$array_name" +trainer.val_after_train False
    ppo_set_arg "$array_name" +trainer.save_at_end True
    if [[ "$TRAINING_METHOD" == ppo ]]; then
        ppo_set_arg "$array_name" rllm.rejection_sample.enable "${REJECTION_SAMPLE_ENABLE:-False}"
        ppo_set_arg "$array_name" rllm.rejection_sample.multiplier "${REJECTION_SAMPLE_MULTIPLIER:-1}"
    else
        ppo_set_arg "$array_name" rllm.rejection_sample.enable "$REJECTION_SAMPLE_ENABLE"
        ppo_set_arg "$array_name" rllm.rejection_sample.multiplier "$REJECTION_SAMPLE_MULTIPLIER"
        # Retain the benchmark's ACE validation sampling independently
        # from the training chain count.
        ppo_set_arg "$array_name" actor_rollout_ref.rollout.val_kwargs.n "${VAL_ROLLOUTS:-8}"
        ppo_set_arg "$array_name" rllm.ace.g_ref "$REFLECTOR_ROLLOUTS"
        ppo_set_arg "$array_name" rllm.ace.g_cur "$CURATOR_ROLLOUTS"
        ppo_set_arg "$array_name" rllm.ace.guidebook_aggregation_level "$GUIDEBOOK_AGGREGATION_LEVEL"
        ppo_set_arg "$array_name" rllm.ace.singlemode "$SINGLEMODE"
        ppo_set_arg "$array_name" rllm.ace.singlemode_curator_one_group "$SINGLEMODE_CURATOR_ONE_GROUP"
        ppo_set_arg "$array_name" rllm.ace.policy_train_source "$POLICY_TRAIN_SOURCE"
        ppo_set_arg "$array_name" rllm.ace.selection_mode "$SELECTION_MODE"
        ppo_set_arg "$array_name" rllm.ace.training_mode "$TRAINING_MODE"
        ppo_set_arg "$array_name" rllm.ace.candidate_eval_rollouts "$CANDIDATE_EVAL_ROLLOUTS"
        ppo_set_arg "$array_name" rllm.ace.local_task_artifact_root "$TASK_ARTIFACT_ROOT"
    fi
    if [[ "$BENCHMARK" == mini_swe_bench ]]; then
        ppo_set_arg "$array_name" trainer.episode_log_dir "$TRAJECTORY_LOG_DIR"
    fi
    ppo_settings+=(
        "algorithm.gamma=$GAE_GAMMA" "algorithm.lam=$GAE_LAMBDA"
        "critic.enable=True" "critic.strategy=fsdp2" "critic.model.path=$MODEL_NAME"
        "critic.model.trust_remote_code=True" "critic.optim.lr=$CRITIC_LR"
        "critic.loss_agg_mode=$CRITIC_LOSS_AGG_MODE"
        "critic.ppo_max_token_len_per_gpu=$actor_budget"
        "+critic.ppo_infer_max_token_len_per_gpu=$actor_budget"
        "critic.model.enable_gradient_checkpointing=${ACTOR_ENABLE_GRADIENT_CHECKPOINTING:-True}"
        "critic.model.enable_activation_offload=${ACTOR_ENABLE_ACTIVATION_OFFLOAD:-True}"
        "critic.fsdp.param_offload=True" "critic.fsdp.optimizer_offload=True"
        "critic.fsdp.forward_prefetch=${ACTOR_FSDP_FORWARD_PREFETCH:-False}"
        "actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra']"
        "actor_rollout_ref.actor.checkpoint.load_contents=['model','optimizer','extra']"
        "critic.checkpoint.save_contents=$CRITIC_SAVE_CONTENTS"
        "critic.checkpoint.load_contents=$CRITIC_SAVE_CONTENTS"
    )
    # KEEP_LAST_N=0 (default) keeps every saved checkpoint for evaluation.
    if (( KEEP_LAST_N > 0 )); then
        ppo_settings+=("trainer.max_actor_ckpt_to_keep=$KEEP_LAST_N" "trainer.max_critic_ckpt_to_keep=$KEEP_LAST_N")
    fi
}
