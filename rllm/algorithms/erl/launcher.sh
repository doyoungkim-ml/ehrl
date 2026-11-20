#!/usr/bin/env bash
# Original ERL recipe. Benchmark presets supply only task/model/runtime bindings.
# Maintained presets use eight initial policy attempts per task.
erl_defaults() {
    local rollouts=$1
    ERL_PRESET=${PRESET:-rollout$rollouts}
    [[ "$ERL_PRESET" == "rollout$rollouts" ]] || { echo "ERL PRESET must be rollout$rollouts" >&2; exit 2; }
    POLICY_ROLLOUTS=${POLICY_ROLLOUTS:-$rollouts}
    [[ "$POLICY_ROLLOUTS" == "$rollouts" ]] || { echo "POLICY_ROLLOUTS=$POLICY_ROLLOUTS conflicts with PRESET=$ERL_PRESET" >&2; exit 2; }
    ERL_USER_EXPERIMENT_NAME=${EXPERIMENT_NAME:-}
    ERL_USER_RUN_NAME=${RUN_NAME:-}
    ERL_USER_WANDB_NAME=${WANDB_NAME:-}
    ERL_USER_WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
    : "${TRAIN_BATCH_SIZE:=8}"
    : "${ACTOR_PPO_MINI_BATCH_SIZE:=32}" "${PPO_MINI_BATCH_SIZE:=32}"
    : "${ERL_UPDATER_TEMPERATURE:=0.7}" "${ERL_UPDATER_TOP_P:=0.9}"
    : "${ERL_UPDATER_MAX_RESPONSE_LENGTH:=8192}" "${ERL_SUCCESS_REWARD_THRESHOLD:=1.0}"
    : "${ERL_VALIDATION_MODE:=plain}" "${VAL_ROLLOUTS:=1}"
    : "${LOSS_AGG_MODE:=seq-mean-token-sum}" "${CRITIC_WARMUP:=0}"
    : "${REJECTION_SAMPLE_ENABLE:=False}" "${REJECTION_SAMPLE_MULTIPLIER:=1}"
    : "${KEEP_LAST_N:=0}"
}

erl_set_identity() {
    PRESET=$ERL_PRESET
    local identity="${BENCHMARK}_erl_${ERL_PRESET}_${MODEL_NAME##*/}_${MODEL_VARIANT}_${N_GPUS_PER_NODE:-2}gpu${RUN_STAMP:-}"
    export RUN_NAME=${ERL_USER_RUN_NAME:-$identity}
    export EXPERIMENT_NAME=${ERL_USER_EXPERIMENT_NAME:-$RUN_NAME}
    export WANDB_NAME=${ERL_USER_WANDB_NAME:-$EXPERIMENT_NAME}
    export WANDB_RUN_NAME=${ERL_USER_WANDB_RUN_NAME:-$EXPERIMENT_NAME}
    export APPWORLD_EXPERIMENT_NAME=${APPWORLD_EXPERIMENT_NAME:-$EXPERIMENT_NAME}
}

erl_set_arg() {
    local -n erl_argv=$1
    local wanted=$2 value=$3 key index normalized
    normalized=${wanted#+}; normalized=${normalized#+}
    for index in "${!erl_argv[@]}"; do
        key=${erl_argv[index]%%=*}; key=${key#+}; key=${key#+}
        if [[ "$key" == "$normalized" ]]; then
            erl_argv[index]="${erl_argv[index]%%=*}=$value"
            return
        fi
    done
    erl_argv+=("$wanted=$value")
}

erl_add_args() {
    local array_name=$1
    local -n erl_settings=$array_name
    # Reflection includes the observed attempt and the previous revised prompt.
    # Reserve role-specific context without shortening the plain task budget.
    local remembered_budget=$MAX_PROMPT_LENGTH
    (( remembered_budget >= ERL_UPDATER_MAX_RESPONSE_LENGTH )) || remembered_budget=$ERL_UPDATER_MAX_RESPONSE_LENGTH
    local updater_prompt=${ERL_UPDATER_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + remembered_budget + 2048))}
    local retry_prompt=${ERL_RETRY_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + ERL_UPDATER_MAX_RESPONSE_LENGTH))}
    local model_window=$((updater_prompt + ERL_UPDATER_MAX_RESPONSE_LENGTH))
    local retry_window=$((retry_prompt + MAX_RESPONSE_LENGTH))
    (( model_window >= retry_window )) || model_window=$retry_window
    local existing_window=${ROLLOUT_MAX_MODEL_LEN:-${MAX_SEQ_LEN:-0}}
    (( model_window >= existing_window )) || model_window=$existing_window
    erl_set_arg "$array_name" actor_rollout_ref.rollout.max_model_len "$model_window"
    erl_set_arg "$array_name" actor_rollout_ref.rollout.max_num_batched_tokens "$model_window"
    erl_set_arg "$array_name" actor_rollout_ref.actor.ppo_max_token_len_per_gpu "$model_window"
    erl_set_arg "$array_name" actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu "$model_window"
    erl_set_arg "$array_name" actor_rollout_ref.ref.log_prob_max_token_len_per_gpu "$model_window"
    # GPU execution settings belong to the benchmark's existing runtime recipe.
    erl_set_arg "$array_name" algorithm.adv_estimator grpo
    erl_set_arg "$array_name" rllm.workflow.use_workflow False
    erl_set_arg "$array_name" rllm.stepwise_advantage.enable False
    erl_set_arg "$array_name" rllm.rejection_sample.enable False
    erl_set_arg "$array_name" rllm.rejection_sample.multiplier 1
    erl_set_arg "$array_name" actor_rollout_ref.actor.loss_agg_mode seq-mean-token-sum
    erl_set_arg "$array_name" actor_rollout_ref.actor.use_kl_loss True
    erl_set_arg "$array_name" actor_rollout_ref.actor.kl_loss_coef 0.001
    erl_set_arg "$array_name" actor_rollout_ref.actor.kl_loss_type low_var_kl
    erl_set_arg "$array_name" actor_rollout_ref.actor.clip_ratio_high 0.28
    erl_set_arg "$array_name" actor_rollout_ref.actor.ppo_mini_batch_size "$ACTOR_PPO_MINI_BATCH_SIZE"
    erl_set_arg "$array_name" actor_rollout_ref.rollout.val_kwargs.n "$VAL_ROLLOUTS"
    erl_set_arg "$array_name" actor_rollout_ref.rollout.val_kwargs.do_sample True
    erl_set_arg "$array_name" actor_rollout_ref.rollout.val_kwargs.temperature 0.7
    erl_set_arg "$array_name" actor_rollout_ref.rollout.val_kwargs.top_p 1.0
    erl_set_arg "$array_name" actor_rollout_ref.rollout.val_kwargs.top_k -1
    erl_set_arg "$array_name" trainer.critic_warmup 0
    # Comparison scores come from the canonical eval/vanilla entrypoint.
    erl_set_arg "$array_name" trainer.val_before_train False
    erl_set_arg "$array_name" trainer.test_freq -1
    erl_set_arg "$array_name" +trainer.val_after_train False
    erl_set_arg "$array_name" +trainer.save_at_end True
    erl_set_arg "$array_name" trainer.episode_log_dir "$EPISODE_LOG_DIR"
    erl_settings+=(
        "critic.enable=False"
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]"
        "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]"
        "+rllm.erl.upstream_commit=20cd6460753973b9a5834769e2feed02736f44c6"
        "+rllm.erl.success_reward_threshold=$ERL_SUCCESS_REWARD_THRESHOLD"
        "+rllm.erl.updater_temperature=$ERL_UPDATER_TEMPERATURE"
        "+rllm.erl.updater_top_p=$ERL_UPDATER_TOP_P"
        "+rllm.erl.updater_max_response_length=$ERL_UPDATER_MAX_RESPONSE_LENGTH"
        "+rllm.erl.updater_max_prompt_length=$updater_prompt"
        "+rllm.erl.retry_max_prompt_length=$retry_prompt"
        "+rllm.erl.validation_mode=$ERL_VALIDATION_MODE"
    )
    # KEEP_LAST_N=0 (default) keeps every saved checkpoint for evaluation.
    if (( KEEP_LAST_N > 0 )); then
        erl_settings+=("trainer.max_actor_ckpt_to_keep=$KEEP_LAST_N")
    fi
}
