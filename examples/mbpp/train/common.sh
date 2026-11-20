#!/usr/bin/env bash
# Benchmark-local runtime and common Hydra arguments; sourced by method runners.
RLLM_TRAIN_ARGS=("$@")  # positional Hydra overrides, replayed by chained follow-up jobs
# MBPP_FEEDBACK=hidden replaces each failed assertion and traceback in the
# policy's test feedback with a neutral label and a one-line outcome; reward is
# unchanged. The default, full, keeps the historical feedback and command. A
# hidden run is a different environment: it needs its own RUN_DIR naming
# hiddenfb, and its experiment name is tagged _hiddenfb.
MBPP_FEEDBACK=${MBPP_FEEDBACK:-full}
case "$MBPP_FEEDBACK" in
    full) ;;
    hidden)
        [[ -n "${RUN_DIR:-}" ]] || { echo "MBPP_FEEDBACK=hidden changes the environment; set its own RUN_DIR (its path must contain hiddenfb)" >&2; exit 2; }
        [[ "$RUN_DIR" == *hiddenfb* ]] || { echo "MBPP_FEEDBACK=hidden requires a RUN_DIR whose path contains hiddenfb, got: $RUN_DIR" >&2; exit 2; } ;;
    *) echo "MBPP_FEEDBACK must be full or hidden, got: $MBPP_FEEDBACK" >&2; exit 2 ;;
esac
for override in ${RLLM_TRAIN_ARGS[@]+"${RLLM_TRAIN_ARGS[@]}"}; do
    key=${override%%=*}; key=${key#+}; key=${key#+}
    [[ "$key" != rllm.env.env_args.feedback ]] || { echo "Select MBPP feedback with MBPP_FEEDBACK, not a Hydra override" >&2; exit 2; }
done
export MBPP_FEEDBACK

mbpp_feedback_identity() {
    [[ "$MBPP_FEEDBACK" == hidden ]] || return 0
    local name
    for name in EXPERIMENT_NAME WANDB_NAME WANDB_RUN_NAME; do
        [[ -z "${!name:-}" || "${!name}" == *hiddenfb* ]] || export "$name=${!name}_hiddenfb"
    done
}

start_runtime() {
    source "$RLLM_DIR/scripts/training_runtime.sh"
}

common_args() {
    mbpp_feedback_identity
    PPO_TOKEN_BUDGET=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$((ROLLOUT_MAX_MODEL_LEN + 4096))}
    args=(
        +ray_init.include_dashboard=False +ray_init.log_to_driver=False
        "+ray_init._temp_dir=$RAY_TMPDIR" "+ray_init.num_cpus=${RAY_NUM_CPUS:-$HOST_CPUS}"
        "+ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY_BYTES:-17179869184}"
        algorithm.adv_estimator=grpo data.train_files=null data.val_files=null
        "data.train_batch_size=$TRAIN_BATCH_SIZE" "data.val_batch_size=${VAL_BATCH_SIZE:-$DEFAULT_VAL_BATCH}"
        "data.max_prompt_length=$MAX_PROMPT_LENGTH" "data.max_response_length=$MAX_RESPONSE_LENGTH"
        data.trust_remote_code=True "actor_rollout_ref.model.path=$MODEL_NAME"
        actor_rollout_ref.model.trust_remote_code=True actor_rollout_ref.hybrid_engine=True
        actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.ref.strategy=fsdp2
        actor_rollout_ref.actor.loss_agg_mode=token-mean
        "actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE:-$DEFAULT_MINI_BATCH}"
        actor_rollout_ref.actor.use_dynamic_bsz=True "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_TOKEN_BUDGET"
        actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.clip_ratio_high=0.28
        actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
        actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.model.enable_activation_offload=True
        actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.actor.fsdp_config.forward_prefetch=False
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.data_parallel_size=1
        actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async
        "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}"
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config={backend:FLASH_ATTN,flash_attn_version:2}'
        actor_rollout_ref.rollout.temperature=0.7
        "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
        "actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE:-True}"
        actor_rollout_ref.rollout.disable_log_stats=False
        "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS:-$ROLLOUT_MAX_MODEL_LEN}"
        "actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN"
        "actor_rollout_ref.rollout.n=$POLICY_ROLLOUTS" "actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUTS:-8}"
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 actor_rollout_ref.rollout.val_kwargs.top_p=0.8 actor_rollout_ref.rollout.val_kwargs.top_k=20
        actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$PPO_TOKEN_BUDGET}"
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$PPO_TOKEN_BUDGET}"
        actor_rollout_ref.actor.entropy_coeff=0 algorithm.kl_ctrl.kl_coef=0.001
        rllm.filter_token_mismatch=True fireworks.return_token_ids=False rllm.mask_truncated_samples=False
        trainer.critic_warmup=0 "trainer.logger=${TRAINER_LOGGERS:-['console','wandb']}"
        "trainer.project_name=$PROJECT_NAME" "trainer.experiment_name=$EXPERIMENT_NAME"
        "trainer.default_local_dir=$CHECKPOINT_DIR"
        "trainer.val_before_train=${TRAINER_VAL_BEFORE_TRAIN:-False}"
        ${TRAINER_VAL_AFTER_TRAIN:+"+trainer.val_after_train=$TRAINER_VAL_AFTER_TRAIN"}
        "trainer.n_gpus_per_node=$N_GPUS_PER_NODE" trainer.nnodes=1
        "trainer.save_freq=${TRAINER_SAVE_FREQ:-10}" "trainer.test_freq=${TRAINER_TEST_FREQ:--1}"
        trainer.default_hdfs_dir=null "trainer.resume_mode=${TRAINER_RESUME_MODE:-auto}"
        "trainer.episode_log_dir=$EPISODE_LOG_DIR"
        "rllm.rejection_sample.enable=$REJECTION_SAMPLE_ENABLE" "rllm.rejection_sample.multiplier=$REJECTION_SAMPLE_MULTIPLIER"
        "rllm.agent.max_steps=$MAX_TURNS" "rllm.agent.trajectory_hard_timeout=${TRAJECTORY_HARD_TIMEOUT:-1800}"
        +rllm.agent.trajectory_reward_aggregation=max rllm.stepwise_advantage.enable=False
        "rllm.disable_thinking=$DISABLE_THINKING" "rllm.accumulate_reasoning=$ACCUMULATE_THINKING"
        "+rllm.agent.engine_args.n_parallel_agents=$N_PARALLEL_AGENTS"
        "+rllm.agent.agent_args.max_steps=$MAX_TURNS" "+rllm.env.env_args.max_turns=$MAX_TURNS"
        "trainer.total_epochs=${TRAINER_TOTAL_EPOCHS:-100}"
        "trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-301}"
    )
    args+=(+rllm.agent.agent_args.accumulate_history=True "+rllm.agent.agent_args.accumulate_thinking=$ACCUMULATE_THINKING")
        export MBPP_MAX_TEST_WORKERS=${MBPP_MAX_TEST_WORKERS:-$N_PARALLEL_AGENTS}
        args+=("+rllm.agent.engine_args.max_workers=${MBPP_ENGINE_MAX_WORKERS:-$N_PARALLEL_AGENTS}" +rllm.env.env_args.timeout=10 rllm.turn_level_cutoff_mode=max)
        # Full feedback keeps the historical command and recorded run identity.
        [[ "$MBPP_FEEDBACK" == full ]] || args+=("+rllm.env.env_args.feedback=$MBPP_FEEDBACK")
}

launch() {
    [[ -z "${TRAINER_RESUME_FROM_PATH:-}" ]] || args+=("trainer.resume_from_path=$TRAINER_RESUME_FROM_PATH")
    rllm_training_launch "$TRAIN_MODULE" "${args[@]}" "$@"
}
