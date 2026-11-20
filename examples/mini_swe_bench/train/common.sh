#!/usr/bin/env bash
# MiniSWEBench runtime and common Hydra plumbing.
RLLM_TRAIN_ARGS=("$@")  # positional Hydra overrides, replayed by chained follow-up jobs
start_runtime() {
    source "$RLLM_DIR/scripts/training_runtime.sh"
    if [[ "${RLLM_DRY_RUN:-0}" != 1 ]]; then
        [[ "$N_GPUS_PER_NODE" == 2 ]] || { echo "MiniSWEBench training requires two GPUs" >&2; exit 1; }
        case "$RLLM_GPU_FAMILY" in
            h100|h200) ;;
            *) echo "MiniSWEBench training requires H100 or H200 GPUs (detected: $RLLM_GPU_FAMILY)" >&2; exit 1 ;;
        esac
        command -v apptainer >/dev/null || command -v singularity >/dev/null || { echo "Apptainer or Singularity is required" >&2; exit 1; }
        "$PYTHON_BIN" -c 'from swegym.harness.grading import get_eval_report; from swebench.harness.grading import get_eval_report'
        export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$CACHE_DIR/apptainer}
        export APPTAINER_SIF_DIR=${APPTAINER_SIF_DIR:-$CACHE_DIR/sif}
        export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$TMPDIR/apptainer}
        mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_SIF_DIR" "$APPTAINER_TMPDIR"
    fi
}

common_args() {
    ARGS=(
        "algorithm.adv_estimator=grpo"
        "data.train_files=null" "data.val_files=null"
        "data.train_batch_size=$TRAIN_BATCH_SIZE" "data.val_batch_size=${VAL_BATCH_SIZE:-8}"
        "data.max_prompt_length=$MAX_PROMPT_LENGTH" "data.max_response_length=$MAX_RESPONSE_LENGTH"
        "data.trust_remote_code=True"
        "actor_rollout_ref.model.path=$MODEL_NAME"
        "actor_rollout_ref.model.trust_remote_code=True"
        "actor_rollout_ref.hybrid_engine=True"
        "actor_rollout_ref.model.use_remove_padding=True"
        "actor_rollout_ref.model.enable_gradient_checkpointing=True"
        "actor_rollout_ref.model.enable_activation_offload=True"
        "actor_rollout_ref.actor.strategy=fsdp2"
        "actor_rollout_ref.actor.optim.lr=1e-6"
        "actor_rollout_ref.actor.loss_agg_mode=token-mean"
        "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
        "actor_rollout_ref.actor.use_dynamic_bsz=True"
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_SEQ_LEN"
        "actor_rollout_ref.actor.use_kl_loss=False"
        "actor_rollout_ref.actor.clip_ratio_high=0.28"
        "actor_rollout_ref.actor.entropy_coeff=0"
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
        "actor_rollout_ref.actor.fsdp_config.param_offload=True"
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=False"
        "actor_rollout_ref.rollout.name=vllm" "actor_rollout_ref.rollout.mode=async"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
        "actor_rollout_ref.rollout.data_parallel_size=1"
        "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}"
        "actor_rollout_ref.rollout.free_cache_engine=True"
        "actor_rollout_ref.rollout.temperature=0.7"
        "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}"
        "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_SEQ_LEN"
        "actor_rollout_ref.rollout.max_model_len=$MAX_SEQ_LEN"
        "actor_rollout_ref.rollout.n=$POLICY_ROLLOUTS"
        "actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUTS:-8}"
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.7"
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.8"
        "actor_rollout_ref.rollout.val_kwargs.top_k=20"
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_SEQ_LEN"
        "actor_rollout_ref.ref.fsdp_config.param_offload=True"
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_SEQ_LEN"
        "algorithm.kl_ctrl.kl_coef=0.001"
        "rllm.rejection_sample.enable=True" "rllm.rejection_sample.multiplier=2"
        "rllm.rejection_sample.mode=variance"
        "rllm.disable_thinking=$DISABLE_THINKING" "rllm.accumulate_reasoning=$ACCUMULATE_REASONING"
        "rllm.filter_token_mismatch=True" "rllm.mask_truncated_samples=False"
        "fireworks.return_token_ids=False"
        "trainer.logger=${TRAINER_LOGGERS:-[console,wandb]}"
        "trainer.project_name=$PROJECT_NAME" "trainer.experiment_name=$EXPERIMENT_NAME"
        "trainer.n_gpus_per_node=2" "trainer.nnodes=1" "trainer.critic_warmup=0"
        "trainer.total_epochs=${TRAINER_TOTAL_EPOCHS:-100}"
        "trainer.total_training_steps=${TRAINER_TOTAL_TRAINING_STEPS:-301}"
        "trainer.save_freq=${TRAINER_SAVE_FREQ:-10}" "trainer.test_freq=-1"
        "trainer.val_before_train=False" "trainer.default_hdfs_dir=null"
        "trainer.resume_mode=${TRAINER_RESUME_MODE:-auto}"
        "trainer.default_local_dir=$CHECKPOINT_DIR"
        "trainer.log_episodes=${LOG_EPISODES:-True}"
        "trainer.episode_log_dir=$EPISODE_LOG_DIR"
        "+ray_init.include_dashboard=False" "+ray_init.log_to_driver=False"
        "+ray_init._temp_dir=$RAY_TMPDIR"
        "+ray_init.num_cpus=${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-32}}"
        "+ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY_BYTES:-17179869184}"
    )
}

launch() {
    [[ -z "${TRAINER_RESUME_FROM_PATH:-}" ]] || ARGS+=("trainer.resume_from_path=$TRAINER_RESUME_FROM_PATH")
    rllm_training_launch "$MODULE" "${ARGS[@]}" "$@"
}
