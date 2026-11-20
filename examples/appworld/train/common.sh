#!/usr/bin/env bash
# Shared AppWorld training configuration; method presets are loaded first.
RLLM_TRAIN_ARGS=("$@")  # positional Hydra overrides, replayed by chained follow-up jobs
export HOST_CPUS=${HOST_CPUS:-${SLURM_CPUS_PER_TASK:-32}}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
export APPWORLD_CPUS_PER_SERVER=${APPWORLD_CPUS_PER_SERVER:-1}
export APPWORLD_MAX_SERVERS=${APPWORLD_MAX_SERVERS:-32}
for value in "$HOST_CPUS" "$APPWORLD_CPUS_PER_SERVER" "$APPWORLD_MAX_SERVERS" "$N_GPUS_PER_NODE"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "CPU/GPU/server counts must be positive integers" >&2; exit 1; }
done
DEFAULT_SERVERS=$(( ${APPWORLD_AVAILABLE_CPUS:-$HOST_CPUS} / APPWORLD_CPUS_PER_SERVER ))
(( DEFAULT_SERVERS >= 2 )) || DEFAULT_SERVERS=2
(( DEFAULT_SERVERS <= APPWORLD_MAX_SERVERS )) || DEFAULT_SERVERS=$APPWORLD_MAX_SERVERS
export N_SERVERS=${N_SERVERS:-$DEFAULT_SERVERS}
export N_PARALLEL_AGENTS=${N_PARALLEL_AGENTS:-$N_SERVERS}
[[ "$N_SERVERS" =~ ^[1-9][0-9]*$ && "$N_PARALLEL_AGENTS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid server/agent counts" >&2; exit 1; }
if (( N_PARALLEL_AGENTS > N_SERVERS )); then
    echo "N_PARALLEL_AGENTS must not exceed N_SERVERS: AppWorld servers own mutable task worlds" >&2
    exit 1
fi
export WANDB_PROJECT=${WANDB_PROJECT:-appworld}
export PROJECT_NAME=${PROJECT_NAME:-$WANDB_PROJECT}
export TRAINER_LOGGERS=${TRAINER_LOGGERS:-"['console','wandb']"}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export CRITIC_WARMUP=${CRITIC_WARMUP:-0}
export REJECTION_SAMPLE_ENABLE=${REJECTION_SAMPLE_ENABLE:-True}
export REJECTION_SAMPLE_MULTIPLIER=${REJECTION_SAMPLE_MULTIPLIER:-2}
export TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ:--1}
export VLLM_USE_V1=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-False}
export RAY_LOG_TO_STDERR=${RAY_LOG_TO_STDERR:-0}
export RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-0}
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-131072}
export APPWORLD_MAX_TURNS=${APPWORLD_MAX_TURNS:-10}
export APPWORLD_TRAIN_SPLIT=${APPWORLD_TRAIN_SPLIT:-train}
export APPWORLD_TEST_SPLIT=${APPWORLD_TEST_SPLIT:-dev}
export POLICY_ROLLOUTS=${POLICY_ROLLOUTS:-8}
export TRAJECTORY_HARD_TIMEOUT=${TRAJECTORY_HARD_TIMEOUT:-1200}
export MAX_TRAIN_TASKS=${MAX_TRAIN_TASKS:-}
export MAX_TEST_TASKS=${MAX_TEST_TASKS:-}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-32768}
export ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS:-$ROLLOUT_MAX_MODEL_LEN}
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}
export ROLLOUT_DATA_PARALLEL_SIZE=${ROLLOUT_DATA_PARALLEL_SIZE:-1}
export ROLLOUT_DISABLE_LOG_STATS=${ROLLOUT_DISABLE_LOG_STATS:-False}
export ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
export TOKEN_BUDGET_HEADROOM=${TOKEN_BUDGET_HEADROOM:-4096}
TOKEN_BUDGET_PER_GPU=$((ROLLOUT_MAX_MODEL_LEN + TOKEN_BUDGET_HEADROOM))
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$TOKEN_BUDGET_PER_GPU}
export ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-8}
export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$TOKEN_BUDGET_PER_GPU}
export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$TOKEN_BUDGET_PER_GPU}
export ACTOR_ENABLE_ACTIVATION_OFFLOAD=${ACTOR_ENABLE_ACTIVATION_OFFLOAD:-True}
export ACTOR_ENABLE_GRADIENT_CHECKPOINTING=${ACTOR_ENABLE_GRADIENT_CHECKPOINTING:-True}
export ACTOR_FSDP_FORWARD_PREFETCH=${ACTOR_FSDP_FORWARD_PREFETCH:-False}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
export TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-False}
export TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-10}
export TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-100}
# The limit is checked after each step increments, so 301 runs updates 1-300.
export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-301}

MODEL_VARIANT=${MODEL_VARIANT:-nonthinking}
case "$MODEL_VARIANT" in
    nonthinking) DEFAULT_MODEL=Qwen/Qwen3-4B; DEFAULT_DISABLE=True; DEFAULT_ACCUMULATE=False ;;
    thinking) DEFAULT_MODEL=Qwen/Qwen3-4B; DEFAULT_DISABLE=False; DEFAULT_ACCUMULATE=True ;;
    instruct) DEFAULT_MODEL=Qwen/Qwen3-4B-Instruct-2507; DEFAULT_DISABLE=True; DEFAULT_ACCUMULATE=False ;;
    thinking_2507) DEFAULT_MODEL=Qwen/Qwen3-4B-Thinking-2507; DEFAULT_DISABLE=False; DEFAULT_ACCUMULATE=True ;;
    *) echo "MODEL_VARIANT must be nonthinking, thinking, instruct, or thinking_2507" >&2; exit 1 ;;
esac
export MODEL_NAME=${MODEL_NAME:-$DEFAULT_MODEL}
export DISABLE_THINKING=${DISABLE_THINKING:-$DEFAULT_DISABLE}
export ACCUMULATE_THINKING=${ACCUMULATE_THINKING:-$DEFAULT_ACCUMULATE}
export ACCUMULATE_REASONING=${ACCUMULATE_REASONING:-$DEFAULT_ACCUMULATE}
MODEL_TAG=$(printf '%s\n' "${MODEL_NAME##*/}" | sed -E 's#[^A-Za-z0-9._-]+#_#g')

appworld_common_args() {
    HYDRA_ARGS=(
        "algorithm.adv_estimator=$ADV_ESTIMATOR"
        "data.train_files=null"
        "data.val_files=null"
        "data.train_batch_size=$TRAIN_BATCH_SIZE"
        "data.val_batch_size=$VAL_BATCH_SIZE"
        "data.max_prompt_length=$MAX_PROMPT_LENGTH"
        "data.max_response_length=$MAX_RESPONSE_LENGTH"
        "data.trust_remote_code=True"
        "actor_rollout_ref.model.path=$MODEL_NAME"
        "actor_rollout_ref.model.trust_remote_code=True"
        "actor_rollout_ref.hybrid_engine=True"
        "actor_rollout_ref.actor.optim.lr=1e-6"
        "actor_rollout_ref.model.use_remove_padding=True"
        "actor_rollout_ref.actor.strategy=fsdp2"
        "actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE"
        "actor_rollout_ref.actor.ppo_mini_batch_size=$ACTOR_PPO_MINI_BATCH_SIZE"
        "actor_rollout_ref.actor.use_dynamic_bsz=True"
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU"
        "actor_rollout_ref.actor.use_kl_loss=False"
        "actor_rollout_ref.actor.clip_ratio_high=0.28"
        "actor_rollout_ref.actor.kl_loss_coef=0.001"
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
        "actor_rollout_ref.model.enable_gradient_checkpointing=$ACTOR_ENABLE_GRADIENT_CHECKPOINTING"
        "actor_rollout_ref.model.enable_activation_offload=$ACTOR_ENABLE_ACTIVATION_OFFLOAD"
        "actor_rollout_ref.actor.fsdp_config.param_offload=True"
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=$ACTOR_FSDP_FORWARD_PREFETCH"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE"
        "actor_rollout_ref.rollout.data_parallel_size=$ROLLOUT_DATA_PARALLEL_SIZE"
        "actor_rollout_ref.rollout.name=vllm"
        "actor_rollout_ref.rollout.mode=async"
        "actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER"
        "actor_rollout_ref.rollout.temperature=0.7"
        "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
        "actor_rollout_ref.rollout.free_cache_engine=$ROLLOUT_FREE_CACHE_ENGINE"
        "actor_rollout_ref.rollout.disable_log_stats=$ROLLOUT_DISABLE_LOG_STATS"
        "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED_TOKENS"
        "actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN"
        "actor_rollout_ref.rollout.n=$POLICY_ROLLOUTS"
        "actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUTS:-8}"
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.7"
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.8"
        "actor_rollout_ref.rollout.val_kwargs.top_k=20"
        "actor_rollout_ref.ref.fsdp_config.param_offload=True"
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
        "actor_rollout_ref.actor.entropy_coeff=0"
        "algorithm.kl_ctrl.kl_coef=0.001"
        "rllm.filter_token_mismatch=True"
        "fireworks.return_token_ids=False"
        "rllm.mask_truncated_samples=False"
        "trainer.critic_warmup=$CRITIC_WARMUP"
        "trainer.logger=$TRAINER_LOGGERS"
        "trainer.project_name=$PROJECT_NAME"
        "trainer.experiment_name=$EXPERIMENT_NAME"
        "trainer.val_before_train=$TRAINER_VAL_BEFORE_TRAIN"
        "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
        "trainer.nnodes=1"
        "trainer.save_freq=$TRAINER_SAVE_FREQ"
        "trainer.test_freq=$TRAINER_TEST_FREQ"
        "trainer.default_hdfs_dir=null"
        "trainer.resume_mode=${TRAINER_RESUME_MODE:-auto}"
        "trainer.episode_log_dir=$EPISODE_LOG_DIR"
        "rllm.rejection_sample.enable=$REJECTION_SAMPLE_ENABLE"
        "rllm.rejection_sample.multiplier=$REJECTION_SAMPLE_MULTIPLIER"
        "rllm.agent.max_steps=$APPWORLD_MAX_TURNS"
        "rllm.agent.trajectory_hard_timeout=$TRAJECTORY_HARD_TIMEOUT"
        "+rllm.agent.agent_args.accumulate_history=True"
        "+rllm.agent.agent_args.accumulate_thinking=$ACCUMULATE_THINKING"
        "rllm.stepwise_advantage.enable=False"
        "rllm.disable_thinking=$DISABLE_THINKING"
        "rllm.accumulate_reasoning=$ACCUMULATE_REASONING"
        "+rllm.agent.engine_args.n_parallel_agents=$N_PARALLEL_AGENTS"
        "+rllm.env.env_args.max_turns=$APPWORLD_MAX_TURNS"
        "+rllm.env.env_args.remote_environment_url=$APPWORLD_URL"
        "+rllm.env.env_args.experiment_name=$APPWORLD_EXPERIMENT_NAME"
        "+rllm.env.env_args.n_servers=$N_SERVERS"
        "+rllm.env.env_args.base_port=$BASE_PORT"
        "trainer.total_epochs=$TRAINER_TOTAL_EPOCHS"
        "trainer.total_training_steps=$TRAINER_TOTAL_TRAINING_STEPS"
        "+ray_init.include_dashboard=$RAY_INCLUDE_DASHBOARD"
        "+ray_init.log_to_driver=$RAY_LOG_TO_DRIVER"
        "+ray_init._temp_dir=$RAY_TMPDIR"
        "+ray_init.object_store_memory=$RAY_OBJECT_STORE_MEMORY_BYTES"
        "+ray_init.num_cpus=$RAY_NUM_CPUS"
    )
    if [[ -n "${TRAINER_DEFAULT_LOCAL_DIR:-}" ]]; then
        HYDRA_ARGS+=("trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR")
    fi
    if [[ -n "${TRAINER_VAL_ONLY:-}" ]]; then
        HYDRA_ARGS+=("trainer.val_only=$TRAINER_VAL_ONLY")
    fi
    if [[ -n "${TRAINER_RESUME_FROM_PATH:-}" ]]; then
        HYDRA_ARGS+=("trainer.resume_from_path=$TRAINER_RESUME_FROM_PATH")
    fi
}

appworld_start_runtime() {
    source "$RLLM_DIR/scripts/training_runtime.sh"
    if [[ "${RLLM_DRY_RUN:-0}" == 1 ]]; then
        # Synthetic paths only: do not create directories, probe GPUs, or bind ports.
        PYTHON_BIN=${PYTHON_BIN:-python3}
        RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rllm-dry-run/ray}
        RAY_NUM_CPUS=${RAY_NUM_CPUS:-$HOST_CPUS}
        RAY_OBJECT_STORE_MEMORY_BYTES=${RAY_OBJECT_STORE_MEMORY_BYTES:-17179869184}
        RAY_LOG_TO_DRIVER=${RAY_LOG_TO_DRIVER:-False}
        BASE_PORT=${BASE_PORT:-20000}
        APPWORLD_URL=${APPWORLD_URL:-http://localhost:$BASE_PORT}
    else
        source "$RLLM_DIR/examples/appworld/train/sidecars.sh"
    fi
}

appworld_launch() {
    rllm_training_launch "$@"
}
