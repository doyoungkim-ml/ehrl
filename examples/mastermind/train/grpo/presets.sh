#!/usr/bin/env bash
# Scientific defaults and accepted overrides for this benchmark/method.
PRESET=${PRESET:-grpo}
[[ "$PRESET" == grpo ]] || { echo "Unknown grpo PRESET: $PRESET (expected grpo)" >&2; exit 2; }
# GRPO_ACTOR_PROMPT=ace_empty_playbook trains under ACE-RL's step-0 actor prompt
# (empty playbook) instead of the agent's built-in prompt; it needs its own RUN_DIR.
GRPO_ACTOR_PROMPT=${GRPO_ACTOR_PROMPT:-agent}
case "$GRPO_ACTOR_PROMPT" in
    agent) ACTOR_PROMPT_TAG= ;;
    ace_empty_playbook)
        [[ "$TRAINING_METHOD" == grpo ]] || { echo "GRPO_ACTOR_PROMPT applies only to the GRPO launcher" >&2; exit 2; }
        [[ -n "${RUN_DIR:-}" ]] || { echo "GRPO_ACTOR_PROMPT=ace_empty_playbook is an ablation; set its own RUN_DIR" >&2; exit 2; }
        ACTOR_PROMPT_TAG=_ace_empty_playbook ;;
    *) echo "GRPO_ACTOR_PROMPT must be agent or ace_empty_playbook, got: $GRPO_ACTOR_PROMPT" >&2; exit 2 ;;
esac
MODEL_VARIANT=${MODEL_VARIANT:-nonthinking}
case "$MODEL_VARIANT" in
    nonthinking) DISABLE_THINKING=True; ACCUMULATE_THINKING=False ;;
    thinking) DISABLE_THINKING=False; ACCUMULATE_THINKING=True ;;
    *) echo "Qwen3-0.6B supports MODEL_VARIANT=nonthinking or thinking" >&2; exit 1 ;;
esac
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}
MAX_TURNS=${MASTERMIND_MAX_TURNS:-5}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-10240}
BASE_VAL_BATCH=128
METHOD_TAG=grpo$ACTOR_PROMPT_TAG
DEFAULT_BATCH=8
DEFAULT_PROMPT=4096
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
DEFAULT_VAL_BATCH=$BASE_VAL_BATCH

POLICY_ROLLOUTS=${POLICY_ROLLOUTS:-8}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$DEFAULT_PROMPT}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

N_PARALLEL_AGENTS=${N_PARALLEL_AGENTS:-24}
if (( N_PARALLEL_AGENTS > HOST_CPUS - 8 )); then N_PARALLEL_AGENTS=$((HOST_CPUS > 8 ? HOST_CPUS - 8 : 1)); fi
REJECTION_SAMPLE_ENABLE=${REJECTION_SAMPLE_ENABLE:-True}
REJECTION_SAMPLE_MULTIPLIER=${REJECTION_SAMPLE_MULTIPLIER:-2}
if [[ "${REJECTION_SAMPLE_ENABLE,,}" == false ]]; then REJECTION_SAMPLE_MULTIPLIER=1; fi

method_args() {
    [[ "$GRPO_ACTOR_PROMPT" == agent ]] || args+=("+rllm.actor_prompt=$GRPO_ACTOR_PROMPT")
}
