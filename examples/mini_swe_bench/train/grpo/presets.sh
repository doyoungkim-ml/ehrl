#!/usr/bin/env bash
# Model and scientific settings for this method.
PRESET=${PRESET:-grpo}
[[ "$PRESET" == grpo ]] || { echo "Unknown GRPO PRESET: $PRESET (expected grpo)" >&2; exit 2; }
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
MAX_STEPS=${MINISW_MAX_STEPS:-10}
PARALLEL_TASKS=${MINISW_N_PARALLEL_TASKS:-8}
POLICY_ROLLOUTS=${POLICY_ROLLOUTS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PROJECT_NAME=${PROJECT_NAME:-mini_swe_bench}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-mini_swe_bench_grpo${ACTOR_PROMPT_TAG}_${MODEL_NAME##*/}_${MODEL_VARIANT}_${MINISW_REWARD_MODE}_tasks${MAX_TRAIN_TASKS:-all}_offset${TRAIN_TASK_OFFSET}_2gpu}

method_args() {
    MODULE=examples.mini_swe_bench.train.grpo.train
    ARGS+=(
        "rllm.workflow.use_workflow=True"
        "rllm.workflow.n_parallel_tasks=$PARALLEL_TASKS"
        "+rllm.workflow.workflow_args.max_steps=$MAX_STEPS"
        "rllm.workflow.workflow_args.timeout=${WORKFLOW_TIMEOUT:-7200}"
        "+rllm.workflow.workflow_args.reward_timeout=${MINISW_REWARD_TIMEOUT:-600}"
        "+rllm.workflow.workflow_args.reward_mode=$MINISW_REWARD_MODE"
        "+rllm.workflow.workflow_args.env_args.max_steps=$MAX_STEPS"
        "+rllm.workflow.workflow_args.env_args.step_timeout=${MINISW_STEP_TIMEOUT:-120}"
        "+rllm.workflow.workflow_args.env_args.reward_timeout=${MINISW_REWARD_TIMEOUT:-600}"
        "+rllm.workflow.workflow_args.agent_args.scaffold=sweagent"
    )
    [[ "$GRPO_ACTOR_PROMPT" == agent ]] || ARGS+=("+rllm.actor_prompt=$GRPO_ACTOR_PROMPT")
}
