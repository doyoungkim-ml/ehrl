#!/usr/bin/env bash
# PPO remains available as a substantive critic/GAE ablation of the same trainer.
PRESET=${PRESET:-grpo}
case "$PRESET" in
    grpo) ;;
    no_rejection) : "${REJECTION_SAMPLE_ENABLE:=False}" "${REJECTION_SAMPLE_MULTIPLIER:=1}" ;;
    ppo)
        : "${ADV_ESTIMATOR:=gae}" "${GAE_GAMMA:=1.0}" "${GAE_LAMBDA:=0.95}"
        : "${CRITIC_WARMUP:=10}" "${CRITIC_LR:=1e-5}" "${CRITIC_LOSS_AGG_MODE:=token-mean}"
        : "${CRITIC_SAVE_CONTENTS:=['model']}" "${KEEP_LAST_N:=0}"
        : "${REJECTION_SAMPLE_ENABLE:=False}" "${REJECTION_SAMPLE_MULTIPLIER:=1}"
        VLLM_COMPILATION_CONFIG=${VLLM_COMPILATION_CONFIG:-'{cudagraph_mode: FULL_AND_PIECEWISE, inductor_compile_config: {use_static_cuda_launcher: false}}'}
        ;;
    *) echo "Unknown GRPO PRESET: $PRESET (grpo, no_rejection, ppo)" >&2; exit 1 ;;
esac
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
: "${MODEL_VARIANT:=nonthinking}" "${TRAJECTORY_HARD_TIMEOUT:=2400}"
: "${VLLM_ATTENTION_BACKEND:=FLASH_ATTN}"
VLLM_ATTENTION_CONFIG=${VLLM_ATTENTION_CONFIG:-'{backend: FLASH_ATTN, flash_attn_version: 2}'}
VLLM_COMPILATION_CONFIG=${VLLM_COMPILATION_CONFIG:-'{cudagraph_mode: FULL_AND_PIECEWISE}'}
export VLLM_ATTENTION_BACKEND
