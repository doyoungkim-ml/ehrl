#!/usr/bin/env bash
# Inherit benchmark settings, then select this method's isolated recipe.
MODEL_VARIANT=${MODEL_VARIANT:-nonthinking}
source "$RLLM_DIR/rllm/algorithms/ppo/launcher.sh"
ppo_select_joint_batch 8
ppo_defaults
PRESET=delayed_oneeval_rollout8
source "$RLLM_DIR/examples/mastermind/train/ace_rl/presets.sh"
ppo_finish_joint_defaults
PRESET=$PPO_PRESET
