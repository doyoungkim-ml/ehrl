#!/usr/bin/env bash
# Inherit benchmark settings, then select this method's isolated recipe.
MODEL_VARIANT=${MODEL_VARIANT:-instruct}
source "$RLLM_DIR/rllm/algorithms/ppo/launcher.sh"
ppo_select_joint_batch 8
ppo_defaults
PRESET=delayed_joint
source "$RLLM_DIR/examples/appworld/train/ace_rl/presets.sh"
ppo_finish_joint_defaults
PRESET=$PPO_PRESET
