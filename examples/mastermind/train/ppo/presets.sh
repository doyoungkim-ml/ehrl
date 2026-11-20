#!/usr/bin/env bash
# Inherit benchmark settings, then select this method's isolated recipe.
PRESET=${PRESET:-ppo}
[[ "$PRESET" == ppo ]] || { echo "Unknown ppo PRESET: $PRESET (expected ppo)" >&2; exit 2; }
MODEL_VARIANT=${MODEL_VARIANT:-nonthinking}
source "$RLLM_DIR/rllm/algorithms/ppo/launcher.sh"
ppo_defaults
PRESET=grpo
source "$RLLM_DIR/examples/mastermind/train/grpo/presets.sh"
PRESET=ppo
