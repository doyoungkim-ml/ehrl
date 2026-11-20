#!/usr/bin/env bash
# Use the plain benchmark binding with the pinned original ERL objective.
source "$RLLM_DIR/rllm/algorithms/erl/launcher.sh"
erl_defaults 8
PRESET=grpo
source "$RLLM_DIR/examples/mbpp/train/grpo/presets.sh"
PRESET=$ERL_PRESET
