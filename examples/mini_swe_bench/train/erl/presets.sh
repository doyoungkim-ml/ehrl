#!/usr/bin/env bash
# Use the plain benchmark binding with the pinned original ERL objective.
source "$RLLM_DIR/rllm/algorithms/erl/launcher.sh"
erl_defaults 8
MODEL_VARIANT=${MODEL_VARIANT:-instruct}
PRESET=grpo
source "$RLLM_DIR/examples/mini_swe_bench/train/grpo/presets.sh"
PRESET=$ERL_PRESET
