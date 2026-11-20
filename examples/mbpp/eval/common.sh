#!/usr/bin/env bash
# Benchmark defaults shared by the ACE visits8 and vanilla pass8 presets.
: "${MODEL_NAME:=Qwen/Qwen3-0.6B}" "${MAX_TURNS:=6}"
: "${MAX_MODEL_LEN:=40960}" "${GEN_MAX_RESPONSE_LENGTH:=8192}" "${GEN_MAX_PROMPT_LENGTH:=4096}"
: "${SPLIT:=test}" "${DISABLE_THINKING:=False}" "${REFLECTOR_MAX_PROMPT_LENGTH:=20480}"
# An explicitly empty MAX_TASKS selects the full split.
MAX_TASKS=${MAX_TASKS-100}
: "${REFLECTOR_SYSTEM_PROMPT:=You are an expert Python coding analyst. Respond with JSON only.}"
: "${CURATOR_SYSTEM_PROMPT:=You are a master curator of knowledge in Python programming. Respond with JSON only.}"
# MBPP_FEEDBACK=full|hidden (unset means full) is read by scripts/evaluation_runtime.py.
# It stays unset here so a hidden-feedback training run must choose it explicitly.
source "$RLLM_DIR/scripts/evaluation_defaults.sh"
