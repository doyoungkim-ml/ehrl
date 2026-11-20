#!/usr/bin/env bash
# Shared defaults; method and benchmark presets set their values first.
: "${LOCAL_EVAL_PROTOCOL:=sequential_adaptation_v1}"
: "${SEED:=42}" "${G_ACT:=1}" "${SELECTION_MODE:=best_rejection}"
: "${ACE_TEMPERATURE:=0.7}" "${GEN_TEMPERATURE:=0.7}" "${GEN_TOP_P:=1.0}"
: "${MAX_PLAYBOOK_LENGTH:=2048}" "${BASE_SYSTEM_PROMPT_FILENAME:=base_system_prompt.txt}"
: "${REFLECTOR_MAX_RESPONSE_LENGTH:=2048}" "${CURATOR_MAX_PROMPT_LENGTH:=8192}" "${CURATOR_MAX_RESPONSE_LENGTH:=4096}"
: "${VLLM_TP:=1}" "${VLLM_DP:=1}" "${GPU_MEMORY_UTILIZATION:=0.90}" "${MAX_NUM_SEQS:=64}"
: "${MODEL_SOURCE:=auto}" "${TASK_OFFSET:=0}"
