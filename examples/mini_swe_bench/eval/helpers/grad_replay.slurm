#!/usr/bin/env bash
#SBATCH --job-name=miniswe-gradreplay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
# ============================================================================
# Per-trajectory policy-gradient replay. NO training.
#   1. merge one FSDP actor checkpoint -> HF dir (verl.model_merger), cached
#   2. python grad_replay.py on the given episode step dirs
# Env (per submit):
#   CKPT_DIR   experiment checkpoint dir (…/checkpoints/<exp>)
#   STEP       global step of the checkpoint to load (or BASE for the HF base model)
#   EPISODES   space-separated episode step dirs (train_step_*_epoch_*)
#   TAG        label written into every row
#   OUT_DIR    default examples/mini_swe_bench/logs/grad_replay
#   N_SKETCH   default 2000000; LIMIT (debug) default 0
# ============================================================================
set -euo pipefail
RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$RLLM_DIR"
source "$RLLM_DIR/scripts/gpu_venv.sh"
rllm_resolve_gpu_env "$RLLM_DIR"
PYTHON_BIN=${PYTHON_BIN:-$VENV_DIR/bin/python}
export VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$PATH" PYTHONPATH="$RLLM_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
OUT_DIR=${OUT_DIR:-$RLLM_DIR/examples/mini_swe_bench/logs/grad_replay}
export HF_HOME=${HF_HOME:-$OUT_DIR/hf_cache}
mkdir -p "$OUT_DIR" "$HF_HOME"
: "${CKPT_DIR:?}" "${STEP:?}" "${EPISODES:?}" "${TAG:?}"
echo "[gradreplay] job=${SLURM_JOB_ID:-local} venv=$VENV_DIR python=$PYTHON_BIN step=$STEP tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$STEP" = "BASE" ]; then
    MODEL=${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
else
    ACTOR="$CKPT_DIR/global_step_$STEP/actor"
    MODEL="$OUT_DIR/hf_$(basename "$CKPT_DIR")_step$STEP"
    if [ ! -f "$MODEL/config.json" ] || [ -z "$(ls "$MODEL" | grep -E 'safetensors|\.bin$' || true)" ]; then
        echo "[gradreplay] merging $ACTOR -> $MODEL"
        if [[ -e "$MODEL" ]]; then
            echo "Incomplete merged model already exists: $MODEL; choose another OUT_DIR or inspect it first" >&2
            exit 1
        fi
        "$PYTHON_BIN" -m verl.model_merger merge --backend fsdp --local_dir "$ACTOR" --target_dir "$MODEL"
        # tokenizer/chat template travel with the actor's huggingface/ dir
        cp -n "$ACTOR"/huggingface/{tokenizer.json,tokenizer_config.json,chat_template.jinja,generation_config.json} "$MODEL"/ 2>/dev/null || true
        [ -f "$MODEL/config.json" ] || cp "$ACTOR/huggingface/config.json" "$MODEL/"
        du -sh "$MODEL"
    else
        echo "[gradreplay] merged model cached at $MODEL"
    fi
fi

OUT="$OUT_DIR/replay_${TAG}.npz"
"$PYTHON_BIN" examples/mini_swe_bench/eval/helpers/grad_replay.py --model "$MODEL" --episodes $EPISODES \
    --out "$OUT" --n-sketch "${N_SKETCH:-2000000}" --limit "${LIMIT:-0}" --tag "$TAG"
ls -la "$OUT"
echo "[gradreplay] done"
