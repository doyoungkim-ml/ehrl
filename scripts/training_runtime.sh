#!/usr/bin/env bash
# Shared paths and process ownership for single-node training.
set -euo pipefail
: "${RLLM_DIR:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${BENCHMARK:?Set BENCHMARK before loading the training runtime}"
: "${TRAINING_METHOD:?Set TRAINING_METHOD before loading the training runtime}"
[[ -d "$RLLM_DIR/rllm" ]] || { echo "RLLM_DIR must point to the source checkout" >&2; exit 1; }
cd "$RLLM_DIR"
export RLLM_DIR=$PWD
export PRESET=${PRESET:-$TRAINING_METHOD}
export MODEL_VARIANT=${MODEL_VARIANT:-nonthinking}
MODEL_PATH_TAG=${MODEL_NAME##*/}
MODEL_PATH_TAG=${MODEL_PATH_TAG,,}
MODEL_PATH_TAG=${MODEL_PATH_TAG//[^a-z0-9._-]/-}
case "$MODEL_VARIANT" in
    instruct|thinking_2507) RUN_MODEL_TAG=$MODEL_PATH_TAG ;;
    *) RUN_MODEL_TAG=$MODEL_PATH_TAG-$MODEL_VARIANT ;;
esac
[[ "$PRESET" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "PRESET must be a simple configuration name" >&2; exit 2; }
export RUN_DIR=${RUN_DIR:-$RLLM_DIR/examples/$BENCHMARK/train/$TRAINING_METHOD/runs/$PRESET/$RUN_MODEL_TAG}
[[ "$RUN_DIR" == /* ]] || RUN_DIR="$RLLM_DIR/$RUN_DIR"
export RLLM_RUN_DIR=$RUN_DIR
for checkpoint_path in "${CHECKPOINT_DIR:-}" "${TRAINER_DEFAULT_LOCAL_DIR:-}"; do
    if [[ -n "$checkpoint_path" && "$checkpoint_path" != "$RUN_DIR/checkpoints" ]]; then
        echo "Checkpoints belong under RUN_DIR/checkpoints; set RUN_DIR to select an experiment: $RUN_DIR" >&2
        exit 2
    fi
done
export CHECKPOINT_DIR=$RUN_DIR/checkpoints TRAINER_DEFAULT_LOCAL_DIR=$RUN_DIR/checkpoints
export TRAINER_RESUME_MODE=${TRAINER_RESUME_MODE:-auto}
# W&B logs to WANDB_ENTITY when set, otherwise to the logged-in account's default entity.
export WANDB_DIR=$RUN_DIR
export RLLM_LOG_DIR=$RUN_DIR/logs
export TASK_ARTIFACT_ROOT=$RUN_DIR/task_artifacts
export TRAJECTORY_LOG_DIR=$RUN_DIR/episodes
export EPISODE_LOG_DIR=$RUN_DIR/episodes

# Chained submission for runs longer than one allocation. With
# TRAIN_CHAIN_PARTS=N, part 1 submits parts 2..N of the same launcher, each
# --dependency=afterany on the previous part; every part is an ordinary resume
# of RUN_DIR. Only part 1 submits, so resuming a later part never duplicates
# the chain. RUN_DIR/train_jobs.json lists every job of the run.
export TRAIN_CHAIN_PARTS=${TRAIN_CHAIN_PARTS:-1} TRAIN_CHAIN_PART=${TRAIN_CHAIN_PART:-1}
[[ "$TRAIN_CHAIN_PARTS" =~ ^[1-9][0-9]*$ && "$TRAIN_CHAIN_PART" =~ ^[1-9][0-9]*$ ]] && (( TRAIN_CHAIN_PART <= TRAIN_CHAIN_PARTS )) || {
    echo "TRAIN_CHAIN_PARTS and TRAIN_CHAIN_PART must be positive integers with TRAIN_CHAIN_PART <= TRAIN_CHAIN_PARTS" >&2; exit 2;
}

rllm_chain_value() {
    case "$1" in ""|N/A|n/a|"(null)"|None|UNLIMITED) return 1 ;; esac
    printf '%s' "$1"
}

rllm_record_train_job() {
    # Usage: rllm_record_train_job JOB_ID PART EVENT [DEPENDENCY]; EVENT is submitted or started.
    "${PYTHON_BIN:-python3}" - "$RUN_DIR/train_jobs.json" "$1" "$2" "$TRAIN_CHAIN_PARTS" "$3" "${4:-}" "${RLLM_CHAIN_JOB_NAME:-}" "$RLLM_CHAIN_LAUNCHER" <<'PY'
import datetime, json, os, sys, tempfile
path, job, part, parts, event, dependency, name, launcher = sys.argv[1:]
ledger = json.load(open(path)) if os.path.exists(path) else {"schema_version": 1, "jobs": []}
entry = next((item for item in ledger["jobs"] if item["job_id"] == job), None)
if entry is None:
    entry = {"job_id": job, "part": int(part), "parts": int(parts), "submitted_at": None, "dependency": None, "job_name": name or None, "launcher": launcher}
    ledger["jobs"].append(entry)
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
entry["submitted_at" if event == "submitted" else "started_at"] = now
if dependency:
    entry["dependency"] = dependency
handle, temporary = tempfile.mkstemp(prefix=".train_jobs.", dir=os.path.dirname(path))
with os.fdopen(handle, "w") as stream:
    json.dump(ledger, stream, indent=2)
    stream.write("\n")
os.replace(temporary, path)
PY
}

rllm_submit_training_chain() {
    RLLM_CHAIN_LAUNCHER=$RLLM_DIR/examples/$BENCHMARK/train/$TRAINING_METHOD/train.slurm
    local dry_run=0 parent=${SLURM_JOB_ID:-} name=${SLURM_JOB_NAME:-} partition="" account="" qos="" reservation="" limit="" features="" gres="" memory="" comment=""
    [[ "${RLLM_DRY_RUN:-0}" != 1 ]] || dry_run=1
    if [[ -n "$parent" ]] && command -v squeue >/dev/null; then
        # Comment is last so a '|' inside it stays part of the comment.
        IFS='|' read -r name partition account qos reservation limit features gres memory comment \
            < <(squeue -h -j "$parent" -o '%j|%P|%a|%q|%v|%l|%f|%b|%m|%k') || true
    fi
    if [[ -z "$name" && -f "$RLLM_CHAIN_LAUNCHER" ]]; then
        name=$(sed -n 's/^#SBATCH --job-name=//p' "$RLLM_CHAIN_LAUNCHER" | head -n 1)
    fi
    RLLM_CHAIN_JOB_NAME=$name
    if (( dry_run == 0 )) && [[ -n "$parent" ]]; then
        rllm_record_train_job "$parent" "$TRAIN_CHAIN_PART" started || echo "[chain] WARNING: could not update $RUN_DIR/train_jobs.json" >&2
    fi
    (( TRAIN_CHAIN_PART == 1 && TRAIN_CHAIN_PARTS > 1 )) || return 0
    if (( dry_run == 0 )) && [[ -z "$parent" ]]; then
        echo "TRAIN_CHAIN_PARTS>1 requires a Slurm job; submit the launcher with sbatch" >&2; exit 2
    fi
    local options=(--parsable "--job-name=$name") value item
    value=$(rllm_chain_value "$comment") && options+=("--comment=$value")
    value=$(rllm_chain_value "$partition") && options+=("--partition=$value")
    value=$(rllm_chain_value "$account") && options+=("--account=$value")
    value=$(rllm_chain_value "$qos") && options+=("--qos=$value")
    value=$(rllm_chain_value "$reservation") && options+=("--reservation=$value")
    value=$(rllm_chain_value "$limit") && options+=("--time=$value")
    value=$(rllm_chain_value "$memory") && options+=("--mem=$value")
    value=$(rllm_chain_value "$features") && options+=("--constraint=$value")
    if value=$(rllm_chain_value "$gres"); then
        # squeue reports TRES names (gres/gpu:2); sbatch --gres takes gpu:2.
        value=${value//gres\//}; value=${value//gres:/}
        options+=("--gres=$value")
    fi
    local extra=()
    [[ -z "${TRAIN_CHAIN_SBATCH_ARGS:-}" ]] || read -r -a extra <<< "$TRAIN_CHAIN_SBATCH_ARGS"
    # Later parts resume the latest committed checkpoint whatever part 1 used.
    local replay=() key
    for item in ${RLLM_TRAIN_ARGS[@]+"${RLLM_TRAIN_ARGS[@]}"}; do
        key=${item%%=*}; key=${key#+}; key=${key#+}
        case "$key" in trainer.resume_mode|trainer.resume_from_path) continue ;; esac
        replay+=("$item")
    done
    local clean_env=(-u TRAINER_RESUME_FROM_PATH) variable
    while IFS= read -r variable; do
        [[ "$variable" == SLURM_CONF ]] || clean_env+=(-u "$variable")
    done < <(compgen -e | grep -E '^(SLURM|SBATCH|SALLOC|SRUN)_' || true)
    local previous=${parent:-<part-1-job-id>} part submitted
    for (( part = 2; part <= TRAIN_CHAIN_PARTS; part++ )); do
        local command=(env "${clean_env[@]}" "TRAIN_CHAIN_PART=$part" "TRAIN_CHAIN_PARTS=$TRAIN_CHAIN_PARTS" TRAINER_RESUME_MODE=auto
            "RLLM_DIR=$RLLM_DIR" "RUN_DIR=$RUN_DIR" sbatch "${options[@]}" "--dependency=afterany:$previous" "${extra[@]}" "$RLLM_CHAIN_LAUNCHER" "${replay[@]}")
        if (( dry_run )); then
            printf '[chain] part %s/%s would submit:' "$part" "$TRAIN_CHAIN_PARTS" >&2
            printf ' %q' "${command[@]:${#clean_env[@]}+1}" >&2
            printf '\n' >&2
            submitted="<part-$part-job-id>"
        elif submitted=$("${command[@]}"); then
            submitted=${submitted%%;*}
            printf '[chain] submitted part %s/%s as job %s (afterany:%s)\n' "$part" "$TRAIN_CHAIN_PARTS" "$submitted" "$previous"
            rllm_record_train_job "$submitted" "$part" submitted "afterany:$previous" || echo "[chain] WARNING: could not update $RUN_DIR/train_jobs.json" >&2
        else
            echo "[chain] WARNING: submitting part $part/$TRAIN_CHAIN_PARTS failed; training continues, resubmit the remaining parts manually" >&2
            return 0
        fi
        previous=$submitted
    done
}

# FP8 vLLM rollout of the BF16 actor, corrected by token-level truncated
# importance sampling against recomputed BF16 log-probabilities. One knob for
# every method; `none` leaves the command byte-identical.
export ROLLOUT_QUANTIZATION=${ROLLOUT_QUANTIZATION:-none}
rllm_rollout_quantization_args() {
    case "$ROLLOUT_QUANTIZATION" in
        none) ROLLOUT_QUANTIZATION_ARGS=() ;;
        fp8) ROLLOUT_QUANTIZATION_ARGS=(
                actor_rollout_ref.rollout.quantization=fp8
                actor_rollout_ref.rollout.calculate_log_probs=True
                algorithm.rollout_correction.rollout_is=token
                algorithm.rollout_correction.rollout_is_threshold=2.0
            ) ;;
        *) echo "ROLLOUT_QUANTIZATION must be none or fp8, got: $ROLLOUT_QUANTIZATION" >&2; return 2 ;;
    esac
}

rllm_training_launch() {
    local module=$1
    shift
    rllm_rollout_quantization_args || return
    set -- "$@" ${ROLLOUT_QUANTIZATION_ARGS[@]+"${ROLLOUT_QUANTIZATION_ARGS[@]}"}
    if [[ "${RLLM_DRY_RUN:-0}" == 1 ]]; then
        "$PYTHON_BIN" -c 'import json,sys; print(json.dumps(sys.argv[1:], indent=2))' "$PYTHON_BIN" -u -m "$module" "$@"
        return
    fi
    local resume_mode=$TRAINER_RESUME_MODE resume_from_path=${TRAINER_RESUME_FROM_PATH:-} total_steps="" val_only=False override key value
    for override in "$@"; do
        key=${override%%=*}; key=${key#+}; key=${key#+}; value=${override#*=}
        case "$key" in
            trainer.default_local_dir)
                [[ "$value" == "$CHECKPOINT_DIR" ]] || { echo "Set RUN_DIR instead of overriding trainer.default_local_dir" >&2; return 2; } ;;
            trainer.resume_mode) resume_mode=$value ;;
            trainer.resume_from_path) resume_from_path=$value ;;
            trainer.total_training_steps) total_steps=$value ;;
            trainer.val_only) val_only=$value ;;
        esac
    done
    local preflight=(--checkpoint-dir "$CHECKPOINT_DIR" --resume-mode "$resume_mode")
    [[ -z "$resume_from_path" ]] || preflight+=(--resume-from-path "$resume_from_path")
    "$PYTHON_BIN" -m rllm.utils.run_state "${preflight[@]}"
    # The step limit is checked after each step increments: after checkpoint N
    # the next step is N + 1, so N + 1 >= limit means training already finished.
    # A chained part that starts then exits successfully without another update.
    local marker=$CHECKPOINT_DIR/latest_checkpointed_iteration.txt saved
    if [[ "$resume_mode" == auto && "$total_steps" =~ ^[0-9]+$ && "${val_only,,}" != true && -f "$marker" ]]; then
        saved=$(<"$marker")
        if [[ "$saved" =~ ^[0-9]+$ ]] && (( saved + 1 >= total_steps )); then
            printf '[training] already complete: checkpoint %s reached trainer.total_training_steps=%s; nothing to run\n' "$saved" "$total_steps"
            return 0
        fi
    fi
    printf '[training] run=%s checkpoints=%s wandb=%s/%s resume=%s rollout_quantization=%s\n' "$RUN_DIR" "$CHECKPOINT_DIR" "${WANDB_ENTITY:-default}" "$PROJECT_NAME" "$resume_mode" "$ROLLOUT_QUANTIZATION"
    "$PYTHON_BIN" -u -m "$module" "$@"
}

if [[ "${RLLM_DRY_RUN:-0}" == 1 ]]; then
    export PYTHON_BIN=${PYTHON_BIN:-python3}
    export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rllm-dry-run/ray}
    export RAY_NUM_CPUS=${RAY_NUM_CPUS:-${HOST_CPUS:-${SLURM_CPUS_PER_TASK:-32}}}
    export RAY_OBJECT_STORE_MEMORY_BYTES=${RAY_OBJECT_STORE_MEMORY_BYTES:-17179869184}
    export RAY_LOG_TO_DRIVER=${RAY_LOG_TO_DRIVER:-False}
    rllm_submit_training_chain
    return 0
fi
RLLM_JOB_ID=${SLURM_JOB_ID:-local-$$}
mkdir -p "$RUN_DIR" "$RLLM_LOG_DIR"
exec {RLLM_WRITER_FD}>"$RUN_DIR/.writer.lock"
flock -n "$RLLM_WRITER_FD" || { echo "Another writer already owns RUN_DIR: $RUN_DIR" >&2; exit 1; }
# Submit follow-up parts while this job holds the writer lock and before any
# job-scoped runtime state is exported into the submission environment.
rllm_submit_training_chain
# Slurm keeps its small submission receipt; the run owns the complete startup
# and training log. Preserve the trainer's status rather than a tee pipeline's.
exec > >(tee -a "$RLLM_LOG_DIR/train-$RLLM_JOB_ID.log") 2>&1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES RAY_ADDRESS
source "$RLLM_DIR/scripts/gpu_venv.sh"
rllm_resolve_gpu_env "$RLLM_DIR"
[[ -n ${PYTHON_BIN:-} ]] || rllm_ensure_gpu_env "$RLLM_DIR"
export PYTHON_BIN=${PYTHON_BIN:-$VENV_DIR/bin/python}
[[ -x "$PYTHON_BIN" ]] || { echo "Missing training Python: $PYTHON_BIN" >&2; exit 1; }
export PYTHONPATH="$RLLM_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${VERL_SOURCE_DIR:-}" ]]; then
    [[ -f "$VERL_SOURCE_DIR/verl/trainer/ppo/ray_trainer.py" ]] || { echo "Missing prepared VeRL source: $VERL_SOURCE_DIR" >&2; exit 1; }
    export PYTHONPATH="$VERL_SOURCE_DIR:$PYTHONPATH"
fi
source "$RLLM_DIR/scripts/gpu_runtime.sh"
rllm_configure_gpu_runtime "$PYTHON_BIN"
export VIRTUAL_ENV=$VENV_DIR PATH="$VENV_DIR/bin:$PATH"
export RLLM_RUNTIME_LD_LIBRARY_PATH=$RUNTIME_LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 HF_HUB_DISABLE_XET=1
export HOST_CPUS=${HOST_CPUS:-${SLURM_CPUS_PER_TASK:-32}}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-$HOST_CPUS}
export RAY_OBJECT_STORE_MEMORY_BYTES=${RAY_OBJECT_STORE_MEMORY_BYTES:-17179869184}
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=${RAY_WORKER_REGISTER_TIMEOUT_SECONDS:-300}
export RAY_worker_register_timeout_seconds=${RAY_worker_register_timeout_seconds:-300}
export RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms:-300000}
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-300}
export RAY_enable_worker_process_setup_hook=true
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_DASHBOARD_ENABLED=0 RAY_DISABLE_DASHBOARD=true RAY_ENABLE_METRICS_COLLECTION=false RAY_DISABLE_USAGE_STATS=true
export RAY_BACKEND_LOG_LEVEL=warning RAY_LOG_TO_DRIVER=False
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export CACHE_DIR=$RLLM_CACHE_DIR
export HF_HOME=${HF_HOME:-$CACHE_DIR/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CACHE_DIR/xdg}
# Runtime/compiler caches and datasets are job-scoped; the model cache is shared.
# The registry rewrites JSON during data preparation, so parallel jobs must not share it.
# Ray sockets, container scratch, and mutable dataset registries are disposable
# and must be isolated even when several jobs use the same source checkout.
RLLM_JOB_TMP=$(mktemp -d "${LOCAL_TMP_ROOT:-/tmp}/rllm-${USER:-user}-${RLLM_JOB_ID}-XXXXXX")
export RLLM_JOB_DIR=$RLLM_JOB_TMP
export RLLM_HOME=${RLLM_HOME:-$RLLM_JOB_DIR/data}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$RLLM_JOB_DIR/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$RLLM_JOB_DIR/torchinductor}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$RLLM_JOB_DIR/cuda}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$RLLM_JOB_DIR/flashinfer}
# Short socket paths are necessary for Ray's Unix sockets.
export TMPDIR="$RLLM_JOB_TMP/tmp" RAY_TMPDIR="$RLLM_JOB_TMP/ray"
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$RLLM_LOG_DIR" "$RLLM_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
if [[ -f /etc/ssl/certs/ca-certificates.crt ]]; then
    export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}
    export REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}
fi
export RLLM_GPU_KEEPALIVE=${RLLM_GPU_KEEPALIVE:-1}
RLLM_GPU_KEEPALIVE_PID=""
rllm_start_gpu_keepalive() {
    if [[ "$RLLM_GPU_KEEPALIVE" != 1 || "${RLLM_DISABLE_GPU_KEEPALIVE:-0}" == 1 ]]; then
        return 0
    fi
    if [[ -n "$RLLM_GPU_KEEPALIVE_PID" ]] && kill -0 "$RLLM_GPU_KEEPALIVE_PID" 2>/dev/null; then
        return 0
    fi
    # Inherit this allocation's visible GPUs. The helper exits if CUDA is absent;
    # callers can disable it or override these per-device defaults. `fill` keeps
    # the reported utilization near the target while yielding to training;
    # `fixed` is the earlier interval/work schedule.
    export RLLM_GPU_KEEPALIVE_MODE=${RLLM_GPU_KEEPALIVE_MODE:-fill}
    export RLLM_GPU_KEEPALIVE_TARGET_UTIL=${RLLM_GPU_KEEPALIVE_TARGET_UTIL:-85}
    export RLLM_GPU_KEEPALIVE_INTERVAL_SEC=${RLLM_GPU_KEEPALIVE_INTERVAL_SEC:-5.0}
    export RLLM_GPU_KEEPALIVE_WORK_SEC=${RLLM_GPU_KEEPALIVE_WORK_SEC:-0.5}
    export RLLM_GPU_KEEPALIVE_SIZE=${RLLM_GPU_KEEPALIVE_SIZE:-4096}
    "$PYTHON_BIN" "$RLLM_DIR/scripts/gpu_keepalive.py" >"$RLLM_LOG_DIR/gpu-keepalive-$RLLM_JOB_ID.log" 2>&1 &
    RLLM_GPU_KEEPALIVE_PID=$!
    printf '[runtime] gpu_keepalive_pid=%s mode=%s target_util=%s interval=%s work=%s size=%s\n' "$RLLM_GPU_KEEPALIVE_PID" "$RLLM_GPU_KEEPALIVE_MODE" "$RLLM_GPU_KEEPALIVE_TARGET_UTIL" "$RLLM_GPU_KEEPALIVE_INTERVAL_SEC" "$RLLM_GPU_KEEPALIVE_WORK_SEC" "$RLLM_GPU_KEEPALIVE_SIZE"
}

rllm_training_cleanup() {
    local status=${1:-$?}
    local attempt
    if [[ -n "$RLLM_GPU_KEEPALIVE_PID" ]]; then
        kill -TERM "$RLLM_GPU_KEEPALIVE_PID" 2>/dev/null || true
        # Bound shutdown even if a CUDA call in the helper is unresponsive.
        for (( attempt=0; attempt<100; attempt++ )); do
            kill -0 "$RLLM_GPU_KEEPALIVE_PID" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$RLLM_GPU_KEEPALIVE_PID" 2>/dev/null; then
            kill -KILL "$RLLM_GPU_KEEPALIVE_PID" 2>/dev/null || true
        fi
        wait "$RLLM_GPU_KEEPALIVE_PID" 2>/dev/null || true
        RLLM_GPU_KEEPALIVE_PID=""
    fi
    if [[ "$status" -ne 0 && -d "$RAY_TMPDIR/session_latest/logs" ]]; then
        mkdir -p "$RLLM_LOG_DIR/ray-$RLLM_JOB_ID"
        cp -a "$RAY_TMPDIR/session_latest/logs/." "$RLLM_LOG_DIR/ray-$RLLM_JOB_ID/" || true
    fi
    # Only the directory created by this invocation is eligible for removal.
    # Never invoke per-user `ray stop` on nodes shared with other jobs.
    if [[ "${RLLM_CLEAN_JOB_TMP_ON_EXIT:-1}" == 1 ]]; then
        rm -rf -- "$RLLM_JOB_TMP"
    fi
    return "$status"
}
trap rllm_training_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Fail before model loading if the selected VeRL lacks the required repairs.
"$PYTHON_BIN" "$RLLM_DIR/scripts/verify_verl_compat.py" --installed
"$PYTHON_BIN" "$RLLM_DIR/scripts/patch_verl_zmq_ipc.py" --check
rllm_start_gpu_keepalive
printf '[runtime] benchmark=%s method=%s gpu=%s python=%s job=%s\n' "$BENCHMARK" "$TRAINING_METHOD" "$RLLM_GPU_FAMILY" "$PYTHON_BIN" "$RLLM_JOB_ID"
