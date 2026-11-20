#!/usr/bin/env bash
# Serving lifecycle for evaluation launchers. Source after benchmark presets.
# No installs or broad process cleanup; only processes created by this launcher.
EVAL_GPU_KEEPALIVE_PID=""

eval_print_command() {
    printf '%q ' "$@"
    printf '\n'
}

eval_check_resume_dir() {
    case ${RESUME:-1} in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; return 2 ;; esac
    [[ -d "$RUN_DIR" ]] || return 0
    local path has_output=0
    for path in "$RUN_DIR"/* "$RUN_DIR"/.[!.]* "$RUN_DIR"/..?*; do
        [[ -e "$path" || -L "$path" ]] || continue
        # The second check runs after eval_lock_run creates this metadata file.
        [[ ${path##*/} == .evaluation.lock && -f "$path" && ! -L "$path" ]] && continue
        has_output=1
        break
    done
    (( has_output )) || return 0
    if [[ ${RESUME:-1} != 1 ]]; then
        echo "Run directory is not empty: $RUN_DIR; use RESUME=1 or another OUTPUT_DIR" >&2
        return 1
    fi
    if [[ ! -f "$RUN_DIR/evaluation_config.json" ]]; then
        for path in "$RUN_DIR"/* "$RUN_DIR"/.[!.]*; do
            [[ -e "$path" ]] || continue
            case ${path##*/} in vllm.log|wandb_run_id.txt|.evaluation.lock) ;; *)
                echo "Run directory is not empty and has no evaluation configuration: $RUN_DIR; choose another OUTPUT_DIR" >&2
                return 1 ;;
            esac
        done
    fi
}

eval_lock_run() {
    command -v flock >/dev/null || { echo "flock is required for resumable evaluations" >&2; return 1; }
    exec {EVAL_RUN_LOCK_FD}>"$RUN_DIR/.evaluation.lock"
    flock -n "$EVAL_RUN_LOCK_FD" || { echo "Evaluation is already running in $RUN_DIR" >&2; return 1; }
}

eval_unlock_run() {
    flock -u "$EVAL_RUN_LOCK_FD"
    exec {EVAL_RUN_LOCK_FD}>&-
}

eval_resolve_paths() {
    local resolved
    resolved=$(RUN_DIR="${RUN_DIR:-}" STEP="${STEP:-0}" STEPS="${STEP:-0}" MODEL_PATH="${MODEL_PATH:-}" MODEL_NAME="$MODEL_NAME" MODEL_SOURCE="${MODEL_SOURCE:-auto}" EVAL_QUANTIZATION="${EVAL_QUANTIZATION:-none}" "${LAUNCHER_PYTHON:-python3}" "$RLLM_DIR/scripts/evaluation_runtime.py" "$BENCHMARK" "$EVAL_MODE" --shell-paths) || return
    eval "$resolved"
}

eval_merge_checkpoint() {
    # The shell-managed paths own their helper before potentially costly merging.
    # Python-managed evaluations have a separate lifecycle and never source this.
    eval_start_gpu_keepalive
    # EVAL_QUANTIZATION=fp8 serves MODEL_PATH, a block-FP8 copy of the BF16
    # QUANTIZATION_SOURCE (merged checkpoint, local model or Hub base).
    local bf16_model=${QUANTIZATION_SOURCE:-$MODEL_PATH}
    eval_merge_bf16_checkpoint "$bf16_model" || return
    [[ ${EVAL_QUANTIZATION:-none} == fp8 ]] || return 0
    local source_flag=--source-dir
    [[ ${MODEL_SOURCE:-hub} != hub ]] || source_flag=--hub-model
    # Runs in the evaluation allocation with its GPU visible. EVAL_FP8_KERNEL=replica
    # is the explicit CPU fallback; the default VeRL kernel fails without a GPU.
    case ${EVAL_FP8_KERNEL:-verl} in verl|replica) ;; *) echo "EVAL_FP8_KERNEL must be verl or replica" >&2; return 2 ;; esac
    local command=("$PYTHON_BIN" "$RLLM_DIR/scripts/convert_fp8_checkpoint.py" "$source_flag" "$bf16_model" --target-dir "$MODEL_PATH" --kernel "${EVAL_FP8_KERNEL:-verl}")
    [[ -z ${EVAL_FP8_MIN_CAPABILITY:-} ]] || command+=(--min-capability "$EVAL_FP8_MIN_CAPABILITY")
    if [[ ${RLLM_DRY_RUN:-0} == 1 ]]; then
        eval_print_command "${command[@]}"
    else
        "${command[@]}"
    fi
}

eval_merge_bf16_checkpoint() {
    local MODEL_PATH=$1
    if [[ -z ${CHECKPOINT_SOURCE:-} ]]; then
        if [[ ${RLLM_DRY_RUN:-0} != 1 && ${MODEL_SOURCE:-hub} != hub && ${MODEL_SOURCE:-hub} != api ]]; then
            "$PYTHON_BIN" - "$MODEL_PATH" <<'PY'
import sys
from pathlib import Path
model = Path(sys.argv[1])
if not (model / "config.json").is_file() or not any(model.glob("*.safetensors")):
    raise SystemExit(f"Expected merged Hugging Face config.json and .safetensors: {model}")
PY
        fi
        return 0
    fi
    local command=("$PYTHON_BIN" "$RLLM_DIR/scripts/merge_eval_checkpoint.py"
        --source-dir "$CHECKPOINT_SOURCE" --target-dir "$MODEL_PATH" --python "$PYTHON_BIN")
    if [[ ${RLLM_DRY_RUN:-0} == 1 ]]; then
        eval_print_command "${command[@]}"
    else
        "${command[@]}"
    fi
}

eval_record_config() {
    "$PYTHON_BIN" - "$RUN_DIR/evaluation_config.json" "${COMMAND[@]}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path
keys = ("BENCHMARK", "EVAL_MODE", "PRESET", "STEP", "CHECKPOINT_SOURCE", "MODEL_PATH", "MODEL_NAME")
payload = {key.lower(): os.environ.get(key) for key in keys}
payload.update(benchmark=os.environ["BENCHMARK"], method=os.environ["EVAL_MODE"],
               preset=os.environ["PRESET"], step=int(os.environ.get("STEP", "0")),
               run_dir=os.environ["RUN_DIR"])
offset = int(os.environ.get("TASK_OFFSET", "0"))
limit = os.environ.get("MAX_TASKS") if payload["benchmark"] == "mini_swe_bench" else os.environ.get("TASK_LIMIT")
if limit:
    end = offset + int(limit) - 1
    if payload["benchmark"] == "mastermind":
        end = min(end, int(os.environ.get("TEST_SPLIT_SIZE", "100")) - 1)
    payload["task_label"] = f"{offset}-{end}" if offset else str(end + 1)
else:
    payload["task_label"] = f"from{offset}" if offset else "all"
command = [arg for arg in sys.argv[2:] if arg not in {"--resume", "--no-resume"}]
for index, token in enumerate(command[:-1]):
    if token in {"--base_url", "--base-url"} and re.fullmatch(r"http://127\.0\.0\.1:[0-9]+/v1", command[index + 1]):
        command[index + 1] = "<managed-local-model-server>"
payload["evaluation"] = command
if "EVAL_TIE_WORD_EMBEDDINGS" in os.environ:
    value = os.environ["EVAL_TIE_WORD_EMBEDDINGS"].lower()
    if value not in {"true", "false", "1", "0"}:
        raise SystemExit("EVAL_TIE_WORD_EMBEDDINGS must be True/False or 1/0")
    if os.environ.get("MODEL_SOURCE") == "api":
        raise SystemExit("EVAL_TIE_WORD_EMBEDDINGS requires a local, Hub or checkpoint model")
    payload["hf_overrides"] = {"tie_word_embeddings": value in {"true", "1"}}
if "--adaptation_mode" in command or "--provider_seed_scheme" in command or "--handoff_protocol" in command:
    def option(name, default=None):
        return command[command.index(name) + 1] if name in command else default
    seeds = None
    if option("--provider_seed_scheme", "legacy") != "legacy":
        seeds = {"scheme": option("--provider_seed_scheme"), "namespace": option("--provider_seed_namespace"), "evaluation_seed": int(option("--seed", "42"))}
    payload["handoff_control"] = {"adaptation_mode": option("--adaptation_mode", "apply"), "provider_seed": seeds}
    if option("--handoff_protocol", "legacy") != "legacy":
        payload["handoff_control"]["handoff_protocol"] = option("--handoff_protocol")
if os.environ.get("EVAL_QUANTIZATION", "none") != "none":
    # Absent for BF16 so earlier records still resume unchanged.
    payload["quantization"] = os.environ["EVAL_QUANTIZATION"]
    payload["quantization_source"] = os.environ.get("QUANTIZATION_SOURCE")
    payload["quantization_kernel"] = os.environ.get("EVAL_FP8_KERNEL") or "verl"
    marker = Path(os.environ["MODEL_PATH"]) / "fp8_conversion.json"
    if marker.is_file():
        # Implementation, device and compute capability that produced the served weights.
        payload["quantization_conversion"] = json.loads(marker.read_text()).get("quantizer")
if "--local_eval_protocol" in command:
    payload["local_eval_protocol"] = command[command.index("--local_eval_protocol") + 1]
elif payload["benchmark"] == "mastermind" and payload["method"] == "vanilla":
    payload["local_eval_protocol"] = "independent_samples_v1"
path = Path(sys.argv[1])
if path.exists():
    previous = json.loads(path.read_text())
    if previous != payload:
        old_command, new_command = previous.get("evaluation"), payload["evaluation"]
        old_others = {key: value for key, value in previous.items() if key != "evaluation"}
        new_others = {key: value for key, value in payload.items() if key != "evaluation"}
        has_completed_work = any(path.parent.glob("task_*/completed_visits/*.json")) or any(path.parent.glob("task_*/completed_final_policy/*.json")) or any(path.parent.glob("completed_batches/*.json"))
        if has_completed_work or old_others != new_others or not isinstance(old_command, list) or old_command[1:] != new_command[1:]:
            raise SystemExit(f"Evaluation configuration differs from {path}; use another OUTPUT_DIR")
        path.write_text(json.dumps(payload, indent=2) + "\n")
else:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
PY
}

eval_complete_run() {
    # Called only after the evaluator returns zero, while the launcher owns its lock.
    "$PYTHON_BIN" "$RLLM_DIR/scripts/evaluation_runtime.py" "$BENCHMARK" "$EVAL_MODE" \
        --complete-recorded "$RUN_DIR/evaluation_config.json"
}

eval_setup_python() {
    if [[ ${RLLM_DRY_RUN:-0} == 1 ]]; then
        PYTHON_BIN=${PYTHON_BIN:-python3}
        return
    fi
    if [[ -z ${PYTHON_BIN:-} ]]; then
        if [[ ${MODEL_SOURCE:-hub} != api && -z ${VENV_DIR:-} ]]; then
            source "$RLLM_DIR/scripts/gpu_venv.sh"
            rllm_resolve_gpu_env "$RLLM_DIR"
            rllm_ensure_gpu_env "$RLLM_DIR"
        fi
        PYTHON_BIN=${VENV_DIR:+$VENV_DIR/bin/python}
        PYTHON_BIN=${PYTHON_BIN:-python3}
    fi
    command -v "$PYTHON_BIN" >/dev/null || { echo "Python interpreter not found: $PYTHON_BIN" >&2; return 1; }
    export PYTHONPATH="$RLLM_DIR${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
    if [[ ${MODEL_SOURCE:-hub} != api ]]; then
        source "$RLLM_DIR/scripts/gpu_runtime.sh"
        rllm_configure_gpu_runtime "$PYTHON_BIN"
    fi
}

eval_prepare_tracking() {
    [[ -n ${WANDB_PROJECT:-} && ${RLLM_DRY_RUN:-0} != 1 ]] || return 0
    export WANDB_DIR=$RUN_DIR
    if [[ -s "$RUN_DIR/wandb_run_id.txt" ]]; then
        WANDB_RUN_ID=$(<"$RUN_DIR/wandb_run_id.txt")
    else
        WANDB_RUN_ID=${REQUESTED_WANDB_RUN_ID:-$("$PYTHON_BIN" -c 'import uuid; print(uuid.uuid4().hex[:8])')}
        printf '%s\n' "$WANDB_RUN_ID" > "$RUN_DIR/wandb_run_id.txt"
    fi
    export WANDB_RUN_ID
    export WANDB_RESUME=${WANDB_RESUME:-allow}
}

eval_start_gpu_keepalive() {
    [[ ${RLLM_DRY_RUN:-0} != 1 && ${MODEL_SOURCE:-hub} != api && -n ${SLURM_JOB_ID:-} ]] || return 0
    [[ ${RLLM_GPU_KEEPALIVE:-1} == 1 && ${RLLM_DISABLE_GPU_KEEPALIVE:-0} != 1 ]] || return 0
    if [[ -n "$EVAL_GPU_KEEPALIVE_PID" ]] && kill -0 "$EVAL_GPU_KEEPALIVE_PID" 2>/dev/null; then
        return 0
    fi
    # Keep the output directory's resume checks independent of helper logs.
    local log_dir
    log_dir=$(dirname -- "$RUN_DIR")
    mkdir -p -- "$log_dir"
    trap eval_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    "$PYTHON_BIN" "$RLLM_DIR/scripts/gpu_keepalive.py" >"$log_dir/gpu-keepalive-$SLURM_JOB_ID.log" 2>&1 &
    EVAL_GPU_KEEPALIVE_PID=$!
    printf '[evaluation] gpu_keepalive_pid=%s log=%s\n' "$EVAL_GPU_KEEPALIVE_PID" "$log_dir/gpu-keepalive-$SLURM_JOB_ID.log"
}

eval_stop_gpu_keepalive() {
    [[ -n "$EVAL_GPU_KEEPALIVE_PID" ]] || return 0
    local attempt
    kill -TERM "$EVAL_GPU_KEEPALIVE_PID" 2>/dev/null || true
    for (( attempt=0; attempt<100; attempt++ )); do
        kill -0 "$EVAL_GPU_KEEPALIVE_PID" 2>/dev/null || break
        sleep 0.1
    done
    kill -KILL "$EVAL_GPU_KEEPALIVE_PID" 2>/dev/null || true
    wait "$EVAL_GPU_KEEPALIVE_PID" 2>/dev/null || true
    EVAL_GPU_KEEPALIVE_PID=""
}

eval_stop_server() {
    if [[ -n ${VLLM_PID:-} ]]; then
        kill -- "-$VLLM_PID" 2>/dev/null || true
        for _ in {1..10}; do
            kill -0 -- "-$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}

eval_cleanup() {
    eval_stop_gpu_keepalive
    eval_stop_server
    local directory
    for directory in "${EVAL_OWNED_TEMP_DIRS[@]:-}"; do
        [[ -z "$directory" ]] || rm -rf -- "$directory"
    done
}

eval_select_server_url() {
    [[ ${MODEL_SOURCE:-hub} != api ]] || return 0
    VLLM_PORT=${VLLM_PORT:-$((20000 + ${SLURM_JOB_ID:-$$} % 30000))}
    BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
}

eval_start_server() {
    [[ ${MODEL_SOURCE:-hub} != api ]] || return 0
    eval_select_server_url
    SERVER_COMMAND=("$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
        --model "$MODEL_PATH" --served-model-name "$MODEL_NAME"
        --trust-remote-code --host 127.0.0.1 --port "$VLLM_PORT"
        --dtype bfloat16 --generation-config vllm
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
        --tensor-parallel-size "${VLLM_TP:-1}" --data-parallel-size "${VLLM_DP:-1}"
        --max-num-seqs "${MAX_NUM_SEQS:-64}" --max-model-len "$MAX_MODEL_LEN")
    [[ ${EVAL_QUANTIZATION:-none} == none ]] || SERVER_COMMAND+=(--quantization "$EVAL_QUANTIZATION")
    # eval_resolve_paths validates and canonicalizes this optional boolean before
    # runtime setup. The checkpoint's config.json remains untouched.
    if [[ ${EVAL_TIE_WORD_EMBEDDINGS+x} ]]; then
        SERVER_COMMAND+=(--hf-overrides "{\"tie_word_embeddings\":$EVAL_TIE_WORD_EMBEDDINGS}")
    fi
    if [[ ${RLLM_DRY_RUN:-0} == 1 ]]; then
        eval_print_command "${SERVER_COMMAND[@]}"
        return
    fi
    # Fail on any bound TCP listener, even a same-model server owned by another job.
    "$PYTHON_BIN" - "$VLLM_PORT" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
    command -v setsid >/dev/null || { echo "setsid is required to isolate the evaluation server" >&2; return 1; }
    setsid "${SERVER_COMMAND[@]}" >"$RUN_DIR/vllm.log" 2>&1 &
    VLLM_PID=$!
    trap eval_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    for _ in $(seq 1 "${VLLM_STARTUP_ATTEMPTS:-240}"); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM exited during startup; see $RUN_DIR/vllm.log" >&2
            return 1
        fi
        if "$PYTHON_BIN" - "$BASE_URL" "$MODEL_NAME" <<'PY' 2>/dev/null
import json
import sys
import urllib.request
with urllib.request.urlopen(sys.argv[1] + "/models", timeout=5) as response:
    models = json.load(response)["data"]
sys.exit(0 if any(model.get("id") == sys.argv[2] for model in models) else 1)
PY
        then
            return 0
        fi
        sleep "${VLLM_POLL_SECONDS:-5}"
    done
    echo "vLLM did not become ready; see $RUN_DIR/vllm.log" >&2
    return 1
}
