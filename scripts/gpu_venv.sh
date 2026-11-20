#!/usr/bin/env bash

rllm_detect_gpu_name() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi

    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/[[:space:]]*$//')
    [ -n "$gpu_name" ] || return 1
    printf '%s\n' "$gpu_name"
}

rllm_gpu_family_from_name() {
    local gpu_name_lower
    gpu_name_lower=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')

    case "$gpu_name_lower" in
        *rtx*pro*5000* | *rtx*pro*6000*)
            printf '%s\n' "rtxpro"
            ;;
        *rtx*6000*ada* | *rtx6000*ada*)
            printf '%s\n' "rtx6000"
            ;;
        *h200*)
            printf '%s\n' "h200"
            ;;
        *h100*)
            printf '%s\n' "h100"
            ;;
        *a100*)
            printf '%s\n' "a100"
            ;;
        *l40s*)
            printf '%s\n' "l40s"
            ;;
        *)
            return 1
            ;;
    esac
}

rllm_gpu_family_from_hint() {
    local hint_lower
    local has_a100=0
    local has_hopper=0
    local has_h100=0
    local has_h200=0
    local has_l40s=0
    local has_rtx6000=0
    local has_rtxpro=0
    local matches=0

    hint_lower=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')

    case "$hint_lower" in
        *a100*)
            has_a100=1
            ;;
    esac

    case "$hint_lower" in
        *h100* | *h200* | *hopper*)
            has_hopper=1
            ;;
    esac
    case "$hint_lower" in
        *h100*)
            has_h100=1
            ;;
    esac
    case "$hint_lower" in
        *h200*)
            has_h200=1
            ;;
    esac

    case "$hint_lower" in
        *l40s*)
            has_l40s=1
            ;;
    esac

    case "$hint_lower" in
        *rtx*pro*5000* | *rtx*pro*6000*)
            has_rtxpro=1
            ;;
        *rtx6000* | *rtx\ 6000*)
            has_rtx6000=1
            ;;
    esac

    matches=$((has_a100 + has_hopper + has_l40s + has_rtx6000 + has_rtxpro))
    [ "$matches" -eq 1 ] || return 1

    if [ "$has_a100" -eq 1 ]; then
        printf '%s\n' "a100"
    elif [ "$has_l40s" -eq 1 ]; then
        printf '%s\n' "l40s"
    elif [ "$has_rtx6000" -eq 1 ]; then
        printf '%s\n' "rtx6000"
    elif [ "$has_rtxpro" -eq 1 ]; then
        printf '%s\n' "rtxpro"
    elif [ "$has_hopper" -eq 1 ]; then
        if [ "$has_h200" -eq 1 ] && [ "$has_h100" -eq 0 ]; then
            printf '%s\n' "h200"
        elif [ "$has_h100" -eq 1 ] && [ "$has_h200" -eq 0 ]; then
            printf '%s\n' "h100"
        else
            printf '%s\n' "hopper"
        fi
    fi
}

rllm_detect_gpu_family_from_slurm() {
    local hint
    local gpu_family
    local job_details
    local node_details
    local node_list

    for hint in \
        "${SLURM_JOB_CONSTRAINTS:-}" \
        "${SLURM_JOB_CONSTRAINT:-}" \
        "${SBATCH_CONSTRAINT:-}" \
        "${SLURM_JOB_PARTITION:-}" \
        "${SLURM_JOB_GRES:-}" \
        "${SLURM_STEP_GRES:-}" \
        "${SLURM_JOB_NODELIST:-}" \
        "${SLURM_NODELIST:-}" \
        "${SLURM_SUBMIT_LINE:-}"; do
        if [ -n "$hint" ] && gpu_family=$(rllm_gpu_family_from_hint "$hint"); then
            printf '%s\n' "$gpu_family"
            return 0
        fi
    done

    if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
        job_details=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null || true)
        if [ -n "$job_details" ] && gpu_family=$(rllm_gpu_family_from_hint "$job_details"); then
            printf '%s\n' "$gpu_family"
            return 0
        fi
    fi

    node_list=${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}
    if command -v scontrol >/dev/null 2>&1 && [ -n "$node_list" ]; then
        node_details=$(scontrol show node "$node_list" 2>/dev/null || true)
        if [ -n "$node_details" ] && gpu_family=$(rllm_gpu_family_from_hint "$node_details"); then
            printf '%s\n' "$gpu_family"
            return 0
        fi
    fi

    return 1
}

rllm_normalize_gpu_family() {
    local gpu_family_lower
    gpu_family_lower=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')

    case "$gpu_family_lower" in
        a100 | hopper | h100 | h200 | l40s | rtx6000 | rtxpro)
            printf '%s\n' "$gpu_family_lower"
            ;;
        *)
            return 1
            ;;
    esac
}

rllm_flash_attn_arch_from_family() {
    case "$1" in
        a100)
            printf '%s %s %s\n' "80" "8.0" "80"
            ;;
        l40s | rtx6000)
            printf '%s %s %s\n' "89" "8.9" "89"
            ;;
        rtxpro)
            printf '%s %s %s\n' "120" "12.0" "120"
            ;;
        hopper | h100 | h200)
            printf '%s %s %s\n' "90" "9.0" "90"
            ;;
        *)
            return 1
            ;;
    esac
}

rllm_gpu_env_key_from_family() {
    case "$1" in
        h100 | h200 | hopper)
            printf '%s\n' "hopper"
            ;;
        rtx6000)
            printf '%s\n' "l40s"
            ;;
        a100 | l40s | rtxpro)
            printf '%s\n' "$1"
            ;;
        *)
            return 1
            ;;
    esac
}

rllm_default_venv_dir() {
    printf '%s/.venv-%s\n' "$1" "$2"
}

rllm_sanitize_inherited_cache_paths() {
    # Some Slurm login environments export cache locations under
    # /scratch/$USER.  Such filesystems can have a small inode quota, so a stale
    # inherited Hugging Face cache can override a launcher's managed cache
    # directory and fail inside a Ray worker.  Launchers set their own managed
    # cache paths after resolving the GPU environment; discard only these
    # inherited per-user scratch locations first.  Set RLLM_ALLOW_SCRATCH_CACHE=1
    # to deliberately retain one.
    if [ "${RLLM_ALLOW_SCRATCH_CACHE:-0}" = "1" ] || [ -z "${USER:-}" ]; then
        return 0
    fi

    local name
    local value
    for name in \
        CACHE_DIR \
        HF_HOME \
        HF_HUB_CACHE \
        HF_DATASETS_CACHE \
        HUGGINGFACE_HUB_CACHE \
        TRANSFORMERS_CACHE \
        XDG_CACHE_HOME; do
        value=${!name:-}
        case "$value" in
            "/scratch/${USER}"|"/scratch/${USER}"/*)
                echo "[launcher] ignoring inherited $name=$value; launcher-managed cache will be used" >&2
                unset "$name"
                ;;
        esac
    done
}

# Create the resolved environment when it is missing. One job per GPU family
# builds it under a lock on the shared checkout; concurrent jobs wait for it. A
# marker beside the environment makes an interrupted build start over.
rllm_ensure_gpu_env() {
    local rllm_dir=${1:?rllm_dir is required}
    local incomplete="$VENV_DIR.setup-incomplete"
    local lock_fd recreate=0
    if [ -x "$VENV_DIR/bin/python" ] && [ ! -e "$incomplete" ]; then
        return 0
    fi
    if [ "${RLLM_AUTO_SETUP:-1}" != "1" ]; then
        echo "Missing GPU environment $VENV_DIR; run scripts/setup_qwen35.sh or unset RLLM_AUTO_SETUP=0." >&2
        return 1
    fi
    mkdir -p "$rllm_dir/.cache/locks"
    exec {lock_fd}>"$rllm_dir/.cache/locks/gpu-env-$RLLM_GPU_ENV_KEY.lock"
    echo "[setup] waiting for the $RLLM_GPU_ENV_KEY environment lock" >&2
    flock "$lock_fd"
    if [ ! -x "$VENV_DIR/bin/python" ] || [ -e "$incomplete" ]; then
        [ ! -e "$incomplete" ] || recreate=1
        echo "[setup] preparing $VENV_DIR for $RLLM_GPU_FAMILY; this runs once per GPU family" >&2
        : > "$incomplete"
        if ! RECREATE_VENV=$recreate bash "$rllm_dir/scripts/setup_qwen35.sh" >&2; then
            exec {lock_fd}>&-
            echo "GPU environment setup failed; see the log above. The next launch retries it." >&2
            return 1
        fi
        rm -f "$incomplete"
    fi
    exec {lock_fd}>&-
}

rllm_resolve_gpu_env() {
    local rllm_dir=${1:?rllm_dir is required}
    local gpu_family=${RLLM_GPU_FAMILY:-${GPU_FAMILY:-}}
    local gpu_name=${RLLM_GPU_NAME:-${GPU_NAME:-}}
    local gpu_env_key
    local flash_attn_cuda_archs
    local torch_cuda_arch_list
    local cuda_archs

    rllm_sanitize_inherited_cache_paths
    export RLLM_CACHE_DIR="${RLLM_CACHE_DIR:-$rllm_dir/.cache}"

    if [ -n "$gpu_family" ]; then
        local requested_gpu_family=$gpu_family
        if ! gpu_family=$(rllm_normalize_gpu_family "$requested_gpu_family"); then
            echo "Unsupported GPU_FAMILY=$requested_gpu_family. Set GPU_FAMILY to one of: a100 hopper l40s rtxpro (h100/h200/rtx6000 aliases are accepted)." >&2
            return 1
        fi
    fi

    if [ -z "$gpu_name" ]; then
        gpu_name=$(rllm_detect_gpu_name || true)
    fi

    if [ -z "$gpu_family" ] && [ -n "$gpu_name" ]; then
        gpu_family=$(rllm_gpu_family_from_name "$gpu_name" || true)
    fi

    if [ -z "$gpu_family" ]; then
        gpu_family=$(rllm_detect_gpu_family_from_slurm || true)
    fi

    if [ -z "$gpu_family" ]; then
        echo "Unable to determine GPU family automatically. Set GPU_FAMILY to one of: a100 hopper l40s rtxpro (h100/h200/rtx6000 aliases are accepted)." >&2
        return 1
    fi

    if ! gpu_env_key=$(rllm_gpu_env_key_from_family "$gpu_family"); then
        echo "Unsupported GPU_FAMILY=$gpu_family. Set GPU_FAMILY to one of: a100 hopper l40s rtxpro (h100/h200/rtx6000 aliases are accepted)." >&2
        return 1
    fi
    if ! read -r flash_attn_cuda_archs torch_cuda_arch_list cuda_archs < <(rllm_flash_attn_arch_from_family "$gpu_env_key"); then
        echo "Unable to resolve CUDA arch settings for GPU_FAMILY=$gpu_family." >&2
        return 1
    fi

    export RLLM_GPU_FAMILY="$gpu_family"
    export RLLM_GPU_ENV_KEY="$gpu_env_key"
    export RLLM_GPU_NAME="${gpu_name:-unknown}"
    export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-$flash_attn_cuda_archs}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$torch_cuda_arch_list}"
    export CUDAARCHS="${CUDAARCHS:-$cuda_archs}"
    export VENV_DIR="${VENV_DIR:-$(rllm_default_venv_dir "$rllm_dir" "$gpu_env_key")}"
}
