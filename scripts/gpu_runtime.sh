#!/usr/bin/env bash
# Source this file, then use the Python interpreter selected by your launcher.
# Discovery never imports torch, installs packages, or creates toolkit symlinks.
rllm_configure_gpu_runtime() {
    local runtime_python=${1:?Python interpreter is required}
    shift
    if [[ ${RLLM_REQUIRE_CUDA_COMPILER:-0} == 1 ]]; then
        set -- --require-compiler "$@"
    fi
    local runtime_script_dir
    local runtime_exports
    runtime_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return
    runtime_exports=$("$runtime_python" "$runtime_script_dir/gpu_runtime.py" --format shell "$@") || return
    # The helper emits only fixed export names with shlex.quote-escaped values.
    eval "$runtime_exports"
}
