#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
RLLM_DIR=${RLLM_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}

source "$SCRIPT_DIR/gpu_venv.sh"
rllm_resolve_gpu_env "$RLLM_DIR"

VENV_PYTHON=${VENV_PYTHON:-$VENV_DIR/bin/python}
UV_BIN=${UV_BIN:-${HOME}/.local/bin/uv}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.8.3.post1}
CUDA_NVCC_VERSION=${CUDA_NVCC_VERSION:-13.0.88}
CUDA_NVVM_VERSION=${CUDA_NVVM_VERSION:-$CUDA_NVCC_VERSION}
CUDA_CRT_VERSION=${CUDA_CRT_VERSION:-$CUDA_NVCC_VERSION}
CUDA_RUNTIME_VERSION=${CUDA_RUNTIME_VERSION:-13.0.96}
CUDA_CCCL_VERSION=${CUDA_CCCL_VERSION:-13.0.85}
DEFAULT_MAX_JOBS=${SLURM_CPUS_PER_TASK:-4}
if [ "$DEFAULT_MAX_JOBS" -gt 8 ]; then
    DEFAULT_MAX_JOBS=8
fi
MAX_JOBS=${MAX_JOBS:-$DEFAULT_MAX_JOBS}

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Project Python not found at $VENV_PYTHON" >&2
    echo "Create the environment first with: GPU_FAMILY=$RLLM_GPU_ENV_KEY bash scripts/setup_qwen35.sh" >&2
    exit 1
fi

if [ ! -x "$UV_BIN" ]; then
    echo "uv not found at $UV_BIN" >&2
    exit 1
fi

SITE_PACKAGES=$("$VENV_PYTHON" - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)

IFS=$'\t' read -r PYTHON_TAG TORCH_BUILD_TAG TORCH_CUDA_BUILD_TAG PLATFORM_BUILD_TAG < <(
    "$VENV_PYTHON" - <<'PY'
import re
import sys
import sysconfig

import torch


def safe(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value)


print(
    "\t".join(
        [
            f"cp{sys.version_info.major}{sys.version_info.minor}",
            safe(torch.__version__),
            safe(torch.version.cuda or "unknown"),
            safe(sysconfig.get_platform()),
        ]
    )
)
PY
)

WHEELHOUSE_DIR="$RLLM_DIR/.flashattn-build/wheels/$RLLM_GPU_ENV_KEY/flash-attn-$FLASH_ATTN_VERSION/$PLATFORM_BUILD_TAG-$PYTHON_TAG/torch-$TORCH_BUILD_TAG/cuda-$TORCH_CUDA_BUILD_TAG"
LEGACY_WHEELHOUSE_DIR="$RLLM_DIR/.flashattn-build/wheels/$RLLM_GPU_ENV_KEY/flash-attn-$FLASH_ATTN_VERSION/-$PYTHON_TAG/torch-/cuda-"

verify_flash_attn() {
    "$VENV_PYTHON" - <<'PY'
import importlib.metadata as md
import torch
import flash_attn
import flash_attn_2_cuda

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flash-attn", md.version("flash-attn"))
print("flash_attn module", flash_attn.__file__)
print("flash_attn_2_cuda", flash_attn_2_cuda.__file__)
PY
}

find_cached_flash_attn_wheel() {
    find "$WHEELHOUSE_DIR" -maxdepth 1 -type f -name 'flash_attn-*.whl' | sort | tail -n 1
}

install_flash_attn_wheel() {
    local wheel_path=${1:?wheel_path is required}

    ensure_pip_for_wheel_builds
    echo "[flashattn-build] installing wheel $wheel_path"
    "$VENV_PYTHON" -m pip install --no-deps --force-reinstall "$wheel_path"
}

ensure_pip_for_wheel_builds() {
    if "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
        return 0
    fi

    echo "[flashattn-build] bootstrapping pip for wheel builds"
    "$VENV_PYTHON" -m ensurepip --upgrade >/tmp/flashattn-ensurepip.log 2>&1 || {
        cat /tmp/flashattn-ensurepip.log >&2
        exit 1
    }
}

adopt_legacy_flash_attn_wheel() {
    local legacy_wheel_path
    local wheel_basename

    if [ ! -d "$LEGACY_WHEELHOUSE_DIR" ] || [ "$LEGACY_WHEELHOUSE_DIR" = "$WHEELHOUSE_DIR" ]; then
        return 1
    fi

    legacy_wheel_path=$(find "$LEGACY_WHEELHOUSE_DIR" -maxdepth 1 -type f -name 'flash_attn-*.whl' | sort | tail -n 1)
    if [ -z "$legacy_wheel_path" ]; then
        return 1
    fi

    mkdir -p "$WHEELHOUSE_DIR"
    wheel_basename=$(basename "$legacy_wheel_path")
    mv "$legacy_wheel_path" "$WHEELHOUSE_DIR/$wheel_basename"
    echo "[flashattn-build] adopted legacy cached wheel into $WHEELHOUSE_DIR"
}

try_cached_flash_attn_wheel() {
    local wheel_path

    wheel_path=$(find_cached_flash_attn_wheel)
    if [ -z "$wheel_path" ]; then
        return 1
    fi

    echo "[flashattn-build] found cached wheel in $WHEELHOUSE_DIR"
    if ! install_flash_attn_wheel "$wheel_path"; then
        echo "[flashattn-build] cached wheel install failed; rebuilding from source"
        rm -f "$wheel_path"
        return 1
    fi

    if verify_flash_attn >/tmp/flashattn-verify-wheel.log 2>&1; then
        cat /tmp/flashattn-verify-wheel.log
        return 0
    fi

    echo "[flashattn-build] cached wheel failed verification; rebuilding from source"
    cat /tmp/flashattn-verify-wheel.log >&2
    rm -f "$wheel_path"
    return 1
}

build_flash_attn_wheel() {
    local wheel_path

    ensure_pip_for_wheel_builds
    mkdir -p "$WHEELHOUSE_DIR"
    rm -f "$WHEELHOUSE_DIR"/flash_attn-*.whl

    echo "[flashattn-build] building wheel into $WHEELHOUSE_DIR"
    "$VENV_PYTHON" -m pip wheel \
        --no-deps \
        --no-build-isolation \
        --no-cache-dir \
        --no-binary flash-attn \
        --wheel-dir "$WHEELHOUSE_DIR" \
        "flash-attn==$FLASH_ATTN_VERSION"

    wheel_path=$(find_cached_flash_attn_wheel)
    if [ -z "$wheel_path" ]; then
        echo "[flashattn-build] expected a built wheel in $WHEELHOUSE_DIR but none was found" >&2
        exit 1
    fi

    install_flash_attn_wheel "$wheel_path"
}

cd "$RLLM_DIR"
mkdir -p "$RLLM_DIR/.flashattn-build"
mkdir -p "$WHEELHOUSE_DIR"
adopt_legacy_flash_attn_wheel || true

LOCK_FILE="$RLLM_DIR/.flashattn-build/rebuild_flash_attn-${RLLM_GPU_ENV_KEY}.lock"
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        echo "[flashattn-build] another rebuild job is already active for $RLLM_GPU_ENV_KEY; waiting for it to finish"
        flock 9
    fi
fi

if verify_flash_attn >/tmp/flashattn-verify-before.log 2>&1; then
    echo "[flashattn-build] flash-attn already imports cleanly for $RLLM_GPU_ENV_KEY; nothing to rebuild"
    cat /tmp/flashattn-verify-before.log
    exit 0
fi

if try_cached_flash_attn_wheel; then
    echo "[flashattn-build] restored flash-attn from cached wheel"
    exit 0
fi

if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
    export CURL_CA_BUNDLE=$SSL_CERT_FILE
    unset SSL_CERT_DIR
elif [ -f /etc/pki/tls/cert.pem ]; then
    export SSL_CERT_FILE=/etc/pki/tls/cert.pem
    export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
    export CURL_CA_BUNDLE=$SSL_CERT_FILE
    unset SSL_CERT_DIR
fi

echo "[flashattn-build] installing CUDA 13 build dependencies into $VENV_DIR"
"$UV_BIN" pip install --python "$VENV_PYTHON" --upgrade \
    "nvidia-cuda-nvcc==$CUDA_NVCC_VERSION" \
    "nvidia-nvvm==$CUDA_NVVM_VERSION" \
    "nvidia-cuda-crt==$CUDA_CRT_VERSION" \
    "nvidia-cuda-runtime==$CUDA_RUNTIME_VERSION" \
    "nvidia-cuda-cccl==$CUDA_CCCL_VERSION"

CUDA_HOME=${CUDA_HOME:-$SITE_PACKAGES/nvidia/cu13}
export CUDA_HOME

ensure_cuda_linker_compat() {
    local cuda_lib_dir=${1:?cuda_lib_dir is required}
    local cuda_lib64_dir="$CUDA_HOME/lib64"
    local libcudart_target

    if [ ! -d "$cuda_lib_dir" ]; then
        echo "Missing CUDA library directory: $cuda_lib_dir" >&2
        exit 1
    fi

    if [ ! -e "$cuda_lib64_dir" ]; then
        ln -s lib "$cuda_lib64_dir"
    fi

    libcudart_target=$(find "$cuda_lib_dir" -maxdepth 1 -type f -name 'libcudart.so.*' | sort -V | tail -n 1)
    if [ -z "$libcudart_target" ]; then
        echo "Missing versioned CUDA runtime library under $cuda_lib_dir" >&2
        exit 1
    fi

    if [ ! -e "$cuda_lib_dir/libcudart.so" ]; then
        ln -s "$(basename "$libcudart_target")" "$cuda_lib_dir/libcudart.so"
    fi
}

verify_cudart_link() {
    local link_test_bin
    link_test_bin=$(mktemp "$RLLM_DIR/.flashattn-build/cudart-linktest-XXXXXX")
    rm -f "$link_test_bin"

    if ! printf 'int main(void) { return 0; }\n' | \
        "$CXX" -x c++ - -o "$link_test_bin" -L"$CUDA_HOME/lib" -L"$CUDA_HOME/lib64" -lcudart >/tmp/flashattn-cudart-link.log 2>&1; then
        echo "[flashattn-build] failed to link a test binary against libcudart" >&2
        cat /tmp/flashattn-cudart-link.log >&2
        rm -f "$link_test_bin"
        exit 1
    fi

    rm -f "$link_test_bin"
}

ensure_cuda_linker_compat "$CUDA_HOME/lib"

TOOLCHAIN_BIN_DIR="$RLLM_DIR/.flashattn-build/toolchain-bin-${RLLM_GPU_ENV_KEY}"
mkdir -p "$TOOLCHAIN_BIN_DIR"
cat >"$TOOLCHAIN_BIN_DIR/ptxas" <<EOF
#!/usr/bin/env bash
exec "$CUDA_HOME/bin/ptxas" "\$@"
EOF
chmod +x "$TOOLCHAIN_BIN_DIR/ptxas"

export PATH="$TOOLCHAIN_BIN_DIR:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
if [ -z "${CC:-}" ]; then
    if [ -x /usr/bin/gcc ]; then
        CC=/usr/bin/gcc
    elif command -v gcc >/dev/null 2>&1; then
        CC=$(command -v gcc)
    else
        CC=
    fi
fi
if [ -z "${CXX:-}" ]; then
    if [ -x /usr/bin/g++ ]; then
        CXX=/usr/bin/g++
    elif command -v g++ >/dev/null 2>&1; then
        CXX=$(command -v g++)
    else
        CXX=
    fi
fi
if [ -z "${CC:-}" ] || [ -z "${CXX:-}" ]; then
    echo "Missing host C/C++ compiler. Set CC/CXX explicitly before rebuilding flash-attn." >&2
    exit 1
fi
export CC
export CXX
export MAX_JOBS
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LDFLAGS="-L$CUDA_HOME/lib -L$CUDA_HOME/lib64${LDFLAGS:+ $LDFLAGS}"

echo "[flashattn-build] toolchain"
echo "  gpu_family=$RLLM_GPU_FAMILY"
echo "  gpu_env_key=$RLLM_GPU_ENV_KEY"
echo "  gpu_name=$RLLM_GPU_NAME"
echo "  venv_dir=$VENV_DIR"
echo "  wheelhouse_dir=$WHEELHOUSE_DIR"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  CUDA_NVCC_VERSION=$CUDA_NVCC_VERSION"
echo "  CUDA_NVVM_VERSION=$CUDA_NVVM_VERSION"
echo "  CUDA_CRT_VERSION=$CUDA_CRT_VERSION"
echo "  CUDA_RUNTIME_VERSION=$CUDA_RUNTIME_VERSION"
echo "  CUDA_CCCL_VERSION=$CUDA_CCCL_VERSION"
echo "  CC=$CC"
echo "  CXX=$CXX"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  CUDAARCHS=$CUDAARCHS"
echo "  TRITON_PTXAS_PATH=$TRITON_PTXAS_PATH"

if [ ! -f "$CUDA_HOME/include/nv/target" ]; then
    echo "Missing CUDA CCCL header: $CUDA_HOME/include/nv/target" >&2
    exit 1
fi

echo "[flashattn-build] verifying libcudart linkability"
verify_cudart_link

echo "[flashattn-build] rebuilding flash-attn==$FLASH_ATTN_VERSION from source and caching the wheel"
build_flash_attn_wheel

echo "[flashattn-build] verifying import"
verify_flash_attn
