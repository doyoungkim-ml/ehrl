#!/usr/bin/env bash
# Prepare the GPU environment used by every training and evaluation launcher.
# Launchers run this automatically (once per GPU family, under a lock) when the
# environment is missing; run it directly to prepare ahead of time, e.g. on a
# CPU node, since the flash-attn build can take hours.
#
# Usage:
#   GPU_FAMILY=a100 bash scripts/setup_qwen35.sh
#   GPU_FAMILY=hopper bash scripts/setup_qwen35.sh
#   GPU_FAMILY=l40s bash scripts/setup_qwen35.sh
#   GPU_FAMILY=rtxpro bash scripts/setup_qwen35.sh
#   bash scripts/setup_qwen35.sh
#
# Installs exact package versions tested with Qwen3.5-0.8B GRPO and builds
# a GPU-specific flash-attn wheel into a matching virtualenv.
# Several packages have conflicting pins (vllm vs transformers, verl vs numpy)
# so they must be installed with --no-deps.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
RLLM_DIR=${RLLM_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "=== Installing uv for this user ==="
    python3 -m pip install --user --quiet uv
fi

# A plain upstream pin lacks the checkpoint continuation fix required by this
# fork. Without an explicit RLLM_VERL_SOURCE, fetch the manifest's exact revision
# once into the checkout cache and apply the vendored patch there.
if [ -z "${RLLM_VERL_SOURCE:-}" ]; then
    read -r verl_upstream verl_revision < <(python3 -c 'import json, sys; m = json.load(open(sys.argv[1])); print(m["upstream"], m["base_revision"])' "$RLLM_DIR/compat/verl/manifest.json")
    RLLM_VERL_SOURCE=$RLLM_DIR/.cache/verl/$verl_revision
    if [ ! -d "$RLLM_VERL_SOURCE/.git" ]; then
        echo "=== Fetching VeRL $verl_revision into $RLLM_VERL_SOURCE ==="
        verl_tmp=$RLLM_VERL_SOURCE.tmp.$$
        rm -rf "$verl_tmp"
        git init -q "$verl_tmp"
        git -C "$verl_tmp" fetch -q --depth=1 "$verl_upstream" "$verl_revision"
        git -C "$verl_tmp" checkout -q --detach FETCH_HEAD
        python3 "$SCRIPT_DIR/verify_verl_compat.py" --source "$verl_tmp" --apply
        # Another family's setup may have finished the same source meanwhile.
        mv -T "$verl_tmp" "$RLLM_VERL_SOURCE" 2>/dev/null || rm -rf "$verl_tmp"
    fi
fi
RLLM_VERL_SOURCE=$(cd -- "$RLLM_VERL_SOURCE" && pwd)
# Verify before creating or changing any environment.
python3 "$SCRIPT_DIR/verify_verl_compat.py" --source "$RLLM_VERL_SOURCE"

source "$SCRIPT_DIR/gpu_venv.sh"
rllm_resolve_gpu_env "$RLLM_DIR"
cd "$RLLM_DIR"

if [ "${RECREATE_VENV:-0}" = "1" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "=== Creating virtualenv at $VENV_DIR ==="
    if [ "${RECREATE_VENV:-0}" = "1" ]; then
        uv venv --python 3.11 "$VENV_DIR" --clear
    else
        uv venv --python 3.11 "$VENV_DIR"
    fi
fi

echo "=== Installing Qwen 3.5 dependencies for $RLLM_GPU_FAMILY (${RLLM_GPU_NAME}) into $VENV_DIR ==="
source "$VENV_DIR/bin/activate"
VENV_PYTHON=${VENV_PYTHON:-$VENV_DIR/bin/python}

SITE_PACKAGES=$("$VENV_PYTHON" - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)

ensure_cuda_lib64_link() {
    local cuda_home=${1:?cuda_home is required}
    local cuda_lib_dir="$cuda_home/lib"

    if [ ! -d "$cuda_lib_dir" ]; then
        return 0
    fi

    if [ ! -e "$cuda_home/lib64" ]; then
        ln -s lib "$cuda_home/lib64"
    fi
}

discover_runtime_library_dirs() {
    local site_packages=${1:?site_packages is required}

    "$VENV_PYTHON" - "$site_packages" <<'PY'
from pathlib import Path
import sys

site = Path(sys.argv[1])
dirs = []
seen = set()

def add(path: Path) -> None:
    if not path.is_dir():
        return
    real = str(path.resolve())
    if real in seen:
        return
    seen.add(real)
    dirs.append(str(path))

for candidate in [
    site / "nvidia" / "cuda_runtime" / "lib64",
    site / "nvidia" / "cuda_runtime" / "lib",
    site / "torch" / "lib",
    site / "nvidia" / "cu13" / "lib64",
    site / "nvidia" / "cu13" / "lib",
    site / "nvidia" / "cudnn" / "lib",
    site / "nvidia" / "cusparselt" / "lib",
    site / "nvidia" / "nccl" / "lib",
    site / "nvidia" / "nvshmem" / "lib",
]:
    add(candidate)

nvidia_root = site / "nvidia"
if nvidia_root.is_dir():
    for candidate in sorted(nvidia_root.glob("*/lib64")):
        add(candidate)
    for candidate in sorted(nvidia_root.glob("*/lib")):
        add(candidate)

for path in dirs:
    print(path)
PY
}

verify_vllm_cuda_runtime() {
    local site_packages=${1:?site_packages is required}
    local vllm_extension="$site_packages/vllm/_C.abi3.so"
    local -a runtime_library_dirs=()
    local runtime_library_path=""
    local ldd_output=""

    if [ ! -f "$vllm_extension" ]; then
        return 0
    fi

    mapfile -t runtime_library_dirs < <(discover_runtime_library_dirs "$site_packages")
    if [ "${#runtime_library_dirs[@]}" -eq 0 ]; then
        echo "Could not determine runtime library directories for vLLM." >&2
        return 1
    fi

    runtime_library_path=$(IFS=:; echo "${runtime_library_dirs[*]}")
    ldd_output=$(LD_LIBRARY_PATH="$runtime_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$vllm_extension" 2>&1 || true)

    printf '%s\n' "$ldd_output" | grep -E 'libcudart\.so\.12|libcuda\.so\.1' || true

    if ! printf '%s\n' "$ldd_output" | grep -Eq 'libcudart\.so\.12 => .*/libcudart\.so\.12'; then
        echo "vLLM extension could not resolve libcudart.so.12 from the virtualenv." >&2
        return 1
    fi

    if printf '%s\n' "$ldd_output" | grep -Fq "version \`libcudart.so.12' not found"; then
        echo "vLLM extension resolved an incompatible CUDA runtime; install a real CUDA 12 runtime." >&2
        return 1
    fi
}

install_vllm_runtime_deps() {
    local -a vllm_runtime_requirements=()

    mapfile -t vllm_runtime_requirements < <("$VENV_PYTHON" - <<'PY'
import importlib.metadata as md
from packaging.requirements import Requirement

blocked = {
    "flash-attn",
    "numpy",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "vllm",
}

for raw in md.requires("vllm") or []:
    req = Requirement(raw)
    marker_text = str(req.marker) if req.marker else ""
    if req.name in blocked:
        continue
    if "extra" in marker_text:
        continue
    print(raw)
PY
)

    if [ "${#vllm_runtime_requirements[@]}" -eq 0 ]; then
        return 0
    fi

    echo "=== Installing vLLM runtime dependencies for $RLLM_GPU_FAMILY ==="
    uv pip install "${vllm_runtime_requirements[@]}"

    echo "=== Installing CUDA 12 runtime for vLLM extensions ==="
    uv pip install "nvidia-cuda-runtime-cu12"
}

uv pip install -e .[verl] --torch-backend=cu130

# Pin the CUDA 13 PyTorch stack so torchvision's compiled ops match torch.
uv pip install --no-deps \
    "torch==2.10.0+cu130" \
    "torchvision==0.25.0+cu130" \
    "torchaudio==2.10.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130

# verl is installed with --no-deps below, so install the runtime deps we rely on here.
# cachetools is imported by verl.experimental.agent_loop but is not currently declared upstream.
uv pip install \
    "accelerate" \
    "cachetools" \
    "dill" \
    "packaging>=20.0" \
    "peft" \
    "ray[default]>=2.41.0" \
    "tensorboard" \
    "wandb" \
    "tensordict>=0.10.0" \
    "torchdata" \
    "flashinfer-python==0.6.11"

# Core packages (--no-deps to avoid conflicts between vllm<->transformers, verl<->numpy)
# VeRL: install the exact checked source, including the required resume fix.
# Rebuild the local source even when its packaging metadata has not changed.
uv pip install --no-deps --reinstall-package verl \
    "$RLLM_VERL_SOURCE" \
    "vllm==0.19.1" \
    "transformers==5.3.0" \
    "huggingface-hub==1.8.0" \
    "numpy==2.4.4"

install_vllm_runtime_deps

# install_vllm_runtime_deps pulls packages (datasets, tokenizers, peft, etc.)
# whose pins may downgrade hub or upgrade tokenizers beyond the range accepted
# by transformers 5.3.0. Restore the tested versions after runtime deps.
uv pip install --no-deps \
    "transformers==5.3.0" \
    "huggingface-hub==1.8.0" \
    "tokenizers==0.22.2"

ensure_cuda_lib64_link "$SITE_PACKAGES/nvidia/cu13"
ensure_cuda_lib64_link "$SITE_PACKAGES/nvidia/cuda_runtime"
verify_vllm_cuda_runtime "$SITE_PACKAGES"

if [ "${SKIP_FLASH_ATTN_REBUILD:-0}" = "1" ]; then
    echo "=== Skipping flash-attn rebuild for $RLLM_GPU_FAMILY ==="
else
    bash "$SCRIPT_DIR/rebuild_flash_attn.sh"
fi

"$VENV_PYTHON" "$SCRIPT_DIR/verify_verl_compat.py" --installed
# Launchers require the IPC/lifecycle repairs in the installed package.
"$VENV_PYTHON" "$SCRIPT_DIR/patch_verl_zmq_ipc.py"

echo "=== Running smoke test ==="
python3 - <<'PY'
import importlib
import importlib.metadata as md

import flash_attn
import flash_attn_2_cuda
import torch
import torchvision
import transformers
import verl
import vllm
from vllm import LLM
from transformers import AutoConfig

print(f"verl: {verl.__version__}")
print(f"vllm: {vllm.__version__}")
print(f"vllm LLM symbol: {LLM.__module__}.{LLM.__name__}")
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"flash-attn: {md.version('flash-attn')}")
print(f"flash_attn module: {flash_attn.__file__}")
print(f"flash_attn_2_cuda module: {flash_attn_2_cuda.__file__}")

for module_name in [
    "verl.experimental.agent_loop.agent_loop",
    "verl.workers.rollout.vllm_rollout.vllm_rollout",
    "rllm.engine.rollout.verl_engine",
    "rllm.trainer.verl.train_agent_ppo",
]:
    importlib.import_module(module_name)
    print(f"import ok: {module_name}")

cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
print(f"Qwen3.5 model_type: {cfg.model_type}")
print("Setup complete!")
PY
