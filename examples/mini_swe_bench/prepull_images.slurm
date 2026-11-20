#!/usr/bin/env bash
#SBATCH --job-name=prepull-miniswe
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00

# Prepare reusable SIF images with an existing Python and mksquashfs installation.
set -euo pipefail
RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
cd "$RLLM_DIR"
export PYTHONPATH="$RLLM_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-${VENV_DIR:-$RLLM_DIR/.venv-hopper}/bin/python}
[[ -x "$PYTHON_BIN" ]] || { echo "Set PYTHON_BIN or VENV_DIR to an existing environment" >&2; exit 1; }
export MKSQUASHFS_BIN=${MKSQUASHFS_BIN:-$(command -v mksquashfs || true)}
[[ -x "$MKSQUASHFS_BIN" ]] || { echo "Set MKSQUASHFS_BIN to an existing working mksquashfs" >&2; exit 1; }
# Match the checked-in MiniSWE train/eval launchers by default.
CACHE_DIR=${CACHE_DIR:-$RLLM_DIR/.cache/mini_swe_bench}
export APPTAINER_SIF_DIR=${APPTAINER_SIF_DIR:-$CACHE_DIR/sif}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$CACHE_DIR/apptainer}
PREP_TMP=$(mktemp -d "${LOCAL_TMP_ROOT:-/tmp}/minisw-prep-${SLURM_JOB_ID:-local}-XXXXXX")
export APPTAINER_TMPDIR=$PREP_TMP
trap 'rm -rf -- "$PREP_TMP"' EXIT
export SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR SINGULARITY_TMPDIR=$APPTAINER_TMPDIR
export MAX_VAL_TASKS=${MAX_VAL_TASKS-100}
mkdir -p "$APPTAINER_SIF_DIR" "$APPTAINER_CACHEDIR"
"$PYTHON_BIN" -u -m examples.mini_swe_bench.prepull_images \
    --max-parallel "${MAX_PARALLEL:-8}" --splits "${PREPULL_SPLITS:-train,validation}" "$@"
