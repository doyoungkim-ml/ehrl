#!/usr/bin/env bash
#SBATCH --job-name=miniswe-regrade-ab
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# CPU-only comparison of stock and transformed grading scripts.
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$RLLM_DIR"
export PYTHONPATH="$RLLM_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-${VENV_DIR:-$RLLM_DIR/.venv-l40s}/bin/python}
command -v apptainer >/dev/null || { echo "Apptainer is required" >&2; exit 1; }
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-${XDG_CACHE_HOME:-$HOME/.cache}/rllm/apptainer}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${SLURM_TMPDIR:-/tmp}/rllm-regrade-${SLURM_JOB_ID:-$$}-$$}
export APPTAINER_SIF_DIR=${APPTAINER_SIF_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/rllm/sif}
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$APPTAINER_SIF_DIR"
exec "$PYTHON_BIN" -m examples.mini_swe_bench.eval.helpers.regrade_ab_swegym \
    --n "${AB_N:-100}" --workers "${AB_WORKERS:-12}" \
    --out "${OUT_DIR:-$RLLM_DIR/examples/mini_swe_bench/logs/regrade-${SLURM_JOB_ID:-$$}}" "$@"
