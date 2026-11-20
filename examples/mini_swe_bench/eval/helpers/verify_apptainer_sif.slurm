#!/usr/bin/env bash
#SBATCH --job-name=miniswe-verify-sif
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# CPU-only image verification. SIF build temporary storage must be node-local.
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$RLLM_DIR"
export PYTHONPATH="$RLLM_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-${VENV_DIR:-$RLLM_DIR/.venv-l40s}/bin/python}
export APPTAINER_SIF_DIR=${APPTAINER_SIF_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/rllm/sif}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-${XDG_CACHE_HOME:-$HOME/.cache}/rllm/apptainer}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${SLURM_TMPDIR:-/tmp}/rllm-apptainer-${SLURM_JOB_ID:-$$}-$$}
exec "$PYTHON_BIN" -m examples.mini_swe_bench.eval.helpers.verify_apptainer_sif --instance "${INSTANCE:-getmoto__moto-5601}" "$@"
