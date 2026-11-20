#!/usr/bin/env bash
#SBATCH --job-name=mini_swe_bench-erl
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200

# Account/partition/QoS are site inputs; submit with ordinary priority.
set -euo pipefail
export RLLM_DIR=${RLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
# Use the paths prepared by this checkout's data and image setup commands.
if [[ -z ${MINISW_DATA_DIR:-} ]]; then
    MINISW_DATA_DIR=$RLLM_DIR/rllm/data/datasets/swegym
    if [[ -f "$RLLM_DIR/examples/mini_swe_bench/data/prepared/train.parquet" && -f "$RLLM_DIR/examples/mini_swe_bench/data/prepared/validation.parquet" ]]; then
        MINISW_DATA_DIR=$RLLM_DIR/examples/mini_swe_bench/data/prepared
    fi
    export MINISW_DATA_DIR
fi
export APPTAINER_SIF_DIR=${APPTAINER_SIF_DIR:-$RLLM_DIR/.cache/mini_swe_bench/sif}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$RLLM_DIR/.cache/mini_swe_bench/apptainer}
export BENCHMARK=mini_swe_bench TRAINING_METHOD=erl
export TRAIN_MODULE=examples.mini_swe_bench.train.erl.train
source "$RLLM_DIR/examples/mini_swe_bench/train/erl/presets.sh"
source "$RLLM_DIR/examples/mini_swe_bench/train/common.sh"
erl_set_identity
start_runtime
common_args
ARGS+=(
    "rllm.agent.max_steps=$MAX_STEPS"
    "rllm.agent.trajectory_hard_timeout=${WORKFLOW_TIMEOUT:-7200}"
    "+rllm.agent.engine_args.n_parallel_agents=$PARALLEL_TASKS"
    "+rllm.agent.agent_args.scaffold=sweagent"
    "+rllm.env.env_args.max_steps=$MAX_STEPS"
    "+rllm.env.env_args.step_timeout=${MINISW_STEP_TIMEOUT:-120}"
    "+rllm.env.env_args.reward_timeout=${MINISW_REWARD_TIMEOUT:-600}"
    "+rllm.env.env_args.reward_mode=$MINISW_REWARD_MODE"
)
erl_add_args ARGS
MODULE=$TRAIN_MODULE
launch "$@"
