# MiniSWEBench

Training uses the existing SWE XML tools and Apptainer-backed repository/grading
workflow. Data preparation and container utilities are shared at the benchmark
root and in `shared/`; evaluation lives in `eval/`. Reusable diagnostics are in
`eval/helpers/`, and regrading/replay diagnostics live in `eval/helpers/`.

- `train/grpo/train.py`: cumulative-workflow GRPO.
- `train/grpo/train.slurm`: canonical GRPO launcher.
- `train/ace_rl/train.py`: the packaged standard Local ACE-RL controller with
  MiniSWEBench agent, environment, and reflector adapters.
- `train/ace_rl/train.slurm`:
  canonical ACE-RL launcher.
- `train/ace_rl/prompts/` and `playbooks/`: initial ACE-RL assets. The initial
  playbook is empty; each SWE instance gets its own playbook during training.
- `train/{grpo,ace_rl}/presets.sh`: model and scientific settings.
  `train/common.sh` owns shared runtime and Hydra arguments, using
  `scripts/training_runtime.sh` for the shared runtime.

Evaluation follows the same method layout: `eval/{ace,vanilla}/eval.slurm` and
`presets.sh`. API evaluation uses the same filename with explicit CPU-only
`sbatch` resource overrides.
See [evaluation usage and defaults](eval/README.md) for model selection, task
slices, external APIs, artifacts, and diagnostics.

Both launchers request **two H200 GPUs, 32 CPUs, and 500 GB RAM** by default.
Training also accepts two H100 GPUs with an explicit scheduler resource override
and the prepared Hopper environment; the model and training settings remain the
same. H100 memory fit must be verified in the allocation. Account, partition,
QoS, and reservation are submission options. Submit from the repository root or
set `RLLM_DIR` explicitly. A missing GPU environment is built on first use and the
SWE-Gym parquet files and container images are prepared automatically; the
`swegym` and `swebench` grading packages must already be importable in the GPU
environment (see `docs/setup.md`).

| `MODEL_VARIANT` | Model | Thinking |
| --- | --- | --- |
| `nonthinking` (default) | `Qwen/Qwen3-4B` | disabled |
| `thinking` | `Qwen/Qwen3-4B` | enabled |
| `instruct` | `Qwen/Qwen3-4B-Instruct-2507` | disabled |
| `thinking_2507` | `Qwen/Qwen3-4B-Thinking-2507` | enabled |

Inspect the exact JSON argv without loading GPU libraries, preparing data,
creating runtime directories, or starting training:

```bash
RLLM_DRY_RUN=1 MODEL_VARIANT=thinking \
  bash examples/mini_swe_bench/train/ace_rl/train.slurm
RLLM_DRY_RUN=1 MODEL_VARIANT=instruct \
  bash examples/mini_swe_bench/train/grpo/train.slurm
```

Use the same checked-in command for a fresh run and its continuation:

```bash
PRESET=oneeval_rollout8 MODEL_VARIANT=thinking \
  sbatch examples/mini_swe_bench/train/ace_rl/train.slurm
PRESET=delayed_oneeval_rollout8 MODEL_VARIANT=thinking \
  sbatch examples/mini_swe_bench/train/ace_rl/train.slurm
PRESET=grpo MODEL_VARIANT=instruct \
  sbatch examples/mini_swe_bench/train/grpo/train.slurm
```

Supply the site's scheduler arguments as ordinary `sbatch` flags. The shared
runtime accepts `VENV_DIR`, `PYTHON_BIN`, and an optional prepared
`VERL_SOURCE_DIR`; it discovers
cache paths and uses a job-scoped dataset registry and temporary directory.
The environment must already contain the supported rLLM/VeRL GPU stack, the
SWE-Gym grading package, SWE-bench, and Apptainer or Singularity.
The launchers use `rllm/data/datasets/swegym` by default, matching
`preprocess_swegym.py`; they reuse `examples/mini_swe_bench/data/prepared` when
it already contains both train and validation Parquet files.

`prepull_images.slurm` prepares compressed SIF images using an existing
`MKSQUASHFS_BIN` or `mksquashfs` on PATH. Image preparation, training, and
evaluation default to `.cache/mini_swe_bench/{sif,apptainer}` in this checkout;
set `APPTAINER_SIF_DIR` and `APPTAINER_CACHEDIR` only for another location.
Preparation
uses both train and validation splits by default; `PREPULL_SPLITS` can restrict
it. Building images requires sufficient node-local temporary space. It does not
install tools.

ACE uses 8 policy rollouts, 4 reflector rollouts, 4 curator rollouts,
`singlemode=True` reflecting on one uniformly random policy rollout, one curator
group, `policy_train_source=seed`, `selection_mode=best_rejection_update`, and
one evaluation rollout per candidate.
Playbooks flush every three task visits; local validation uses three epochs.
`PRESET=delayed_oneeval_rollout8` preserves eight policy rollouts with delayed
joint scheduling and aligned 1/1 reflector/curator samples.
The agent keeps the existing SWE tool system prompt when ACE adds a playbook,
and candidate evaluation clones independent containers for each issue.

Both methods default to the existing `shaped` rewards (0 / 0.05 / 0.2 / 1).
`MINISW_REWARD_MODE` also accepts `binary`, `frac`, `frac2`, and `frac3`; GRPO and
ACE use the same reward implementation. Only a resolved issue counts as success.
Changing reward mode is an experimental ablation; select a distinct `RUN_DIR`. Container failures raise retryable errors in ACE instead of becoming
zero-reward programming attempts.

Useful controls: `MINISW_DATA_DIR`, `MAX_TRAIN_TASKS`, `TRAIN_TASK_OFFSET`,
`MAX_VAL_TASKS` (default 100; empty means all), `MINISW_N_PARALLEL_TASKS`,
`MINISW_MAX_STEPS`, and `TRAINER_TOTAL_TRAINING_STEPS`. Datasets are registered
with a fingerprint of the source contents and selection. Training defaults to
8 tasks per batch, 10 tool steps (the evaluation limit), 301 total steps with a
checkpoint every 10, and 8192/32768 prompt/response budgets; all methods use
eight policy rollouts or complete chains per task (see `docs/experiments.md`).
Additional Hydra overrides may be passed as launcher arguments. Runs
retain automatic checkpoint resume and skip periodic validation (`test_freq=-1`).

Focused CPU tests use actual Hydra configs and packaged ACE helpers; they mock
container execution and never load a model or start a Ray cluster. With an
existing compatible environment and prepared VeRL runtime source:

```bash
PYTHONPATH="$VERL_SOURCE_DIR:$PWD" "$PYTHON_BIN" \
  -m unittest discover -s tests/mini_swe_bench -v
```

These checks cover adapter rewards, prompt resets, task identity, candidate
cloning, cleanup, dataset refresh, and all eight launcher/model combinations.
They do not validate distributed GPU execution or installed cluster prerequisites.

Runs live at `train/<method>/runs/<preset>/<model>/`, for example
`train/ace_rl/runs/oneeval_rollout8/qwen3-4b-thinking/`. Repeating the command
resumes its latest committed checkpoint. Checkpoints, W&B files, logs, episodes,
and task artifacts remain below the same run; W&B logs to `WANDB_ENTITY`
(default: your account's entity). `EXPERIMENT_NAME` is a display name, while
`RUN_DIR` explicitly selects storage.

Evaluate and plot that run through the same simple interfaces:

```bash
RUN_DIR=/absolute/path/to/training/run STEP=30 sbatch examples/mini_swe_bench/eval/ace/eval.slurm
python plot.py --run /absolute/path/to/training/run --preset eval
```

Evaluation requires an explicit checkpoint step and records results under
`RUN_DIR/eval/ace/visits8/step_<N>/` or `RUN_DIR/eval/vanilla/pass8/step_<N>/`
(10 policy turns for both); rerun the command to resume. Training and evaluation presets have distinct
budgets and model defaults; see [the evaluation documentation](eval/README.md).
