# MBPP

One benchmark directory contains data preparation, training, and evaluation:

- `train/grpo/`: GRPO entrypoint and one Slurm preset.
- `train/ace_rl/`: ACE-RL entrypoint, adapter, prompts, playbook, and one Slurm preset.
- `eval/ace/`: ACE evaluation and its helpers.
- `eval/vanilla/`: vanilla evaluation presets and `eval.slurm`.

Run training from the repository root and pass the cluster account and partition
as `sbatch` flags. Both presets request **two GPUs**, 32 CPUs, and 500G memory.
They default to **Qwen/Qwen3-0.6B** with `MODEL_VARIANT=nonthinking`; set
`MODEL_VARIANT=thinking` to enable thinking for the policy and ACE roles.

```bash
PRESET=grpo sbatch examples/mbpp/train/grpo/train.slurm
PRESET=oneeval_rollout8 MODEL_VARIANT=thinking sbatch \
  examples/mbpp/train/ace_rl/train.slurm
# Delayed-joint scheduling with aligned 1/1 reflector/curator sampling.
PRESET=delayed_oneeval_rollout8 MODEL_VARIANT=thinking sbatch \
  examples/mbpp/train/ace_rl/train.slurm

# Print the exact Python command as JSON without creating a runtime or training.
RLLM_DRY_RUN=1 bash examples/mbpp/train/grpo/train.slurm
```

Each method owns `presets.sh`, `train.py`, and one `train.slurm`.
Benchmark-local `train/common.sh` builds common Hydra arguments and loads
`scripts/training_runtime.sh`. `RLLM_DIR` defaults to the submission directory;
`VENV_DIR` selects another environment. Runtime data registries and temporary
directories belong to the job. A missing GPU environment is built on first use
(see `docs/setup.md`).

The same `PRESET` and `MODEL_VARIANT` always use
`train/<method>/runs/<preset>/<model>/`. For example, the ACE thinking run is
`train/ace_rl/runs/oneeval_rollout8/qwen3-0.6b-thinking/`. Run the same command
again to resume its latest committed checkpoint. `checkpoints/`, `wandb/`,
`logs/`, `episodes/`, and `task_artifacts/` remain under that run directory.
W&B logs to `WANDB_ENTITY` (default: your account's entity). `RUN_DIR` selects a separate explicit experiment;
changing `EXPERIMENT_NAME` alone does not change storage.

MBPP preparation downloads `google-research-datasets/mbpp` (`full`) when the
job's registry is empty. It requires the training environment's `datasets`,
`pandas`, and parquet dependencies. Code execution uses a 10-second test timeout.

`MBPP_FEEDBACK` selects what the policy sees after a failed attempt. The
default, `full`, shows each failed assertion and its traceback, with an
unchanged command and output. With `hidden`, each test result shows only a
label (`hidden test 2`) and a one-line outcome: `wrong answer` when the test
assertion fails, the timeout or early-exit message, or the final exception
line (reduced to the exception type when it contains the assertion or one of
its values). Reward still uses the same three tests. The environment redacts
results itself, so the actor, ACE reflector and curator, ERL updater and
episode logs receive the same view. Training with `MBPP_FEEDBACK=hidden`
requires its own `RUN_DIR` whose path contains `hiddenfb`, tags the experiment
name `_hiddenfb`, and records `rllm.env.env_args.feedback=hidden` in `run.json`.
Evaluation takes the same variable (or `evaluate.py --feedback hidden`), writes
to `<preset>-hiddenfb/`, and refuses a hidden-feedback training run unless
`MBPP_FEEDBACK` is set explicitly in the evaluation environment.

Training uses 5 turns by default, equal to the evaluation limit. `MAX_TRAIN_TASKS`,
`TRAIN_TASK_OFFSET`, and `MAX_VAL_TASKS` select task subsets in both methods;
unset caps use full splits. `TRAIN_BATCH_SIZE`, `POLICY_ROLLOUTS`,
`TOTAL_TRAINING_STEPS`, and other shared launcher settings accept environment
overrides. For small task subsets, ensure enough tasks remain for the generated
batch (`train_batch_size` times the rejection multiplier).

The canonical ACE preset uses 8 policy, 4 reflector, and 4 curator rollouts;
reflection on one uniformly random policy rollout;
singlemode with one curator group; seed policy training; one candidate evaluation
rollout; and `best_rejection_update`. Its standard, iterative, and delayed-joint
controllers all come from `rllm.algorithms.ace`. Change experiment settings through
configuration and use a distinct `RUN_DIR` for each ablation:

| Ablation | Configuration |
| --- | --- |
| Rejection disabled | `REJECTION_SAMPLE_ENABLE=False` (multiplier becomes 1) |
| Selected-candidate policy updates | `POLICY_TRAIN_SOURCE=selected_candidate` |
| Both policy sources | `POLICY_TRAIN_SOURCE=both` |
| Multiple curator groups | `SINGLEMODE_CURATOR_ONE_GROUP=False` |
| Task-level aggregation | `SINGLEMODE=False`, append `++rllm.ace.guidebook_aggregation_level=task` |
| Historical selection rule | `SELECTION_MODE=best_rejection` |
| Iterative schedule | `ACE_TRAINING_MODE=iterative`; append `++rllm.ace.iterative_schedule=[policy,reflector,curator]` and `++rllm.ace.iterative_steps_per_role=1` |
| Delayed joint | `ACE_TRAINING_MODE=delayed_joint POLICY_TRAIN_SOURCE=selected_candidate REFLECTOR_ROLLOUTS=1 CURATOR_ROLLOUTS=1` |

Positional arguments after the Slurm filename are Hydra overrides. For example:

```bash
ACE_TRAINING_MODE=iterative RUN_DIR=/absolute/path/to/mbpp-iterative-thinking \
PRESET=oneeval_rollout8 MODEL_VARIANT=thinking sbatch \
  examples/mbpp/train/ace_rl/train.slurm \
  '++rllm.ace.iterative_schedule=[policy,reflector,curator]' \
  ++rllm.ace.iterative_steps_per_role=1
```

Evaluation reads its prompts and initial playbook from `train/ace_rl/`.
Select one training checkpoint and inspect the command before running:

```bash
RUN_DIR=/absolute/path/to/training/run STEP=20 \
  bash examples/mbpp/eval/ace/eval.slurm --dry-run
python plot.py --run /absolute/path/to/training/run --preset eval
```

Use `sbatch` on the same `eval.slurm` to request resources, or `bash` inside an
existing allocation. `eval/vanilla/eval.slurm` selects vanilla evaluation.
ACE runs the `visits8` preset and vanilla the `pass8` preset, both with a
5-turn policy budget. Results stay under `RUN_DIR/eval/ace/visits8/step_<N>/`
and `RUN_DIR/eval/vanilla/pass8/step_<N>/`; rerun the same command to resume.
See [evaluation settings and outputs](../../docs/evaluation.md).
