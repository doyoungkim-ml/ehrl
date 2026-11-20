# AppWorld training

Use one launcher for each method, from the repository root:

```bash
PRESET=grpo MODEL_VARIANT=nonthinking sbatch examples/appworld/train/grpo/train.slurm
PRESET=oneeval_rollout8 MODEL_VARIANT=thinking sbatch examples/appworld/train/ace_rl/train.slurm
PRESET=delayed_oneeval_rollout8 MODEL_VARIANT=thinking sbatch examples/appworld/train/ace_rl/train.slurm
```

Both request one node, **2 H200 GPUs**, 32 CPUs, 500 GB and 48 hours. Account,
partition, QoS, reservation and any validated hardware override are submit-time
inputs. `RLLM_DIR` defaults to `SLURM_SUBMIT_DIR`, then the current directory.

Launchers source `examples/appworld/server/runtime.sh`, which uses existing
AppWorld data in `data/appworld/` in the checkout, `appworld_data/` beside it, or
`/scratch/$USER/appworld_data`, and the server environment `.venv_appworld_server`.
When either is missing, it runs `server/setup_server_env_uv.sh` once (under a lock)
to create them in the checkout. Set `APPWORLD_DATA_DIR` or `APPWORLD_SERVER_VENV`
for other locations. The GPU environment is resolved, and built if missing, by
`scripts/training_runtime.sh`; startup checks the installed VeRL compatibility
contract and IPC repair.

| `MODEL_VARIANT` | Model | Thinking / accumulated reasoning |
| --- | --- | --- |
| `nonthinking` | Qwen/Qwen3-4B | Off |
| `thinking` | Qwen/Qwen3-4B | On |
| `instruct` | Qwen/Qwen3-4B-Instruct-2507 | Off |
| `thinking_2507` | Qwen/Qwen3-4B-Thinking-2507 | On |

The main ACE-RL preset and GRPO default to `nonthinking`. Other ACE-RL presets
default to base `thinking`. Explicit environment settings override preset values;
additional Hydra arguments can be appended to the launcher command.

## Harness presets

Select `PRESET` on the same ACE-RL launcher. The default is `oneeval_rollout8`.
All presets use the same prompts and initial playbook in `ace_rl/`.

| `PRESET` | Policy / reflector / curator rollouts | Policy training source | Controller and selection |
| --- | --- | --- | --- |
| `oneeval_rollout8` | 8 / 4 / 4 | seed | Singlemode on one uniformly random policy rollout, one curator group, one candidate eval, `best_rejection_update` |
| `delayed_oneeval_rollout8` | 8 / 1 / 1 | seed | The same main preset with delayed joint scheduling and aligned auxiliary samples |
| `standard` | 8 / 4 / 4 | seed | Joint, task playbooks, `best_rejection` |
| `singlemode` | 8 / 4 / 4 | selected candidate | Singlemode, all curator groups, `best_rejection` |
| `one_curator_group` | 8 / 4 / 4 | selected candidate | Singlemode, one curator group, `best_rejection` |
| `one_curator_group_no_rejection` | 8 / 4 / 4 | selected candidate | Same controller; trainer rejection sampling disabled |
| `seed_oneeval` | 8 / 4 / 4 | seed | Singlemode, one curator group, one candidate eval, `best_rejection` |
| `seed_oneeval_no_rejection` | 8 / 4 / 4 | seed | Same controller; trainer rejection sampling disabled |
| `iterative_singlemode` | 8 / 4 / 4 | seed | Policy → reflector → curator, 10 steps each |
| `delayed_joint` | 8 / 1 / 1 | selected candidate | Delayed joint, task playbooks, `best_rejection` |

Unless one evaluation is specified, each candidate uses the policy rollout count
(`CANDIDATE_EVAL_ROLLOUTS=0`). Rejection sampling defaults to enabled with a 2×
multiplier. Disabling trainer rejection does not change the ACE selection mode.
`standard` and `iterative_singlemode` retain eager
rollouts with the cache kept resident. Other presets release the rollout cache;
`one_curator_group` with `instruct` retains eager mode.

The main and seed-oneeval presets bound reflector history to 20,000 tokens and
the prompt to 30,720 tokens. The base-model `one_curator_group` preset has those
same bounds. Other variants retain their 32,768-token prompt limit and unbounded
adapter history. Override `REFLECTOR_MAX_PROMPT_LENGTH` and
`REFLECTOR_MAX_HISTORY_TOKENS` explicitly when comparing different truncation
settings (`REFLECTOR_MAX_HISTORY_TOKENS=null` disables the adapter history bound).
The main and one-curator-group variants retain their 1,200-second timeout;
the other controller presets use 2,400 seconds.

Run the same command again to continue the same experiment. Its visible home is
`ace_rl/runs/<preset>/<model>/`, for example
`ace_rl/runs/oneeval_rollout8/qwen3-4b-thinking/`. Checkpoints, W&B files, logs,
episodes, and task artifacts stay in `checkpoints/`, `wandb/`, `logs/`,
`episodes/`, and `task_artifacts/` below that directory. Checkpoint continuation
uses `trainer.resume_mode=auto` and the latest committed checkpoint marker.
The launch preflight rejects incomplete checkpoint roots instead of silently
starting over. A writer lock prevents two jobs from changing the same experiment.
The complete startup/training log is `logs/train-<job>.log`; Slurm's small
`%x-%j.out`/`%x-%j.err` receipt remains in the submission directory so errors
before the run directory is resolved are still visible.

Use an explicit `RUN_DIR` for an intentional fork or a reviewed migration.
Changing the experiment display name does not change storage. W&B logs to `WANDB_ENTITY`
(default: your account's entity).

## GRPO presets

`PRESET=grpo` uses the 8-rollout baseline; `PRESET=no_rejection` disables trainer
rejection and uses multiplier 1. `PRESET=ppo` retains the GAE/critic ablation:
γ=1, λ=0.95, critic learning rate 1e-5, 10 critic warmup steps, token-mean losses,
and every checkpoint kept (`KEEP_LAST_N=0`; a positive value retains that many). PPO uses the
vLLM compilation setting `use_static_cuda_launcher: false`; use
`VLLM_COMPILATION_CONFIG` to select this setting for GRPO as well.

GRPO/PPO keep task slicing through `MAX_TRAIN_TASKS`, `TRAIN_TASK_OFFSET`, and
`MAX_VAL_TASKS` (`MAX_TEST_TASKS` also supplies the validation limit). Their
outputs follow the same `grpo/runs/<preset>/<model>/` structure. Every launcher
runs 301 steps (updates 1–300), saves every 10 steps and uses 10 turns by
default; `TRAIN_CHAIN_PARTS` chains successor jobs past the 48-hour limit (see
`docs/experiments.md`).

## Inspect before running

```bash
RLLM_DRY_RUN=1 MODEL_VARIANT=thinking_2507 bash examples/appworld/train/ace_rl/train.slurm
RLLM_DRY_RUN=1 PRESET=ppo bash examples/appworld/train/grpo/train.slurm
```

Dry-run mode prints the command arguments as a raw JSON array (no object wrapper)
before GPU discovery, runtime setup,
filesystem writes, or sidecar startup. Synthetic Ray paths and ports are used
unless overridden. It requires Bash and Python 3; Hydra can consume the emitted
arguments for configuration validation in a prepared environment. It does not
prove GPU or AppWorld runtime compatibility.

Real runs use job-scoped task registries, temporary files and sidecar roots.
The launcher checks that each server process remains alive and its HTTP endpoint
is ready. It stops only the sidecars/optional GPU keepalive it started, preserves
Ray logs on failure, and delegates temporary cleanup to the shared runtime.
`RLLM_GPU_KEEPALIVE=0` disables the keepalive when it is unnecessary.
