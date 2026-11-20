# Experiments

Each benchmark has one directory under `examples/`:

```text
examples/<benchmark>/
  prepare_*.py                 # dataset preparation (run automatically)
  train/common.sh              # common Hydra arguments for the benchmark
  train/<method>/
    train.slurm                # the entrypoint
    presets.sh                 # named scientific settings
    runs/<preset>/<model>/     # one experiment; see artifacts.md
  eval/<ace|vanilla>/
    eval.slurm
    presets.sh
    results/                   # pretrained-model evaluations
  figures/output/<model>/      # benchmark gallery
```

## Models

| Benchmark | Model | `MODEL_VARIANT` | GPUs |
| --- | --- | --- | --- |
| MBPP | Qwen/Qwen3-0.6B | `nonthinking`, `thinking` | 2 |
| Mastermind | Qwen/Qwen3-0.6B | `nonthinking`, `thinking` | 2 |
| AppWorld | Qwen3-4B family | `nonthinking`, `thinking`, `instruct`, `thinking_2507` | 2 H200 |
| MiniSWEBench | Qwen3-4B family | `nonthinking`, `thinking`, `instruct`, `thinking_2507` | 2 H200 |

`nonthinking` and `thinking` use the same Qwen checkpoint and switch thinking
through the chat parser and history settings. `instruct` selects
`Qwen/Qwen3-4B-Instruct-2507`; `thinking_2507` selects `Qwen/Qwen3-4B-Thinking-2507`.
`nonthinking` is the default; set it explicitly in a run matrix.

## Methods and presets

Every method defaults to `actor_rollout_ref.rollout.n=8` on every benchmark.
This is the number of initial policy rollouts per task, or complete ACE chains
per task for delayed joint PPO. Auxiliary sampling remains method-specific;
equal policy group sizes do not imply equal generated or gradient-update tokens.

| Launcher | Preset | Training `rollout.n` | Tasks per step |
| --- | --- | --- | --- |
| `train/grpo/train.slurm` | `grpo` | 8 | 8 |
| `train/ace_rl/train.slurm` | `oneeval_rollout8` (default) | 8 | 8 |
| `train/ace_rl/train.slurm` | `delayed_oneeval_rollout8` | 8 | 8 |
| `train/ace_rl/train.slurm` | `shared_rollout8` (AppWorld, MiniSWEBench, Mastermind, MBPP) | 4 | 8 |
| `train/ppo/train.slurm` | `ppo` | 8 | 8 |
| `train/erl/train.slurm` | `rollout8` | 8 | 8 |
| `train/delayed_joint_ppo/train.slurm` | `batch<N>_rollout8` | 8 | MBPP 6; others 8 |

For delayed joint PPO, N is tasks per step and the actor/critic mini batch size:
`batch6_rollout8` for MBPP and `batch8_rollout8`
for Mastermind, AppWorld and MiniSWEBench. Each of the eight chains has one
reflector and one curator response. Standard ACE-RL retains `g_ref=g_cur=4`;
delayed ACE-RL retains `1/1` per policy rollout.
Obsolete lower-rollout preset names and joint PPO's old `batch<N>` names are
rejected. Historical runs keep their original paths and recorded settings;
changing a run's sampling requires a fresh `RUN_DIR`.
GRPO, ACE-RL, PPO and ERL train on 8 tasks per step. GRPO and ACE-RL generate
16 tasks (rejection multiplier 2); PPO, ERL and delayed joint PPO generate
their task batch.
Internal validation, where enabled, samples 8 rollouts per task regardless of
the training group size (ERL keeps its single validation sample). Details of
the baselines are in [baselines](baselines.md).

`GRPO_ACTOR_PROMPT` selects the GRPO actor prompt on Mastermind, MBPP, AppWorld
and MiniSWEBench; the default, `agent`, keeps each agent's built-in prompt and
an unchanged command. `ace_empty_playbook` is the actor-only control: GRPO
trains under ACE-RL's step-0 actor prompt (`ace_rl/prompts/base_system_prompt.txt`
with an empty playbook, in the same message roles), while GRPO's turns, parsing
and feedback stay unchanged. It records `rllm.actor_prompt` and the prompt text
in `run.json`, tags the experiment name and requires its own `RUN_DIR`. ACE-RL
launchers default to 20 turns on Mastermind, AppWorld and MiniSWEBench, so set
`MASTERMIND_MAX_TURNS`, `APPWORLD_MAX_TURNS` or `MINISW_MAX_STEPS` deliberately.

```bash
PRESET=oneeval_rollout8 MODEL_VARIANT=thinking sbatch examples/mbpp/train/ace_rl/train.slurm
```

`oneeval_rollout8` uses the standard local ACE-RL controller with:

- 8 policy, 4 reflector and 4 curator rollouts;
- single-mode reflection on one uniformly random policy rollout and one
  curator group. `REFLECTION_SOURCE_SELECTION=min_reward` instead chooses the
  **lowest-reward** rollout (exact ties broken uniformly at random, a missing
  reward counts as 0). `rllm.ace.reflection_source_selection` records the mode;
- seed-policy training and `best_rejection_update` candidate selection;
- strict-improvement admission of reflector and curator groups to the gradient
  batch (`rllm.ace.auxiliary_update_gate=strict_improvement`);
- one candidate-evaluation rollout ("one evaluation" means one rollout per
  candidate, not one local evaluation epoch);
- per-task playbooks, flushed after three visits, and three local evaluation epochs.

The auxiliary gate changes only training-batch admission. Actor groups remain
eligible independently of playbook improvement; reflector and curator groups
are admitted only when the selected playbook's measured reward is strictly
better than its selection baseline. Admitted groups keep every candidate,
including unsuccessful candidates, for relative credit. Rollout generation,
candidate selection, and playbook updates keep their existing rules. Equal
actor rewards still give zero GRPO task advantage. Auxiliary generation cost
is unchanged, while the retained gradient-token budget can be smaller.

`AUXILIARY_UPDATE_GATE=none` selects the historical objective. The gate is part
of the run identity and saved ACE controller state: changing it cannot silently
resume an existing run. Use a distinct `RUN_DIR` for the new objective; changing
the experiment name alone is insufficient. Historical checkpoints without the
gate field are interpreted as `none`. Delayed ACE-RL keeps `none` and its existing
training-batch rules. See [checkpoint compatibility](ace-checkpoint-resume.md).

With `min_reward`, the seed is the lowest-reward rollout, so
`best_rejection_update` (which keeps candidates whose evaluation reward is at
least their seed's) accepts
nearly every candidate under binary rewards: whenever any rollout of the task
fails, the seed reward is 0. This is intended, not a selection bug; compare
acceptance rates only between runs with the same `reflection_source_selection`.
Changing the selection mode requires a fresh `RUN_DIR`; changing the experiment
name alone does not isolate checkpoints. To continue an existing minimum-reward
run, preserve `REFLECTION_SOURCE_SELECTION=min_reward` explicitly.

`delayed_oneeval_rollout8` keeps those settings with delayed joint scheduling
and aligned 1/1 reflector/curator sampling; it reflects on every policy rollout,
so the reflection-source choice does not apply. Each method's `presets.sh`
lists further ablations. Prompts and initial playbooks live with each
benchmark's ACE-RL binding; the shared algorithm is in `rllm/algorithms/ace`.

`shared_rollout8` keeps GRPO's budget of eight actor rollouts per task: four
seed rollouts plus at most four candidate-scoring rollouts (2 reflector samples,
one 3-sample curator group, one evaluation per candidate).
`policy_train_source=seed_and_candidates` trains all of them in the task's seed
GRPO group, and an actor mini-batch of 16 prompts keeps GRPO's 64 rows. It is
otherwise the standard one-evaluation recipe, including the strict-improvement
gate (`AUXILIARY_UPDATE_GATE=none` selects the ungated objective) and the
benchmark's standard turn default (Mastermind 20, MBPP 6). It runs on AppWorld,
MiniSWEBench, Mastermind and MBPP. The training opt-in
`+rllm.ace.handoff_protocol=feedback_v1 +rllm.agent.preserve_terminal_feedback=True`
selects the [feedback-preserving role inputs](troubleshooting.md#explicit-feedback-preserving-ace-training)
on all five benchmarks; it is not part of the automatic run directory, so set a
distinct `RUN_DIR`. Mastermind keeps its guess table under `feedback_v1`, because
the raw terminal reply reveals the secret code; only its role prompts change.

## Training length, turns and checkpoints

Every method and benchmark defaults to 301 total training steps and a
checkpoint every 10 steps. The trainer checks the limit after incrementing its
step, so 301 runs updates 1–300 and saves `global_step_10` … `global_step_300`.
PPO and delayed joint PPO spend their first 10 steps on critic warmup: their
actor updates run on steps 10–300. MiniSWE GRPO and PPO use the workflow
trainer, whose updates are zero-based: 301 runs updates 0–300 (one more than
the others) and also saves `global_step_0`, so checkpoint labels line up with
the other methods. Every saved checkpoint is kept for evaluation
(`KEEP_LAST_N=0`; GRPO and ACE-RL never prune). Budget the disk accordingly:
with FP32 FSDP weights plus Adam state, one actor checkpoint is roughly 7 GB
for Qwen3-0.6B and 48 GB for Qwen3-4B, doubled for PPO and delayed joint PPO's
critic, times 30 checkpoints per run (estimate from model size, not measured).

Standard ACE-RL defaults to MBPP 6 turns and 20 for Mastermind,
AppWorld and MiniSWE. Other methods retain their previous defaults: MBPP 6,
Mastermind 5, and AppWorld/MiniSWE 10. The shared training override
names are `MBPP_MAX_TURNS`, `MASTERMIND_MAX_TURNS`,
`APPWORLD_MAX_TURNS` and `MINISW_MAX_STEPS`. ACE reflector adapters receive the
same cap as the actor through `rllm.ace.max_turns`.

For the six-method comparison (GRPO, PPO, ERL, standard ACE-RL, delayed ACE-RL,
and delayed joint PPO), set those overrides explicitly to MBPP 6 and all other
benchmarks 20. Set `MAX_TURNS` to the same value for checkpoint evaluations;
the standalone evaluation defaults do not inherit training overrides. Preserve
other methods' algorithms and scientific settings. A checkpoint trained at
another cap is historical evidence rather than exact continuation of the
selected configuration.

## Small-model defaults

MBPP and Mastermind share their runtime settings. Default policy
response limits are MBPP 8192 and Mastermind 10240 tokens.
GRPO and ACE-RL use 8 tasks per batch and two GPUs, with rollout TP=1/DP=1,
actor/reference offload and a 0.35 rollout memory fraction. Thinking and
nonthinking use the same task and sampling settings.

Mastermind rewards are normalized per task: a solved game scores 1, an exact
match contributes `1 / code_length`, and a value-only match `0.5 / code_length`.
Its reflector identifies success from `won`, so partial credit is never mistaken
for solving the game.

`TRAINER_VAL_AFTER_TRAIN=False` skips the trainer's final validation pass after
the last step for MBPP and Mastermind GRPO/ACE-RL (PPO and ERL already
skip it). Checkpoints are evaluated separately with visits8/pass8.

## Overrides and resume

Useful overrides include `TRAIN_BATCH_SIZE`, `VAL_BATCH_SIZE`, `N_PARALLEL_AGENTS`,
`TRAINER_TOTAL_EPOCHS`, `TOTAL_TRAINING_STEPS`, `TRAINER_SAVE_FREQ`,
`TRAINER_TEST_FREQ`, `TRAINER_RESUME_MODE` and `EXPERIMENT_NAME`. Positional
arguments are appended as Hydra overrides.

Changing method, sampling or rejection settings creates an ablation: give it its
own `RUN_DIR`. `EXPERIMENT_NAME` changes only the display name, not storage.

Run the same training command again to resume the latest committed checkpoint.
The first job copies the effective train/validation parquet files to
`RUN_DIR/data/`; later jobs use those copies. An existing checkpoint tree with a
missing or invalid marker fails preflight instead of starting over, and a writer
lock prevents two jobs from writing one run. A job that starts after the run
already reached its step limit (latest checkpoint + 1 ≥ total steps) exits
successfully without training. Continuation details are in
[ACE checkpoint resume](ace-checkpoint-resume.md).

Training launchers request 48 hours, 2 GPUs, 32 CPUs and 500G. For runs that
need longer, set `TRAIN_CHAIN_PARTS=N` at submission:

```bash
TRAIN_CHAIN_PARTS=3 PRESET=oneeval_rollout8 sbatch examples/appworld/train/ace_rl/train.slurm
```

Part 1 takes the writer lock, then submits parts 2..N of the same launcher,
each with `--dependency=afterany:<previous part>`. Follow-ups keep part 1's job
name, comment, partition, account, QoS, reservation, time limit, memory,
constraint and GRES (`TRAIN_CHAIN_SBATCH_ARGS` appends other `sbatch` options),
replay its positional Hydra overrides and resume `RUN_DIR` with
`TRAINER_RESUME_MODE=auto`. Each job sees `TRAIN_CHAIN_PART`/`TRAIN_CHAIN_PARTS`;
only part 1 submits, so resubmitting a later part never duplicates the chain.
A part that starts after training finished exits successfully, so extra parts
cost only startup; `scancel` them if they are not needed. `RUN_DIR/train_jobs.json`
lists every job of the run (job ID, part, submission/start time, dependency).
Slurm spools each part when part 1 submits it, so keep the checkout unchanged
until the chain finishes. `RLLM_DRY_RUN=1` prints the `sbatch` commands on
stderr instead of submitting them.

## FP8 rollout with truncated importance sampling

`ROLLOUT_QUANTIZATION=fp8` (default `none`) applies to every training launcher
and method. The actor keeps BF16 master weights from the usual base model; only
the vLLM rollout replica serves FP8 weights, quantized by VeRL at every weight
sync. The knob appends:

```text
actor_rollout_ref.rollout.quantization=fp8
actor_rollout_ref.rollout.calculate_log_probs=True
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=2.0
```

Each sampled token's vLLM log-probability (temperature-scaled `processed_logprobs`)
is stored with the trajectory and compared with the BF16 actor's recomputed
`old_log_probs`; the policy-gradient loss of that token is multiplied by
`min(pi_bf16 / pi_fp8, 2)` (token-level truncated importance sampling). Tokens
whose sampled context differs from the training context, or that were sampled
with other sampling parameters, keep weight 1: environment/template tokens (no
loss anyway), the turn whose generated Qwen `<think>\n` was dropped from history,
every turn after a text-equivalent retokenization, ERL updater rows (top-p 0.9)
and internalized copies (replaced prompt), and workflow rows whose prompt is
left-truncated. Rejection sampling, bypass mode, top-p/top-k training sampling
and stepwise advantages are rejected with this knob. Metrics appear under
`rollout_corr/` (see [troubleshooting](troubleshooting.md#metrics-and-learning-signal)).

The setting is part of `training_identity` in `run.json`, so it requires its own
`RUN_DIR`; it is an ablation of the BF16 runs. FP8 rollout is VeRL's
rollout-only feature; FSDP FP8 training and pre-quantized `*-FP8` checkpoints
as the actor are not supported. VeRL does not claim this feature for the pinned
vLLM 0.19.1, so GPU startup, repeated weight sync with sleep/wake, and the first
optimizer update are unverified until a GPU smoke test passes:

```bash
RUN_DIR=/absolute/scratch/mbpp-grpo-fp8 ROLLOUT_QUANTIZATION=fp8 \
PRESET=grpo MODEL_VARIANT=nonthinking TOTAL_TRAINING_STEPS=4 TRAINER_SAVE_FREQ=-1 \
TRAINER_RESUME_MODE=disable TRAINER_LOGGERS="['console']" WANDB_MODE=disabled \
  sbatch --gres=gpu:2 examples/mbpp/train/grpo/train.slurm
```

Run the same command with `ROLLOUT_QUANTIZATION=none` and another `RUN_DIR` as
the baseline. Evaluate FP8-trained checkpoints with `EVAL_QUANTIZATION=fp8`
([evaluation](evaluation.md#fp8-evaluation)) to keep training and evaluation numerics aligned.

## Bounded validation runs

AppWorld and MiniSWEBench accept `TRAINER_TOTAL_TRAINING_STEPS`; MBPP and
Mastermind use `TOTAL_TRAINING_STEPS`. The trainer checks the limit after
incrementing its step, so a limit of **3** runs logged updates 1 and 2:

```bash
RUN_DIR=/absolute/path/to/validation/run \
PRESET=oneeval_rollout8 MODEL_VARIANT=thinking \
TRAINER_TOTAL_TRAINING_STEPS=3 TRAINER_SAVE_FREQ=2 \
TRAINER_RESUME_MODE=disable TRAINER_VAL_BEFORE_TRAIN=False MAX_TEST_TASKS=16 \
TRAINER_LOGGERS="['console']" WANDB_MODE=disabled \
  sbatch examples/appworld/train/ace_rl/train.slurm
```

Rejection can consume batches without optimizer updates, and a checkpoint save
does not test resume or learning quality. Evaluation-only initialization still
builds the training dataloader, so keep at least `data.gen_batch_size` training
records.

The epoch count is a second limit. A skipped or rejected batch uses data
without an update, so set `TRAINER_TOTAL_EPOCHS` (default 100) to give more
batches than the step target. AppWorld has 5 batches per epoch, so 100 epochs
is exactly 500 batches and a 500-step run has no room for a skipped batch. If
the data runs out first, the trainer saves the last completed update and
fails with an error. Raise `TRAINER_TOTAL_EPOCHS` and resume; the epoch count
is not part of the run identity.

For AppWorld, `TRAINER_VAL_ONLY=True TRAINER_VAL_BEFORE_TRAIN=True` replays
validation without updates; add `TRAINER_RESUME_MODE=resume_path
TRAINER_RESUME_FROM_PATH=/absolute/path/to/global_step_N` to replay a saved
checkpoint with its original world size and data configuration.

## Checks

`bash -n` checks launcher syntax, and `RLLM_DRY_RUN=1 bash <launcher>` prints the
resolved command without starting Ray, a model or benchmark services. Neither is
evidence that GPU training works; inspect the first optimizer update of a real job.
