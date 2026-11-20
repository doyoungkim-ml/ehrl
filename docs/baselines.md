# Baseline training and evaluation

The maintained baseline matrix has three methods and four benchmarks:

| Benchmark | Model | Mode |
| --- | --- | --- |
| Mastermind, MBPP | `Qwen/Qwen3-0.6B` | Nonthinking |
| AppWorld, Mini-SWE-Bench | `Qwen/Qwen3-4B-Instruct-2507` | Instruct, thinking disabled |

Use `examples/<benchmark>/train/<method>/train.slurm`. Methods are `erl`
(`PRESET=rollout8`), `ppo` (`PRESET=ppo`), and `delayed_joint_ppo`
(`PRESET=batch<N>_rollout8`). All use eight initial rollouts or complete chains
per task. Joint PPO's task batch size N is 6 for MBPP and 8 for
the other benchmarks; see [experiments](experiments.md#methods-and-presets).
Each has its own canonical run directory under
`train/<method>/runs/<preset>/<model>/`. Launchers resolve prepared runtime,
dataset, and model locations from the checkout and benchmark setup defaults.
Explicit environment exports select nonstandard locations; `RUN_DIR` selects
the canonical artifacts when evaluating a source snapshot.

## Original ERL

The executable reference is Microsoft's [released ERL implementation](https://github.com/microsoft/experiential_rl/tree/20cd6460753973b9a5834769e2feed02736f44c6),
commit `20cd6460753973b9a5834769e2feed02736f44c6`, using the full-objective
Hotpot recipe with its default one-example workflow episodes. This is a port
of that release to the four benchmarks and the current pinned VeRL backend;
it is not a reproduction of the paper's original benchmark/model results.
Provenance and the upstream license live in `rllm/algorithms/erl/`.

1. Every initial attempt uses the benchmark's ordinary agent prompt.
2. Only a failed attempt invokes the prompt updater and a fresh retry. The
   updater receives the complete observed attempt, reward, feedback, and the
   workflow slot's remembered prompt (or the original prompt initially).
3. The updater's reward is the retry's raw reward. A successful retry updates
   that slot's remembered prompt. Memory conditions future updates; it never
   modifies ordinary first attempts or vanilla evaluation.
4. First attempts, updater outputs, and guided retries participate in GRPO.
   A copy of every actual retry is also trained under its original first-attempt
   prompt using the released RAFT rule: positive raw reward becomes the
   assistant-token advantage, while nonpositive reward contributes zero.
   This is not an additional separately weighted SFT optimizer step.

The controller has no ACE curator, playbook candidate search, or reward-group
rejection. The revised prompt is extracted from `<prompt>...</prompt>` using
the release's parser. Models, environment adapters, task feedback formatting,
and task context limits necessarily differ from the released Hotpot/FrozenLake
examples. MiniSWE attempts use the ordinary SWEAgent and MiniSweWorkflow
termination/grading path. AppWorld retains its task and demonstration messages
when replacing retry instructions.

The default recipe uses eight tasks per batch, eight initial rollouts per task
on every benchmark, 32-task PPO minibatches, GRPO plus RAFT, sequence-mean
token-sum loss, KL loss coefficient 0.001, and upper clipping 0.28.
The updater samples at temperature
0.7/top-p 0.9 with up to 8192 generated tokens. Role-specific context capacity
holds the failed attempt and remembered prompt without shortening the normal
task budget. An updater input longer than its budget drops that whole rollout,
first attempt included, and leaves the remembered prompt unchanged. The release
does the same: a generation prompt over its limit ends the episode with no
trajectories. Nothing is summarized or truncated, and
`erl/updater_prompt_overflow_rows` counts the dropped rollouts. Variable-size
mixed batches use zero-loss padding with corrected per-minibatch normalization.

Known differences from the release:

- The updater uses the release's generic instruction from
  `examples/erl/erl_flow.py`. The release's task-specific instructions cover
  only its own tasks.
- The release appends a fixed action-format reminder to each revised prompt.
  This port does not; a revision replaces the instruction text only.
- A first attempt that reaches the context limit keeps its zero reward, as in
  the other baselines. The release would end that episode without trajectories.

Checkpoints include actor/optimizer/RNG state, the dataloader, and ERL's
per-slot prompt memory and available-slot order. Controller sidecars are saved
before the backend publishes the latest checkpoint marker. CPU continuation
tests do not establish bitwise reproducibility of asynchronous GPU generation.

## PPO and delayed joint PPO

`ppo` uses the ordinary task agent, a learned critic, token-level GAE
(`gamma=1`, `lambda=0.95`), and clipped PPO, with 8 tasks × 8 rollouts per step.
Its default critic learning rate is `1e-5`, with ten critic-warmup steps and
reward-group rejection disabled. Within the 301-step budget the actor updates
on steps 10–300, after the warmup.
Actor and critic checkpoints include model, optimizer, and extra state.

`delayed_joint_ppo` selects the dedicated `LocalACEJointPPOTrainer` with
`training_mode=joint_ppo`. It collects eight complete ACE chains per task
(`actor_rollout_ref.rollout.n=8`):

```text
initial actor attempt -> reflection -> curation -> actor with updated playbook
```

Each chain produces one initial actor rollout, one reflection, one curator
response, and one downstream actor rollout (`g_ref=g_cur=1` per chain).
`batch<N>_rollout8` specifies tasks per step (MBPP 6, Mastermind 8,
AppWorld 8, MiniSWE 8) with an N-task actor/critic mini batch. With group size
G, collection produces `4*N*G` training rows. All four segments receive their
own chain's downstream verifier reward; ordinary GAE remains per row, without
group normalization or reward-based selection. All chains start from the same
task-local playbook, curate private copies, and use those copies downstream.
The first chain's playbook persists for the next task visit, independently of
reward; artifacts record every chain and the persisted chain ID. Task visit
and reset counters advance once per task, not once per chain. Invalid or empty
curation retains that chain's preceding playbook and trains every segment.
Collection uses fixed policy weights; the critic and actor update afterward.
Changing the group size changes the sample and token budget and requires a
fresh run rather than resuming an incompatible checkpoint.

This is an ACE-style adaptation of
[CompactionRL](https://arxiv.org/html/2607.05378v1#S4), with a learned value
critic, ordinary segment-local GAE (`gamma=1`, `lambda=0.95`), and token-mean
PPO loss. It does not implement context replacement or CompactionRL's
cross-segment advantage correction. The critic predicts returns; it is not
a learned reward judge. Critic warmup remains configured to 10; actor updates
run on steps 10–300 of the 301-step budget.

The GRPO `ace_rl` / `delayed_oneeval_rollout8` method is separate: it
samples eight initial actor rollouts, one reflection and curation per rollout,
and one evaluation per candidate, with seed-policy training and its existing
grouped credit/selection rules. Older `delayed_joint_ppo` runs used eight
evaluations per candidate, selected-candidate policy training, and reward-group
rejection. Those checkpoints and metrics describe that former objective; they
must not be resumed or relabeled as the complete-chain method; restart the run
from its base model in a fresh `RUN_DIR`.

## Evaluation

Evaluation uses the existing benchmark `eval.slurm` entrypoints on L40S:

| Training method | Evaluation launcher |
| --- | --- |
| PPO | `examples/<benchmark>/eval/vanilla/eval.slurm` |
| Original ERL | `examples/<benchmark>/eval/vanilla/eval.slurm` |
| Delayed joint PPO | `examples/<benchmark>/eval/ace/eval.slurm` |

PPO and ERL vanilla evaluation uses one policy sample and one visit without
reflection, prompt memory, curation, or adaptation. Delayed joint PPO uses
the canonical ACE evaluation protocol and its reflection/curation settings.
Benchmark-specific prompts, task splits, generation budgets, and sampling
remain those of the selected canonical evaluator.
For AppWorld use `EVAL_PROTOCOL=training_input_parity`. The evaluator reads the
model and thinking mode from the run (Instruct-2507 evaluates as nonthinking).

The new training recipes disable initial, periodic, and final internal
validation. Internal trainer validation is not interchangeable with the
standalone evaluator. `trainer.save_at_end=True` saves the final
optimized state even when the last update falls between periodic saves.
Select an actual saved checkpoint step; do not infer a checkpoint number
from the requested training-step budget. The recipes keep every checkpoint
(`KEEP_LAST_N=0`, every 10 steps up to step 300) so each one can be evaluated;
a positive `KEEP_LAST_N` set before the run retains only that many. See
[experiments](experiments.md#training-length-turns-and-checkpoints) for the
disk estimate.

Submit evaluation only after the selected checkpoint exists. Pass its actual
`RUN_DIR` and `STEP` to the matching existing `eval.slurm`; do not queue
speculative evaluation dependencies alongside a new training run. Outputs go
under `RUN_DIR/eval/<vanilla|ace>/<preset>/step_N`. Training's Hopper runtime
settings do not select evaluation hardware.

## Submission and evidence

Training launchers request two GPUs, 32 CPUs, 500G RAM and 48 hours;
`TRAIN_CHAIN_PARTS` chains longer runs (see [experiments](experiments.md#overrides-and-resume)). Each benchmark
inherits the working runtime settings of its corresponding GRPO or ACE
parent, including attention backend, compilation/eager behavior, cache,
memory settings, and shared startup/helper/cleanup behavior. There is no
matrix-wide backend or optimization-level override. Compare effective configurations against successful runs on the
same model, hardware, and prepared runtime; retain intentional algorithm
differences described above. GPU keepalive follows the shared training runtime,
just as for GRPO and ACE; baseline helpers and submission environments must not
add a method-specific disable. Include exported environment variables in the
comparison and disclose any operational exception before submitting.
Use ordinary `sbatch` with the applicable account and hardware constraint.
Keep the source checkout unchanged while jobs run. Evaluation is a separate
submission after the checkpoint exists.

Validate shell syntax, the full 15-command Hydra matrix, the maintained CPU
regression gate, actual prepared-runtime imports, and engine arguments
against the installed vLLM API before submitting. These checks do not prove
GPU execution. Verify engine startup and an optimizer update after launch;
report queued jobs and CPU/schema checks separately from measured training
progress. A successful `sbatch --test-only` or accepted job is not evidence
that training works.
