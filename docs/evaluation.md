# Evaluation

Each benchmark has `eval/ace/eval.slurm` and `eval/vanilla/eval.slurm`, with
scientific settings in `presets.sh` and the benchmark's `eval/common.sh`. Use the
same file with `bash` inside an allocation or with `sbatch` to request one.

```bash
RUN=examples/appworld/train/ace_rl/runs/oneeval_rollout8/qwen3-4b-instruct-2507

# One checkpoint, each method (ACE visits8, vanilla pass8); rerun the same command to resume.
RUN_DIR=$RUN STEP=215 sbatch examples/appworld/eval/ace/eval.slurm
RUN_DIR=$RUN STEP=215 sbatch examples/appworld/eval/vanilla/eval.slurm

# Pretrained model; no training run needed.
MODEL_NAME=Qwen/Qwen3-4B MODEL_VARIANT=thinking sbatch examples/appworld/eval/vanilla/eval.slurm

# Review the planned command without creating files or starting services.
RUN_DIR=$RUN STEP=215 bash examples/appworld/eval/ace/eval.slurm --dry-run
```

`RUN_DIR` is the **training run**, not the evaluation destination. `STEP=N`
selects exactly `RUN_DIR/checkpoints/global_step_N`; there is no latest-step
guess. The launcher merges inference weights into
`RUN_DIR/checkpoints_merged/global_step_N` and leaves the training state in place.
The model and thinking mode come from `RUN_DIR/run.json` (or, for runs without a
record, from the run directory's `-thinking`/`-nonthinking` suffix); explicit
`MODEL_NAME`, `MODEL_VARIANT` or `DISABLE_THINKING` still win.

Results go to `RUN_DIR/eval/ace/visits8/step_N/` or
`RUN_DIR/eval/vanilla/pass8/step_N/` (`metrics_stepN.json`), so the
two methods never share a directory. Rerunning a command resumes that
directory; a different configuration for the same method and preset stops with
an error instead of mixing results, so give such a variant its own `OUTPUT_DIR`.
Without `RUN_DIR`, results go to
`examples/<benchmark>/eval/<method>/results/<preset>/<model>/`. `OUTPUT_DIR`
chooses an exact destination, for example for a task slice.

A missing GPU environment or benchmark runtime is prepared on first use; see
[setup](setup.md).

`RLLM_DRY_RUN=1` is equivalent to `--dry-run`. `VENV_DIR` or `PYTHON_BIN`
selects another evaluator environment. Submit from the source root or set
`RLLM_DIR`. GPU defaults request one L40S/A100/RTX 6000 GPU.

GPU evaluation launchers put FlashInfer's workspace under the allocation's
temporary directory, keyed by job ID, so service startup does not write to a
login-home cache. `FLASHINFER_WORKSPACE_BASE` remains an explicit override.

`#SBATCH` directives are static; request other resources with ordinary `sbatch`
overrides on the same entrypoint, such as `--time` or `--gres=gpu:2`.
Keep `VLLM_TP * VLLM_DP` equal to the allocated GPU count for local serving.
`sbatch --test-only` checks scheduler acceptance; it does not establish model fit
or successful execution.

## Presets

Each method has exactly one preset, which is also its default; any other
`PRESET` value is rejected before anything starts.

| Preset | Method | Protocol |
|---|---|---|
| `visits8` | ACE | 8 measured policy visits with 7 adaptation slots (`LOCAL_EVAL_PROTOCOL=sequential_adaptation_v1`, `G_ACT=G_REF=G_CUR=1`); defaults `TASK_CONCURRENCY=8 N_PARALLEL_AGENTS=8` (Mastermind: `PARALLEL=8`) |
| `pass8` | vanilla | 8 independent attempts per task (`G_ACT=8`), one visit, reflection and curation disabled; defaults `TASK_CONCURRENCY=1 N_PARALLEL_AGENTS=8` (Mastermind: `PARALLEL=8`) |

Capacity defaults fit one GPU; raise them with the variables in the capacity table
below without changing the protocol.

## Benchmark settings

Both methods of a benchmark share its generator settings and policy turn budget.

| Benchmark | Generator settings | Max policy turns | ACE role budgets |
|---|---|---|---|
| AppWorld | Qwen3-4B; 8192 prompt / 32768 response; 40960 model window; all dev tasks | 10 | 20480/2048 reflector, 8192/4096 curator |
| MBPP | Qwen3-0.6B; 4096/8192; 40960 window; first 100 test tasks | 6 | 20480/2048 reflector, 8192/4096 curator |
| Mastermind | Qwen3-0.6B; ACE thinking 8192/32768/40960, vanilla nonthinking 8192/10240/32768; all 100 test games | 5 | 30720/2048 reflector, 8192/4096 curator |
| MiniSWE | Qwen3.5-4B; 8192/32768; ACE 40000 / vanilla 40960 window; first 100 validation tasks | 10 | see `examples/mini_swe_bench/eval/README.md` |

`MAX_TURNS` overrides the budget for either method. Mastermind vanilla passes it
as `--max-turns`, which replaces every dataset row's `max_turns` (the registered
games carry 20), so the recorded games and `metrics_stepN.json` report the
effective budget.

Generator sampling is temperature 0.7, top-p 1.0 (MiniSWE: 0.6/0.95). Vanilla
disables both roles; `G_ACT` selects its attempts per task.
AppWorld vanilla uses the agent's built-in GRPO prompt; MBPP uses `base_system_prompt_v1.txt`.
AppWorld and MBPP enable thinking by default. Checkpoints of a
training run use the run's recorded thinking mode. For other AppWorld local
checkpoints, set `MODEL_VARIANT=thinking|nonthinking`, or set `DISABLE_THINKING`
and `ACCUMULATE_THINKING` to match training.
`ACCUMULATE_REASONING` defaults to the latter. API generators require it off.
These evaluation settings are distinct from the training presets; compare the
actual prompts, role counts, budgets and selection rule before comparing runs.
An API model (`MODEL_NAME=gemini-2.5-pro`, or `MODEL_SOURCE=api`) uses the same
presets without a local model server; request CPU resources with `sbatch` overrides.

## Sequential ACE and parallel vanilla pass@k

The default local ACE protocol is `LOCAL_EVAL_PROTOCOL=sequential_adaptation_v1`.
For `LOCAL_EVAL_EPOCHS=N`, each task receives **N measured policy visits and
N−1 adaptation slots**:

```text
policy visit 1 → reflection → curator delta → policy visit 2 → ... → policy visit N
```

Visit 1 runs the policy with a fresh copy of the configured initial playbook,
before any evaluation-time reflection or curation. Its reward covers the whole
policy attempt, including the allowed environment turns and their feedback; it
is not the first action's reward. With `LOCAL_EVAL_EPOCHS=1`, the sequential
evaluator makes no reflector or curator inference calls. An @1 point extracted
from an eight-visit evaluation has the same pre-adaptation meaning, but the full
evaluation still pays for the later visits and adaptation calls.

A valid curator delta updates the playbook directly. There is no separate
candidate-scoring policy rollout, rejection test or extra final rollout. An
unusable or no-op curator delta leaves the playbook unchanged; the next scheduled
policy visit still runs. Runtime exceptions still fail the evaluation. `G_REF=G_CUR=1` selects full ACE. Branching search with
larger role counts belongs to the explicit legacy protocol below. These changes
apply to standalone local evaluation, not ACE-RL training.

`visits8` uses `G_ACT=1 LOCAL_EVAL_EPOCHS=8`: eight total policy trajectories and
seven adaptation slots per task. An explicit `LOCAL_EVAL_EPOCHS=7` means seven
trajectories in the same `visits8` directory, so give such a variant its own
`OUTPUT_DIR`. Every selected task runs every visit even after a success; `FREEZE_SOLVED=True` is rejected.
Successes and later failures remain measured results, without replacing later
results by an earlier success.

Vanilla `pass8` uses `G_ACT=8 LOCAL_EVAL_EPOCHS=1`. The vanilla preset disables
reflection and curation. All k independent attempts are scheduled in parallel,
within the configured capacity; one successful attempt never cancels the rest.
Each attempt has a fresh environment and no context adaptation. The retained
sample records and trajectories support pass@1 through pass@k after evaluation.
Mastermind assigns distinct policy sampling seeds to attempts while keeping each
game's task seed unchanged.

Do not assume the evaluator's `SEED` controls model sampling. AppWorld's direct
ACE/vanilla evaluators and Mastermind's ACE evaluator seed host RNGs but do not
forward that value to actor or ACE-role model requests; the shared server
launchers also omit `--seed`. Mastermind vanilla forwards `seed + attempt_id`,
so pass8 panels labeled 42 and 43 share seven request seeds. These labels do not
establish independent sampling replications or matched streams across methods.
Verify provider request seeds before treating repeated panels as seed evidence;
future seed-control changes must preserve existing runs and use separate outputs.

| Benchmark | Capacity for vanilla `G_ACT=8` | Capacity for ACE `G_ACT=1` |
|---|---|---|
| AppWorld | `TASK_CONCURRENCY=C N_PARALLEL_AGENTS=8*C N_SERVERS=8*C` | `TASK_CONCURRENCY=C N_PARALLEL_AGENTS=C N_SERVERS=C` |
| MBPP | `TASK_CONCURRENCY=C N_PARALLEL_AGENTS=8*C` | `TASK_CONCURRENCY=C N_PARALLEL_AGENTS=C` |
| MiniSWE | Same rule; start with `TASK_CONCURRENCY=1 N_PARALLEL_AGENTS=8` and budget for eight containers | `TASK_CONCURRENCY=C N_PARALLEL_AGENTS=C` |
| Mastermind | `PARALLEL=8*C`; separate evaluator batches C complete groups of eight | `TASK_CONCURRENCY=C PARALLEL=C` |

`C` and `8*C` in this table denote numeric values to supply, not literal shell
values. Shared local evaluators require
`TASK_CONCURRENCY * G_ACT <= N_PARALLEL_AGENTS`; Mastermind ACE calls the capacity
`PARALLEL`. AppWorld also requires one stateful server per parallel agent.
Mastermind vanilla requires `PARALLEL>=G_ACT` and rounds its batch size down to a
whole number of task sample groups. Capacity is not proof that the GPU's KV cache,
host memory or benchmark containers fit; use the same task/sampling protocol
when measuring throughput.

Larger throughput settings that have been used for the eight-sample presets on
larger GPUs (override the preset defaults with these variables):

| Benchmark | `visits8` task workers / agents / servers | `pass8` task workers / agents / servers |
|---|---|---|
| AppWorld | 16 / 16 / 16 | 1 / 8 / 8 |
| MiniSWE | 16 / 16 / not applicable | 1 / 8 / not applicable |
| MBPP | 64 / 64 / not applicable | 8 / 64 / not applicable |
| Mastermind | 64 task workers, `PARALLEL=64` | `PARALLEL=64` processes eight complete pass@8 groups; `TASK_CONCURRENCY` does not apply |

PPO and ERL checkpoints are evaluated with vanilla; delayed joint PPO and ACE-RL
with ACE; GRPO with both.

For example, evaluate one MBPP checkpoint with each method:

```bash
RUN_DIR=$RUN STEP=210 sbatch examples/mbpp/eval/vanilla/eval.slurm
RUN_DIR=$RUN STEP=210 sbatch examples/mbpp/eval/ace/eval.slurm

# Base model with the same protocol
MODEL_NAME=Qwen/Qwen3-0.6B DISABLE_THINKING=False \
  sbatch examples/mbpp/eval/vanilla/eval.slurm
```

Eight attempts or visits can exceed a launcher's static wall-time default,
especially for MiniSWE; pass `--time` to `sbatch`. For more throughput, raise the
capacity controls according to the table above. For a checkpoint series, submit
one job per checkpoint and protocol; each job shares one serving model across its
attempts. Results made with `legacy_search_v1` have different rollout accounting
and must not be combined with sequential results.

All four benchmarks expose `pass_at_1_success_rate` through
`pass_at_k_success_rate` (including the numeric alias such as
`pass_at_8_success_rate`). For r≤k, `pass_at_r_success_rate` is the usual
without-replacement estimate `1 − C(k−c,r)/C(k,r)`, averaged over tasks, where c is
the number of successful retained samples. At r=k this is the fraction of tasks
solved by at least one attempt. At r=1 it is the mean success of all retained
attempts, not just sample zero. Use these explicit fields for cross-benchmark
plots: shared evaluator `success_rate` averages sample success, whereas Mastermind
vanilla `success_rate` is the fraction of tasks solved by any sample. Partial
reward averages are separate from either success metric.

Shared evaluators store sample identities/outcomes in each visit's task rows and
all policy trajectories in task artifacts. Mastermind stores grouped `tasks[].samples`
in `metrics_stepN.json` and every raw sample in `trajectories.json`; its optional
`prefix_success_rates` measure ordered prefixes separately from the estimator.
No successful task or failed attempt is omitted from pass@k accounting.

## AppWorld training-input protocol

`EVAL_PROTOCOL=training_input_parity` is an explicit opt-in for AppWorld
(ACE `visits8` and vanilla `pass8`) with a local checkpoint or Hub model. The default remains
`EVAL_PROTOCOL=legacy`. This setting selects input/sampling conventions; it is
independent of `LOCAL_EVAL_PROTOCOL`, which selects the local evaluation algorithm.
An `EVAL_PROTOCOL=legacy` input convention therefore does not select legacy search.
For the Qwen3-4B thinking comparison, select training-input parity for vanilla,
GRPO + ACE, ACE-RL + ACE, and their base-model evaluations:

```bash
EVAL_PROTOCOL=training_input_parity \
MODEL_NAME=Qwen/Qwen3-4B MODEL_VARIANT=thinking \
TEST_SPLIT=test_normal MAX_TASKS= \
bash examples/appworld/eval/ace/eval.slurm --dry-run

# Use eval/vanilla/eval.slurm with the same protocol and model/task settings
# for the vanilla curve. Use MODEL_PATH or CKPT_BASE/STEPS for checkpoints.
```

The comparison fixes these settings across evaluated checkpoints:

| Setting | Value |
|---|---|
| Thinking | Enabled; retain thinking and reasoning in policy history |
| Policy, reflector and curator sampling | Temperature 0.7, top-p 1.0, top-k -1 (unrestricted) |
| Serving context | 32768 tokens |
| Policy limits | 8192 initial prompt tokens, 32768 cumulative trajectory response budget, 10 turns (the shared AppWorld budget) |
| Reflector | 30720 prompt / 2048 response tokens; 20000-token history cap |
| Curator | 8192 prompt / 4096 response tokens; include reflector reasoning and failure reason |
| Playbook limit | 2048 tokens |
| ACE adaptation | `visits8`: `G_ACT=1`, `G_REF=1`, `G_CUR=1`; eight policy visits / seven adaptation slots |
| Vanilla | `pass8`: one local visit with eight attempts; reflector and curator disabled |
| Configured hard trajectory timeout | 7200 seconds; see the local ACE enforcement limitation below |

The response budget is cumulative across a policy trajectory, including
tokenized environment additions; it is not a fresh 32768-token allowance on
every turn. Serving uses `--generation-config vllm`, so a checkpoint's Hugging
Face sampling defaults do not replace the settings above. Policy top-k remains
at vLLM's unrestricted default; the ACE role requests explicitly set it to -1.
Numeric environment overrides remain possible and appear in the planned command;
changing them changes the comparison protocol and should be recorded explicitly.

Within one task, the 1/1/1 sequential protocol awaits policy, reflection and
curation in order; the next measured policy visit scores the updated context.
`N_PARALLEL_AGENTS=4` alone provides capacity; changing
`train_batch_size` does not make this local loop concurrent, and increasing
`G_ACT` changes the per-task sampling budget.

AppWorld `visits8` runs `TASK_CONCURRENCY=8` (CLI `--task_concurrency 8`) with
`G_ACT=1`, eight parallel-agent slots, and eight stateful servers. Each
worker owns a separate generator engine, runner, context rule, and single-server
endpoint. With `G_ACT>1`, workers instead share one server port pool; each active
trajectory exclusively checks out a server until its task world is closed.
For two concurrent pass@8 tasks, set `TASK_CONCURRENCY=2 N_PARALLEL_AGENTS=16 N_SERVERS=16`.
Task playbooks remain independent across visits, all workers finish a
visit before the next starts, and metrics/artifacts retain input task order.
`pass8` keeps one task worker by default. Concurrent requests change server RNG draw order;
the sampling parameters and per-task sample counts stay fixed, but bitwise sample
or score equivalence is not promised. Checkpoint-level Slurm arrays provide a
separate level of parallelism.

MiniSWEBench (ACE and vanilla launchers) accepts the same `TASK_CONCURRENCY=N`
(CLI `--task_concurrency N`). It needs no extra servers: every worker builds its
own generator engine and every `MiniSWEnv` already owns a private apptainer
testbed, overlay and grading directory. The constraint is capacity,
`N_PARALLEL_AGENTS >= TASK_CONCURRENCY * G_ACT` (default 8 agents: eight
single-sample `visits8` workers or one `pass8` task). MBPP (ACE and vanilla launchers, both
through `examples/<benchmark>/eval/ace/evaluate.py`) and the Mastermind ACE
launcher accept it under the same capacity rule (Mastermind names its capacity
`PARALLEL`); their environments are per-trajectory resources, so each worker only
needs its own generator engine. Per-task playbooks keep visits independent, so
the protocol is unchanged apart from server RNG draw order. Mastermind vanilla
uses its own batch evaluator with `--parallel` and `--g-act` and does not take
`TASK_CONCURRENCY`. The selected protocol and concurrency appear in the recorded
command; changed scientific settings require a separate output directory.
Standalone evaluations resume in the same output directory by default. Each
completed ACE task visit has an atomic receipt tied to its saved trajectory,
playbook, and ACE log. A retry reuses verified visits, imports complete task
artifacts from older runs, and reruns unfinished visits.
Startup failures with no task work restart in place. Mastermind vanilla saves
complete task batches and retries unfinished batches. An interrupted in-flight
trajectory is rerun, and model-server RNG state is not restored, so resumed
sampling need not be bitwise identical to an uninterrupted run. `RESUME=0`
refuses a nonempty output.
Keep Slurm logs, registration files and submission receipts outside the actual
evaluator `OUTPUT_DIR`. An output directory containing only these files is not
a resumable evaluation: without a matching `evaluation_config.json`, the
launcher rejects it before model startup. Check the output guard as well as the
no-write command plan; a successful dry run does not validate existing outputs.
Each worker can add `G_ACT` sandboxes and in-flight policy trajectories, so check the
job's CPU and host memory (the maintained launcher reserves 16 CPUs and 64 GB)
and expect KV-cache preemption before raising workers past the model server's
comfortable batch for the configured `MAX_MODEL_LEN`.

## Controlling evaluation throughput

`TASK_CONCURRENCY` is the application worker count, not a search budget. The
maintained concurrent local adapters support **AppWorld**,
**MiniSWEBench**, **MBPP** and **Mastermind ACE**. Without it every
local ACE loop visits one task at a time (`TASK_CONCURRENCY=1`); increasing an engine's parallel-agent
capacity alone does not parallelize it. A new benchmark needs a worker
adapter that isolates its mutable context, engine and environment before using
the same task-worker mechanism.

| Control | Meaning and constraint |
|---|---|
| `TASK_CONCURRENCY` | Independent task workers; `visits8` defaults to 8, `pass8` to 1 |
| `N_PARALLEL_AGENTS` | Generator capacity; must be at least `TASK_CONCURRENCY * G_ACT` |
| `N_SERVERS` | AppWorld servers with exclusive checkout per trajectory; must cover all parallel agents |
| `MAX_NUM_SEQS` | vLLM scheduling ceiling, default 64; does not create application requests or guarantee KV-cache fit |
| `MAX_MODEL_LEN` | Maximum model context; 32768 for the AppWorld training-input protocol; longer contexts require more KV memory |
| `GPU_MEMORY_UTILIZATION` | vLLM memory allocation fraction, default 0.90; distinguish allocated cache from occupied cache |
| `RLLM_OPENAI_REQUEST_TIMEOUT_S` | Positive finite SDK timeout in seconds for both OpenAIEngine endpoints, default 3600; constructor `request_timeout_s` takes precedence |
| `RLLM_GPU_KEEPALIVE` | Default `1`: inside a Slurm allocation, GPU evaluations run the shared utilization keepalive (`scripts/gpu_keepalive.py`, fill mode) as an owned process, logged beside the outputs as `gpu-keepalive-<job>.log`; `0` disables it. API models start none |
| `VLLM_TP`, `VLLM_DP` | Model/data parallelism; their product must match allocated GPU count |
| `MODEL_PATH` / `MODEL_NAME` / `CKPT_BASE` + `STEPS` | Explicit local model, Hub model, or checkpoint series |
| Slurm CPUs and memory | Host resources for the model server, worker engines and isolated environments; verify these when adding workers |

MiniSWE and Mastermind's shell launchers start the helper before checkpoint
merging and stop only their own helper during job cleanup. AppWorld and MBPP
keep their existing Python-managed lifecycle. `RLLM_DISABLE_GPU_KEEPALIVE=1`
also disables the helper; dry runs and runs outside Slurm do not start it.

Keep `G_ACT`, `G_REF`, `G_CUR`, local epochs, task set, thinking mode, sampling,
precision, response budgets and turn limits fixed when measuring concurrency.
`G_ACT` changes samples per task; it is not an interchangeable speed control.
AppWorld uses `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH`; MBPP uses
`GEN_MAX_PROMPT_LENGTH` / `GEN_MAX_RESPONSE_LENGTH`. Check the planned command
for the actual benchmark/model before submitting:

```bash
TASK_CONCURRENCY=8 N_PARALLEL_AGENTS=8 N_SERVERS=8 \
EVAL_PROTOCOL=training_input_parity MODEL_VARIANT=thinking \
bash examples/appworld/eval/ace/eval.slurm --dry-run
```

A serving benchmark should use exact input/output token lengths and the same
model/runtime, then confirm the gain on matched real task cohorts. Record actual
tokens, elapsed time, cache occupancy/preemption, active/waiting requests and
host memory/CPU. Different sampled runs can produce different amounts of work;
wall time alone does not isolate the concurrency effect.

Long non-streaming generations can exceed the OpenAI SDK's usual ten-minute
timeout as per-request decoding slows under concurrency. `OpenAIEngine` uses
the same one-hour default for tokenizer-backed completions and chat, with the
override above; an explicit per-call SDK `timeout` remains authoritative.
Sampling, response budgets and retry counts are unchanged. This is separate
from a trajectory deadline, a whole evaluator timeout, or the measurement
runner's short health/metrics HTTP timeout.

Enable SDK retry logging with `OPENAI_LOG=info` before Python imports OpenAI,
and verify that launch adapters preserve it in the evaluator environment.
In the inspected OpenAI 2.53 runtime, recovered retries log at INFO; the
default WARNING level can hide them entirely. Log validators must recognize
the actual formatter, including the SDK's bracketed timestamp and source-line
format. An empty retry scan alone does not establish that no retries occurred.

Check client retry logs as well as server finish counters. In the inspected
vLLM 0.19.1 runtime, client cancellations can contribute generated tokens
without appearing in the finished-request histogram or abort counter. At
stable idle boundaries, reconcile `generation_tokens_total` with
`request_generation_tokens_sum`, and histogram request counts with published
finish counts, for the same model/engine labels. A mismatch or recovered
client retry needs review before ranking a trial. Missing counters are
unavailable evidence, not zero discarded work. The measurement runner retains
raw scrapes, waits for available finished-request histograms to settle and
records the evaluator log hash; `metrics_valid` certifies counter integrity,
not absence of client retries or scientific equivalence.

## Uncertain AppWorld execution

The AppWorld client sends `/execute` once. HTTP 5xx, a request timeout or a lost
connection can occur after the action has changed state; these outcomes end the
trajectory with `ENV_EXECUTION_UNCERTAIN` rather than automatically replaying
the action or retrying the whole trajectory. Other endpoints retain their
existing bounded retries. An outer deadline that interrupts an in-flight
execute fails with `UncertainExecutionError` when no safe terminal result exists.

Evaluation and training trajectory records preserve `execution_uncertain=True`,
`execution_outcome="unknown"`, `execute_retry_attempted=False`, the error type
and available HTTP status. A simultaneous trajectory limit is retained as
`limit_termination_reason` so existing overlong filtering still applies. This
is a distinct reported failure, not evidence that the action was unapplied.
There is no server-side exactly-once recovery or automatic replay.

The flagged terminal result keeps reward zero and remains subject to the
existing GRPO/ACE selection and loss-mask rules; this change does not introduce
an infrastructure-error exclusion policy. It can save failed-request backoff,
duplicate actions and subsequent generation against poisoned state, including
in training. It does not accelerate healthy generation, and a training speedup
has not been measured. The separate AppWorld request-log serialization defect
is not repaired by disabling retries.

## Adaptation and final-policy results

Under `sequential_adaptation_v1`, all N policy visits remain in
`evaluation.local_epochs`, with one-based visit numbers and zero-based `iter_`
artifact filenames. `evaluation.adaptation_evaluation` references visits 1…N−1;
`evaluation.final_policy_evaluation` references visit N. The latter has
`kind="last_measured_policy_visit"` and provenance `extra_policy_rollouts=0`.
The final visit performs no reflection or curation. It is the measured policy
result after the preceding adaptation slots, including any slots that produced
no valid delta. `actually_applied_context_updates` records successful changes
separately from the number of scheduled update slots.

`evaluation.metrics` and its `final_policy/*` fields report the last measured
visit. They do not launch a second evaluation. Plot current success or reward
from each visit to show adaptation; plot `final_policy/avg_reward` or the explicit
final-policy pass@r metric for the final result. `cumulative_success_rate` records
whether each task succeeded in any preceding visit and is distinct from current
measured success. A later failure remains a failure in the current/final metrics.
`max_reward_mean` is the current visit's mean best sample reward, without padding
later visits from earlier successes.

The final summary is always present in sequential evaluation; no
`FINAL_POLICY_EVAL=True` is needed. With `G_ACT=1 LOCAL_EVAL_EPOCHS=8`, the final
summary therefore accounts for the eighth of eight policy trajectories per task.
Legacy-search runs, whose final pass was an additional rollout, write to a separate `_final_policy` directory.

### Legacy search and separate final-policy replay

`LOCAL_EVAL_PROTOCOL=legacy_search_v1` explicitly retains the previous algorithm:
each of N adaptation cycles scores the current playbook, reflects, curates,
tests candidates, and selects the next playbook. Its 1/1/1 setting can use one
acting and one candidate-scoring trajectory per cycle. `FINAL_POLICY_EVAL=True`
then adds an extra policy-only visit on the final selected playbook. For N=3,
that is up to seven policy trajectories per task (three acting, three candidate,
one final). Invalid candidates can reduce that count. Solved-task freezing and
branching candidate selection belong to this legacy protocol only.

AppWorld's ACE preset defaults `FINAL_POLICY_EVAL=True` only when the legacy
protocol is selected; `FINAL_POLICY_EVAL=False` retains adaptation-only legacy
evaluation. The additional pass scores every selected task, including solved
tasks. Its `final_policy_evaluation.kind` is `final_policy_only`.
`final_policy/avg_reward` is the fresh result and `final_policy/max_reward_mean`
uses the legacy cumulative-success padding rule. The separate replay controls
below are exposed by the AppWorld launcher and require the legacy protocol.

For a completed evaluation, `FINAL_POLICY_FROM_RUN_DIR` reads its saved
final playbooks and runs **only** the added policy pass in a separate output
directory. Keep the original model, task set, N and scientific controls:

```bash
# Retain the source run's exact model, assets, sampling and task settings.
LOCAL_EVAL_PROTOCOL=legacy_search_v1 \
FINAL_POLICY_EVAL=True FINAL_POLICY_FROM_RUN_DIR=/results/original_step150_tasksall \
OUTPUT_DIR=/results/final_policy_step150_tasksall LOCAL_EVAL_EPOCHS=3 \
CKPT_BASE=/models/merged STEPS=150 MODEL_VARIANT=thinking \
EVAL_PROTOCOL=training_input_parity \
TASK_CONCURRENCY=4 N_PARALLEL_AGENTS=4 N_SERVERS=4 \
bash examples/appworld/eval/ace/eval.slurm --dry-run
```

Continuation verifies the completed source command/configuration, metrics hash,
task/epoch coverage and agreement between its final ACE logs and saved playbook
text. Source and destination remain distinct. It preserves the old adaptation
metrics and binds the source artifacts in the new result's provenance; it does
not rerun the old adaptation cycles. This restores rendered inference playbook
content, not training controller state or an interrupted task's RNG. Historical
assets must be checked against their retained source provenance before reusing
old runs. This path is currently exposed by the AppWorld launcher only.

When the old command omitted `--prompts_dir` or `--initial_playbook`, supply
`FINAL_POLICY_SOURCE_MANIFEST=/absolute/path/source-receipt.json` (Python CLI:
`--final_policy_source_manifest`) together with `FINAL_POLICY_FROM_RUN_DIR`.
The replay command must explicitly set `PROMPTS_DIR` and `INITIAL_PLAYBOOK` to
files whose bytes match the frozen source assets. Current checkout defaults
cannot establish what a historical run used.

A version-1 receipt records absolute `source_run_dir` and `source_root` paths;
`source_configuration_sha256`, `source_completion_sha256` and
`source_metrics_sha256` for the unchanged old records; and
`deployment_manifest`/`launch_manifest` objects containing each retained
manifest's absolute `path` and `sha256`. The launch reference also contains an
`entry_path` list of object keys/list indices, for example
`["entries", 7, "evaluations", 0]` or `["base_evaluations", 1]`.

The verifier checks the deployment's `remote_root` and `source_files` hashes,
merges the launch's `common_env` with ancestor/entry `env` maps, and checks the
entry's run directory, model, step and ACE method. `RLLM_DIR` must select the
frozen source, and launch `TASK_CONCURRENCY` must match the recorded command
(both default to one if omitted). This distinguishes retained
manifests that include the same run entries from different source deployments.
The AppWorld evaluator and utility modules must match deployment
hashes and the supported default asset resolver. The entire frozen prompt
directory and initial playbook must also match deployment hashes. Missing or
inconsistent proof stops replay before clients or output writes.

The new result records the receipt hash, retained manifest hashes, selected
launch entry, resolved default paths and verified source-file hashes under
`final_policy_evaluation.provenance.source_asset_receipt`. This is an explicit
binding to retained launch/deployment evidence; it does not claim that the old
completion record contained asset hashes or establish exact RNG continuation.
Creating this separate receipt never requires rewriting an old source record.

`PROMPT_TRUNCATION` is an acting-trajectory termination label, not evidence that
the reflector's history-fitting guard failed. The acting conversation can exhaust
the model window even before its separate cumulative response allowance is spent.
Training's VeRL backend clips generation to remaining context and rejects oversized
prompts directly. The standalone OpenAI backend currently requests at least one
token, retries a context-length HTTP rejection, then the execution loop records
`PROMPT_TRUNCATION`. Reflector output `finish_reason=length` is a separate response
budget limit. Neither label by itself proves a crashed evaluation service.

The reflector receives the training representation reconstructed from chat
messages: user observations become strings, and each assistant `content` is
used for both reasoning and action. Separate `raw_completion` or reasoning
metadata is not substituted. Each observation is capped at 300 characters;
the history cap is measured in tokens. The training formatter keeps the largest
equal number of complete first/last steps that fits, dropping middle steps and
using a prefix/suffix fallback for oversized endpoints. After the history cap,
a tokenizer/chat-template-aware guard shrinks history further when needed to
fit the complete reflector prompt; it rejects fixed fields that exceed the
budget even with an empty history. The separate reflector `task_instr` is read
from the original environment task payload. In the audited AppWorld training
dataset this payload has no instruction field, so `task_instr` is empty; the
task text remains in generator chat history, subject to the observation cap
above. Curator input includes the reflector's `reasoning` and `failure_reason`
when present and uses the training task-batch header convention (`Task 0` for
a single-task evaluation).

This aligns inputs and sampling, without making the evaluation algorithm the
training algorithm. The selected ACE-RL training setup samples eight policy
trajectories, four reflections and up to seven curator candidates per task.
Sequential evaluation instead applies one valid delta between measured visits,
without candidate scoring. Only `legacy_search_v1` retains evaluation's
`best_rejection` selection rather than training's `best_rejection_update`
grouping and random tie handling. Curator output validation also remains the
evaluator behavior: ADD/UPDATE tips shorter than 10 stripped characters are discarded,
direct JSON operation lists are rejected, and `tip_id` is accepted as an alias
for `bullet_id`. The training path accepts short tips and operation lists but
requires `bullet_id`. Fresh per-checkpoint playbooks, held-out tasks and absence of
optimizer updates remain evaluation differences.

The launcher passes a 7200-second hard trajectory timeout, but the current
local ACE acting, candidate and final-policy paths call `execute_tasks`
directly. They bypass the retry wrapper that enforces this deadline in
`AgentExecutionEngine`. Do not treat the configured flag as an enforced local
ACE deadline; the inference request timeout and whole-evaluator timeout are
separate limits. Neither the protocol name nor a CPU test establishes full
training equivalence or GPU runtime validation.

The [AppWorld defaults](../examples/appworld/eval/common.sh),
[launcher planner](../scripts/evaluation_runtime.py),
[AppWorld role setup](../examples/appworld/eval/ace/appworld_ace_utils.py),
[training-input helpers](../rllm/algorithms/ace/context/training_inputs.py) and
[AppWorld reflector](../examples/appworld/eval/ace/training_inputs.py) define the
executable contract. The planner records the protocol in
`evaluation_config.json`.
See [visualization](visualization.md) for plotting these curves.

## Models, outputs and overrides

For AppWorld and MBPP:

```bash
# One merged checkpoint directory. AppWorld additionally requires its thinking mode.
MODEL_PATH=/path/to/global_step_20 \
bash examples/mbpp/eval/ace/eval.slurm --dry-run

# A checkpoint series. STEPS is required and selects global_step_N directories.
CKPT_BASE=/path/to/merged/checkpoints STEPS="10 20 30" \
MODEL_VARIANT=nonthinking bash examples/appworld/eval/ace/eval.slurm --dry-run

# API generation. Authentication is read by the client from its normal environment.
MODEL_NAME=gemini-2.5-pro PYTHON_BIN=/path/to/api-environment/bin/python \
bash examples/mbpp/eval/vanilla/eval.slurm
```

`MODEL_NAME` accepts a Hub ID or recognized API model name. `MODEL_PATH`
selects one already merged local model. `MODEL_SOURCE=auto|hub|local|checkpoint|api`
makes the choice explicit. Single models use step 0. Legacy merged series remain
supported only with an explicit `CKPT_BASE` and distinct `STEPS`; there is no
implicit checkpoint-root fallback. For training shards, use `RUN_DIR` and `STEP`
as shown above. AppWorld local checkpoints require an explicit thinking mode.

All four benchmarks accept the optional serving control
`EVAL_TIE_WORD_EMBEDDINGS=False` (or `True`; case-insensitive, with `0`/`1`
aliases). It passes an explicit `tie_word_embeddings` boolean through vLLM's
`--hf-overrides` without changing the saved checkpoint. Use `False` when the
checkpoint has verified independent embedding and output-head weights but its
configuration declares them tied. The launcher does not infer this override.
Unset preserves the model configuration and existing evaluation records; an
empty or malformed value and API-only models are rejected before runtime setup.
The explicit override is recorded in `evaluation_config.json` and the completion
receipt. Adding, removing, or changing it requires a different output directory,
including when the previous evaluation is already complete.

For an inventoried two-rank checkpoint, [merge_eval_checkpoint.py](../scripts/merge_eval_checkpoint.py)
wraps that merger with CPU-only execution, `--merge-dtype preserve`, source
fingerprint checks, byte-identical tokenizer/prompt assets and output validation.
It writes a complete model in a unique hidden sibling attempt before publishing
the requested inference-model target. Publication never replaces an existing target, including an
empty directory. Where `renameat2(RENAME_NOREPLACE)` is unavailable or unsupported,
the final path is an atomically created relative symlink to that attempt's
`model` directory. Keep the referenced attempt directory with the published
checkpoint; it contains the actual weights. Reuse accepts only the exact sibling
layout with matching target/attempt provenance, unchanged source fingerprints,
verified HF assets and valid weight metadata/headers. Unrelated or altered
symlinks are rejected. Logs and failed attempts are retained for inspection;
there is no automatic cleanup or overwrite policy. Source weight fingerprints
use size and modification time, not full-file cryptographic hashes.

### FP8 evaluation

`EVAL_QUANTIZATION=fp8` (default `none`) evaluates with the numerics of the FP8
training rollout (`ROLLOUT_QUANTIZATION=fp8`, see [experiments](experiments.md)).
Every launcher converts the BF16 model with
[convert_fp8_checkpoint.py](../scripts/convert_fp8_checkpoint.py) and serves the
copy with `--quantization fp8 --dtype bfloat16`:

| Model | BF16 source | Served FP8 copy |
|---|---|---|
| `RUN_DIR` + `STEP` | `checkpoints_merged/global_step_N` (merged first) | `checkpoints_merged/global_step_N-fp8` |
| `CKPT_BASE` + `STEPS` | `CKPT_BASE/global_step_N` | `CKPT_BASE/global_step_N-fp8` |
| Hub base (step 0) or `MODEL_PATH` | cached BF16 snapshot or local directory | `.cache/fp8_models/<name>-fp8` (`EVAL_FP8_CACHE_DIR`) |

The converter applies VeRL's rollout quantization offline: each decoder Linear
weight (q/k/v/o, gate/up/down) is cast to BF16 and stored as e4m3 in 128x128
blocks with a float32 `weight_scale_inv` (block absmax / 448, partial edge
blocks zero-padded); embeddings, `lm_head`, norms and biases are copied unchanged,
and `config.json` gains the `quantization_config` VeRL injects
(`quant_method=fp8`, `fmt=e4m3`, dynamic activations, `weight_block_size=[128,128]`,
its MoE `mlp.gate` ignore list). These are the fields of Qwen's published FP8
checkpoints as far as known; they were not compared with those repositories,
which are not needed because the copy is made locally.

The conversion runs inside the evaluation job, before vLLM starts, with the
job's GPU visible. By default (`EVAL_FP8_KERNEL=verl`) it quantizes with
`scaled_fp8_blockwise` from the evaluation environment's VeRL, the function the
rollout weight sync calls, using its Triton kernel; without a visible GPU or Triton
the job fails instead of falling back. `EVAL_FP8_KERNEL=replica` explicitly uses
the CPU restatement of the same arithmetic (direct round-to-nearest-even e4m3
cast), which can differ from a kernel by one e4m3 code on a small fraction of
elements. Triton itself can lower the kernel's float32-to-e4m3 cast through
float16 (its FP16-intermediate path, expected below compute capability 9.0 and
not yet checked on an H100/H200): on an L40S the kernel differed from
the CPU replica in 0.37% of Qwen3-0.6B weights, each by one code, and an
emulated float16 cast accounted for 80% of those differences. The L40S kernel
therefore need not equal the H100/H200 training rollout bit for bit;
`EVAL_FP8_MIN_CAPABILITY=9.0` refuses conversion on older GPUs. A copy converted
once on an H100/H200 (run the converter in such an allocation) is reused by later
L40S evaluations. `fp8_conversion.json` and `evaluation_config.json`
(`quantization_kernel`, `quantization_conversion`) record the implementation,
kernel hash, GPU and compute capability. A copy is reused only when the source,
converter, implementation and kernel hash match; a kernel copy never replaces
a replica copy or the reverse, and any other existing target is refused.
`--verify-against-verl [<verl-source>]` re-quantizes every tensor with VeRL's
kernel on the visible device and reports code and scale differences; it fails
only when bit identity is expected (a kernel copy checked on a GPU of the same
compute capability).

FP8 results never share a directory with BF16 results: the preset directory gains
a `-fp8` suffix (`RUN_DIR/eval/<method>/<preset>-fp8/step_N`, or
`results/<preset>-fp8/<model>`), the served model path differs, and
`evaluation_config.json` records `quantization` and `quantization_source`. A
reused `OUTPUT_DIR` with the other precision is refused. API models reject the
setting. The L40S/A100 block-FP8 GEMM path (Triton on pre-Hopper GPUs) differs
from the H100/H200 training kernels and is unverified until the FP8 GPU smoke
test passes.

`MAX_TASKS` and `TASK_OFFSET` select the task slice; an explicitly empty
`MAX_TASKS` selects the full remaining split. All launchers record benchmark,
method, preset, step, task label, output path and the evaluator command in
`evaluation_config.json`. Standalone `RUN_TAG` changes the model-directory label,
and `RUNS_BASE` changes the explicit results root. `OUTPUT_DIR` selects one exact
destination. `EVAL_LANE_SUFFIX` appends one name to the preset directory, so a
step series with changed settings gets its own lane beside earlier results.
Use separate destinations for different slices or scientific settings;
conflicting output is refused rather than inferred from a long name.
AppWorld launchers use split-specific task-ID JSON files exported by setup to
`examples/appworld/data/`, or an existing
`examples/appworld/eval/support/task-ids/` export. `APPWORLD_TASK_IDS_DIR`
selects another directory explicitly. AppWorld evaluation does not reuse the training
registry's generic `test` entry.

Set supported evaluator options through uppercase environment variables;
`MAX_MODEL_LEN` maps to `--max_model_length`. Run the benchmark's `evaluate.py`
CLI directly for options outside the launcher contract. AppWorld and
MBPP reject raw passthrough overrides that could invalidate the planned model,
step or output identity. `WANDB_PROJECT` opts into W&B logging under `WANDB_ENTITY` (default: the account's
default entity). Local W&B files stay under the evaluation
output directory, and local metrics are always saved. Key files are explicit `API_KEY_FILE` / `ACE_KEY_FILE` inputs;
launchers do not scrape personal credential files.

AppWorld starts one local stateful server per parallel agent, preserving the
requested concurrency. `APPWORLD_PORT` selects the start of its port range;
unset ports are chosen from available local ports. An existing `APPWORLD_URL`
can be used with `N_PARALLEL_AGENTS=1`. Each owned server process group is stopped
on completion, failure or interruption. An externally supplied server is not
stopped by the launcher.

All maintained benchmark launchers write `evaluation_config.json`, task artifacts,
and `metrics_stepN.json` under their run directory. AppWorld and MBPP
also retain server/evaluation logs there. A completion record
(`evaluation_complete.json`) is published under the output lock only after
successful evaluator exit and validation of the model, step, protocol, policy
budget and task/sample outcomes. Mastermind vanilla uses the independent-sample
validator; current ACE and other vanilla evaluations use the sequential validator.
The record binds the command to the metrics SHA-256. Reuse requires the same
recorded command and unchanged metrics. Incomplete directories with matching
recorded commands resume in place; conflicting configuration or changed
completed-task artifacts are rejected. Nothing is deleted. A launcher output
is complete only with every measured visit from 1 to N in one run; there is no
launcher path that extends a saved evaluation with further visits.

## Context modes and validation

MiniSWEBench exposes `reflector_only` and action/output-only reflection history
as explicit evaluation ablations. Reflector-only skips the Curator and turns
reflector insights into playbook additions. Sequential mode applies valid
additions for the next measured visit; legacy mode retains candidate scoring and
selection. Action/output history reads post-action environment observations and
omits model reasoning. These change the algorithm or its evidence, rather than
being required for basic MiniSWEBench compatibility. The full ACE path still uses
reflector insights followed by a curator delta;
the default history formatter is unchanged.

Mastermind and MiniSWEBench training and ACE evaluation share the prompts under
`train/ace_rl/prompts`, as AppWorld and MBPP do. Shared adapters accept the
environment's `won` and the trainer's `all_passed` success fields.

The [figure guide](visualization.md) covers the extract-then-plot local gallery, exact
slice selection, padded turn rewards and ACE iteration rewards.
`scripts/merge_task_chunks.py` validates derived aggregation inputs; use `--help`
for its explicit arguments.
Some summaries cannot be reconstructed from chunk averages and are omitted with
provenance rather than fabricated.

The maintained CPU gate includes context and launcher regressions. Additional
focused evaluation and plotting checks are:

```bash
python -m pytest --confcutdir=tests/evaluation tests/evaluation -q
python -m unittest discover -s tests/figures -p 'test_*.py'
```

CPU tests and synthetic renders do not establish GPU, API, container or remote
AppWorld runtime compatibility. Those require actual benchmark execution in the
selected environment.

## Controlled experience handoff

With the default `PROVIDER_SEED_SCHEME=legacy`, do not assume the evaluator's
`SEED` controls model sampling. AppWorld's direct ACE/vanilla evaluators and
Mastermind's ACE evaluator seed host RNGs but do not
forward that value to actor or ACE-role model requests; the shared server
launchers also omit `--seed`. Mastermind vanilla forwards `seed + attempt_id`,
so pass8 panels labeled 42 and 43 share seven request seeds. These labels do not
establish independent sampling replications or matched streams across methods.
Verify provider request seeds before treating repeated panels as seed evidence;
use separate outputs for provider seed controls.

ACE evaluation on all four benchmarks supports a frozen-weight handoff comparison:
`ADAPTATION_MODE=apply` applies each valid delta; `ADAPTATION_MODE=discard`
generates the same scheduled reflection and curator work but discards the delta,
so every actor visit receives the initial book. Both use the original actor
wrapper, measured visit budget and final-visit reuse. Role calls can use different
numbers of tokens and may fail; this is matched scheduled work, not equal token
cost. The default remains `apply` with historical unseeded provider requests.

For paired request sampling, set `PROVIDER_SEED_SCHEME=sha256_v1`, a nonempty
`PROVIDER_SEED_NAMESPACE` containing the benchmark, and `SEED`. Use the same
namespace and seed across the two arms. The SHA256 schedule includes namespace,
replicate seed, task, actor attempt or auxiliary visit, role, sample and actor
turn. It excludes the arm, execution order and checkpoint. Transport retries
reuse the logical call seed. This controls provider sampling inputs; it does
not guarantee deterministic GPU execution. Keep seed schedules fixed when
comparing checkpoints.

Opt-in preset directories include the application mode, namespace hash and
replicate seed. An explicit `OUTPUT_DIR` still selects the exact directory.
`evaluation_config.json` records the control and command. Each task-visit row
and ACE log records generated versus applied versus changed context, the
provider identities, actual dispatched seeds and canonical request hashes.
The discarded candidate and its generated operations remain in the ACE log.
Resume rejects a different arm or seed schedule. The completion validator
checks control identity, unchanged discard-arm books and seed receipts. Each
auxiliary call also records its actual dispatch count or an explicit failure
before dispatch; removing a dispatched reflector or curator receipt fails
validation.

Request receipts retain the provider's actual prompt, completion and total
usage immediately after the SDK returns, including responses followed by a
parse failure and retry. A failed dispatch or missing provider usage has
`usage=null`. Task, visit and whole-evaluation `provider_usage` summaries sum
the retained receipts, including visits loaded on resume. `observed_*_tokens`
sum known usage; `*_tokens` is null when any dispatch has unknown usage. This
measures returned-provider usage. SDK-internal HTTP retries do not expose
separate generation usage; the receipts do not establish unseen server work.

For repaired-feedback evaluations, set `HANDOFF_PROTOCOL=feedback_v1` (direct
CLI `--handoff_protocol feedback_v1`). This selects the repaired adapter and
`curator_prompt_handoff_v1.txt`; it also selects
`reflector_prompt_handoff_v1.txt` when that benchmark supplies it. The actor
prompt stays selected by its existing argument. `legacy` keeps the original
role files even when repaired variants exist beside them. The protocol enters
the immutable control configuration and automatic output directory, so it
cannot silently resume a legacy run.

Direct
CLI equivalents are `--adaptation_mode`, `--provider_seed_scheme` and
`--provider_seed_namespace`.
