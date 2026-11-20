# Troubleshooting and diagnostics

Verified failure patterns and how to read the evidence. Match the exact symptom,
versions, and workload before applying a fix here, and prefer the benchmark's
last successful run over any rule of thumb. When a new failure is understood and
fixed, record the pattern in the matching section below rather than in a new
file.

## Establish what actually ran

- Resolve the submitted launcher, its stdout/stderr, the selected Python, the
  imported rLLM and VeRL revisions, GPU family, resources, realized Hydra
  arguments, W&B identity, checkpoint root, and resume evidence
  (`scontrol show job`, `sacct`, `RUN_DIR/logs/`, `RUN_DIR/run.json`).
- Slurm runs a spooled copy of the submitted file. Editing a launcher does not
  change a queued job, and `BASH_SOURCE` may point into the spool; launchers
  therefore locate the checkout through `RLLM_DIR`/`SLURM_SUBMIT_DIR`. Code the
  launcher sources at run time (`scripts/`, `examples/`) is read when the job
  starts.
- Before calling a job stalled, check completion markers and phase-specific log
  timestamps. Normal cleanup can briefly leave only sidecars after metrics and
  completion markers are written.
- Find the first causal error in preserved Ray/EngineCore and sidecar logs.
  `Connection refused`, `EngineDeadError`, or a trajectory-retry wrapper is a
  consequence, not a root cause. Failed training jobs copy Ray logs to
  `RUN_DIR/logs/ray-<job>/`.
- A dry run (`RLLM_DRY_RUN=1` or `--dry-run`) and `sbatch --test-only` validate
  configuration and scheduling only. Report GPU engine startup and the first
  optimizer update separately.

## Changing a runtime setting

Compare a new method or rerun with a successful parent run on the same
benchmark, model, hardware, and environment: submitted environment, shared
runtime helpers, startup and cleanup, not only Hydra arguments. Keep the
parent's backend, compilation/eager, cache, and memory settings unless the task
requires a change, and state each intentional difference. One benchmark's
backend or memory setting is not a default for the others.

For joint PPO group-size overrides, candidate-evaluation coverage guards from
ACE presets do not apply: `rollout.n` counts complete chains per task and each
chain has one downstream attempt. Keep the dedicated pre-Ray validation
(`g_ref=g_cur=candidate_eval_rollouts=1`, GAE, no rejection); do not increase
candidate evaluation samples to work around a parent preset guard.

## vLLM engine and GPU failures

First decide which phase failed: argument validation, model/engine
initialization, graph capture, generation, or actor/critic backward.

| First causal evidence | Check next |
| --- | --- |
| Invalid or unsupported engine argument | Validate the value and type against the installed vLLM with the job's Python. Hydra accepting an override does not mean vLLM does. |
| Attention-kernel or TMA descriptor failure | Read the kernel stack and resolved backend; compare with a successful run on the same hardware and environment. |
| Failure during graph capture, autotuning, or generation | Inspect that phase's compilation settings. Eager execution or another backend is not a universal fix. |
| OOM during actor/critic backward | Check resident engine/KV-cache memory and the colocated layout before changing batch sizes or model limits. |

Do not repair a configuration error by reinstalling the environment. Eager
execution and `free_cache_engine` address different phases: keeping the vLLM KV
cache during FSDP backward can exhaust memory, while releasing it exercises
version-specific sleep/wake paths. A failure on one GPU family does not justify
moving the experiment to another.

For a low-utilization cancellation, check the job's keepalive log through the
cancellation time, actual optimizer updates in `metrics.jsonl`, and the
scheduler's per-device readings and averaging window. Quiet driver stdout alone
does not show a stalled trainer. Helper duty-cycle percentages are timing
estimates, not scheduler utilization averages. If the scheduler reports zero
despite recorded GPU work, preserve both reports and compare CUDA device UUIDs
with the monitoring device UUIDs inside that allocation; do not assume a mapping
fault or increase helper load without evidence.

## Rejected batches and weight restoration

All-zero reward groups can be rejected repeatedly before any logging,
checkpoint, or optimizer step, which looks like a stalled server but is
rejection starvation. A rejected or rounded-away batch can also skip weight
restoration after hybrid rollout replicas sleep; the PPO helper
`_maybe_restore_rollout_after_skipped_batch` restores unchanged weights through
`checkpoint_manager.update_weights(self.global_steps - 1)`, since a level-2
sleep releases weights and a bare wake-up is insufficient.

## AppWorld server processes

- Servers run in the same allocation as the trainer; a separate Slurm job does
  not share `localhost`. The server environment (`.venv_appworld_server`) is
  separate from the GPU environment.
- Check process liveness and HTTP readiness (`GET /docs`), then exclusive task
  ownership of each port, acquired at `reset()` inside the rollout concurrency
  limit.
- Use bounded transport/5xx retries and quarantine unhealthy ports. Do not
  replay a possibly completed state-changing `/execute` after an ambiguous
  timeout; restart a fresh trajectory instead.
- If every port is missing, read per-server logs before raising readiness
  timeouts. `mkdir ... Disk quota exceeded` can mean per-user inode exhaustion
  with free bytes remaining; clean only paths owned by finished jobs.
- Repeated `/initialize Connection refused` after allocation teardown is
  downstream of the scheduler time limit, Ray shutdown, or memory; check those.
- A reflector prompt that overflows before generation is deterministic; retries
  cannot help. Budget tokenizer-measured input against model length minus the
  response budget. Silently shortening inputs changes the method.

## Ray and rollout throughput

Each job uses its own cache/temp roots and Ray temp directory, and cleanup
touches only that job's paths; never run a user-wide `ray stop` on a shared
node. Check benchmark imports in the selected GPU environment separately from
server startup. A shared rollout-engine death should fail the batch promptly
rather than consume per-environment retries.

For slow ACE candidate evaluation, separate the environment cap, the
candidate-only concurrency cap, and the wave size
`candidate_eval_batch_size * candidate_eval_rollouts_per_candidate`. Task
artifact names such as `stepNNNN` count local ACE steps, not optimizer updates.
Heartbeats and file growth do not prove learning.

## Metrics and learning signal

- `RUN_DIR/metrics.jsonl` is the local scalar journal. `training/global_step` is
  the optimizer step; `logging/index` and W&B's `_step` advance with logging. A
  resume can repeat optimizer steps, and both observations are kept.
- `train/actor_reward_sum`, `train/actor_reward_count`, and
  `train/actor_reward_invalid_count` describe initial actor outcomes before
  selection, augmentation, rejection, or padding. The epoch tracker consumes
  these even when a collection produces no optimizer update. Its completed
  epoch mean uses native reward sums/counts, not averages of logged batch means.
  Missing or invalid outcomes make that epoch ineligible for stopping. Use the
  checkpointed reward state and current-job live receipt for epoch patience;
  historical scalar journals cannot establish complete epoch coverage.
- Token reports use step-local totals under `rollout/total_*`, summed over every
  generation phase (policy, reflector, curator, candidate evaluation) including
  discarded samples. They are totals within one logged step, not run-to-date
  counters. `tokens_for_gradient_update` equals completion length in
  `verl_engine.py`; it is not a loss-mask measurement.
- When reading W&B history, use `scan_history` with run identity and step
  semantics; requesting keys together can drop rows that lack one of them.
  Compare matched steps and configurations, and keep throughput separate from
  learning signal.
- At ACE role transitions, raw `solve_none` counts can include masked inactive
  components; reconstruct active task groups before diagnosing collapse.
- Standard ACE's `auxiliary_update_gate=strict_improvement` changes training-row
  admission only. Actor rows stay eligible; complete reflector and curator
  groups enter when the selected playbook reward strictly exceeds its source
  seed under `best_rejection_update` (the pre-update task mean for other modes).
  Ties exclude auxiliary groups. Rollouts, playbook acceptance and updates,
  validation, and delayed methods keep their existing behavior. Admission
  metrics (`ace/*_training_rows`, `ace/*_training_groups`, and
  `ace/auxiliary_tasks_admitted` / `_dropped`) precede generic equal-reward
  filtering and padding. Generated-token totals still include excluded rows
  and candidate evaluations; actor eligibility does not imply nonzero advantage.
- Keep candidate selection, auxiliary credit, training rewards, and reported
  pass@k distinct. `best_rejection_update` keeps candidates whose evaluation
  reward is at least their own seed reward, then picks a maximum with random
  exact ties. With `rllm.ace.reflection_source_selection=min_reward`, singlemode
  ACE-RL reflects on the task's lowest-reward rollout, so with binary rewards
  the seed reward is 0 whenever any rollout fails and nearly every candidate
  passes the threshold. That high acceptance rate is intended; it is not
  comparable with `random`-source runs. Standard launchers default to `random`;
  preserve the recorded selection mode when continuing a run. With `policy_train_source=both`,
  every reward row needs its own correctly ordered task key; a stale shorter
  list silently turns grouped pass@k into a trajectory mean.
- With `ROLLOUT_QUANTIZATION=fp8`, `rollout_corr/*` compares the FP8 sampler with
  the BF16 actor over tokens with comparable log-probabilities only:
  `available_token_fraction` (share of action tokens corrected; the rest keep
  weight 1), `rollout_is_mean`/`_max`, `rollout_is_clipped_fraction` (ratio above
  the 2.0 cap), `log_prob_abs_diff_mean`/`_max`, and `kl`/`k3_kl` (sampler versus
  actor). A mean far from 1, a large clipped fraction, or a log-prob gap that
  grows after weight syncs points at the rollout weights, not at learning.
- Loss aggregation (`token-mean` and alternatives) is experimental semantics,
  not logging. Existing W&B history is not corrected by a later source change.
- Curator prompts must request a single JSON object with a flat `operations`
  array, string `content`, and `bullet_id` for UPDATE/REMOVE. The training
  playbook mutator does not normalize evaluation's `tip_id` alias. Classifying
  an insight in `reasoning` does not make that classification useful tip text.
  Check applied operations and before/after playbooks as well as candidate
  rewards: a high downstream reward can come from an unchanged playbook.
  Prompt edits change the recorded training identity; use a fresh run and a
  distinct evaluation output directory when comparing changed prompts.
- Separate first-visit policy reward from the gain over later visits when
  diagnosing ACE evaluation. A first-visit drop occurs before evaluation-time
  reflection and curation. Later-visit gains with unchanged empty playbooks can
  come from repeated policy attempts. Inspect action sequences and supplied
  context before attributing either result to adaptation. Positive scalar
  rewards alone do not establish positive GRPO advantages or gradient credit.
- Audit the actual reflector input before judging a turn budget. The generic
  `format_trajectory_as_conversation` used by MBPP and AppWorld
  emits each pre-action observation (at most 300 characters), then the action;
  it omits the final action's reply. A one-turn trajectory therefore contains
  no executed-action feedback in that history, although the outcome summary
  can retain a scalar reward. Mastermind's custom guess table retains final
  feedback. MiniSWE's maintained launchers use actions-only history, which
  retains the final reply with a 1,500-character per-reply limit; its direct
  Python evaluator needs `--reflector_simplified_history` for that path.
  Enough turns to produce a useful lesson and enough turns to finish a task
  are different claims. Check budget-hit rates, concrete feedback, applied
  playbook changes and later-visit reward before attributing gains to handoff.

## Qwen multi-turn token history

- AppWorld native agents store the exact rendered environment
  reply in the preceding step's `next_observation`, including terminal replies.
  `Step.to_dict()` and `Step.from_dict()` preserve it through serialization;
  older payloads without the field restore `None`. This also preserves
  MiniSWE's existing native reply for reflection after serialization.
- The shared Qwen completion parser processes `<tool_call>` blocks before
  benchmark agents receive actions, including AppWorld's code-only agent.
  Syntactically valid JSON without a tool `name` or `arguments` used to crash
  the rollout. Accept only an object with a nonempty string name and object
  arguments; skip malformed calls and retain later valid calls. Do not invent
  a name, coerce arguments, or execute text from a rejected block. The chat
  parser still removes tool blocks from action content and retains surrounding
  content; exact raw completion history remains unchanged. The same shared
  parser serves Qwen runs on all four benchmarks, so this is a runtime repair,
  not evidence against a model or turn budget.
- Use parsed `content` for environment actions and code execution, and keep the
  exact raw completion text and IDs for training history.
- The generation prompt can prefill `<think>\n`; completion IDs start after that
  prefix. Rebuild history from the prefill plus the exact raw completion: no
  `.strip()`, whitespace normalization, dropped or reinserted sentinels, or an
  extra terminator.
- Thinking-disabled Qwen history retains `raw_completion` even when
  `accumulate_reasoning=False`. Parsed action content strips whitespace and tool
  blocks; using it to rebuild history can change the next sampled token prefix
  and mask the entire trajectory from training. Thinking-enabled runs still
  honor an explicit choice to omit reasoning from subsequent prompts.
- Check each benchmark agent's message builder as well as the shared parser.
  A parser that supports raw history cannot preserve it if an agent drops
  `ModelOutput.text` when `accumulate_thinking=False`. Store raw structured
  output independently of that flag; let the parser choose whether to render
  reasoning. Test the actual agent-to-parser path with whitespace before the
  terminal token. Manually inserting `raw_completion` into a test message misses
  this failure. Do not recover old lossy trajectories by changing their masks.
  Chat API requests omit the local `raw_completion` and `reasoning` metadata
  while retaining other message fields and the same role/content values.
  Tokenizer-backed completion requests render the local Qwen history, so the
  preserved raw text can change later prompts when filtered content lost
  whitespace or tool blocks. These are distinct request paths.
- Check thinking settings for each role. MiniSWE's policy factory passes
  `disable_thinking` explicitly, but its local reflector/curator engine omits
  that argument and defaults to `False`. The Qwen parser therefore uses an open
  `<think>\n` generation prefix for those roles even with an Instruct checkpoint.
  A policy flag or model name does not prove auxiliary prompt formatting.
  Compare actual requests for each role with the successful parent's behavior
  before changing settings.
- Test `prompt_ids_t + completion_ids_t == prefix(prompt_ids_{t+1})` with the
  real tokenizer, including repeated thinking sentinels, terminal tokens,
  whitespace, and later turns. Decode mismatching spans to tell lost content
  from BPE boundary differences.
- A thinking-disabled run is an isolation control, not a substitute for the
  requested thinking run; small response limits add truncation that hides the
  mismatch.

## Evaluation

- Rerunning the same evaluation command resumes from per-task-visit receipts
  after checking the recorded configuration; in-flight trajectories and
  model-server RNG are not restored.
- When running several maintained evaluation commands serially in one Slurm
  job, assign a distinct explicit `VLLM_PORT` to each command. The shell
  launcher otherwise derives the same port from `SLURM_JOB_ID`, and its bind
  preflight can fail after the preceding server exits. An address-in-use error
  alone does not distinguish a remaining listener from TCP `TIME_WAIT`; inspect
  socket and process ownership before changing cleanup. Managed localhost ports
  are normalized in evaluation identity, so a diagnosed port-only retry can
  resume the same outputs. Verify completed resumes leave result, request and
  server-log hashes unchanged.
- Do not infer generation health from zero server abort counters alone;
  reconcile client retries with generated and finished tokens.
- `legacy_search_v1` results have different rollout accounting and must not be
  combined with `sequential_adaptation_v1` results.
- Evaluation launchers source their settings under `set -a`, which also exports
  shell functions. Apptainer passes the environment into MiniSWE containers,
  whose bash cannot import `eval_start_server` (it contains a heredoc), so every
  command reply and the graded `git diff` gained an `eval_start_server ... syntax
  error` line and the model tried to repair it. The launchers now unexport all
  functions after `set +a`; MiniSWE results from before that change carry this
  noise in their observations, patches and playbooks.
- Slurm task prologs and `BASH_ENV` can overwrite `TMPDIR` and Apptainer cache
  settings after allocation. Login and compute startup files can differ without
  a site-wide change. Record actual node file fingerprints and environment
  values before validating them against the registered versions. Restore
  registered settings after site initialization and through nested host Bash
  wrappers; verify that `--cleanenv` excludes the host startup hook from task
  containers. Before model startup, check the actual tempfile parent and owner,
  node-local disk capacity and inodes, write/fsync, allocated resources, and
  container isolation. A login-node rehearsal does not establish these checks.
- MiniSWE's editor tools edit files through a generated Python snippet inside
  the task container. Older task images run Python versions that ignore
  `PYTHONUTF8`, so those snippets now open files explicitly as UTF-8 with
  `surrogateescape`; before that, editing any file with non-ASCII text failed
  with an `'ascii' codec can't decode` error that looked like a model mistake.
- MiniSWE creates testbeds and overlays with Python `tempfile`; setting
  `MINISW_SCRATCH_ROOT` alone does not select that storage. For an allocated
  runtime gate, set `MINISW_PREFLIGHT_MANIFEST` to a JSON manifest and supply an
  explicit `MINISW_SCRATCH_ROOT`. The maintained launcher then exports `TMPDIR`
  from that root after site startup and runs the existing SIF verifier before
  starting the model server. Without this opt-in, launcher behavior is unchanged.
  The gate requires owned local-disk scratch, registered free bytes/inodes,
  matching prepared SIF metadata, and successful clean-container probes. It does
  not pull, build, replace, or delete images. It preserves `BASH_ENV`; register
  cache paths that agree with site startup instead of assuming an inherited
  cache value survives nested Bash wrappers.
  The manifest uses `schema_version: 1`, `scratch_gate` (`literal_path`,
  `required_free_bytes`, `required_free_inodes`, optional
  `allowed_local_disk_filesystems`), and `prepared_container_images` (each with
  `task_id`, absolute `.sif` `path`, and `size`; optional `mtime_ns` and
  `base_commit`). Optional `expected_environment`, `input_files` (`path` and
  `sha256`), and `probe_task_ids` bind paths, inputs, and representative images.
  The default probe is the first image; select each required repository
  explicitly when registering a mixed panel. For allocation blocks above 4KiB,
  also supply `concurrency`, `max_full_image_entries`, and `max_testbed_entries`
  in `scratch_gate`; their extra allocation is added to the byte floor. Set
  `MINISW_PREFLIGHT_RECEIPT` to select a receipt outside the evaluation directory;
  the default is its sibling `<output_dir>.runtime-preflight.json`. Keeping this
  receipt outside the output preserves the startup-only resume guard. A passing
  receipt proves readiness on that node at that time, not model execution or
  future free-space availability. The free-space check does not reserve storage.
  Jobs on the same filesystem can each pass while their combined scratch
  allowance exceeds capacity. Before colocating jobs, sum their remaining peak
  byte and inode allowances, account for other filesystem users, and retain a
  shared reserve. If that bound does not fit, use separate verified filesystems
  or run the jobs sequentially; separate scratch directory names do not isolate
  capacity. GPU execution remains a separate check.

A complete-looking evaluation without `evaluation_complete.json` is not accepted
as complete. Older Mastermind and MiniSWE shell launchers omitted
this step. Before recovering a receipt, hold the evaluation output lock, verify
the recorded configuration and exact task/sample coverage, and tie the metrics
to a successful evaluator job. Do not certify files being written by active
jobs or infer success from a metrics file alone. Preserve partial results and
resume them through the maintained evaluator when evidence is insufficient.
The shared `evaluation_runtime.write_completion` validates and atomically writes
the receipt; it does not by itself establish historical process success.

## Explicit feedback-preserving ACE training

`rllm.ace.handoff_protocol=feedback_v1` selects a fresh experimental input and
mutation contract. The default `legacy` keeps historical prompts and formatting.
The new protocol requires `rllm.agent.preserve_terminal_feedback=true`; missing
feedback configuration or formatter errors fail rather than silently falling
back. It needs a separate run identity and cannot resume a legacy controller.

AppWorld retains action/reply pairs, including the terminal reply,
with bounded head/tail excerpts. Its new curator assets use `bullet_id`.

AppWorld's feedback prompts include the original task in both roles and the
current playbook in the reflector. This lets each role check inherited advice
against the requested result, even when trajectory excerpts omit task details.
The curator receives cited insights, not the full trajectory. Training resolves
the instruction from the matching actor when environment metadata has only a
task ID. Pin the same final role assets in a new training run and its
evaluations; an evaluation-only prompt swap on older weights does not establish
learning with the revised prompts.

Training normalizes the legacy alias, records per-operation mutation outcomes,
and excludes unchanged proposals before candidate scoring. Candidate counts and
actual generated-token costs can therefore differ from legacy training.
MiniSWE retains only a bounded terminal verdict, submission/patch flags and
aggregate test counts; raw grading errors, diffs and test names are excluded.
`env_reward_reason` is the native verdict; `reward_reason` can be a shaped reward
tier. Correct/helpful-tip retention is a prompt constraint, not proof that the
model obeys it; inspect actual deletions and subsequent actor context.

`rllm.ace.log_role_credit=true` records role rows, group sizes, action-token
counts and positive/negative/nonzero advantages after rejection and truncation,
before transport padding. These are loss-eligible advantages, not measured
parameter gradients. Compare them with pre-filter auxiliary admission and keep
all generated work in `rollout/total_*`.

For a matched fresh ACE training pair, pass
`+rllm.ace.controller_seed=42 data.seed=42` to both maintained training
launchers. The first seeds Python, NumPy and torch CPU in the Ray trainer
process before construction; the second seeds VeRL's separate task sampler.
An environment variable named `SEED` or `PYTHONHASHSEED` does not establish
either control. Historical runs recorded `data.seed=null` and did not opt in
to controller initialization, so explicit seeding is a recorded difference
when comparing new runs with those parents. It does not guarantee identical
GPU execution or sampling streams after the arms take different control paths.
