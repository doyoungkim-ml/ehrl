# Local ACE checkpoint continuation

## Status and scope

This implements controller checkpoint preservation. CPU verification and source
deployment do not by themselves establish distributed GPU continuation.
AppWorld's standard `train/ace_rl/train.slurm` entrypoint resolves
to `rllm.algorithms.ace.trainer.LocalACERLAgentPPOTrainer`.
Its iterative and delayed-joint subclasses inherit the checkpoint hooks.

The local implementation needs the tested VeRL dataloader-resume patch. It does
**not** silently work around an unpatched VeRL installation by reloading data.pt
or modifying site-packages. A resumed run fails with a clear compatibility error
if VeRL did not restore dataloader state and expose iterator-finished metadata.
Fresh training remains allowed. Do not deploy the rLLM changes alone without a
verified prepared dependency. Ordinary restarts use the same `PRESET` command
and run directory; see [artifact locations](artifacts.md).

## Preserved state

Each `global_step_<step>/` contains VeRL's normal `actor/`, optional critic state,
and `data.pt`, plus a versioned UTF-8 `ace_state.json` sidecar containing:

- Per-task training playbooks: full content, explicit flat/section mode, next
  bullet ID (including IDs no longer present in the text).
- Task training visit counts, preserving the third/sixth/etc.-visit flush schedule.
- Initial/global playbook, flush configuration, last global flush step.
- Iterative component schedule configuration and position, when applicable.
- Auxiliary gradient-admission objective (`auxiliary_update_gate`).
- Actual zero-based data epoch, independent of optimizer steps skipped through
  rejection sampling.
- Python, NumPy, and torch CPU controller RNG state.

Evaluation caches are transient. Validation is isolated from training controller
state, including on exceptions. Iterator construction is also isolated from
controller RNG: torchdata draws a worker base seed when rebuilding an iterator,
which otherwise changes subsequent CPU random decisions on resume.

Fresh ACE runs can set `+rllm.ace.controller_seed=42`. The shared ACE Ray
entrypoint seeds Python, NumPy and the torch CPU generator before constructing
the trainer. The default is `null` and leaves existing RNG state unchanged.
Accepted seeds are integers from 0 through `2**32 - 1`; booleans, other types
and out-of-range values fail before backend construction. This does not seed
CUDA or change worker determinism. Set the separate existing `data.seed=42`
override to seed the training-task sampler.

`controller_seed` is an initialization setting recorded in `run.json`. A
missing historical field means `null`; changing an explicit seed fails the
maintained run-identity guard. No ACE sidecar field is needed: checkpoint
restoration replaces the Python, NumPy and torch CPU states after construction,
so saved controller RNG remains authoritative. An initialization seed cannot
reconstruct RNG state missing from an old checkpoint.

The ACE sidecar is atomically replaced before calling the parent's checkpoint
writer, so its latest marker cannot be advanced after a failed sidecar write.
The sidecar does not change VeRL's actor/optimizer serialization or introduce a
cross-file transactional checkpoint protocol.

All maintained agent and workflow trainers also save `reward_epoch_state.json`
before the parent publishes the checkpoint marker. It preserves the current
loop epoch, partial initial-actor reward sums/counts and coverage, completed
epoch records, historical best, and consecutive-below-best count. This separate
monitoring state does not change the optimization objective or controller RNG.
Ordinary agent GRPO/PPO also recover their loop epoch from this sidecar, rather
than inferring it from optimizer steps that can skip rejected collections.

The trainer restores reward state only from the checkpoint actually loaded and
replaces the current job's live receipt on startup. Mid-epoch and empty saved
epoch-tail resumes retain their exact partial aggregate; a finished iterator
advances once. Genuine dataloader exhaustion finalizes an epoch. Training limits
and exceptions inside an epoch do not finalize partial data. A malformed present
sidecar fails resume. A missing legacy sidecar warns, starts a new monitoring
baseline, and excludes the resumed partial epoch; it cannot recover historical
epoch averages. Monitoring changes neither the historical checkpoint's weights
nor its original controller-state coverage.

## Dataloader and epoch boundaries

VeRL must restore raw `data.pt` even at optimizer-step modulo boundaries.

- Mid-epoch: continue with the remaining saved batches.
- Last batch saved before `StopIteration`: re-enter the saved epoch, let its empty
  tail finish, then start the exact next shuffled epoch.
- Saved after `StopIteration`: torchdata starts the next epoch automatically;
  advance the epoch label once, without draining or discarding a batch.

The training loop records `_training_epoch`; task epoch labels use it rather than
assuming consumed batches equal optimizer steps. VeRL's serialized loader state
retains the within-epoch position. Exact continuation assumes unchanged dataset,
sampler, batch size, worker configuration and compatible torch/torchdata versions.

## Historical and incompatible checkpoints

Missing ACE sidecar: restore weights/dataloader, warn explicitly, and cold-start
ACE playbooks/visit counts. These older checkpoints never saved that information;
this change cannot reconstruct it. Legacy data-epoch estimation from optimizer
steps is not exact when batches were rejected. This fallback applies only to
the historical `auxiliary_update_gate=none` objective. Strict-improvement
training rejects a missing sidecar, since it cannot establish the saved objective.

The standard launcher now sets `rllm.ace.auxiliary_update_gate=strict_improvement`.
Actor groups stay eligible; complete reflector/curator groups enter the gradient
batch only after strict measured playbook improvement. This does not change
rollout or playbook-update rules. Delayed ACE-RL remains `none`. The sidecar
records the effective gate and requires it to match before restoring controller
state; old sidecars without this optional field mean `none`. `run.json` applies
the same normalization, so explicit `none` can resume a historical run, but the
new objective cannot overwrite that run.

An objective change needs a separate weight-only run initialized from a merged
checkpoint, with a fresh optimizer, scheduler, dataloader and ACE controller.
It is not exact continuation or a from-scratch baseline. Record its source
checkpoint and changed configuration. Full-state migration between different
auxiliary objectives is not supported. Ordinary same-objective continuation
preserves the existing state and run identity.

Malformed JSON, unknown schema, a missing actual data epoch in a present sidecar,
mismatched checkpoint step, invalid playbook IDs, corrupt RNG state, or
incompatible flush/component configuration fail before controller state is
mutated. Epoch-less in-memory snapshots before training are allowed; disk
checkpoints require the actual epoch. The saved initial template is authoritative.
Remote HDFS sidecars are unsupported; local/shared filesystem checkpoints are the
supported path. Files named `data.pt` remain trusted VeRL checkpoints; its existing
`torch.load(..., weights_only=False)` behavior is unchanged.

## Workflow GRPO continuation

The workflow trainer restores unchanged rollout weights and cache before a
rejection-refill wave. The pinned checkpoint manager is called without a new
step version: this is not an optimizer update. The SDK refill path uses the
same restoration behavior; the existing agent PPO lifecycle remains unchanged.

Workflow updates are zero-based. Fresh training starts at update 0; a checkpoint
saved after update N resumes at update N+1, including checkpoint 0. The workflow
loads its validated `epoch_state.json` from the checkpoint actually restored and
uses the pinned VeRL iterator-finished metadata to distinguish an empty saved
epoch tail from the next epoch. It writes the epoch sidecar atomically before
the parent can publish the checkpoint marker. Missing or incompatible epoch or
loader metadata prevents a workflow resume rather than guessing from update
counts, which can diverge from consumed batches under rejection.

Historical workflow checkpoints preserve actor/optimizer state, actor-worker
RNG and loader position, but do not save driver Python/NumPy/torch RNG. They do
not provide bit-identical controller continuation; unsaved updates and retained
refill groups are recomputed. The workflow resume and rejected-batch regression
suites exercise the actual loop and pinned checkpoint method bodies with real
CPU torchdata loaders. Distributed GPU recovery still requires target-runtime
verification.

## Sustainable VeRL patch, no broad upgrade

The exact backport and its test matrix are now repository-owned in
`compat/verl/`. `compat/verl/manifest.json` pins the runtime base to
`bcb638649a50e58494a8ddd92085ad1174f674b8` and records the patch/file checksums.
The upstream source candidate was based on
`23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3`; that broader revision is not the
supported runtime pin.

Follow [VeRL compatibility](verl-compatibility.md) to prepare an exact source
checkout, verify it, and run the tests. Both Qwen environment installers now
require this checked source and verify protected installed-file hashes. This
makes new installations reproducible; it does not deploy into an existing
shared environment. Verify that target's installed protected hashes. Historical
machine-specific candidate locations belong in the locally ignored context.

[PR 5725](https://github.com/verl-project/verl/pull/5725) introduced the boundary
skip; it is not full sampler-state preservation.
[PR 7402](https://github.com/verl-project/verl/pull/7402) concerns SFT and is not the
required PPO fix. Read-only upstream searches found no equivalent open PPO fix;
recheck before publication. VeRL's contribution policy requires a human submitter
to review each change and run the tests. No PR has been published.

## Reproduce local verification

The CPU environment uses Python 3.11, torch 2.10.0+cpu and torchdata 0.11.0+cpu.
From this repository, in that compatible CPU environment:

```bash
python scripts/run_cpu_regressions.py --verl-source /path/to/prepared/verl
```

For the focused controller subset:

```bash
VERL_RESUME_SOURCE=/path/to/prepared/verl python -m pytest \
  --confcutdir=tests/trainer \
  tests/trainer/test_local_ace_checkpoint_resume.py \
  tests/trainer/test_local_ace_checkpoint_hooks.py \
  tests/trainer/test_agent_ppo_resume_loop.py \
  tests/trainer/test_local_ace_resume_integration.py -q
```

The `--confcutdir` flag avoids the existing broad conftest's fake optional-module
injections (for example a fake pandas breaks real torch.optim imports).
The integration tests require `VERL_RESUME_SOURCE`; without it they explicitly
skip, not silently pass. They execute the actual save/load method bodies with
real torchdata, Playbooks, CPU model updates and SGD optimizer state, replacing
only distributed workers. The separate loop tests isolate the actual production
epoch control flow from PPO computation.

From either VeRL checkout:

```bash
python -m pytest \
  tests/trainer/ppo/test_dataloader_resume_on_cpu.py -q
```

This covers single/multiprocess loading, persistent workers, stale snapshots,
drop-last choices, initial/mid/last/finished positions, subsequent sample order,
and sampler/dataset state.

## Verification limits

The full project suite does not collect in the lightweight CPU environment:
optional training dependencies are absent and the broad `tests/conftest.py`
substitutes several modules. Use the CPU gate in [development](development.md);
do not report a full-suite pass. CPU results do not establish distributed GPU
continuation.

## Kinds of continuation

Weight-only, full optimizer/scheduler/RNG, and full controller/data continuation
are different claims; state which one a run achieved. When splicing a resumed
W&B curve, splice at the checkpoint actually loaded: logged updates after the
last save are recomputed. An intentional fork of a run needs its own `RUN_DIR`
while its source checkpoint stays untouched. World-size shard-name errors and
optimizer parameter-group/topology mismatches are different failures, and
neither is a reason to delete valid progress.

## Experience input protocol compatibility

ACE state records `handoff_protocol` (`legacy` or `feedback_v1`) as well as the
auxiliary admission gate. Old sidecars without the protocol mean `legacy`.
A changed protocol fails full-state restoration, including when the legacy
missing-sidecar adapter would otherwise cold-start the controller. Use a
separate weight-only run with a fresh optimizer, scheduler, dataloader and
controller for an input/protocol ablation. Compare two such fresh runs when
isolating a repair from the restart itself.

The run identity fingerprints the prompt assets actually selected by the
protocol. Optional unused handoff assets do not alter legacy fingerprints.
Missing `preserve_terminal_feedback=false`, `handoff_protocol=legacy` and
`log_role_credit=false` fields normalize to their historical defaults.
