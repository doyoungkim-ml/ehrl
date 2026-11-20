# Development and maintenance

This fork carries two kinds of changes: algorithm implementations and repairs to
its execution stack. Keep them independently reviewable. A backend update must
preserve the repair behavior or replace it with an equivalent upstream fix.
Folder cleanup and version numbers are not evidence that a workaround is obsolete.

## History behind the protected behavior

Each row names a repair this code carries, the failure that caused it, and what
a later change must keep. The compatibility manifest contains the exact VeRL
backport provenance and checksums.

| Why it exists | What a future change must preserve |
| --- | --- |
| Qwen 3.5 runtime adaptation and exact VeRL pin | Supported worker APIs, IPC/lifecycle and training-output repairs; use the manifest pin |
| Qwen thinking history lost exact generated text and token boundaries | Raw completion including whitespace/repeated sentinels; filtered action text stays separate |
| Different dynamic microbatch counts can deadlock distributed collectives | Correct group synchronization and already-imported actor aliases |
| Workflow steps lacked per-step token/model-output information | Actual generated token IDs and per-step model outputs reach training |
| Loss aggregation and metric names drifted across paths; token work was hard to account for | Selected loss aggregation and step-local rollout totals, including discarded generation |
| ACE rollout-source expansion and task grouping disagreed | Policy/auxiliary row alignment, per-task pass@k grouping, selected-candidate provenance |
| Unscoped Ray cleanup could stop another job | Job-scoped cleanup and unchanged launcher identity/resources |
| AppWorld request logging failed on non-JSON values | Request serialization compatibility; retain benchmark-specific repairs |
| Special chat tokens leaked into environment actions | Use filtered content for actions while preserving raw history for training |
| Fatal Ray/vLLM failures were obscured by retries or missing evidence | Fatal-error propagation and preserved diagnostic logs |
| Checkpoints omitted ACE controller state, raw loader boundaries were lost, rejected batches could leave rollout weights asleep | Atomic controller sidecars; actual epoch and RNG continuation; unchanged-weight recovery on skipped updates |
| Candidate selection lost seed provenance or assigned wrong auxiliary credit | Each candidate compared with its own seed, maximum eligible selection, random exact ties, strategy-specific credit |

The `rollout/total_*` metrics are totals within a logged training step, not
run-to-date counters. Count all generation phases before filtering. Completion
length is not proof of loss-masked optimizer participation; the legacy gradient
count metric's name does not change that limitation.

## Updating rLLM or VeRL

1. Keep the algorithm and benchmark changes separate from the proposed backend
   update. Review original patch diffs and their current callers, including
   worker-side imports. Do not remove `experimental` code merely by its name.
2. Select an exact VeRL commit. Inspect each cataloged repair against that source;
   rebase only the fixes still needed. Update the patch, file hashes, modes,
   provenance, and tests together. The verifier intentionally rejects unrelated
   edits rather than overwriting another backport.
3. Run the CPU gates from a dedicated environment and inspect the reports.
   Remove a patch only after its regression passes with equivalent upstream
   behavior. Keep algorithm ablations outside global compatibility patches.
4. Build the wheel and check it away from the source tree. Import paths that work
   only because the checkout supplies `examples` are not package boundaries.
5. For a separately authorized runtime deployment, verify actual imported modules
   and installed hashes in driver and workers. Run the supported distributed
   smoke tests: at least rollout → update → rollout, rejected batch → rollout,
   and save/resume at a loader boundary with controller state. Preserve topology,
   hardware, thinking mode, sampling and data choices while testing infrastructure.

Installed protected-file hashes prove those files match the repair. They do not
prove that the entire environment, worker processes, optimizer topology, or GPU
execution is compatible. The source-only sync and checkpoint deployment rules
still apply.

## Repeatable local CPU gate

Create a dedicated Python 3.11 environment, then install:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r tests/requirements/cpu.txt
```

Prepare a VeRL checkout using [the compatibility instructions](verl-compatibility.md),
then run:

```bash
python scripts/run_cpu_regressions.py \
  --verl-source /path/to/prepared/verl \
  --report-dir /path/to/reports
```

The runner launches separate pytest processes for compatibility, foundation,
context, controllers, and upstream dataloader tests. It supplies
`VERL_RESUME_SOURCE`, puts that VeRL source on `PYTHONPATH` for the controller
suite, disables ambient pytest plugins/addopts, uses explicit
`--confcutdir` boundaries, writes JUnit plus `summary.json`, and fails on skips,
empty collections, or errors. In particular, it never loads the broad
`tests/conftest.py`, whose optional-module stand-ins invalidate real optimizer
tests. The selected CPU suite is explicit in `scripts/run_cpu_regressions.py`.

Compatibility tests cover the patch manager, real IPC transformations, installed
file verification, and runtime-patch routing with backend doubles. Controller
cases execute real production blocks with real Playbooks, torchdata and CPU
optimization where required; they replace distributed workers. Foundation tests
exercise a non-ACE trainer through the shared setup and the actual Qwen parser
with a tiny offline text tokenizer. They do not validate a real model tokenizer.

Full trainer-import tests such as `test_delayed_joint_selection.py`,
`test_train_pass_at_k_grouping.py`, and other integration suites remain in the
repository and require the complete runtime. CPU selection coverage instead
executes the actual producer/selection/credit blocks across all four ACE
controller variants. The inherited broad test suite also has stale upstream
interfaces, including the removed policy-loss override monkey patch; this
refactor does not claim that suite passes or silently skip it into success.

Build and verify the package with:

```bash
python -m pip install build
python -m build --wheel
python scripts/check_foundation_wheel.py dist/*.whl
```

Use a clean output directory when multiple package versions have been built.
CI runs these gates for pull requests; a locally passing run is not a claim that
GitHub Actions or distributed training has executed.

## Repository layout rules

The layout is deliberately small so a fork can run experiments directly:

- Entry points are `examples/<benchmark>/train/<method>/train.slurm`,
  `examples/<benchmark>/eval/<ace|vanilla>/eval.slurm`, and `plot.py`. Presets
  (`presets.sh`) select variants; do not add wrapper scripts, shortcut launchers,
  per-cluster copies, or generated submission files.
- Run outputs live inside their run directory (see [artifacts](artifacts.md));
  nothing else writes into the tracked tree. Machine-specific paths belong in
  submit-time environment variables, never in tracked files.
- Documentation is the files in `docs/` plus `README.md` and the benchmark
  READMEs. Update the narrowest existing page; do not add appendices, dated
  status notes, incident logs, or history narration. Git history is the record.
- `tests/test_repository_layout.py` enforces the allowed top-level entries,
  documentation pages, and launcher hygiene. Changing its allowlists is a
  deliberate decision for the maintainer, not a way to make a check pass.

## Where durable knowledge lives

Durable technical knowledge (verified failure patterns, metric semantics,
protocol definitions) lives in the docs page for its topic, most often
[troubleshooting](troubleshooting.md). When a change verifies a new lesson,
update that page in the same change and remove guidance the lesson contradicts.
