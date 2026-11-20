# VeRL compatibility contract

The maintained PPO backend uses VeRL commit
`bcb638649a50e58494a8ddd92085ad1174f674b8` with a narrow dataloader-resume
backport. The portable contract is [`compat/verl/manifest.json`](../compat/verl/manifest.json):
one exact base revision, one patch checksum, and before/after checksums for every
changed file, including its executable mode. The patch includes its upstream-facing
CPU regression tests.
Algorithm code should depend on the backend's documented behavior, not edit an
installed dependency or hide a missing fix with a second loader restoration.

This contract covers source compatibility for checkpoint continuation. It is
neither a complete environment lock nor evidence of cluster deployment. The
existing setup scripts pin the broader Qwen runtime; retain their hardware and
library choices until a separately validated upgrade. The local continuation
deployment hold in [ace-checkpoint-resume.md](ace-checkpoint-resume.md) still applies.

## Inspect or prepare a source checkout

Use a dedicated VeRL Git checkout at the manifest's exact revision. From this
repository, verification is offline, uses only Python's standard library and
Git, and imports no training packages:

```bash
python scripts/verify_verl_compat.py --source /path/to/verl-source
```

The command prints JSON and changes no source files. Exit status is `0` for the
exact applied patch, `1` for clean unpatched source, and `2` for a wrong pin,
checksum mismatch, partial patch, unrelated changes, or an invalid source root.
Both forward and reverse applicability are checked as appropriate. Verification
also checks every patched file's full content, rather than trusting markers.

To prepare that dedicated checkout explicitly:

```bash
python scripts/verify_verl_compat.py --source /path/to/verl-source --apply
```

Application requires an unchanged index and either clean exact-pin source or the
exact already-applied patch. It changes only the inventoried source/test files,
leaves the index untouched, and is idempotent. Installed `site-packages` and
`dist-packages` trees, symlink patch targets, path escapes, and source drift are
rejected. Ignored build artifacts are not part of Git's cleanliness check.
There is no network access, package installation, remote write, job submission,
or forced reset. A source checkout already carrying IPC or other backports must
be reviewed as a new patch set; this tool deliberately rejects extra edits and
never removes them.

`scripts/setup_qwen35.sh` (run automatically by the launchers when an
environment is missing) installs VeRL only from a verified patched checkout.
Without `RLLM_VERL_SOURCE`, it fetches the manifest's exact revision into
`.cache/verl/<revision>/` and applies the patch with the command above. To use a
checkout prepared elsewhere:

```bash
export RLLM_VERL_SOURCE=/path/to/verl-source
python scripts/verify_verl_compat.py --source "$RLLM_VERL_SOURCE"
```

Setup runs this check before changing the environment, so a bare upstream pin
cannot silently omit the resume fix, and verifies the installed files afterwards.
The manifest owns the VeRL revision.

## Verify installed protected files

After installing the checked source, use the selected environment's interpreter:

```bash
/path/to/environment/bin/python scripts/verify_verl_compat.py --installed
```

This read-only mode resolves the package using Python's import path without
importing VeRL. It checks the manifest's production-file hashes and reports the
selected package directory. Tests are not expected inside the installed wheel.
A mismatch, missing file, or ambiguous package fails. `--apply` is never allowed
with `--installed`.

Both Qwen setup scripts force `--reinstall-package verl` so a cached local wheel
cannot retain unpatched Python files, then run this installed check. This verifies
the protected files only, not a complete environment lock or GPU execution.
Driver and worker import/hash verification remains part of runtime deployment.

## Regression checks and provenance

The vendored artifact is the full `runtime-bcb638.patch` exported on 2026-09-05,
including the test file and its Apache-2.0 header. It was narrowly backported from
a source candidate based on `23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3`.
Its historical result was 338 CPU cases passing with Python 3.11,
torch 2.10.0+cpu, and torchdata 0.11.0+cpu. Those versions describe the tested
state envelope; do not silently assume another torchdata version serializes the
same schema. The manifest retains portable provenance and hashes instead of
requiring the original author's local evidence directories.

Guard the patch workflow and retained dynamic-batch/output routing without
installing training dependencies:

```bash
python -m unittest discover -s tests/compat -v
```

The runtime-patch tests execute the production functions with lightweight backend
doubles. They cover explicit/default distributed groups, alias updates,
idempotence, inference passthrough, and training loss/metric preservation; they
do not exercise distributed collectives or GPU tensor reconstruction.

After applying the patch, run the actual VeRL tests with the compatible CPU
interpreter, from the VeRL source checkout:

```bash
python -m pytest tests/trainer/ppo/test_dataloader_resume_on_cpu.py -q
```

They execute the actual checkpoint method bodies with real torchdata and cover
single/multiprocess workers, persistent workers, stale snapshots, sampler and
dataset state, epoch-boundary order, and invalid-state rejection. They isolate
distributed imports; they do not validate GPU training. Run the project's
controller/optimizer integration suite with `VERL_RESUME_SOURCE` pointing at this
checkout and the documented `--confcutdir=tests/trainer` boundary; see
[ace-checkpoint-resume.md](ace-checkpoint-resume.md). A skipped integration test
is missing coverage.

The generic VeRL epoch helper assumes one consumed batch per PPO step. Algorithms
that reject batches without optimizer steps must save their actual epoch and use
the restored iterator metadata, as the project's controller continuation does.
Do not substitute the optimizer-step modulo for the algorithm's consumed-data
position.

## Other carried fixes

The manifest catalogs existing compatibility work separately from the applied
resume patch. These implementations remain in place; the tool does not claim to
validate or apply them.

| Fix | Location and reason | Upgrade/removal gate |
| --- | --- | --- |
| IPC and lifecycle | `scripts/patch_verl_zmq_ipc.py` generates per-update job/rank/process socket handles and routes HYBRID sleep/wake through EngineCore. Existing launchers still apply this patch in the selected environment; all three source files are now validated before writing. | Review equivalent behavior at the next pin, preserve concurrent weight-sync isolation, and test repeated sleep/wake on the supported GPU topology. |
| Dynamic microbatch synchronization | `rllm/experimental/verl/patch.py:patch_verl_dynamic_batch_sync` supplies a distributed group and patches the imported actor reference to prevent unequal microbatch collective counts. | Test the actual data-parallel topology; defaulting to WORLD is not automatically correct for every future topology. |
| Training output reconstruction | `rllm/experimental/verl/patch.py:patch_verl_training_output_postprocess` skips unused nested training outputs while retaining metrics/losses. | Test training metrics and inference/log-prob outputs against the next backend before removing it. |

The first three fixes have code/history provenance in the manifest but no fresh
distributed validation from this cataloging work. The dynamic-batch and training
output patches live under `experimental` but are also called by maintained PPO
entrypoints; directory naming alone is not a removal criterion. The IPC patcher
still edits the selected environment when an existing launcher calls it. Its transformed output was compared byte-for-byte with the legacy
patcher on disposable copies of the exact pinned VeRL source. Hermetic tests
cover its preparation failures and import resolution; they do not establish
runtime lifecycle correctness. Moving it into a versioned source patch should
preserve the supported runtime's actual diff and lifecycle tests.

### IPC patcher safeguards

The no-argument `scripts/patch_verl_zmq_ipc.py` invocation retains the existing
launcher's apply behavior. It resolves the VeRL package selected by the
interpreter's import path with `find_spec("verl")`, without importing VeRL.
It no longer chooses the first directory returned by `site.getsitepackages()`.
A dedicated source checkout can be selected explicitly; inspect it without writes:

```bash
python scripts/patch_verl_zmq_ipc.py --check --source /path/to/verl-source
```

Omitting `--source` checks the selected interpreter's package. Exit status is `0`
for valid patched source, `1` if changes are needed, and `2` on validation errors.
Omitting `--check` applies the validated plan. Every file is read first, every
expected old/new occurrence and marker is checked, and every resulting Python
module is compiled before any write. Mixed worker-extension states or partial
steps within a transform fail; complete earlier IPC patches can receive the
later lifecycle backport. Markers alone cannot establish success.

Application checks for changed source snapshots before writes and preserves file
permission bits through atomic file replacement. This is not a multi-file
transaction: an operating-system write failure can leave complete patched files
alongside unchanged ones. Inspect again before retrying. Pattern checks cover
these transformations, not arbitrary dependency drift; the strict checksum
contract of the separate resume verifier is unchanged.

## FP8 rollout and rollout correction

`ROLLOUT_QUANTIZATION=fp8` (see [experiments](experiments.md)) uses unpatched
features of the pinned VeRL; the compatibility patch is unchanged:

| Behaviour | Pinned VeRL location |
|---|---|
| `rollout.quantization=fp8` injects dynamic e4m3 128x128 block `quantization_config` (with an MoE `mlp.gate` ignore list) and VeRL's vLLM FP8 patches | `workers/rollout/vllm_rollout/vllm_async_server.py` `_apply_quantization` |
| Each weight sync quantizes the BF16 actor weights with `scaled_fp8_blockwise` (Triton kernel on GPU) before `load_weights` | `workers/rollout/vllm_rollout/utils.py` `_update_weights` -> `utils/vllm/vllm_fp8_utils.py` `load_quanted_weights` |
| Sampler log-probabilities follow temperature scaling (`logprobs_mode=processed_logprobs`) | `workers/config/rollout.py`, vLLM `v1/sample` |
| Truncated IS weights and mismatch metrics | `trainer/ppo/rollout_corr_helper.py` `compute_rollout_correction_and_rejection_mask` |
| The actor multiplies its per-token policy loss by `rollout_is_weights` | `workers/utils/losses.py` `ppo_loss` (selects the field) -> `trainer/ppo/core_algos.py` `compute_policy_loss_vanilla` |

VeRL's own trainer computes the weights in `RayPPOTrainer.fit`; the rLLM
trainers (`AgentPPOTrainer`, `AgentWorkflowPPOTrainer`) do not use that loop, so
`rllm/trainer/verl/rollout_correction.py` calls the helper after recomputing
`old_log_probs`, over tokens whose sampled context is exactly the training
context (the per-token `rollout_log_prob_mask`). `scripts/patch_verl_zmq_ipc.py`
edits the same weight-sync module but leaves the FP8 branch intact; the installed
patched `update_weights_from_ipc` still routes FP8 models to
`load_quanted_weights` and skips the generic post-load transform. VeRL does not
claim FP8 rollout for vLLM 0.19.1. The Triton kernel's float32-to-e4m3 cast can be lowered through float16 (expected below compute capability 9.0), so its output can depend on the GPU generation (see [evaluation](evaluation.md#fp8-evaluation)); repeated FP8 weight sync with sleep/wake on
H100/H200 is unverified until the GPU smoke test passes, and must be retested at
the next VeRL or vLLM pin.

## Keeping future forks maintainable

For each backend upgrade, select one exact candidate commit and compare the
carried fixes against its implementation. Remove a backport only after its
regression test passes against equivalent upstream behavior. Re-export remaining
patches and update their inventory/checksums together with provenance. Re-run the
CPU continuation and algorithm tests, then the supported distributed smoke tests
before declaring that runtime supported. Source hashes and CPU results alone
must not become a claim of numerical equivalence or cluster readiness.

Keep dependency repairs in this compatibility catalog, reusable algorithm logic
in the algorithm package, and benchmark choices in examples/configuration. A new
algorithm needing different optimization semantics should implement that choice
at its extension boundary; it should not silently add another global VeRL patch.
