# Experiment files

Every training command resolves one directory,
`examples/<benchmark>/train/<method>/runs/<preset>/<model>/` (`RUN_DIR`). Set
`RUN_DIR` only to choose a deliberately separate experiment; changing
`EXPERIMENT_NAME` does not create another checkpoint root.

| Contents | Location |
|---|---|
| Full resumable checkpoints | `RUN_DIR/checkpoints/global_step_<N>/` |
| Latest committed checkpoint | `RUN_DIR/checkpoints/latest_checkpointed_iteration.txt` |
| Configuration, model, thinking mode, W&B identity | `RUN_DIR/run.json` |
| The run's dataset files and their hashes | `RUN_DIR/data/` |
| Scalar training history | `RUN_DIR/metrics.jsonl` |
| Current job's completed-epoch reward state | `RUN_DIR/training_reward_state.json` |
| Resumable epoch reward aggregates and history | `RUN_DIR/checkpoints/global_step_<N>/reward_epoch_state.json` |
| Local W&B files | `RUN_DIR/wandb/` |
| Logs, episodes, task evidence | `RUN_DIR/logs/`, `episodes/`, `task_artifacts/` |
| Inference weights merged for evaluation | `RUN_DIR/checkpoints_merged/global_step_<N>/` |
| Checkpoint evaluation | `RUN_DIR/eval/ace/visits8/step_<N>/`, `RUN_DIR/eval/vanilla/pass8/step_<N>/` |
| Run figures | `RUN_DIR/figures/<preset>/` |
| Pretrained-model evaluation | `examples/<benchmark>/eval/<method>/results/<preset>/<model>/` |
| Benchmark gallery (`python plot.py <benchmark>`) | `examples/<benchmark>/figures/output/<model>/` |
| Environments | `.venv-<gpu-family>/`, `.venv_appworld_server/` |
| Reusable caches (VeRL source, uv, models, images, locks) | `.cache/` (`RLLM_CACHE_DIR`) |
| Job-owned Ray/container scratch | node-local temporary storage, removed with the job |

Do not create a top-level `archive/`. Keep requested reports and their source
tables beside the figures in the documented output directory. Keep temporary checks in temporary storage and durable lessons in `docs/`.
Before removing old files, check open files, symbolic links, Python `.pth`
references, and Git `.git`/`commondir` files. Git worktrees can depend on metadata
outside their source directory. An unused launcher can still contain packages
used by another environment. After relocation, verify the source pin and run
the affected checks before removing any remaining dependency.

The same training command resumes the checkpoint named by the marker. Missing,
invalid or incomplete saved state raises an error instead of starting over. An
explicit resume source must be a full training checkpoint; merged inference
weights contain no optimizer, RNG, dataloader or ACE controller state.

`run.json` records the W&B entity, project and run ID on the first job, and
restarts reuse them. Optimizer steps are logged as `training/global_step`; W&B's
own step advances with logging, so updates replayed after an unsaved interruption
remain visible.

Epoch reward summaries use the dedicated reward state, not reconstructed
averages from `metrics.jsonl`: older journals omit rejected collections and can
contain replayed updates. The reward sidecar rolls back with the checkpoint.

Evaluations resume by default: rerun the same command and completed task visits
are reused while unfinished ones run again. See [evaluation](evaluation.md).

Relocate a run only after its jobs have stopped, and move the whole directory.
Never overwrite or combine conflicting checkpoints.
