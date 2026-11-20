# AppWorld

Use the checked-in method entrypoints from the repository root:

```bash
PRESET=oneeval_rollout8 MODEL_VARIANT=thinking \
  sbatch examples/appworld/train/ace_rl/train.slurm
PRESET=delayed_oneeval_rollout8 MODEL_VARIANT=thinking \
  sbatch examples/appworld/train/ace_rl/train.slurm
PRESET=grpo MODEL_VARIANT=nonthinking \
  sbatch examples/appworld/train/grpo/train.slurm
```

Training requests two H200 GPUs, 32 CPUs, and 500 GB. Supply the applicable
account and other scheduler settings as ordinary `sbatch` flags. On first use the
launchers create the GPU environment, the AppWorld server environment and the
AppWorld data; see [training](train/README.md).

The same command resumes the latest committed checkpoint in
`train/<method>/runs/<preset>/<model>/`, for example
`train/ace_rl/runs/oneeval_rollout8/qwen3-4b-thinking/`. Checkpoints, W&B files,
logs, episodes, and task artifacts remain under that run. W&B logs to `WANDB_ENTITY` (default: your account's entity).
Set `RUN_DIR` explicitly for a separate experiment. `MODEL_VARIANT` also accepts
`nonthinking`, `instruct`, and `thinking_2507`.

Evaluate an exact training checkpoint and plot its persisted results:

```bash
RUN_DIR=/absolute/path/to/training/run STEP=215 sbatch examples/appworld/eval/ace/eval.slurm
python plot.py --run /absolute/path/to/training/run --preset eval
```

Use `eval/vanilla/eval.slurm` for vanilla evaluation. ACE runs the `visits8`
preset and vanilla the `pass8` preset, both with a 10-turn policy budget. Results
go under `RUN_DIR/eval/ace/visits8/step_<N>/` and
`RUN_DIR/eval/vanilla/pass8/step_<N>/`; rerun the same command to resume.
Evaluation presets have their own
budgets; see [evaluation](../../docs/evaluation.md). Plot presets are listed by
`python plot.py --list`.

`RLLM_DRY_RUN=1 bash .../train.slurm` and `bash .../eval.slurm --dry-run`
show resolved commands without creating outputs or starting services.
