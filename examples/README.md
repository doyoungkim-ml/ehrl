# Benchmark clients and experiments

Each maintained benchmark has one folder: `mbpp`, `appworld`, `mastermind`, or
`mini_swe_bench`.

| Location within a benchmark | Responsibility |
| --- | --- |
| Root helpers | Dataset preparation and environment setup |
| `train/<method>/` | `train.slurm`, `presets.sh` and the method binding (ACE-RL adds adapters, prompts and an initial playbook) |
| `eval/ace/`, `eval/vanilla/` | Evaluation `eval.slurm`, `presets.sh`, and Python entrypoints |
| `figures/output/` | Benchmark gallery written by `python plot.py <benchmark>` |

Training launchers accept model variants and method settings through configuration.
All methods default to eight policy rollouts or complete chains per task;
reflector and curator group sizes follow the selected ACE method.
Shared algorithms live in `rllm/algorithms`, backend setup in `rllm/trainer`, and
cluster-neutral shell setup in `scripts/training_runtime.sh`. Evaluation can share
the benchmark's harness prompt assets. No benchmark imports another benchmark's
controller.

See [experiments](../docs/experiments.md) for the model and preset matrix,
[evaluation](../docs/evaluation.md) for evaluation presets, and
[setup](../docs/setup.md) for environments and submission. Distinct sampling,
selection and training schedules remain explicit ablations; they share setup code
instead of duplicating complete launchers. For new algorithms, follow the
[extension guide](../docs/architecture.md).
