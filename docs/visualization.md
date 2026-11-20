# Visualization

Run from the checkout root with a Python that has Matplotlib.

```bash
python plot.py appworld                         # gallery of every local appworld run
python plot.py --run RUN_DIR                    # one run's training curve
python plot.py --run RUN_DIR --preset eval      # one run's checkpoint evaluations
python plot.py appworld --preset comparison --config comparison.json
```

Replace `appworld` with `mbpp`, `mini_swe_bench` or `mastermind`.
`plot.py` uses `environments/plotting/bin/python` automatically when the
invoking Python lacks Matplotlib and that environment exists.

## Benchmark gallery

`python plot.py BENCHMARK` finds every run under
`examples/BENCHMARK/train/<method>/runs/<preset>/<model>/`, extracts the plotted
values, and draws them. Each model gets its own directory:

```text
examples/<benchmark>/figures/output/<model>/
  data/{training,evaluation,turns}.csv, manifest.json
  learning.pdf/png, comparison*.pdf/png, variants.pdf/png, rollout.pdf/png, turns_<method>*.pdf/png
  index.html
```

| Option | Effect |
|---|---|
| `--model M` | Only the run directory name `M` (for example `qwen3-4b-thinking`); default: every model with runs |
| `--plots a,b` | `learning`, `comparison`, `comparison_extended`, `variants`, `rollout`, `turns`; default: all |
| `--methods a,b` | `base`, `grpo`, `ppo`, `joint_ppo`, `erl`, `ace_rl`, `delayed_joint`, `alternating`, `ace_n_evals`; default: all |
| `--append` | Merge into existing data: re-extracted rows replace rows with the same identity, other rows stay |
| `--source HOST` | Read runs from an SSH host's checkout (`--source-root PATH` or `RLLM_PLOT_<HOST>_ROOT`) |
| `--offline` | With `--source`, reuse the cached rows without connecting |
| `--no-open` | Do not open `index.html` in a browser |

Runs are matched by method and preset: `grpo`, `ppo`,
`delayed_joint_ppo` `batch<N>_rollout8` (`joint_ppo`), ERL `rollout<N>` (`erl`), and ACE-RL `oneeval_rollout<N>` (`ace_rl`),
`delayed_oneeval_rollout<N>` (`delayed_joint`) and `iterative_singlemode` (`alternating`).
Current training defaults use eight policy rollouts or complete chains per task.
Historical rollout counts and joint PPO's older `batch<N>` run names remain
readable; plotting does not rename them or change their recorded settings.
Base results come from `examples/BENCHMARK/eval/ace/results/visits8/<model>/` and
`examples/BENCHMARK/eval/vanilla/results/pass8/<model>/`. Evaluations come from
each run's `eval/<method>/<preset>/step_N/` ([evaluation layout](evaluation.md));
each method has one preset:

| Directory | Used for |
|---|---|
| `eval/ace/visits8` | RSI@1–8 and eight-visit turn curves; its first three visits give RSI@3, the three-visit turn curves and the learning reward |
| `eval/vanilla/pass8` | pass@1 (`pass_at_1_success_rate`, the mean over all eight attempts) and pass@1–8 |

A base model without a pass8 result uses its first ACE visit (one unadapted
attempt per task) as pass@1.

Figures:

- `learning`: training reward (centered five-step mean per logging segment;
  `training.csv` keeps raw values) and evaluation reward through visit 3, with
  earlier full successes carried as 1.
- `comparison`: pass@1 and RSI@3 against checkpoint step for every method, with
  the measured Base result at step 0. `comparison_extended`: pass@K and RSI@K for
  K = 1, 2, 4, 8. RSI@K is the share of tasks that reached reward 1 by visit K.
- `variants`: ACE-RL variants. `rollout`: every recorded `rollout/*` metric.
- `turns_<method>`: turn-level reward across the first three visits for each
  checkpoint; `_extended` uses all eight visits; `_nopad` variants do not carry
  solved tasks into later visits.
- Every step-axis figure also has `_total_gradient_update_tokens` and
  `_total_completion_tokens` versions whose x axis sums the recorded step-local
  `rollout/total_*` counts within the latest checkpoint lineage.

Missing series are labeled on the figure; a figure without data is drawn empty.
An evaluation directory without metrics yet is missing input; invalid present
metrics are reported as failures in `manifest.json`, and the command exits
nonzero after writing what it could. Evaluations whose `evaluation_config.json`
lacks a preset or task label are not selected.

When reporting best-of-K ACE reward, take each task's maximum reward over visits
1 through K, then average over tasks. Taking a maximum after averaging tasks
gives a different statistic. Carrying earlier full successes as 1 also differs
from best-of-K when rewards have partial credit. Label these quantities
separately. K counts complete visits, not environment turns within a visit.
See [evaluation call order](evaluation.md) for the first-visit policy budget.

Checkpoint comparison reports must keep an `index.html` with a checkpoint
selector, visit curves for K = 1–8, and an overview across all saved checkpoints.
Show ACE-RL, delayed ACE-RL, GRPO and GRPO + ACE where measured; label missing
evaluations rather than filling them in. Preserve separate views for best
reward through K, cumulative full success, and the reward of visit K itself.
Keep turn-budget ablations separate, with the same tasks and metric definitions.
Include training reward curves in the same report, with raw CSV values, exact
metric names, optimizer steps, and any smoothing stated. For policy rollout
reward, `turn/trajectory_max_reward_mean` averages each recorded policy
trajectory's best turn reward; identify the configured policy training source.
Keep this separate from `critic/score/mean`, which
uses retained update rows and can include ACE auxiliary roles; label it as a
training score rather than treating it as held-out policy performance. Record
the snapshot time, run identities, checkpoint coverage, and source hashes.

For a paper comparison, trace each displayed point to its source table and
preparation code. Check whether that point was observed or projected. A cached
task-level result establishes its recorded value, but does not replace original
trajectories when auditing behavior or rerunning the evaluation.

With `--source HOST`, the reader code runs on the host against its files and
only computed rows come back, cached under `.cache/figures/remote/`. Turn curves
are recomputed only for checkpoints whose metrics, configuration or completion
record changed; `--full-turn-check` rechecks every trajectory file.

## Single-run figures

`python plot.py --run RUN_DIR --preset NAME` writes to `RUN_DIR/figures/NAME/`.
The benchmark is read from the run path; name it first (`python plot.py mbpp
--run ...`) when the path does not contain `examples/<benchmark>/`.

| Preset | Horizontal axis | Input |
|---|---|---|
| `training` (default) | Optimizer step | `RUN_DIR/metrics.jsonl`; default `critic/score/mean`, repeat `--metric` for others |
| `eval` | Checkpoint step | Evaluation scalar; default `max_reward_mean` |
| `turns` | Turn across visits | Average task reward, padding finished trajectories with their maximum reward |
| `iterations` | Visit | Recorded `max_reward_mean` per visit |
| `comparison` | Checkpoint step | Explicit JSON configuration (below) |
| `all` | — | Every applicable preset; `--jobs N` runs N at a time |

```bash
python plot.py --list-presets --run RUN_DIR
python plot.py --run RUN_DIR --preset turns --eval-preset visits8 --local-epoch 3 --step 100
python plot.py --run RUN_DIR --preset all --jobs 4
python plot.py --run RUN_DIR --preset turns --help      # preset options
```

`--eval-preset` picks the evaluation preset (`visits8` or `pass8`), `--local-epoch K`
limits ACE views to measured visits 1…K (no extrapolation), `--step` (repeatable)
picks checkpoints, `--task-label` picks a task slice (slices are never pooled),
`--runs-dir DIR` points at an evaluation directory directly, `--output-dir` and
`--out` override the destination. `--check` validates inputs without rendering;
`--resolve` prints the plan. `--run HOST:PATH` mirrors only plotting inputs
(metrics, configurations, completion records and, for turn plots, policy
trajectories) over SSH into `.cache/figures/remote/`; `--offline` reuses it.

Turn plots read `task_*/chat_completions/eval/policy_stepNNNNNN_iter_I.json`
beside the metrics file (Mastermind vanilla: `trajectories.json`), average
rollouts within a task, then tasks. A task that reached reward 1 stays at 1 in
later visits; missing unsolved tasks or trajectories fail. The padding width is
the recorded turn budget: `max_turns` in `evaluation_config.json`, else the
recorded `--max_turns`/`--max-turns` option, else (Mastermind vanilla) each game's
`task.max_turns` in `trajectories.json`; `--max-turns` overrides it.

## Comparison configuration

`--preset comparison` reads only the files named in its JSON. `--check`
validates every input without Matplotlib; rendering writes PDF, PNG and a CSV of
every plotted point with its source path and SHA256. The same configuration can
drive `turns` and `iterations`.

```json
{
  "title": "AppWorld — Qwen3-4B (thinking)",
  "benchmark": "appworld",
  "split": "test_normal",
  "task_label": "all",
  "require_completion": true,
  "series": [
    {"id": "grpo_vanilla", "run": "grpo-run-name", "points": [
      {"step": 0, "file": "base/vanilla/metrics_step0.json", "provenance": "Qwen/Qwen3-4B; vanilla"},
      {"step": 20, "file": "grpo/eval/vanilla/pass8/step_20/metrics_step20.json", "provenance": "GRPO step 20; vanilla"}]},
    {"id": "ace_rl", "run": "ace-rl-run-name", "points": [
      {"step": 0, "file": "base/ace/metrics_step0.json", "provenance": "Qwen/Qwen3-4B; ACE"},
      {"step": 20, "file": "ace_rl/eval/ace/visits8/step_20/metrics_step20.json", "provenance": "ACE-RL step 20; ACE"}]}
  ],
  "training": [
    {"series": "grpo_vanilla", "run": "grpo-run-name", "metric": "critic/score/mean",
     "data": [[1, 0.2], [2, 0.3]], "provenance": "exported training history"}
  ]
}
```

- Relative paths resolve against the configuration file. Every point is
  required; the plotter never inserts or substitutes a baseline, and reusing a
  file across series other than a shared step-0 base is rejected.
- `--metric` defaults to `max_reward_mean`. A series may declare its own
  `metric`; mixed metrics need a top-level `evaluation_ylabel`. Final-policy
  scores (`final_policy/avg_reward`) require a recorded policy-only pass.
- Each metrics file must hold one evaluation at the declared step with task IDs;
  the task-ID union must match across points. With a completion record, its
  hash, split, model, step, task slice and visit count must agree.
  `require_completion: true` rejects missing records, including older MiniSWE
  and Mastermind results produced before their launchers wrote completion receipts.
- `training` overlays name an exact run and metric, with `data` inline or a
  `file` mapping run name to `[[step, value], ...]`.
- An optional `data_timestamp` (ISO with timezone) is the only timestamp shown.

For comparable curves, evaluate every point, including the base model, with the
same protocol, split and task IDs (for the AppWorld Qwen3-4B thinking comparison,
`EVAL_PROTOCOL=training_input_parity`).

## Tests

```bash
python -m pytest -q tests/figures
```

Rendering tests require Matplotlib; a skipped rendering test is not evidence that
figures render.
