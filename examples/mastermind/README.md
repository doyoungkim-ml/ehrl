# Mastermind

Mastermind uses Qwen3-0.6B with thinking either enabled or disabled. Benchmark
assets and entrypoints live together here:

- `prepare_mastermind_data.py`: registers the training and held-out test tasks.
- `train/grpo/`: GRPO entrypoint and launcher.
- `train/ace_rl/`: ACE-RL entrypoint, launcher, prompts and initial playbook.
- `eval/evaluate.py`: evaluates an existing serving endpoint on registered test tasks.

Training registers its datasets automatically. To register them ahead of time in
the registry selected by `RLLM_HOME` (default `~/.rllm`):

```bash
python -m examples.mastermind.prepare_mastermind_data
```

The two training launchers default to two GPUs and accept
`MODEL_VARIANT=nonthinking` or `MODEL_VARIANT=thinking`:

```bash
PRESET=grpo MODEL_VARIANT=nonthinking sbatch examples/mastermind/train/grpo/train.slurm
PRESET=oneeval_rollout8 MODEL_VARIANT=thinking sbatch examples/mastermind/train/ace_rl/train.slurm
# Delayed-joint scheduling with aligned 1/1 reflector/curator sampling.
PRESET=delayed_oneeval_rollout8 MODEL_VARIANT=thinking sbatch examples/mastermind/train/ace_rl/train.slurm
```

The same command resumes the latest committed checkpoint in
`train/<method>/runs/<preset>/<model>/`, for example
`train/ace_rl/runs/oneeval_rollout8/qwen3-0.6b-thinking/`. Checkpoints, W&B files,
logs, episodes, and task artifacts stay below that directory. W&B logs to
`WANDB_ENTITY` (default: your account's entity). Set an explicit `RUN_DIR` for a separate experiment.

## Evaluation

Each evaluation method has `presets.sh` and one `eval.slurm`, matching the
training folder layout. The launchers build a missing GPU environment. ACE
evaluation registers the test split when it is missing; vanilla evaluation needs
it registered once with `python -m examples.mastermind.prepare_mastermind_data`
(same `RLLM_HOME`):

```bash
bash examples/mastermind/eval/vanilla/eval.slurm
bash examples/mastermind/eval/ace/eval.slurm
```

`RLLM_DRY_RUN=1` prints the complete serving and evaluation commands without
loading a model, making network calls, or creating output directories.
`VENV_DIR` or `PYTHON_BIN` selects another environment; otherwise the shared GPU
helpers select, and if needed build, the environment for the allocated GPU. Slurm wrappers request one GPU, 8 CPUs, and 64 GB RAM; account,
partition, QoS, and reservation are submission options.

| Method / preset | Default thinking | Prompt / response / model budgets | Samples per game | Turns | Sampling |
| --- | --- | --- | --- | --- | --- |
| Vanilla / `pass8` | disabled | 8192 / 10240 / 32768 | 8 independent attempts | 5 | 0.7 temperature / 1.0 top-p |
| ACE / `visits8` | enabled | 8192 / 32768 / 40960 | 8 sequential visits | 5 | 0.7 temperature / 1.0 top-p |

Both methods use a five-guess budget (`MAX_TURNS`). Vanilla passes it as
`--max-turns`, which replaces the registered games' own `max_turns` of 20, and
records the effective budget in each saved game and in the metrics.

The method defaults intentionally have different generation budgets and
thinking modes. Set `MODEL_VARIANT=thinking|nonthinking` and explicit budgets
when constructing a matched comparison. ACE uses one reflector and one curator
per attempt, 0.7 ACE temperature, and a 2048-token playbook cap.

`RUN_DIR=/absolute/path/to/training/run STEP=100` selects a saved training
checkpoint. `MODEL_PATH=/path/to/merged` selects already merged inference weights;
`MODEL_NAME` is the serving alias and `TOKENIZER` identifies the matching
tokenizer. The base-model defaults are `Qwen/Qwen3-0.6B`. `TASK_OFFSET` and
`TASK_LIMIT` select a test slice; an empty limit selects all remaining registered
test tasks. The launchers default to eight concurrent games. Training-run
evaluations write `RUN_DIR/eval/<method>/<preset>/step_N/` and take the model's
thinking mode from the run; standalone models write
`eval/<method>/results/<preset>/<model>/` (`pass8` for vanilla, `visits8` for ACE).
Set `OUTPUT_DIR` only for a task slice.

ACE can use Gemini for reflection/curation while keeping the Qwen generator:

```bash
ACE_MODEL=gemini-2.5-pro ACE_THINKING_BUDGET=2048 bash examples/mastermind/eval/ace/eval.slurm
```

Set `GEMINI_API_KEY` in the environment or `ACE_KEY_FILE` for Vertex credentials.
`ACE_MODEL`, `ACE_BASE_URL`, `ACE_PROJECT_ID`, and `ACE_THINKING_BUDGET` are
explicit overrides. No credential paths are inferred from another user's home.

Training and ACE evaluation share the prompts in `train/ace_rl/prompts`;
`PROMPTS_DIR` overrides them for an evaluation. The shared reflector adapter
shows the reflector one row per guess with its feedback, rejections, repeats and
contradictions, and takes the solved outcome from the environment (`won`, or
the training reconstruction's `all_passed`); partial reward alone does not
establish success.

Each run writes `metrics_step<N>.json`; vanilla also writes full per-game
`trajectories.json`, including raw model output. ACE writes per-task playbooks
and evaluation artifacts. W&B is optional (`WANDB_PROJECT`); checkpoint step
metadata is recorded independently of W&B. Existing run directories resume
when their recorded configuration matches; ACE reuses completed task visits
and vanilla reuses complete task batches. Failure exits
nonzero, and the launcher stops only its own vLLM process group.

Evaluate and plot a training run with explicit inputs:

```bash
RUN_DIR=/absolute/path/to/training/run STEP=100 \
  sbatch examples/mastermind/eval/ace/eval.slurm
python plot.py --run /absolute/path/to/training/run --preset eval
```

Use `python plot.py --list` to inspect retained plot presets.

For an already-running endpoint, call the Python evaluator directly. The
endpoint must support OpenAI-compatible `/v1/completions`; the execution stack
renders Qwen prompts with the matching tokenizer. `--help` only needs Python's
standard library.

```bash
python -m examples.mastermind.eval.evaluate \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-0.6B --model-variant nonthinking \
  --limit 20 --output /tmp/mastermind-nonthinking.json
```

Direct vanilla evaluation retains top-p 0.8 unless explicitly overridden;
the method launcher passes the training rollout value of 1.0. `--playbook`
accepts fixed UTF-8 hints and uses `train/ace_rl/prompts/base_system_prompt.txt`.
The generic runner uses `adapt=False`, so these hints remain fixed. Without
`--playbook`, it uses `MastermindAgent.SYSTEM_PROMPT`. Set `OPENAI_API_KEY` if
an existing endpoint requires authentication.

The evaluator uses only registered `mastermind/test` tasks with their original
seeds and turn limits. It never prepares the dataset automatically. Use the
same `RLLM_HOME` as data preparation. Success requires the environment's `won`
flag; reward is the best normalized per-guess reward. Empty trajectories have
zero reward and are unsuccessful.

CPU launcher checks are in `tests/evaluation/test_mini_mastermind_launchers.py`.
They verify parser parity, training-prompt preservation, output protection,
and owned-process cleanup with a fake server. GPU/API evaluation still requires
an appropriately prepared runtime.
