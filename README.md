# Experience-Handoff Reinforcement Learning for Recursive Self-Improvement

Code for [Experience-Handoff Reinforcement Learning for Recursive
Self-Improvement](https://arxiv.org/abs/2611.16409) ([PDF](ehrl.pdf)).

ACE-RL trains one model to act, reflect, and curate task-local playbooks inside
the Agentic Context Engineering (ACE) harness. This repository contains the
training and evaluation code for ACE-RL and the baselines (GRPO, PPO, ERL and
delayed-joint PPO) on Mastermind, MBPP, AppWorld and MiniSWEBench. It is built on
[rLLM](https://github.com/rllm-org/rllm) and [VeRL](https://github.com/verl-project/verl).

| Path | Contents |
| --- | --- |
| `examples/<benchmark>/train/<method>/train.slurm` | Training launchers |
| `examples/<benchmark>/eval/{ace,vanilla}/eval.slurm` | Evaluation with the ACE harness or without it |
| `examples/<benchmark>/train/ace_rl/prompts/` | Role prompts |
| `rllm/algorithms/` | ACE-RL, PPO and ERL implementations |
| `scripts/paper_metrics.py` | pass@K, RSI@K and training compute |
| `docs/` | Setup, presets, evaluation and troubleshooting |

## Usage

```bash
# Train
sbatch examples/<benchmark>/train/<method>/train.slurm

# Evaluate a saved checkpoint
RUN_DIR=<run-dir> STEP=<step> MAX_TURNS=<turns> sbatch examples/<benchmark>/eval/ace/eval.slurm
RUN_DIR=<run-dir> STEP=<step> MAX_TURNS=<turns> sbatch examples/<benchmark>/eval/vanilla/eval.slurm

# Score an evaluation
python scripts/paper_metrics.py <run-dir>/eval/<ace|vanilla>/<lane>/step_<step>/metrics_step<step>.json
```

See [docs](docs/) for setup, presets and options.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

```bibtex
@misc{kim2026experiencehandoffreinforcementlearningrecursive,
      title={Experience-Handoff Reinforcement Learning for Recursive Self-Improvement},
      author={Doyoung Kim and Byeongguk Jeon and Kailin Zhang and Sherry Yang},
      year={2026},
      eprint={2611.16409},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2611.16409},
}
```
