"""Train an AppWorld agent (AppWorldReactAgent + AppWorldEnv) with GRPO.

Provides:

- ``preflight_checks``: validates the token-length config before GPU init.
- ``MAX_TRAIN_TASKS`` / ``TRAIN_TASK_OFFSET`` / ``MAX_VAL_TASKS`` env knobs that
  limit the train/validation task counts (default: all). The limits are applied
  here rather than at prepare time because ``prepare_appworld_data`` returns
  early once a full-size dataset is registered, silently ignoring
  ``--train-size``.

The AppWorld env server(s) are started by the launcher before this entrypoint
runs; this process connects to them over ``remote_environment_url``.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from examples.appworld.prepare_appworld_data import prepare_appworld_data
from rllm.agents.appworld_react_agents import AppWorldReactAgent
from rllm.agents.appworld_react_agents_ace import AppWorldReactAgentACE
from rllm.algorithms.ace.agents import ace_empty_playbook_prompt
from rllm.data import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

ACE_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "ace_rl" / "prompts"


def select_agent_class(config):
    """Use ACE-RL's agent and empty-playbook template when rllm.actor_prompt selects it."""
    prompt = ace_empty_playbook_prompt(config, ACE_PROMPTS_DIR)
    if prompt is None:
        return AppWorldReactAgent
    OmegaConf.update(config, "rllm.agent.agent_args.ace_prompt_template", prompt[0], force_add=True)
    return AppWorldReactAgentACE


def preflight_checks(config) -> None:
    errors = []
    max_prompt = config.data.max_prompt_length
    max_response = config.data.max_response_length
    max_seq = max_prompt + max_response

    ppo_max_tok = OmegaConf.select(config, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu")
    if ppo_max_tok is not None and ppo_max_tok < max_seq:
        errors.append(f"ppo_max_token_len_per_gpu ({ppo_max_tok}) < max_prompt + max_response ({max_prompt}+{max_response}={max_seq})")

    lp_max_tok = OmegaConf.select(config, "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu")
    if lp_max_tok is not None and lp_max_tok < max_seq:
        errors.append(f"log_prob_max_token_len_per_gpu ({lp_max_tok}) < max_seq ({max_seq})")

    max_model_len = OmegaConf.select(config, "actor_rollout_ref.rollout.max_model_len")
    if max_model_len is not None and max_model_len < max_seq:
        errors.append(f"max_model_len ({max_model_len}) < max_seq ({max_seq})")

    if errors:
        formatted = "\n".join(f"  [{idx}] {error}" for idx, error in enumerate(errors, 1))
        # WARN, don't raise: max_prompt + max_response overstates the real max
        # sequence (max_model_len limits generation), and with use_dynamic_bsz=True
        # these budgets are soft packing targets rather than hard limits.
        print(
            f"\n{'=' * 60}\n[preflight] WARNING — token budgets below max_prompt+max_response (soft for use_dynamic_bsz):\n{'=' * 60}\n{formatted}\n{'=' * 60}",
            flush=True,
        )
    else:
        print("[preflight] All config checks passed.")


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    from rllm.environments.appworld.appworld_env import AppWorldEnv

    preflight_checks(config)
    agent_class = select_agent_class(config)

    if not DatasetRegistry.dataset_exists("appworld", "train") or not DatasetRegistry.dataset_exists("appworld", "test"):
        print("AppWorld datasets not found. Preparing datasets...")
        prepare_appworld_data()

    train_dataset = DatasetRegistry.load_dataset("appworld", "train")
    val_dataset = DatasetRegistry.load_dataset("appworld", "test")
    if train_dataset is None or val_dataset is None:
        raise ValueError("Failed to load AppWorld datasets. Run: python examples/appworld/prepare_appworld_data.py")

    # Optional train task-count limit; unset/empty -> all tasks.
    max_tasks_env = os.environ.get("MAX_TRAIN_TASKS", "").strip()
    train_offset = int(os.environ.get("TRAIN_TASK_OFFSET", "0") or "0")
    max_train_tasks = int(max_tasks_env) if max_tasks_env else None

    if max_train_tasks is not None:
        train_data = train_dataset.get_data()[train_offset : train_offset + max_train_tasks]
        # Own registry name per slice, so a persistent full-size registration
        # cannot shadow it.
        train_name = f"appworld_train_off{train_offset}_n{len(train_data)}"
        if not DatasetRegistry.dataset_exists(train_name, "train"):
            DatasetRegistry.register_dataset(train_name, train_data, "train")
        train_dataset = DatasetRegistry.load_dataset(train_name, "train")
        print(
            f"[data] training on {len(train_data)} tasks (MAX_TRAIN_TASKS={max_tasks_env}, offset={train_offset}, registry='{train_name}')",
            flush=True,
        )
    else:
        print(
            f"[data] training on all {len(train_dataset.get_data())} tasks (MAX_TRAIN_TASKS=all)",
            flush=True,
        )

    # Optional val task-count limit, mirroring MAX_TRAIN_TASKS (bounds the final
    # validation pass).
    max_val_env = os.environ.get("MAX_VAL_TASKS", "").strip()
    max_val_tasks = int(max_val_env) if max_val_env else None
    if max_val_tasks is not None:
        val_data = val_dataset.get_data()[:max_val_tasks]
        val_name = f"appworld_test_n{len(val_data)}"
        if not DatasetRegistry.dataset_exists(val_name, "test"):
            DatasetRegistry.register_dataset(val_name, val_data, "test")
        val_dataset = DatasetRegistry.load_dataset(val_name, "test")
        print(
            f"[data] validating on {len(val_data)} tasks (MAX_VAL_TASKS={max_val_env}, registry='{val_name}')",
            flush=True,
        )
    else:
        print(
            f"[data] validating on all {len(val_dataset.get_data())} tasks (MAX_VAL_TASKS=all)",
            flush=True,
        )

    print(f"Train verl path: {train_dataset.get_verl_data_path()}")
    print(f"Val verl path: {val_dataset.get_verl_data_path()}")
    print(f"Train examples: {len(train_dataset.get_data())}")
    print(f"Val examples: {len(val_dataset.get_data())}")

    trainer = AgentTrainer(
        agent_class=agent_class,
        env_class=AppWorldEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
