"""Named ACE roles sharing the generic single-turn agent implementation.

``ace_empty_playbook_prompt`` also gives ACE-RL's actor prompt to other
trainers, such as the GRPO actor-prompt control.
"""

import os

from omegaconf import OmegaConf

from rllm.agents.single_turn_agent import SingleTurnAgent
from rllm.algorithms.ace.context.training_inputs import render_training_template

ACTOR_PROMPTS = ("agent", "ace_empty_playbook")


class SingleTurnReflectorAgent(SingleTurnAgent):
    """Generate a reflection from a trajectory prompt."""


class SingleTurnCuratorAgent(SingleTurnAgent):
    """Generate playbook operations from a curation prompt."""


def ace_empty_playbook_prompt(config, prompts_dir) -> tuple[str, str] | None:
    """Return ACE-RL's actor template and its empty-playbook rendering, if selected.

    ``rllm.actor_prompt=ace_empty_playbook`` selects the prompt that ACE-RL
    gives the actor at step 0 when the initial playbook is empty. The default,
    ``agent``, keeps the agent's built-in prompt and returns None.
    """
    choice = OmegaConf.select(config, "rllm.actor_prompt", default="agent")
    if choice not in ACTOR_PROMPTS:
        raise ValueError(f"rllm.actor_prompt must be one of {', '.join(ACTOR_PROMPTS)}, got {choice!r}")
    if choice == "agent":
        return None
    # Read and render the template as the ACE trainer does.
    with open(os.path.join(os.path.abspath(prompts_dir), "base_system_prompt.txt")) as f:
        template = f.read()
    return template, render_training_template(template, {"playbook": ""})
