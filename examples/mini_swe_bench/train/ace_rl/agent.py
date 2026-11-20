"""SWE tool agent with reset-safe ACE playbook injection."""

from rllm.agents.swe_agent import SWEAgent
from rllm.agents.system_prompts import SWEAGENT_SYSTEM_PROMPT
from rllm.algorithms.ace.context.training_inputs import render_training_template


class MiniSweACEAgent(SWEAgent):
    def __init__(self, ace_prompt_template=None, ace_playbook="", **kwargs):
        kwargs.setdefault("scaffold", "sweagent")
        super().__init__(**kwargs)
        # A fixed template lets trainers without an ACE controller use the
        # prompt that ACE-RL injects before each rollout.
        if ace_prompt_template is not None:
            rendered = render_training_template(ace_prompt_template, {"playbook": ace_playbook})
            self.set_ace_playbook_prompt(template=ace_prompt_template, playbook=ace_playbook, rendered_prompt=rendered)

    def set_ace_playbook_prompt(self, *, template, playbook, rendered_prompt):
        # Preserve the existing XML tool contract; ACE assets supply only the
        # additional playbook instructions. SWEAgent.reset reads system_prompt.
        self.system_prompt = SWEAGENT_SYSTEM_PROMPT + "\n\n" + rendered_prompt
        if self.messages:
            self.messages[0] = {"role": "system", "content": self.system_prompt}
