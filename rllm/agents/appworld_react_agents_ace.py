"""AppWorld React agent variant that supports ACE playbook/template injection."""

from __future__ import annotations

from jinja2 import Template

from rllm.agents.appworld_react_agents import AppWorldReactAgent


class AppWorldReactAgentACE(AppWorldReactAgent):
    def __init__(self, *args, ace_prompt_template: str | None = None, ace_playbook: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        # A fixed template lets trainers without an ACE controller use the
        # prompt that ACE-RL injects before each rollout.
        if ace_prompt_template is not None:
            self.set_ace_playbook_prompt(template=ace_prompt_template, playbook=ace_playbook)

    def set_ace_playbook_prompt(
        self,
        *,
        template: str,
        playbook: str,
        rendered_prompt: str | None = None,
    ) -> None:
        # AppWorld prompts need task/user fields from the reset observation, so
        # keep the Jinja template and render after reset rather than accepting
        # the playbook-only rendered prompt.
        self._injected_template = template
        self._injected_playbook = playbook

    def _initialize_from_task(self, observation: dict, **kwargs):
        if "user_info" in observation:
            self.user_info = observation["user_info"]
        elif "user_info" in kwargs:
            self.user_info = kwargs["user_info"]
        else:
            self.user_info = {
                "first_name": "User",
                "last_name": "Test",
                "email": "user@example.com",
                "phone_number": "+1234567890",
            }

        self.task_instruction = observation.get("instruction", "")
        self._trajectory.task = {"instruction": self.task_instruction}

        app_descriptions = observation.get("app_descriptions", "") or "[List of available apps will be shown here]"
        injected_template = getattr(self, "_injected_template", None) or getattr(self.__class__, "_injected_template", None)
        if injected_template:
            prompt_template = injected_template
            playbook = getattr(self, "_injected_playbook", None) or getattr(self.__class__, "_injected_playbook", "") or ""
        else:
            prompt_template = self.REACT_PROMPT
            playbook = ""

        template = Template(prompt_template)
        react_prompt = template.render(
            main_user=self.user_info,
            app_descriptions=app_descriptions,
            input_str=self.task_instruction,
            playbook=playbook,
        )
        self.messages = self.text_to_messages(react_prompt)
        self._initial_message_count = len(self.messages)
