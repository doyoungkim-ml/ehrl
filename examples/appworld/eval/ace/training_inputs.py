"""Optional AppWorld reflector inputs matching the ACE training path."""

from types import SimpleNamespace

from rllm.algorithms.ace.context.reflector import Reflector
from rllm.algorithms.ace.context.training_inputs import (
    coerce_training_reflection_payload,
    fit_training_reflector_prompt,
    parse_training_json_response,
    steps_from_training_chat,
)


class AppWorldTrainingInputReflector(Reflector):
    """Use training chat reconstruction and full-prompt fitting during eval."""

    def _build_prompt(self, traj_data: dict, playbook_snapshot: str) -> str:
        trajectory = traj_data["trajectory"]
        native_steps = getattr(trajectory, "steps", []) or []
        # Each Step stores cumulative chat through its assistant message. A
        # final environment reply creates no reconstructed training step.
        messages = getattr(native_steps[-1], "chat_completions", None) if native_steps else []
        if native_steps and (not isinstance(messages, list) or not messages):
            raise ValueError("AppWorld training input parity requires cumulative Step.chat_completions")
        info = getattr(trajectory, "info", {}) or {}
        step_rewards = list(info.get("step_rewards", [step.reward for step in native_steps]) or [])
        reward = float(traj_data.get("reward", 0.0) or 0.0)
        all_passed = bool(info.get("all_passed", reward == 1.0))
        if "training_task" not in traj_data:
            raise ValueError("AppWorld training input parity requires the original environment task payload")
        reconstructed = SimpleNamespace(
            steps=steps_from_training_chat(messages, step_rewards, all_passed=all_passed),
            reward=reward,
            # Training reads env.task, not the agent's task populated at reset.
            task=traj_data["training_task"],
            info=info,
        )
        termination_reason = traj_data.get("termination_reason") or info.get("termination_reason")
        variables = self.adapter.get_template_variables(
            reconstructed,
            reward,
            playbook_snapshot,
            termination_reason=termination_reason,
            max_turns=traj_data.get("max_turns"),
        )
        variables.setdefault("playbook", playbook_snapshot)
        return fit_training_reflector_prompt(
            self.prompt_template,
            variables,
            tokenizer=self.rollout_engine.tokenizer,
            chat_parser=getattr(self.rollout_engine, "chat_parser", None),
            system_prompt=self.system_prompt,
            max_prompt_tokens=self.rollout_engine.max_prompt_length,
        )

    def _parse_response(self, response: str) -> dict:
        return coerce_training_reflection_payload(parse_training_json_response(response))

    def _get_default_reflection(self, traj_data: dict) -> dict:
        # Training's empty/failed generation parses to no insight.
        return {}
