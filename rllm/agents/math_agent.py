import copy
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory, make_assistant_message, resolve_model_response_content


class MathAgent(BaseAgent):
    """
    A math agent that solves mathematical problems step by step, following the BaseAgent interface.
    """

    def __init__(self, accumulate_thinking=True):
        """
        Initialize the MathAgent.
        """
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Process environment feedback and update internal state."""

        # Reward update for existing step (None OR empty dict)
        if observation is None or (isinstance(observation, dict) and observation == {}):
            if self.trajectory.steps:
                cur_step = self.get_current_state()
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info
            return

        # This is a new observation, create a new step
        if isinstance(observation, dict):
            if "question" not in observation:
                raise ValueError(f"Observation dict missing required 'question' field: {observation}")
            formatted_observation = observation["question"]
        elif isinstance(observation, str):
            formatted_observation = observation
        else:
            raise ValueError(f"Invalid observation type: {type(observation)}")

        self.messages.append({"role": "user", "content": formatted_observation})

        new_step = Step(observation=formatted_observation)
        self._trajectory.steps.append(new_step)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state based on the model's response.
        """

        # Update the latest step. Preserve exact raw model text when available so
        # Qwen thinking history can be rendered losslessly by the parser.
        model_output = kwargs.get("model_output")
        content = resolve_model_response_content(response, model_output)
        self.messages.append(
            make_assistant_message(
                content,
                model_output,
                preserve_raw_completion=self.accumulate_thinking,
            )
        )

        cur_step = self.get_current_state()
        cur_step.chat_completions = self.chat_completions
        cur_step.model_response = content

        if content.count("</think>") == 1:
            thought, sep, action = content.partition("</think>")
            thought = thought + sep
            action = Action(action=action.strip())
        else:
            thought = None
            action = Action(action=content.strip())

        cur_step.thought = thought
        cur_step.action = action

        return action

    def reset(self) -> None:
        """Reset agent state for new episode (wipes trajectory and messages)."""
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        # remove thinking from assistant messages if not accumulate_thinking except the last one
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
