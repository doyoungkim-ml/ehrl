"""Task-protocol preserving prompt replacement for an ERL second attempt."""

from rllm.agents.agent import BaseAgent


class PromptOverrideAgent(BaseAgent):
    """Delegate all actions/history to the plain agent; replace one instruction.

    Most benchmark agents start with a system message. AppWorld's canonical
    plain prompt instead begins with a user instruction followed by demos; its
    first message is replaced in the same position without removing the demos
    or task query. The internalization target restores the original full input.
    """

    def __init__(self, agent, prompt):
        self.agent = agent
        self.prompt = prompt
        self._applied = False

    def reset(self):
        self.agent.reset()
        self._applied = False

    def update_from_env(self, *args, **kwargs):
        result = self.agent.update_from_env(*args, **kwargs)
        if not self._applied:
            messages = self.agent.chat_completions
            if not messages or messages[0]["role"] not in ("system", "user"):
                raise ValueError("ERL requires an initial system/user instruction message")
            messages[0]["content"] = self.prompt
            self._applied = True
        return result

    def update_from_model(self, *args, **kwargs):
        return self.agent.update_from_model(*args, **kwargs)

    def get_current_state(self):
        return self.agent.get_current_state()

    @property
    def trajectory(self):
        return self.agent.trajectory

    @property
    def chat_completions(self):
        return self.agent.chat_completions
