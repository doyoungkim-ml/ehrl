"""Batch execution with explicit context rendering and adaptation hooks."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, StrictUndefined

from .base import ContextRule, ContextUpdateLevel

if TYPE_CHECKING:
    from rllm.agents.agent import BaseAgent, Trajectory
    from rllm.engine.agent_execution_engine import AgentExecutionEngine


class ContextAwareRunner:
    """Inject one context snapshot per batch, then update from its trajectories.

    Templates receive context; playbook remains an alias for existing prompts.
    Plain prompts have nonempty context appended. A custom
    prompt_renderer(template, context) can replace this behavior, or
    render_in_agent=True can leave task-dependent rendering to the agent.

    Agent wrappers expose _injected_system_prompt, _injected_template,
    and _injected_context (also available as _injected_playbook). Their
    reset method sets the first system message. Agents that render themselves
    consume the template and raw context when they receive task information.

    The runner temporarily replaces engine.agent_class during execution;
    that engine must not be shared with a concurrent runner.
    """

    def __init__(
        self,
        engine: AgentExecutionEngine,
        context_rule: ContextRule,
        base_system_prompt: str | Path,
        batch_size: int = 64,
        empty_context_text: str = "(No context available)",
        verbose: bool = True,
        *,
        prompt_renderer: Callable[[str, str], str] | None = None,
        render_in_agent: bool = False,
        empty_playbook_text: str | None = None,
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if render_in_agent and prompt_renderer is not None:
            raise ValueError("Choose either prompt_renderer or render_in_agent")
        self.engine = engine
        self.context_rule = context_rule
        self._check_update_level()
        self.batch_size = batch_size
        self.empty_context_text = empty_context_text if empty_playbook_text is None else empty_playbook_text
        self.verbose = verbose
        self.prompt_renderer = prompt_renderer
        self.render_in_agent = render_in_agent
        source = os.fspath(base_system_prompt)
        self.base_system_prompt_raw = Path(source).read_text(encoding="utf-8") if os.path.isfile(source) else source
        self.use_jinja2 = any(marker in self.base_system_prompt_raw for marker in ("{{", "{%", "{#"))
        self.prompt_template = None
        if self.use_jinja2 and not render_in_agent and prompt_renderer is None:
            environment = Environment(undefined=StrictUndefined, autoescape=False)
            self.prompt_template = environment.from_string(self.base_system_prompt_raw)
        self.batch_metrics: list[dict[str, Any]] = []

    def _check_update_level(self) -> None:
        if self.context_rule.update_level != ContextUpdateLevel.BATCH:
            raise NotImplementedError("ContextAwareRunner supports only BATCH updates; STEP and TRAJECTORY require execution-time callbacks")

    async def run(self, tasks: list[dict], *, adapt: bool = True) -> list[Trajectory]:
        """Execute batches; adapt=False skips every reset and update hook."""
        self._check_update_level()
        all_trajectories = []
        for batch_idx, batch_start in enumerate(range(0, len(tasks), self.batch_size)):
            batch_tasks = tasks[batch_start : batch_start + self.batch_size]
            if adapt:
                self.context_rule.reset_batch()
            trajectories = await self._process_batch(batch_tasks, batch_idx, adapt=adapt)
            all_trajectories.extend(trajectories)
            metrics = self._compute_batch_metrics(trajectories, batch_idx)
            self.batch_metrics.append(metrics)
            if self.verbose:
                print(f"[Context] Batch {batch_idx + 1}: {metrics['success_count']}/{metrics['num_trajectories']} successes, average reward {metrics['avg_reward']:.2f}")
        return all_trajectories

    async def _process_batch(self, batch_tasks: list[dict], batch_idx: int, adapt: bool = True) -> list[Trajectory]:
        """Execute one batch without resetting state owned by an outer loop."""
        self._check_update_level()
        context = self.context_rule.get_context()
        rendered_prompt = self._render_prompt(context)
        original_agent_class = self.engine.agent_class
        self.engine.agent_class = self._create_context_aware_agent_class(
            original_agent_class,
            system_prompt=rendered_prompt,
            template=self.base_system_prompt_raw,
            context=context,
        )
        try:
            trajectories = await self.engine.execute_tasks(batch_tasks)
        finally:
            self.engine.agent_class = original_agent_class

        if adapt:
            for trajectory in trajectories:
                self.context_rule.on_trajectory_complete(trajectory)
            await self.context_rule.on_batch_complete(trajectories)
        return trajectories

    def _render_prompt(self, context: str) -> str:
        if self.render_in_agent:
            return self.base_system_prompt_raw
        context_content = context if context and context.strip() else self.empty_context_text
        if self.prompt_renderer is not None:
            return self.prompt_renderer(self.base_system_prompt_raw, context_content)
        if self.prompt_template is not None:
            return self.prompt_template.render(context=context_content, playbook=context_content)
        if context and context.strip():
            return f"{self.base_system_prompt_raw}\n\n{context}"
        return self.base_system_prompt_raw

    @staticmethod
    def _create_context_aware_agent_class(original_class: type[BaseAgent], system_prompt: str, template: str, context: str) -> type[BaseAgent]:
        class ContextAwareAgent(original_class):
            _injected_system_prompt = system_prompt
            _injected_template = template
            _injected_context = context
            _injected_playbook = context

            def reset(self, *args, **kwargs):
                result = super().reset(*args, **kwargs)
                system_message = {"role": "system", "content": self._injected_system_prompt}
                if not self.messages:
                    self.messages = [system_message]
                elif self.messages[0].get("role") == "system":
                    self.messages[0]["content"] = self._injected_system_prompt
                else:
                    self.messages.insert(0, system_message)
                return result

        ContextAwareAgent.__name__ = f"ContextAware{original_class.__name__}"
        ContextAwareAgent.__qualname__ = f"ContextAware{original_class.__qualname__}"
        return ContextAwareAgent

    @staticmethod
    def _compute_batch_metrics(trajectories: list[Trajectory], batch_idx: int) -> dict[str, Any]:
        count = len(trajectories)
        success_count = sum(trajectory.reward > 0 for trajectory in trajectories)
        return {
            "batch_idx": batch_idx,
            "num_trajectories": count,
            "success_count": success_count,
            "success_rate": success_count / count if count else 0.0,
            "avg_steps": sum(len(trajectory.steps) for trajectory in trajectories) / count if count else 0.0,
            "avg_reward": sum(trajectory.reward for trajectory in trajectories) / count if count else 0.0,
        }

    def get_batch_metrics(self) -> list[dict[str, Any]]:
        """Return independent metric records for completed batches."""
        return [metrics.copy() for metrics in self.batch_metrics]

    def get_context(self) -> str:
        return self.context_rule.get_context()

    def get_rendered_prompt(self) -> str:
        """Render the current context without executing or adapting."""
        return self._render_prompt(self.context_rule.get_context())

    def save_context(self, path: str) -> None:
        self.context_rule.save(path)

    def load_context(self, path: str) -> None:
        self.context_rule.load(path)
