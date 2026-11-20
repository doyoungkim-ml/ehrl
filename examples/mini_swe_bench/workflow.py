"""Workflow-based mini SWE-bench training surface using the current rLLM stack."""

from __future__ import annotations

from examples.mini_swe_bench.env import MiniSWEnv
from examples.mini_swe_bench.shared.rewards import MiniSweRewardMixin
from rllm.agents.swe_agent import SWEAgent
from rllm.workflows.cumulative_workflow import CumulativeWorkflow


class MiniSweWorkflow(MiniSweRewardMixin, CumulativeWorkflow):
    def __init__(
        self,
        agent_cls=None,
        env_cls=None,
        agent_args=None,
        env_args=None,
        max_steps=20,
        reward_timeout=600,
        reward_mode=None,
        frac3_lambda=None,
        frac3_floor=None,
        **kwargs,
    ):
        agent_cls = agent_cls or SWEAgent
        env_cls = env_cls or MiniSWEnv
        agent_args = dict(agent_args or {})
        env_args = dict(env_args or {})
        agent_args.setdefault("scaffold", "sweagent")
        env_args.setdefault("max_steps", max_steps)
        env_args.setdefault("reward_timeout", reward_timeout)
        super().__init__(
            agent_cls=agent_cls,
            env_cls=env_cls,
            agent_args=agent_args,
            env_args=env_args,
            max_steps=max_steps,
            **kwargs,
        )
        self.reward_timeout = reward_timeout
        self.configure_reward(reward_mode, frac3_lambda, frac3_floor)
