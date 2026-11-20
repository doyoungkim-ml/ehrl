"""Adapt the existing Apptainer environment's terminal reward for agent ACE."""

import re

from examples.mini_swe_bench.env import MiniSWEnv
from examples.mini_swe_bench.shared.rewards import MiniSweRewardMixin
from rllm.agents.agent import Step, Trajectory


class MiniSweACEEnv(MiniSweRewardMixin, MiniSWEnv):
    def __init__(self, *, reward_mode=None, frac3_lambda=None, frac3_floor=None, **kwargs):
        super().__init__(**kwargs)
        self.configure_reward(reward_mode, frac3_lambda, frac3_floor)

    @property
    def task_id(self):
        instance = self.task.get("instance", self.task)
        task_id = instance.get("instance_id") or self.task.get("task_id")
        if not task_id or not re.fullmatch(r"[A-Za-z0-9._-]+", str(task_id)):
            raise ValueError("MiniSWEBench ACE requires a path-safe instance_id for each task")
        return str(task_id)

    def reset(self, task=None):
        result = super().reset(task)
        if not self._host_testbed:
            self.close()
            raise RuntimeError(f"MiniSWEBench container initialization failed for {self.task_id}")
        return result

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if info.get("infra_failed"):
            # The agent engine retries environment failures; a broken container
            # must never masquerade as a scored programming attempt.
            raise RuntimeError("MiniSWEBench container backend failed: " + observation)
        if done:
            trajectory = Trajectory(steps=[Step(metadata=info)], metadata={})
            self.compute_trajectory_reward(trajectory)
            reward = trajectory.reward
            info = {**info, **trajectory.info, "all_passed": reward >= self.REWARD_RESOLVED}
        return observation, reward, done, info

    @staticmethod
    def from_dict(info):
        return MiniSweACEEnv(
            task=info.get("task", info),
            max_steps=info.get("max_steps", 30),
            step_timeout=info.get("step_timeout", 120),
            reward_timeout=info.get("reward_timeout", 600),
            scratch_root=info.get("scratch_root"),
            reward_mode=info.get("reward_mode"),
            frac3_lambda=info.get("frac3_lambda"),
            frac3_floor=info.get("frac3_floor"),
        )
