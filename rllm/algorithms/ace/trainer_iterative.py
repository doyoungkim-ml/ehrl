"""Iterative Local ACE+RL trainer for per-problem playbooks."""

import logging

from rllm.algorithms.ace.trainer import LocalACERLAgentPPOTrainer

logger = logging.getLogger(__name__)

_VALID_ITERATIVE_ROLES = {"policy", "reflector", "curator"}


class LocalACERLIterativeAgentPPOTrainer(LocalACERLAgentPPOTrainer):
    """Local ACE+RL with iterative role scheduling."""

    def _init_ace_components(self):
        super()._init_ace_components()
        ace = self.ace_config

        self.training_mode = "iterative"
        self.policy_only_phase_enabled = True
        self.role_reduced_sampling = True

        default_schedule = ["policy", "reflector", "curator"]
        self.iterative_schedule = self._parse_iterative_schedule(
            ace.get("iterative_schedule", default_schedule),
            default_schedule,
        )

        try:
            self.iterative_steps_per_role = int(ace.get("iterative_steps_per_role", 20))
        except (TypeError, ValueError):
            self.iterative_steps_per_role = 20
        if self.iterative_steps_per_role < 1:
            self.iterative_steps_per_role = 20

        self._iter_role_index = 0
        self._iter_role_step = 0

        logger.info(
            "[LOCAL-ACE-ITER] schedule=%s steps_per_role=%s",
            self.iterative_schedule,
            self.iterative_steps_per_role,
        )

    def _parse_iterative_schedule(self, raw_schedule, default_schedule: list[str]) -> list[str]:
        if isinstance(raw_schedule, str):
            text = raw_schedule.strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            tokens = [x.strip().strip("'").strip('"') for x in text.split(",")]
            roles = [x.lower() for x in tokens if x]
        elif isinstance(raw_schedule, list | tuple):
            roles = [str(x).strip().lower() for x in raw_schedule if str(x).strip()]
        elif raw_schedule is not None and not isinstance(raw_schedule, dict):
            try:
                roles = [str(x).strip().lower() for x in raw_schedule if str(x).strip()]
            except TypeError:
                roles = []
        else:
            roles = []

        parsed = [r for r in roles if r in _VALID_ITERATIVE_ROLES]
        if not parsed:
            logger.warning(
                "[LOCAL-ACE-ITER] Invalid iterative_schedule=%s. Falling back to %s.",
                raw_schedule,
                default_schedule,
            )
            return list(default_schedule)
        return parsed

    def _get_active_component(self) -> str:
        return self.iterative_schedule[self._iter_role_index % len(self.iterative_schedule)]

    def _advance_component_schedule(self):
        self._iter_role_step += 1
        if self._iter_role_step >= self.iterative_steps_per_role:
            self._iter_role_step = 0
            self._iter_role_index = (self._iter_role_index + 1) % len(self.iterative_schedule)
