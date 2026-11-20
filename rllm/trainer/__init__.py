"""Training interfaces; benchmark registries load only when requested."""

from .agent_trainer import AgentTrainer

__all__ = ["AgentTrainer"]


def __getattr__(name):
    if name in {
        "safe_import",
        "ENV_CLASSES",
        "AGENT_CLASSES",
        "WORKFLOW_CLASSES",
        "ENV_CLASS_MAPPING",
        "AGENT_CLASS_MAPPING",
        "WORKFLOW_CLASS_MAPPING",
    }:
        from . import env_agent_mappings

        value = getattr(env_agent_mappings, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
