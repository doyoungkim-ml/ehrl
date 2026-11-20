"""Base environment exports, loaded without optional reward/tool dependencies."""

from importlib import import_module

_EXPORTS = {
    "BaseEnv": "rllm.environments.base.base_env",
    "SingleTurnEnvironment": "rllm.environments.base.single_turn_env",
    "MultiTurnEnvironment": "rllm.environments.base.multi_turn_env",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
