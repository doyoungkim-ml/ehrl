"""Environment exports; optional benchmark modules load only when requested."""

from importlib import import_module

_CORE_EXPORTS = {
    "BaseEnv": "rllm.environments.base.base_env",
    "SingleTurnEnvironment": "rllm.environments.base.single_turn_env",
    "ToolEnvironment": "rllm.environments.tools.tool_env",
}

ENVIRONMENT_IMPORTS = [
    ("rllm.environments.browsergym.browsergym", "BrowserGymEnv"),
    ("rllm.environments.frozenlake.frozenlake", "FrozenLakeEnv"),
    ("rllm.environments.swe.swe", "SWEEnv"),
    ("rllm.environments.code.competition_coding", "CompetitionCodingEnv"),
    ("rllm.environments.mbpp.mbpp", "MBPPEnv"),
    ("rllm.environments.appworld.appworld_env", "AppWorldEnv"),
    ("rllm.environments.mastermind.mastermind", "MastermindEnv"),
]
_EXPORTS = {**_CORE_EXPORTS, **{name: module for module, name in ENVIRONMENT_IMPORTS}}
__all__ = list(_CORE_EXPORTS)


def safe_import(module_path, class_name):
    """Return an optional environment class if its dependencies are available."""
    try:
        return getattr(import_module(module_path), class_name)
    except ImportError:
        return None


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
