"""Utilities for the rllm package."""

from importlib import import_module

__all__ = ["EpisodeLogger", "compute_pass_at_k", "extract_source_metadata", "visualize_trajectories", "VisualizationConfig", "colorful_print", "colorful_warning", "Tracking"]


def __getattr__(name):
    modules = {
        "EpisodeLogger": "episode_logger",
        "compute_pass_at_k": "compute_pass_at_k",
        "extract_source_metadata": "source_extraction",
        "Tracking": "tracking",
        "VisualizationConfig": "visualization",
        "colorful_print": "visualization",
        "colorful_warning": "visualization",
        "visualize_trajectories": "visualization",
    }
    if name in modules:
        value = getattr(import_module(f"rllm.utils.{modules[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
