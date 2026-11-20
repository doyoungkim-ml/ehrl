"""Tool interfaces with integrations loaded on demand.

Importing tool schemas or parsers does not initialize execution/search tools.
The public ``tool_registry`` still supplies the built-in tools on first access.
"""

from importlib import import_module

__all__ = ["PythonInterpreter", "GoogleSearchTool", "FirecrawlTool", "TavilyExtractTool", "TavilySearchTool", "ToolRegistry", "tool_registry"]

_TOOL_MODULES = {
    "PythonInterpreter": "rllm.tools.code_tools",
    "GoogleSearchTool": "rllm.tools.web_tools",
    "FirecrawlTool": "rllm.tools.web_tools",
    "TavilyExtractTool": "rllm.tools.web_tools",
    "TavilySearchTool": "rllm.tools.web_tools",
}
_DEFAULT_NAMES = {
    "python": "PythonInterpreter",
    "google_search": "GoogleSearchTool",
    "firecrawl": "FirecrawlTool",
    "tavily-extract": "TavilyExtractTool",
    "tavily-search": "TavilySearchTool",
}


def __getattr__(name):
    if name in _TOOL_MODULES:
        value = getattr(import_module(_TOOL_MODULES[name]), name)
    elif name == "ToolRegistry":
        from .registry import ToolRegistry

        value = ToolRegistry
    elif name in {"DEFAULT_TOOLS", "tool_registry"}:
        from .registry import ToolRegistry

        defaults = {key: __getattr__(class_name) for key, class_name in _DEFAULT_NAMES.items()}
        registry = ToolRegistry()
        # Preserve registrations made through the explicit registry API before
        # this first lazy access, including user overrides of built-in names.
        for key, tool_cls in defaults.items():
            if registry.get(key) is None:
                registry.register(key, tool_cls)
        globals()["DEFAULT_TOOLS"] = defaults
        globals()["tool_registry"] = registry
        return globals()[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
