"""Guard package ownership and optional-dependency boundaries in fresh interpreters."""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_python(source):
    result = subprocess.run([sys.executable, "-c", source], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_foundation_imports_do_not_load_backends_or_benchmarks():
    run_python("""
import importlib.abc
import sys
class BlockIntegrations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("examples", "ray", "verl", "datasets", "httpx", "rllm.engine", "rllm.experimental")
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("Unexpected integration import: " + fullname)
sys.meta_path.insert(0, BlockIntegrations())
from rllm.context import ContextRule, ContextAwareRunner, NoContextRule
from rllm.trainer import AgentTrainer
from rllm.trainer.verl.task import RayAgentTrainer, run_agent_task
from rllm.algorithms.ace.context import Playbook
from rllm.agents.single_turn_agent import SingleTurnAgent
from rllm.environments.base.prompt_response_env import SingleTurnEnv
from rllm.parser.chat_template_parser import QwenChatTemplateParser
from rllm.tools.tool_base import ToolCall
assert NoContextRule().get_context() == ""
assert Playbook("example").to_string()
""")


def test_algorithms_and_context_do_not_import_example_implementations():
    for folder in ["rllm/algorithms", "rllm/context", "rllm/trainer"]:
        for path in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                assert not any(name == "examples" or name.startswith("examples.") for name in modules), path
                if folder == "rllm/context":
                    assert not any(name.startswith("rllm.algorithms") for name in modules), path


def test_lazy_tool_registry_defaults_and_user_overrides():
    run_python("""
import sys
from types import ModuleType
import rllm.tools as tools
from rllm.tools.registry import ToolRegistry
from rllm.tools.tool_base import Tool
assert "rllm.tools.web_tools" not in sys.modules
assert "rllm.tools.code_tools" not in sys.modules
class CustomTool(Tool):
    pass
class DefaultTool(Tool):
    pass
for module_name in ("rllm.tools.web_tools", "rllm.tools.code_tools"):
    module = ModuleType(module_name)
    for name in tools._TOOL_MODULES:
        setattr(module, name, DefaultTool)
    sys.modules[module_name] = module
registry = ToolRegistry()
registry.register("python", CustomTool)
assert tools.tool_registry is registry
assert registry.get("python") is CustomTool
assert set(registry.list_tools()) == {"python", "google_search", "firecrawl", "tavily-extract", "tavily-search"}
assert tools.PythonInterpreter is DefaultTool
assert tools.tool_registry is tools.tool_registry
registry.clear()
assert tools.tool_registry.list_tools() == []
""")
