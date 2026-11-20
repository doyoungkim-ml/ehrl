#!/usr/bin/env python3
"""Check that the built wheel owns algorithm code and imports away from the source tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REQUIRED = (
    "rllm/algorithms/ace/trainer.py",
    "rllm/algorithms/ace/trainer_base.py",
    "rllm/algorithms/ace/trainer_iterative.py",
    "rllm/algorithms/ace/trainer_delayed_joint.py",
    "rllm/algorithms/ace/checkpoint.py",
    "rllm/algorithms/ace/entrypoints.py",
    "rllm/algorithms/ace/context/playbook.py",
    "rllm/agents/single_turn_agent.py",
    "rllm/environments/base/prompt_response_env.py",
    "rllm/trainer/verl/task.py",
    "rllm/trainer/config/agent_ppo_trainer.yaml",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    wheel = parser.parse_args().wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = set(REQUIRED) - names
        if missing:
            raise SystemExit(f"Wheel is missing foundation modules: {sorted(missing)}")
        if any(name.startswith("examples/") or name.startswith("rllm/context/ace/") for name in names):
            raise SystemExit("Wheel contains obsolete algorithm/example paths")
    source = """
import importlib
import importlib.resources
import sys
sys.path.insert(0, sys.argv[1])
modules = [
    "rllm.context", "rllm.algorithms.ace.context",
    "rllm.algorithms.ace.entrypoints", "rllm.algorithms.ace.agents",
    "rllm.algorithms.ace.environments", "rllm.algorithms.ace.checkpoint",
    "rllm.trainer.verl.task", "rllm.parser.chat_template_parser",
]
for name in modules:
    module = importlib.import_module(name)
    assert module.__file__.startswith(sys.argv[1] + "/"), (name, module.__file__)
config = importlib.resources.files("rllm.trainer.config").joinpath("agent_ppo_trainer.yaml")
assert config.is_file()
assert "defaults:" in config.read_text()
assert not any(name == "examples" or name.startswith("examples.") for name in sys.modules)
assert "verl" not in sys.modules and "ray" not in sys.modules
print("Built-wheel imports and Hydra resource passed; full GPU trainers not imported")
"""
    with tempfile.TemporaryDirectory(prefix="rllm-wheel-check-") as cwd:
        subprocess.run([sys.executable, "-I", "-c", source, str(wheel)], cwd=cwd, check=True)


if __name__ == "__main__":
    main()
