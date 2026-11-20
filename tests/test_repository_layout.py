"""The tracked tree stays small and forkable; see docs/development.md.

Changing an allowlist here is a maintainer decision, not a way to pass a check.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = {"appworld", "mastermind", "mbpp", "mini_swe_bench"}
TRAINING_METHODS = {"ace_rl", "delayed_joint_ppo", "erl", "grpo", "ppo"}
TOP_LEVEL = {
    ".github", ".gitattributes", ".gitignore", ".pre-commit-config.yaml", "LICENSE", "NOTICE", "README.md",
    "compat", "docs", "ehrl.pdf", "examples", "plot.py", "pyproject.toml", "rllm", "scripts", "tests",
}
DOCS = {
    "ace-checkpoint-resume.md", "architecture.md", "artifacts.md", "baselines.md", "development.md", "evaluation.md",
    "experiments.md", "setup.md", "troubleshooting.md", "verl-compatibility.md", "visualization.md",
}
# Scheduler identities and personal or site paths belong in submit-time settings.
SITE_SPECIFIC = re.compile(r"#SBATCH\s+--(account|partition|qos)\b|(?<![\w.])/(home|projects|mnt/home)/[A-Za-z0-9_-]+/|\.site\.env")


def tracked() -> list[str]:
    output = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [path for path in output.split("\0") if path and (ROOT / path).exists()]


def test_top_level_entries_are_the_maintained_set():
    assert {path.split("/", 1)[0] for path in tracked()} <= TOP_LEVEL


def test_documentation_pages_are_the_maintained_set():
    assert {path.removeprefix("docs/") for path in tracked() if path.startswith("docs/")} <= DOCS


def test_examples_hold_only_maintained_benchmarks_with_their_entry_points():
    assert {path.split("/")[1] for path in tracked() if path.startswith("examples/") and path.count("/") > 1} <= BENCHMARKS
    for benchmark in BENCHMARKS:
        for method in TRAINING_METHODS:
            assert (ROOT / f"examples/{benchmark}/train/{method}/train.slurm").is_file()
        for method in ("ace", "vanilla"):
            assert (ROOT / f"examples/{benchmark}/eval/{method}/eval.slurm").is_file()


def test_tracked_text_has_no_site_specific_settings():
    offenders = []
    for path in tracked():
        # Tests may use such strings as fixtures.
        if path.startswith("tests/") or not path.endswith((".sh", ".slurm", ".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt")):
            continue
        for number, line in enumerate((ROOT / path).read_text(errors="replace").splitlines(), 1):
            if SITE_SPECIFIC.search(line):
                offenders.append(f"{path}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
