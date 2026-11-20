#!/usr/bin/env python3
"""Run the supported CPU gates in isolated pytest processes; skipped coverage fails."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from verify_verl_compat import verify

ROOT = Path(__file__).resolve().parents[1]
# Full trainer-import suites require the GPU stack. The CPU selection suite
# below executes real candidate construction and credit blocks across all four
# ACE variants; it does not replace or claim those integration tests.
CONTROLLER_TESTS = [
    "test_local_ace_checkpoint_resume.py",
    "test_local_ace_checkpoint_hooks.py",
    "test_agent_ppo_resume_loop.py",
    "test_training_reward_state.py",
    "test_local_ace_resume_integration.py",
    "test_standard_rejection_update.py",
    "test_ace_auxiliary_update_gate.py",
    "test_ace_handoff_protocol.py",
    "test_ace_training_task_context.py",
    "test_agent_ppo_rejected_batch_lifecycle.py",
    "test_agent_ppo_trainer_regressions.py",
    "test_agent_ppo_component_padding.py",
    "test_agent_ppo_rejection_mode.py",
    "test_ace_package_boundaries.py",
    "test_workflow_actor_batching.py",
    "test_workflow_rejected_batch_lifecycle.py",
    "test_workflow_checkpoint_resume.py",
    "test_ace_candidate_pool.py",
    "test_agent_masked_padding.py",
    "test_mixed_advantages.py",
    "test_ppo_empty_value_loss.py",
    "test_workflow_ppo_padding.py",
    "test_final_validation_routing.py",
    "test_erl_original.py",
    "test_erl_checkpoint.py",
    "test_erl_miniswe.py",
    "test_joint_ppo.py",
    "test_ace_reflection_source.py",
    "test_rollout_correction.py",
]


def read_report(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    return {key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-source", type=Path, required=True, help="Verified exact-pin VeRL checkout with the vendored resume patch")
    parser.add_argument("--report-dir", type=Path, default=ROOT / ".pytest_cache/rl-foundation")
    args = parser.parse_args(argv)
    source = args.verl_source.resolve()
    contract = verify(source)
    if contract["state"] != "applied":
        parser.error("VeRL source is unpatched; prepare it using scripts/verify_verl_compat.py --apply")
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    suites = [
        ("compatibility", ROOT, "tests/compat", ["tests/compat"]),
        ("run-state", ROOT, "tests", ["tests/test_run_state.py"]),
        ("repository-layout", ROOT, "tests", ["tests/test_repository_layout.py"]),
        (
            "foundation",
            ROOT,
            "tests/foundation",
            ["tests/foundation", "tests/agents/test_model_output_content_filtering.py", "tests/agents/test_terminal_feedback.py"],
        ),
        ("context", ROOT, "tests/context", ["tests/context"]),
        ("launchers", ROOT, "tests/scripts", ["tests/scripts"]),
        (
            "benchmarks",
            ROOT,
            "tests/envs",
            [
                "tests/envs/test_mastermind_env.py",
                "tests/envs/test_mastermind_eval.py",
                "tests/envs/test_mbpp_grader.py",
                "tests/envs/test_mbpp_feedback.py",
            ],
        ),
        ("appworld-execute", ROOT, "tests/envs", ["tests/envs/test_appworld_execute_uncertainty.py"]),
        (
            "appworld-engine",
            ROOT,
            "tests/engine",
            ["tests/engine/test_appworld_execution_uncertainty.py", "tests/engine/test_agent_execution_engine_retryable_env_errors.py", "tests/engine/test_rollout_log_prob_alignment.py"],
        ),
        ("mini-swe-bench", ROOT, "tests/mini_swe_bench", ["tests/mini_swe_bench"]),
        ("evaluation", ROOT, "tests/evaluation", ["tests/evaluation"]),
        ("controllers", ROOT, "tests/trainer", [f"tests/trainer/{name}" for name in CONTROLLER_TESTS]),
        ("verl-resume", source, "tests/trainer/ppo", ["tests/trainer/ppo/test_dataloader_resume_on_cpu.py"]),
    ]
    env = dict(os.environ, VERL_RESUME_SOURCE=str(source))
    # Never allow pytest's ambient addopts or global plugin autoload to weaken
    # the supported gate; these cases require no optional pytest plugins.
    env["PYTEST_ADDOPTS"] = ""
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    summary = {"python": sys.version, "contract": contract, "suites": {}}
    failed = False
    for name, cwd, boundary, tests in suites:
        report = report_dir / f"{name}.xml"
        command = [sys.executable, "-m", "pytest", f"--confcutdir={boundary}", *tests, "-q", f"--junitxml={report}"]
        print(f"\nRunning {name} CPU regressions", flush=True)
        suite_env = env
        if name == "controllers":
            # Controller tests build real VeRL DataProto batches from the pinned source.
            suite_env = dict(env, PYTHONPATH=os.pathsep.join(filter(None, (str(source), env.get("PYTHONPATH")))))
        result = subprocess.run(command, cwd=cwd, env=suite_env)
        counts = read_report(report) if report.exists() else {"tests": 0, "failures": 0, "errors": 1, "skipped": 0}
        passed = result.returncode == 0 and counts["tests"] > 0 and not any(counts[key] for key in ("failures", "errors", "skipped"))
        summary["suites"][name] = {**counts, "passed": passed, "returncode": result.returncode, "junit": str(report)}
        failed |= not passed
    summary["passed"] = not failed
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nCPU reports: {report_dir}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
