from __future__ import annotations

import ast
import atexit
import logging
import multiprocessing as mp
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment

logger = logging.getLogger(__name__)

_TEST_EXECUTOR: ProcessPoolExecutor | None = None
_TEST_EXECUTOR_LOCK = threading.Lock()
_MAX_TEST_WORKERS = int(os.environ.get("MBPP_MAX_TEST_WORKERS", 8))
_TEST_EXECUTOR_START_METHOD = os.environ.get("MBPP_TEST_EXECUTOR_START_METHOD", "spawn").strip() or "spawn"


def _get_test_mp_context():
    try:
        return mp.get_context(_TEST_EXECUTOR_START_METHOD)
    except ValueError:
        logger.warning(
            "Invalid MBPP_TEST_EXECUTOR_START_METHOD=%s; falling back to spawn",
            _TEST_EXECUTOR_START_METHOD,
        )
        return mp.get_context("spawn")


def _get_test_executor() -> ProcessPoolExecutor:
    global _TEST_EXECUTOR
    if _TEST_EXECUTOR is None:
        with _TEST_EXECUTOR_LOCK:
            if _TEST_EXECUTOR is None:
                _TEST_EXECUTOR = ProcessPoolExecutor(
                    max_workers=_MAX_TEST_WORKERS,
                    mp_context=_get_test_mp_context(),
                )
    return _TEST_EXECUTOR


def _shutdown_test_executor():
    global _TEST_EXECUTOR
    with _TEST_EXECUTOR_LOCK:
        if _TEST_EXECUTOR is not None:
            _TEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)
            _TEST_EXECUTOR = None


atexit.register(_shutdown_test_executor)

# ``full`` shows each failed assertion and its stderr (the historical
# observation). ``hidden`` shows only a per-test outcome, so the policy cannot
# copy test inputs or expected values into its next submission.
FEEDBACK_MODES = ("full", "hidden")
_HIDDEN_SUMMARY_MAX_CHARS = 200
_TRACEBACK_FRAME = re.compile(r'^\s*File "(?P<path>[^"]*)", line (?P<line>\d+), in (?P<name>.*)$', re.MULTILINE)


def validate_feedback_mode(feedback: Any) -> str:
    if feedback not in FEEDBACK_MODES:
        raise ValueError(f"MBPP feedback must be one of {', '.join(FEEDBACK_MODES)}; got {feedback!r}")
    return feedback


def _assertion_texts(test_assertion: str) -> list[str]:
    """Source and literal values of a test assertion, which hidden feedback must not show."""
    texts = {test_assertion.strip()}
    try:
        tree = ast.parse(test_assertion)
    except SyntaxError:
        return [text for text in texts if text]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict)):
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
            texts.update((str(value), repr(value)))
    return [text for text in texts if text.strip()]


def _contains_text(summary: str, text: str) -> bool:
    # Match whole tokens so a literal 5 is found in "got 5." but not in "line 50".
    pattern = r"(?<![A-Za-z0-9_])(?<!\d\.)" + re.escape(text) + r"(?![A-Za-z0-9_]|\.\d)"
    return re.search(pattern, summary) is not None


def _hidden_error_summary(stderr: str, test_assertion: str, test_path: str, assertion_lines: range) -> str:
    """One-line outcome of a failed run, without assertion source or values.

    The traceback body repeats the assertion line, so only the final exception
    line is kept. A bare AssertionError raised by the test assertion itself
    becomes ``wrong answer``. A final line that still contains the assertion or
    one of its literal values (for example from an exception raised in a
    candidate ``__eq__``) is reduced to its exception type.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "Test failed"
    final = lines[-1]
    frames = list(_TRACEBACK_FRAME.finditer(stderr))
    if final.split(":", 1)[0] == "AssertionError" and frames:
        frame = frames[-1]
        if frame["path"] == test_path and frame["name"] == "<module>" and int(frame["line"]) in assertion_lines:
            return "wrong answer"
    if any(_contains_text(final, text) for text in _assertion_texts(test_assertion)):
        kind = final.split(":", 1)[0].strip()
        if re.fullmatch(r"[A-Za-z_][\w.]*", kind) and not any(_contains_text(kind, text) for text in _assertion_texts(test_assertion)):
            return kind
        return "runtime error"
    if len(final) > _HIDDEN_SUMMARY_MAX_CHARS:
        final = final[:_HIDDEN_SUMMARY_MAX_CHARS] + "..."
    return final


def _run_single_test(
    code: str,
    test_assertion: str,
    timeout: int,
    setup_code: str = "",
    feedback: str = "full",
) -> dict[str, Any]:
    # Exit code 0 alone is not a pass: code that calls sys.exit(0) or
    # os._exit(0) never reaches the assertion. Only a run that prints this
    # fresh marker after the assertion counts.
    marker = f"RLLM_MBPP_TEST_PASSED_{secrets.token_hex(16)}"
    parts = [code, setup_code, test_assertion, f"print({marker!r}, flush=True)"]
    test_code = "\n\n".join(part for part in parts if part) + "\n"
    preceding = [part for part in (code, setup_code) if part]
    first_assertion_line = sum(part.count("\n") + 2 for part in preceding) + 1
    assertion_lines = range(first_assertion_line, first_assertion_line + test_assertion.count("\n") + 1)

    try:
        with tempfile.TemporaryDirectory(prefix="rllm-mbpp-test-") as temp_dir:
            temp_path = os.path.join(temp_dir, "candidate_test.py")
            with open(temp_path, "w", encoding="utf-8") as file:
                file.write(test_code)

            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=temp_dir,
            )
            if result.returncode == 0 and marker in result.stdout:
                return {"test": test_assertion, "passed": True, "error": None}

            if result.returncode == 0:
                error_message = "Test did not run: the program exited before the test assertion finished"
            elif feedback == "hidden":
                error_message = _hidden_error_summary(result.stderr or "", test_assertion, temp_path, assertion_lines)
            else:
                error_message = result.stderr.strip() if result.stderr else "Test failed"
            if len(error_message) > 500:
                error_message = error_message[:500] + "..."
            return {
                "test": test_assertion,
                "passed": False,
                "error": error_message,
            }
    except subprocess.TimeoutExpired:
        return {
            "test": test_assertion,
            "passed": False,
            "error": f"Timeout: execution exceeded {timeout} seconds",
        }
    except Exception as exc:
        return {
            "test": test_assertion,
            "passed": False,
            "error": f"Execution error: {type(exc).__name__}" if feedback == "hidden" else f"Execution error: {exc}",
        }


def extract_code_from_response(response: str) -> str | None:
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    code_blocks = re.findall(r"```\n?(.*?)```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    return None


def run_code_with_tests(
    code: str,
    test_list: list[str],
    timeout: int = 10,
    parallel: bool = True,
    setup_code: str = "",
    feedback: str = "full",
) -> dict[str, Any]:
    """Run each assertion separately; ``feedback="hidden"`` redacts what results show."""
    validate_feedback_mode(feedback)
    results = {
        "test_results": [],
        "passed_tests": 0,
        "total_tests": len(test_list),
        "all_passed": False,
        "execution_error": None,
    }

    if not test_list:
        results["all_passed"] = True
        return results

    if parallel and len(test_list) > 1:
        executor = _get_test_executor()
        futures = {executor.submit(_run_single_test, code, test_assertion, timeout, setup_code, feedback): test_assertion for test_assertion in test_list}

        test_to_result = {}
        for future in as_completed(futures):
            test_assertion = futures[future]
            try:
                test_to_result[test_assertion] = future.result()
            except Exception as exc:
                test_to_result[test_assertion] = {
                    "test": test_assertion,
                    "passed": False,
                    "error": f"Execution error: {type(exc).__name__}" if feedback == "hidden" else f"Execution error: {exc}",
                }

        for test_assertion in test_list:
            test_result = test_to_result[test_assertion]
            results["test_results"].append(test_result)
            if test_result["passed"]:
                results["passed_tests"] += 1
    else:
        for test_assertion in test_list:
            test_result = _run_single_test(code, test_assertion, timeout, setup_code, feedback)
            results["test_results"].append(test_result)
            if test_result["passed"]:
                results["passed_tests"] += 1

    if feedback == "hidden":
        # Copies, because duplicate assertions share one parallel result.
        results["test_results"] = [{**result, "test": f"hidden test {index}"} for index, result in enumerate(results["test_results"], start=1)]
    results["all_passed"] = results["passed_tests"] == results["total_tests"]
    return results


class MBPPEnv(MultiTurnEnvironment):
    """Multi-turn environment for MBPP code-generation tasks."""

    def __init__(
        self,
        task: dict | None = None,
        max_turns: int = 6,
        timeout: int = 10,
        feedback: str = "full",
        **kwargs,
    ):
        super().__init__(task=task, max_turns=max_turns, **kwargs)
        self.timeout = timeout
        # Redaction happens here, so every consumer of observations, results
        # and logs receives the same view the policy saw.
        self.feedback = validate_feedback_mode(feedback)
        self.text = ""
        self.test_list: list[str] = []
        self.task_id: int | str = ""
        self.reference_code = ""
        self.test_setup_code = ""
        self.last_code: str | None = None
        self.last_test_results: dict[str, Any] | None = None
        self.best_reward = 0.0

    def reset(self, task: dict | None = None):
        if task is not None:
            self.task = task
        if self.task is None:
            raise ValueError("Task must be provided either in __init__ or reset()")

        self.text = self.task.get("text", "")
        self.test_list = self.task.get("test_list", [])
        self.task_id = self.task.get("task_id", "")
        self.reference_code = self.task.get("code", "")
        self.test_setup_code = self.task.get("test_setup_code", "")

        self.current_turn = 0
        self.done = False
        self.history = []
        self.last_code = None
        self.last_test_results = None
        self.best_reward = 0.0

        return self._render_observation(), {}

    def _extract_function_name(self) -> str | None:
        if not self.test_list:
            return None
        pattern = r"assert\s+\(?(\w+)\s*[\(\[]"
        for test in self.test_list:
            match = re.search(pattern, test)
            if match:
                return match.group(1)
        return None

    def _render_observation(self) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "turn": self.current_turn,
            "max_turns": self.max_turns,
            "remaining_turns": self.max_turns - self.current_turn,
            "task_id": self.task_id,
            "task_description": self.text,
            "num_tests": len(self.test_list),
            "done": self.done,
        }

        if self.last_test_results is not None:
            observation["test_results"] = self.last_test_results["test_results"]
            observation["passed_tests"] = self.last_test_results["passed_tests"]
            observation["total_tests"] = self.last_test_results["total_tests"]
            observation["all_passed"] = self.last_test_results["all_passed"]

        if self.current_turn == 0:
            observation["instructions"] = "Write Python code to solve the problem described above. Your code will be tested against hidden test cases. Output your code in a markdown code block."
            function_name = self._extract_function_name()
            if function_name:
                observation["expected_function_name"] = function_name

        return observation

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        if isinstance(action, str):
            code = extract_code_from_response(action)
        else:
            code = extract_code_from_response(str(action))

        if code is None:
            self.last_code = None
            self.last_test_results = {
                "test_results": [],
                "passed_tests": 0,
                "total_tests": len(self.test_list),
                "all_passed": False,
                "execution_error": "No code block found in response",
            }
            observation = self._render_observation()
            observation["error"] = "No code block found. Please provide your code in a markdown code block (```python ... ```)."
            return 0.0, observation

        self.last_code = code
        self.last_test_results = run_code_with_tests(
            code,
            self.test_list,
            self.timeout,
            setup_code=self.test_setup_code,
            feedback=self.feedback,
        )

        if self.last_test_results["total_tests"] > 0:
            reward = self.last_test_results["passed_tests"] / self.last_test_results["total_tests"]
        else:
            reward = 1.0

        self.best_reward = max(self.best_reward, reward)
        if self.last_test_results["all_passed"]:
            self.done = True

        observation = self._render_observation()
        if self.last_test_results["all_passed"]:
            observation["result"] = "All tests passed! Your solution is correct."
        else:
            failed_count = self.last_test_results["total_tests"] - self.last_test_results["passed_tests"]
            observation["result"] = f"{failed_count} test(s) failed. Please review the errors and try again."
        return reward, observation

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        self.history.append(action)
        reward, next_obs = self.get_reward_and_next_obs(self.task or {}, action)
        self.current_turn += 1

        if self.current_turn >= self.max_turns:
            self.done = True

        info = {
            "task_id": self.task_id,
            "turns_used": self.current_turn,
            "best_reward": self.best_reward,
            "all_passed": (self.last_test_results["all_passed"] if self.last_test_results is not None else False),
        }
        return next_obs, reward, self.done, info

    @staticmethod
    def from_dict(env_args: dict) -> MBPPEnv:
        return MBPPEnv(
            task=env_args,
            max_turns=env_args.get("max_turns", 6),
            timeout=env_args.get("timeout", 10),
            feedback=env_args.get("feedback", "full"),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
