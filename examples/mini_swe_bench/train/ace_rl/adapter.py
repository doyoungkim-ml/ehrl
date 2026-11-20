"""
Mini SWE-Bench Reflector Adapter (training and evaluation)

Environment-specific adapter for mini_swe_bench (SWE-bench Verified + SWE-Gym)
that produces a pre-digested trajectory summary for the Reflector prompt.

Why pre-digested (Option B): MiniSWEnv only populates step.info with rich
signals (reward_reason, git_diff, tool_stats) on clean terminal trajectories —
i.e. when the agent calls `submit` or hits max_steps. Engine-level cuts
(PROMPT_TRUNCATION, TIMEOUT, ENV_TIMEOUT, malformed actions) leave info empty.
We therefore recompute behavioral signals from trajectory.steps[*].action and
step.next_observation, which are always present. Training rebuilds steps
from the chat, where the action is the whole reply. The default legacy path
has no grader report. The explicit feedback_v1 path preserves bounded terminal
facts through Token rollouts, and uses aggregate counts rather than test names.

Summary contract:
    Outcome: Success | Failed
    Verdict: <grader verdict> — <gloss>
    Submit Attempted: YES|NO
    Turns Used: <N>/<max>
    Tool Calls, Edit Outcomes, First Successful Edit, Repeated Actions
    Patch Produced, and FAIL_TO_PASS / PASS_TO_PASS counts when graded
"""

from __future__ import annotations

import copy
import random
import re
from types import SimpleNamespace
from typing import Any

from rllm.engine.terminal_feedback import sanitize_terminal_feedback

# Matches the env's grammar (examples/mini_swe_bench/env.py:_parse_xml_action).
_FUNCTION_RE = re.compile(r"<function=([^>]+)>")
_PARAM_COMMAND_RE = re.compile(r"<parameter=command>(.*?)</parameter>", re.DOTALL)
_BARE_COMMAND_RE = re.compile(r"<command>(.*?)</command>", re.DOTALL)

LAST_OBS_MAX_CHARS = 500


def _parse_action_function(action_str: str) -> tuple[str, str]:
    """Return (top-level function name, editor subcommand) — both possibly ''."""
    if not action_str:
        return "", ""
    fn_match = _FUNCTION_RE.search(action_str)
    fn_name = fn_match.group(1).strip() if fn_match else ""

    if fn_name != "str_replace_editor":
        return fn_name, ""

    cmd_match = _PARAM_COMMAND_RE.search(action_str) or _BARE_COMMAND_RE.search(action_str)
    subcommand = cmd_match.group(1).strip() if cmd_match else "view"
    return fn_name, subcommand


def _count_tools(trajectory: Any) -> dict[str, int]:
    counts = {
        "bash": 0,
        "view": 0,
        "str_replace": 0,
        "create": 0,
        "insert": 0,
        "submit": 0,
        "unknown": 0,
    }
    for step in getattr(trajectory, "steps", []) or []:
        action_str = getattr(step, "action", "") or ""
        fn_name, subcommand = _parse_action_function(action_str)
        if fn_name == "execute_bash":
            counts["bash"] += 1
        elif fn_name == "submit":
            counts["submit"] += 1
        elif fn_name == "str_replace_editor":
            key = subcommand if subcommand in {"view", "str_replace", "create", "insert"} else "unknown"
            counts[key] += 1
        else:
            counts["unknown"] += 1
    return counts


def _step_next_observation(step: Any) -> str:
    """Read the env's response to step.action, regardless of where it's stored.

    SWEAgent stores it in step.info['next_observation'] (Step pydantic model
    doesn't have a next_observation field — see webarena_agent.py:663).
    """
    text = getattr(step, "next_observation", None)
    if text is None:
        info = getattr(step, "info", None) or {}
        if isinstance(info, dict):
            text = info.get("next_observation")
    return str(text) if text else ""


def _count_edit_outcomes(trajectory: Any) -> tuple[int, int]:
    successes = 0
    failures = 0
    for step in getattr(trajectory, "steps", []) or []:
        action_str = getattr(step, "action", "") or ""
        fn_name, subcommand = _parse_action_function(action_str)
        if not (fn_name == "str_replace_editor" and subcommand == "str_replace"):
            continue
        # The env writes the outcome of str_replace into the NEXT observation:
        # success → contains "has been edited"; failure → contains "Error:"
        # (see env.py:_editor_str_replace).
        next_obs = _step_next_observation(step)
        if "has been edited" in next_obs:
            successes += 1
        elif "Error:" in next_obs:
            failures += 1
        # else: ambiguous — skip
    return successes, failures


def _first_successful_edit_turn(trajectory: Any) -> int | None:
    for turn, step in enumerate(getattr(trajectory, "steps", []) or [], start=1):
        fn_name, subcommand = _parse_action_function(getattr(step, "action", "") or "")
        if fn_name == "str_replace_editor" and subcommand in {"str_replace", "insert"}:
            if "Error:" not in _step_next_observation(step):
                return turn
    return None


_CALL_RE = re.compile(r"<function=.*?</function>", re.DOTALL)


def _repeated_actions(trajectory: Any) -> tuple[int, int | None, str]:
    """Count calls identical to the immediately preceding one; report where the first run starts.

    Only the function call is compared, so reworded reasoning around the same call still counts.
    """
    repeats, first_turn, first_action = 0, None, ""
    previous = None
    for turn, step in enumerate(getattr(trajectory, "steps", []) or [], start=1):
        action = str(getattr(step, "action", "") or "").strip()
        call = _CALL_RE.search(action)
        action = call.group(0) if call else action
        if action and action == previous:
            repeats += 1
            if first_turn is None:
                first_turn, first_action = turn - 1, action
        previous = action
    compact = " ".join(first_action.split())
    return repeats, first_turn, compact[:160] + ("..." if len(compact) > 160 else "")


def _last_step_info(trajectory: Any) -> dict:
    steps = getattr(trajectory, "steps", []) or []
    if not steps:
        return {}
    info = getattr(steps[-1], "info", None)
    return info if isinstance(info, dict) else {}


def _patch_produced_label(last_info: dict) -> str:
    if isinstance(last_info.get("patch_present"), bool):
        if not last_info["patch_present"]:
            return "no"
        size = last_info.get("patch_bytes")
        return f"yes (~{size} bytes)" if size is not None else "yes"
    if "git_diff" not in last_info:
        return "unknown"
    diff = last_info.get("git_diff") or ""
    if isinstance(diff, str) and diff.strip():
        return f"yes (~{len(diff)} bytes)"
    return "no"


def _last_observation_tail(trajectory: Any) -> str:
    steps = getattr(trajectory, "steps", []) or []
    if not steps:
        return ""
    last = steps[-1]
    text = _step_next_observation(last) or getattr(last, "observation", None) or ""
    text = str(text).strip()
    if len(text) <= LAST_OBS_MAX_CHARS:
        return text
    return text[-LAST_OBS_MAX_CHARS:].lstrip()


# Parse the ftp_fail / ptp_fail counts out of a `tests_failed (...)` reason.
_FTP_FAIL_RE = re.compile(r"ftp_fail=(\d+)")
_PTP_FAIL_RE = re.compile(r"ptp_fail=(\d+)")


def _derive_verdict(
    *,
    reward: float,
    reward_reason: str | None,
    git_diff: str | None,
    termination_reason: str | None,
    resolved: bool | None = None,
    grading_error: bool = False,
    patch_present: bool | None = None,
) -> tuple[str, str]:
    """Map authoritative env signals to a single (verdict, gloss) pair.

    Verdict is one of RESOLVED / WRONG_FIX / REGRESSION_ONLY /
    PATCH_NOT_APPLIED / NO_PATCH / OTHER for a graded trajectory, or
    UNGRADED when the engine cut the trajectory before grading. Derived
    only from grader ground truth (reward, reward_reason, git_diff) — no
    action-string heuristics. Check order matters: the distinct
    reward_reason strings (OTHER, PATCH_NOT_APPLIED) must be matched
    before the ftp/ptp branch, and NO_PATCH (empty diff) before WRONG_FIX.
    """
    if resolved if resolved is not None else reward >= 1.0:
        return "RESOLVED", "the patch resolved the issue; all target tests pass."

    reason = (reward_reason or "").strip()
    if not reason and termination_reason == "ENV_DONE":
        # The environment graded the attempt, but training rollouts keep only its outcome.
        return "NOT_RESOLVED", "the issue was not resolved; the grader's per-test report is not available for this attempt."
    if not reason:
        cut = termination_reason or "engine cut"
        return "UNGRADED", (f"the trajectory was cut by the engine ({cut}) before grading — no graded result exists.")

    if grading_error or reason.startswith("make_test_spec_failed") or reason.startswith("get_eval_report_failed"):
        return "OTHER", ("the grading harness errored — the outcome is inconclusive and is NOT attributable to the Generator.")

    if "patch_not_applied" in reason:
        return "PATCH_NOT_APPLIED", ("the submitted diff did not apply cleanly to the repository.")

    if patch_present is False and reason in {"no_patch", "tests_failed"}:
        return "NO_PATCH", "no patch was produced; the repository was left unchanged."

    if reason.startswith("tests_failed"):
        if patch_present is False or (git_diff is not None and not git_diff.strip()):
            return "NO_PATCH", ("no patch was produced; the repository was left unchanged.")
        ftp_m = _FTP_FAIL_RE.search(reason)
        ptp_m = _PTP_FAIL_RE.search(reason)
        ftp_fail = int(ftp_m.group(1)) if ftp_m else 0
        ptp_fail = int(ptp_m.group(1)) if ptp_m else 0
        if ftp_fail > 0:
            return "WRONG_FIX", (f"a patch was submitted, but {ftp_fail} FAIL_TO_PASS test(s) still fail — the issue is not resolved.")
        if ptp_fail > 0:
            return "REGRESSION_ONLY", (f"the patch made all FAIL_TO_PASS tests pass but broke {ptp_fail} previously-passing PASS_TO_PASS test(s).")

    return "OTHER", (f"unrecognized grading result ({reason!r}) — outcome inconclusive.")


class MiniSweReflectorAdapter:
    """Pre-digested trajectory summary for the Reflector prompt."""

    def __init__(
        self,
        max_turns: int = 30,
        verbose_summary: bool = False,
        obs_max_chars: int = 300,
        test_sample_n: int = 0,
        simplified_history: bool = True,
        tokenizer: Any = None,
        max_history_tokens: int | None = 20000,
        handoff_protocol: str = "legacy",
    ):
        if handoff_protocol not in {"legacy", "feedback_v1"}:
            raise ValueError(f"Unsupported MiniSWE handoff protocol: {handoff_protocol}")
        self.handoff_protocol = handoff_protocol
        self.tokenizer = tokenizer
        self.max_history_tokens = int(max_history_tokens or 20000)
        self.max_turns = max_turns
        self.verbose_summary = verbose_summary
        self.obs_max_chars = obs_max_chars
        self.test_sample_n = test_sample_n
        self.simplified_history = simplified_history

    def format_conversation_history(self, trajectory: Any) -> str:
        if self.handoff_protocol == "feedback_v1":
            feedback = self._terminal_feedback(trajectory)
            steps = list(getattr(trajectory, "steps", None) or [])
            if steps and feedback.get("submitted") is True and feedback.get("reward_reason"):
                # Submit's reply contains the grader reason, including arbitrary
                # exception messages. Replace that report with the same bounded
                # facts as the summary. Preserve the original trajectory.
                terminal = copy.copy(steps[-1])
                terminal.next_observation = "Submitting solution. Episode complete. Result: " + feedback["reward_reason"]
                terminal.info = {**(getattr(terminal, "info", None) or {}), "next_observation": terminal.next_observation}
                trajectory = SimpleNamespace(steps=[*steps[:-1], terminal])
        history = self._format_history(trajectory)
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(history, add_special_tokens=False)
            if len(tokens) > self.max_history_tokens:
                history = "[Earlier tool history omitted]\n" + self.tokenizer.decode(tokens[-self.max_history_tokens :])
        return history

    @staticmethod
    def _terminal_feedback(trajectory: Any) -> dict:
        last_info = _last_step_info(trajectory)
        trajectory_info = getattr(trajectory, "info", None) or {}
        feedback = trajectory_info.get("terminal_feedback") if isinstance(trajectory_info, dict) else None
        if not isinstance(feedback, dict):
            feedback = last_info.get("terminal_feedback", last_info)
        return sanitize_terminal_feedback(feedback)

    def _format_history(self, trajectory: Any) -> str:
        if self.simplified_history:
            from rllm.algorithms.ace.context.utils import format_trajectory_actions_only

            # Skip the initial-user-message preamble because the Reflector
            # template already injects it via {{problem_statement}}.
            # Default cap is 1500 (set in the util); fall back to obs_max_chars
            # only when the user explicitly set a non-default value.
            cap = self.obs_max_chars if self.obs_max_chars != 300 else 1500
            return format_trajectory_actions_only(
                trajectory,
                obs_max_chars=cap,
                include_initial_user_message=False,
            )
        from rllm.algorithms.ace.context.utils import format_trajectory_as_conversation

        return format_trajectory_as_conversation(trajectory, obs_max_chars=self.obs_max_chars)

    def format_trajectory_summary(self, trajectory: Any, reward: float, **kwargs) -> str:
        turns_used = len(getattr(trajectory, "steps", []) or [])

        tool_counts = _count_tools(trajectory)
        last_info = _last_step_info(trajectory)
        if self.handoff_protocol == "feedback_v1":
            last_info = self._terminal_feedback(trajectory)

        # The reflector framework passes max_turns defaulting to 5 (the mbpp
        # default); traj_data never sets it. Use the env's authoritative
        # per-trajectory max_steps, falling back to the adapter's own value.
        max_turns = last_info.get("max_steps") or self.max_turns

        engine_reason = kwargs.get("termination_reason")
        if engine_reason is None:
            # The engine records its cut reason on trajectory.info, not as an attribute.
            traj_info = getattr(trajectory, "info", None) or {}
            if isinstance(traj_info, dict):
                engine_reason = traj_info.get("termination_reason")
        resolved = self._resolved(trajectory, reward)
        verdict, verdict_gloss = _derive_verdict(
            reward=reward,
            reward_reason=last_info.get("reward_reason"),
            git_diff=last_info.get("git_diff"),
            termination_reason=engine_reason,
            resolved=resolved,
            grading_error=bool(last_info.get("grading_error", False)),
            patch_present=last_info.get("patch_present"),
        )

        submit_attempted = "YES" if tool_counts["submit"] > 0 or last_info.get("submitted") is True else "NO"
        outcome = "Success" if resolved else "Failed"
        edit_successes, edit_failures = _count_edit_outcomes(trajectory)
        first_edit = _first_successful_edit_turn(trajectory)
        repeats, repeat_turn, repeat_action = _repeated_actions(trajectory)
        lines = [
            f"Outcome: {outcome}",
            f"Verdict: {verdict} — {verdict_gloss}",
            f"Submit Attempted: {submit_attempted}",
            f"Turns Used: {turns_used}/{max_turns}",
            "Tool Calls: " + ", ".join(f"{name}={count}" for name, count in tool_counts.items()),
            f"Edit Outcomes: str_replace succeeded {edit_successes}, failed {edit_failures}",
            f"First Successful Edit: turn {first_edit}" if first_edit else "First Successful Edit: never",
        ]
        if repeats:
            lines.append(f"Repeated Actions: {repeats} turn(s) repeated the previous action exactly; the first repeat run starts at turn {repeat_turn}: {repeat_action}")
        else:
            lines.append("Repeated Actions: none")
        lines.append(f"Patch Produced: {_patch_produced_label(last_info)}")
        test_block = self._format_test_results(last_info)
        if test_block:
            lines.append(test_block)
        return "\n".join(lines)

    def _format_test_results(self, last_info: dict) -> str:
        """Decoded per-test outcome for the Trajectory Summary.

        feedback_v1 uses aggregate ground-truth-denominator counts only.
        Counts are always shown (they decode the verdict, leak nothing).
        Named cases are added only when test_sample_n > 0: up to
        test_sample_n names per category, split 2:3 success:failure,
        randomly sampled, no backfill when a bucket is short. tests_status
        is populated by MiniSWEnv only on the graded-failure path, so a
        resolved / patch-not-applied / ungraded trajectory yields "".
        """
        if self.handoff_protocol == "feedback_v1":
            counts = last_info.get("test_counts") or {}
            out = []
            for prefix, category in (("ftp", "FAIL_TO_PASS"), ("ptp", "PASS_TO_PASS")):
                passed, total = counts.get(f"{prefix}_passed"), counts.get(f"{prefix}_total")
                if passed is not None and total is not None:
                    # Totals include tests omitted from the grader report after
                    # a collection crash; missing cases must count as failures.
                    out.append(f"  {category}: {passed} passed, {max(0, total - passed)} failed ({total} expected)")
            return "Test Results:\n" + "\n".join(out) if out else ""
        tests_status = last_info.get("tests_status") if isinstance(last_info, dict) else None
        if not isinstance(tests_status, dict) or not tests_status:
            return ""
        rng = random.Random(str(last_info.get("instance_id") or ""))
        out = ["Test Results:"]
        for cat, note in (
            ("FAIL_TO_PASS", "target tests for this issue -- any failure means the issue is NOT resolved"),
            ("PASS_TO_PASS", "regression-guard tests -- any failure means the patch broke existing behavior"),
        ):
            bucket = tests_status.get(cat) or {}
            passed = list(bucket.get("success") or [])
            failed = list(bucket.get("failure") or [])
            out.append(f"  {cat}: {len(passed)} passed, {len(failed)} failed  ({note})")
            if self.test_sample_n > 0:
                out.extend(self._sample_cases(cat, passed, failed, rng))
        return "\n".join(out)

    def _sample_cases(self, cat: str, passed: list, failed: list, rng: random.Random) -> list[str]:
        """Up to test_sample_n names for one category, split 2:3 success:failure."""
        n = self.test_sample_n
        n_pass = round(n * 2 / 5)
        n_fail = n - n_pass
        pick_pass = rng.sample(passed, min(n_pass, len(passed)))
        pick_fail = rng.sample(failed, min(n_fail, len(failed)))
        if not (pick_pass or pick_fail):
            return []
        lines = [f"  Sampled {cat} cases:"]
        lines += [f"    [passed] {t}" for t in pick_pass]
        lines += [f"    [failed] {t}" for t in pick_fail]
        return lines

    def format_problem_statement(self, trajectory: Any) -> str:
        """The GitHub issue the Generator was tasked to resolve.

        Sourced from trajectory.task["problem_statement"] -- identical to the
        text the SWEAGENT_USER_PROMPT template wraps in <github_issue> tags,
        but pulled from the task dict directly (robust to 0-step trajectories
        and to any literal '</github_issue>' substring inside the issue).
        """
        task = getattr(trajectory, "task", None)
        if not isinstance(task, dict):
            return "(problem statement unavailable)"
        instance = task.get("instance", task)
        if not isinstance(instance, dict):
            instance = task
        ps = (instance.get("problem_statement") or task.get("problem_statement") or "").strip()
        return ps or "(problem statement unavailable)"

    @staticmethod
    def _resolved(trajectory: Any, reward: float) -> bool:
        """Training rewards can be shaped; all_passed, when recorded, is the resolved flag."""
        info = getattr(trajectory, "info", None) or {}
        if isinstance(info, dict) and "all_passed" in info:
            return bool(info["all_passed"])
        if any((getattr(step, "info", None) or {}).get("all_passed") for step in getattr(trajectory, "steps", []) or []):
            return True
        return reward >= 1.0

    def get_template_variables(
        self,
        trajectory: Any,
        reward: float,
        playbook: str,
        **kwargs,
    ) -> dict[str, str]:
        task = getattr(trajectory, "task", None)
        instance = task.get("instance", task) if isinstance(task, dict) else {}
        return {
            "instance_id": (instance or {}).get("instance_id", "") if isinstance(instance, dict) else "",
            "outcome": "SUCCESS" if self._resolved(trajectory, reward) else "FAILED",
            "problem_statement": self.format_problem_statement(trajectory),
            "trajectory_summary": self.format_trajectory_summary(trajectory, reward, **kwargs),
            "conversation_history": self.format_conversation_history(trajectory),
            "playbook": playbook,
        }
