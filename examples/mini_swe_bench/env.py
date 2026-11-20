"""Example-local mini SWE-bench environment using Apptainer-backed tool execution."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from examples.mini_swe_bench.shared.apptainer_utils import run_apptainer_exec
from examples.mini_swe_bench.shared.eval_transforms import (
    apply_all_speed_transforms,
)
from examples.mini_swe_bench.shared.skyrl_utils import get_docker_image_name
from rllm.environments.base.base_env import BaseEnv, is_transient_env_error

# Grading-harness resolver. SWE-Gym (training) and SWE-bench_Verified (eval) need
# DIFFERENT spec maps: stock `swebench` only knows SWE-bench_Verified repos and
# KeyErrors on SWE-Gym repos like getmoto/moto. `swegym` is a fork of the swebench
# package (github.com/SWE-Gym/SWE-Bench-Package) with the SWE-Gym repo maps baked
# in — same API except its `get_eval_report` takes `log_path=` instead of
# `test_log_path=`, and `make_test_spec` lives one module up. We select by the
# instance's `data_source`, mirroring SkyRL-OpenHands eval_infer.py. Imports are
# lazy + cached so a venv missing `swegym` only fails when it actually grades a
# SWE-Gym instance (with a clear install hint), not at import time.
_GRADING_HARNESS_CACHE: dict[str, dict[str, Any]] = {}


def _test_list_size(value: Any) -> int:
    """Count tests in an instance FAIL_TO_PASS/PASS_TO_PASS field. The field is
    a real list/ndarray in the train parquet but a JSON string in the
    validation parquet; missing or unparseable values count as 0."""
    if value is None:
        return 0
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _resolve_grading_harness(data_source: str) -> dict[str, Any]:
    flavor = "swegym" if "swe-gym" in (data_source or "").lower() else "swebench"
    cached = _GRADING_HARNESS_CACHE.get(flavor)
    if cached is not None:
        return cached
    if flavor == "swegym":
        try:
            from swegym.harness.grading import get_eval_report
            from swegym.harness.test_spec import make_test_spec
        except ImportError as exc:
            raise ImportError(f"SWE-Gym instances require the `swegym` package (uv pip install 'git+https://github.com/SWE-Gym/SWE-Bench-Package.git'). data_source={data_source!r}") from exc
        log_kwarg = "log_path"
    else:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec

        log_kwarg = "test_log_path"
    resolved = {
        "flavor": flavor,
        "make_test_spec": make_test_spec,
        "get_eval_report": get_eval_report,
        "log_kwarg": log_kwarg,
    }
    _GRADING_HARNESS_CACHE[flavor] = resolved
    return resolved


OBS_MAX_CHARS = int(os.environ.get("MINISW_OBS_MAX_CHARS", "3000"))
APPTAINER_NOISE_RE = re.compile(
    r"^(WARNING: (SINGULARITY|APPTAINER)_CACHEDIR.*|"
    r"INFO:\s+Using cached SIF image.*|"
    r"WARNING: Error changing the container working directory.*)$",
    re.MULTILINE,
)


def _clean_apptainer_output(text: str) -> str:
    cleaned = APPTAINER_NOISE_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# Container-backend failures (image pull/build, Docker Hub rate limit) — the
# observation is infra noise, not a real tool result. Matches only apptainer
# FATAL/build/registry errors, NOT the informational "Fetching OCI" lines that
# also appear on a successful first pull, so a normal pull is never flagged.
INFRA_FAILURE_RE = re.compile(
    r"(TOOMANYREQUESTS"
    r"|pull rate limit"
    r"|while building SIF from layers"
    r"|FATAL:\s+(?:container creation failed|Unable to handle docker://)"
    r"|no space left on device)",
    re.IGNORECASE,
)


def _truncate(text: str, max_chars: int = OBS_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n<response clipped>"


def _parse_xml_action(action_xml: str) -> tuple[str, dict]:
    fn_match = re.search(r"<function=([^>]+)>", action_xml)
    if not fn_match:
        return "", {}
    fn_name = fn_match.group(1).strip()
    params = {}
    for match in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", action_xml, re.DOTALL):
        params[match.group(1).strip()] = match.group(2).strip()

    for tag in {"command", "path", "file_text", "old_str", "new_str", "insert_line", "view_range"}:
        if tag not in params:
            match = re.search(rf"<{tag}>(.*?)</{tag}>", action_xml, re.DOTALL)
            if match:
                params[tag] = match.group(1).strip()
    return fn_name, params


class MiniSWEnv(BaseEnv):
    """Apptainer-backed mini SWE-bench environment with bind-mounted /testbed persistence."""

    DOCKER_WORKDIR = "/testbed"

    def __init__(
        self,
        task: dict | None = None,
        max_steps: int = 30,
        step_timeout: int = 120,
        reward_timeout: int = 300,
        scratch_root: Path | str | None = None,
    ):
        self.task = task or {}
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.scratch_root = Path(scratch_root) if scratch_root else Path("/scratch") / "unknown"
        self.step_count = 0
        self._last_output = ""
        self._commands_history: list[str] = []
        self._host_testbed: str | None = None
        self._overlay_dir: str | None = None
        self._infra_failed = False
        self._tool_stats = self._empty_tool_stats()

    @staticmethod
    def _empty_tool_stats() -> dict:
        return {
            "bash_calls": 0,
            "view_calls": 0,
            "edit_calls": 0,
            "edit_successes": 0,
            "edit_failures": 0,
            "create_calls": 0,
            "insert_calls": 0,
            "submit_calls": 0,
            "unknown_action": 0,
        }

    def _exec(self, command: str, timeout: int = 30) -> tuple[str, int]:
        bind_dirs: list[tuple[str, str]] = []
        if self._host_testbed:
            bind_dirs.append((self._host_testbed, self.DOCKER_WORKDIR))
        output, code = run_apptainer_exec(
            self._get_image_uri(),
            command,
            bind_dirs=bind_dirs,
            overlay_path=self._overlay_dir,
            timeout=timeout,
        )
        cleaned = _clean_apptainer_output(output or "")
        if INFRA_FAILURE_RE.search(cleaned):
            self._infra_failed = True
        return cleaned, code

    def _get_image_uri(self) -> str:
        instance = self.task.get("instance", self.task)
        image = instance.get("image_name") or self.task.get("image_uri") or self.task.get("instance_image_key")
        if image:
            return image if image.startswith("docker://") else f"docker://{image}"
        data_source = self.task.get("data_source", "SumanthRH/SWE-Gym-Subset")
        return f"docker://{get_docker_image_name(instance, data_source).replace('docker.io/', '')}"

    def reset(self, task: dict | None = None) -> tuple[str, dict]:
        self._cleanup_host_testbed()
        if task is not None:
            self.task = task
        self.step_count = 0
        self._last_output = ""
        self._commands_history = []
        self._infra_failed = False
        self._tool_stats = self._empty_tool_stats()
        self._init_host_testbed()

        instance = self.task.get("instance", self.task)
        return instance.get("problem_statement", ""), {
            "instance_id": instance.get("instance_id", ""),
            "max_steps": self.max_steps,
        }

    def _init_host_testbed(self) -> None:
        host_dir = tempfile.mkdtemp(prefix="minisw_testbed_")
        # Sibling directory overlay so writes inside the container (pip installs,
        # __pycache__, /tmp, etc.) land on /scratch rather than the apptainer
        # session-dir tmpfs (capped at 64 MiB and not user-overridable).
        overlay_dir = tempfile.mkdtemp(prefix="minisw_overlay_")
        self._host_testbed = host_dir
        self._overlay_dir = overlay_dir
        try:
            output, code = run_apptainer_exec(
                self._get_image_uri(),
                f"cp -a {self.DOCKER_WORKDIR}/. /opt/",
                bind_dirs=[(host_dir, "/opt")],
                overlay_path=overlay_dir,
                timeout=120,
            )
        except Exception:
            self._cleanup_host_testbed()
            raise
        if code != 0:
            self._cleanup_host_testbed()
            if INFRA_FAILURE_RE.search(_clean_apptainer_output(output or "")):
                self._infra_failed = True

    def _cleanup_host_testbed(self) -> None:
        if self._host_testbed:
            shutil.rmtree(self._host_testbed, ignore_errors=True)
            self._host_testbed = None
        if self._overlay_dir:
            shutil.rmtree(self._overlay_dir, ignore_errors=True)
            self._overlay_dir = None

    def __del__(self) -> None:
        try:
            self._cleanup_host_testbed()
        except Exception:
            pass

    def _get_git_diff(self) -> str:
        output, code = self._exec(f"cd {self.DOCKER_WORKDIR} && git diff HEAD")
        if code == 0 and output.strip():
            return output.strip()
        output, code = self._exec(f"cd {self.DOCKER_WORKDIR} && git diff")
        if code == 0 and output.strip():
            return output.strip()
        return ""

    def _compute_reward(self) -> tuple[float, str]:
        """Grade the agent's current /testbed state. Thin wrapper around
        _run_one_instance — pulls (instance, agent_diff) from self and delegates."""
        instance = self.task.get("instance", self.task)
        agent_diff = self._get_git_diff()
        return self._run_one_instance(instance, agent_diff)

    def _run_one_instance(
        self,
        instance: dict,
        agent_diff: str,
    ) -> tuple[float, str]:
        """Run the official SWE-bench eval pipeline for one (instance,
        agent_diff) and return (reward, reason). Mirrors
        ``regrade_ab_swegym._run_one`` — same logical pipeline, just inside a
        live rollout instead of offline regrading.

        Phases:
            A. ``make_test_spec(instance)``             → bash script + grading metadata
            B. run ``test_spec.eval_script`` via ``_exec`` → captures pytest output
            C. write log + ``get_eval_report``          → resolved bool + per-test buckets
            D. format reward + reason string

        We don't re-apply the agent's patch: it's already on ``/testbed`` via
        the host_testbed bind-mount. The eval_script's reset_commands handle
        test-file overlap so the agent's source-code edits persist while any
        test-file edits get overwritten by test_patch (matches official
        ``run_evaluation.run_instance`` semantics).
        """
        instance_id = instance["instance_id"]
        # Per-test PASS/FAIL buckets + pass/total counts; populated whenever
        # get_eval_report succeeds, read back by step() into
        # info["tests_status"] / info["test_counts"].
        self._last_tests_status: dict = {}
        self._last_test_counts: dict = {}

        # Select the grading harness by dataset: SWE-Gym (training) -> swegym,
        # SWE-bench_Verified (eval) -> swebench. See _resolve_grading_harness.
        data_source = self.task.get("data_source") or instance.get("data_source", "")
        try:
            harness = _resolve_grading_harness(data_source)
        except ImportError as e:
            print(f"[eval] {instance_id}: grading_harness_unavailable: {e}", flush=True)
            return 0.0, f"grading_harness_unavailable: {e}"
        make_test_spec = harness["make_test_spec"]
        get_eval_report = harness["get_eval_report"]

        # Phase A — build the official eval spec for this instance
        try:
            test_spec = make_test_spec(instance)
        except Exception as e:
            # Previously a silent early return — a SWE-Gym repo missing from the
            # swebench spec map (the original bug) produced a 0.0 with no trace.
            print(
                f"[eval] {instance_id}: make_test_spec_failed (flavor={harness['flavor']}): {type(e).__name__}: {e}",
                flush=True,
            )
            return 0.0, f"make_test_spec_failed: {type(e).__name__}: {e}"

        # Phase B — run eval_script inside the container.
        # HOME=/tmp because we run apptainer with --no-home; eval_script's
        # `git config --global` needs a writable HOME, and /tmp is on the overlay.
        # The speed transforms (strip-git-audit / -locale-gen / -all-conda,
        # skip-install-when-safe) are A/B-verified verdict-safe on BOTH flavors:
        #   swebench: 100/100 agreement (SWE-bench_Verified regrade 2026-05-19;
        #             median 8.8s -> 3.0s, 2.85x).
        #   swegym:   74/74 agreement on resolved+applied+FTP/PTP sets AND
        #             74/74 stock-vs-original-verdict (regrade_ab_swegym
        #             job 15222309, 2026-08-03, moto+mypy pool with the
        #             make-init / compound-pip install idioms added to
        #             skip-install; median 20.5s -> 7.6s, 2.70x).
        # 18/92 A/B samples were unreconstructable (agent diff didn't apply to
        # a fresh base — symmetric, both sides) and are outside the verified
        # set; the guard (buildable-file edits keep the real install) applies
        # per trajectory at runtime either way.
        eval_script = apply_all_speed_transforms(test_spec.eval_script, agent_diff, instance_id)
        full_script = 'export HOME=/tmp\nmkdir -p "$HOME"\n' + eval_script
        t0 = time.time()
        output, code = self._exec(
            f"bash -c {shlex.quote(full_script)}",
            timeout=self.reward_timeout,
        )
        elapsed = time.time() - t0
        print(
            f"[eval] {instance_id}: elapsed={elapsed:.1f}s exit={code}",
            flush=True,
        )

        # Phase C — write log + score via official function.
        # get_eval_report reads the log from disk (path-based API); we put it on
        # the overlay so it shares the apptainer container's filesystem view but
        # persists past the exec call.
        # IMPORTANT: the log MUST live in a directory named exactly {instance_id}.
        # swegym's get_logs_eval infers the repo from `Path(log_fp).parent.stem`
        # (it expects the SWE-bench layout .../{instance_id}/test_output.txt) and
        # then indexes MAP_REPO_TO_PARSER[repo]; a flat `{instance_id}.eval.log`
        # makes the parent a random temp-dir name -> bogus repo -> KeyError. This
        # layout is correct for both flavors (swebench gets its parser from
        # test_spec and ignores the path). Mirrors SkyRL-OpenHands eval_infer.py.
        # Where to write the eval log:
        #   - If MINISW_EVAL_LOG_DIR is set (launcher points it at
        #     .../logs/<run>/eval_logs when KEEP_EVAL_LOGS=true), PERSIST logs
        #     there for debugging — separated per run, so different jobs never
        #     mix. We do NOT delete them in that case.
        #   - Otherwise use a throwaway temp dir (overlay on /scratch, else /tmp)
        #     and clean it up after grading.
        persist_root = os.environ.get("MINISW_EVAL_LOG_DIR", "").strip()
        if persist_root:
            scratch_root = Path(persist_root)
            cleanup = False
        else:
            scratch_root = Path(self._overlay_dir) if self._overlay_dir else Path(tempfile.gettempdir())
            cleanup = True
        scratch_root.mkdir(parents=True, exist_ok=True)
        # Unique per-grade dir so concurrent rollouts of the SAME instance never
        # collide on a shared path. Uniqueness must live ABOVE the {instance_id}
        # dir, because swegym derives the repo from that dir's name. The outer dir
        # is named {instance_id}-<rand> so persisted logs are browsable by task.
        grade_root = Path(tempfile.mkdtemp(prefix=f"{instance_id}-", dir=str(scratch_root)))
        log_dir = grade_root / instance_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "test_output.txt"
        log_path.write_text(output or "")

        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": "minisw_inproc",
            "model_patch": agent_diff,
        }
        try:
            report = get_eval_report(
                test_spec=test_spec,
                prediction=prediction,
                include_tests_status=True,
                # swegym fork takes `log_path`; modern swebench takes `test_log_path`.
                **{harness["log_kwarg"]: str(log_path)},
            )
        except Exception as e:
            print(
                f"[eval] {instance_id}: get_eval_report_failed (flavor={harness['flavor']}): {type(e).__name__}: {e}",
                flush=True,
            )
            return 0.0, f"get_eval_report_failed: {type(e).__name__}: {e}"
        finally:
            if cleanup:
                shutil.rmtree(grade_root, ignore_errors=True)

        # Phase D — format reward + reason
        inst = report.get(instance_id, {})
        tests_status = inst.get("tests_status") or {}
        ftp = tests_status.get("FAIL_TO_PASS") or {}
        ptp = tests_status.get("PASS_TO_PASS") or {}
        self._last_tests_status = tests_status
        # Pass/total counts for the workflow's frac reward mode. Denominators
        # come from the instance's ground-truth test lists, NOT from
        # success+failure in the report: a patch that breaks test collection
        # drops tests from BOTH report buckets, so a report-derived denominator
        # would let a collection crash inflate the pass fraction. A test that
        # didn't run counts as failed.
        self._last_test_counts = {
            "ftp_passed": len(ftp.get("success", [])),
            "ftp_total": _test_list_size(instance.get("FAIL_TO_PASS")),
            "ptp_passed": len(ptp.get("success", [])),
            "ptp_total": _test_list_size(instance.get("PASS_TO_PASS")),
        }
        if inst.get("resolved"):
            return 1.0, "tests_passed"
        if not inst.get("patch_successfully_applied"):
            return 0.0, "tests_failed (patch_not_applied)"
        return 0.0, (f"tests_failed (ftp_fail={len(ftp.get('failure', []))}, ptp_fail={len(ptp.get('failure', []))})")

    def _editor_view(self, path: str, params: dict) -> str:
        if not path:
            output, _ = self._exec("ls -la")
            return output or "(empty)"

        output, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && test -d {path} && echo DIR || echo FILE")
        if output.strip() == "DIR":
            listing, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && find {path} -maxdepth 2 ! -name '.*' ! -path '*/.*' -print | head -200")
            return listing or "(empty directory)"

        view_range_raw = params.get("view_range", "")
        if view_range_raw:
            nums = re.findall(r"-?\d+", view_range_raw)
            if len(nums) >= 2:
                start, end = int(nums[0]), int(nums[1])
                end_cond = "" if end == -1 else f" && NR<={end}"
                range_output, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && awk 'NR>={start}{end_cond} {{printf \"%6d\\t%s\\n\", NR, $0}}' {path}")
                return range_output or "(empty range)"

        output, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && cat -n {path}")
        return output or "(empty)"

    def _editor_str_replace(self, path: str, params: dict) -> str:
        if not path:
            return "Error: path is required for str_replace"
        old_str = params.get("old_str", "")
        new_str = params.get("new_str", "")
        if not old_str:
            return "Error: old_str is required for str_replace"
        old_b64 = base64.b64encode(old_str.encode()).decode()
        new_b64 = base64.b64encode(new_str.encode()).decode()
        python_snippet = (  # noqa: UP031 — preserve generated Python snippets.
            "import base64,sys; path=%r; o=base64.b64decode(%r).decode(); n=base64.b64decode(%r).decode(); "
            "c=open(path,encoding='utf-8',errors='surrogateescape').read(); cnt=c.count(o); "
            "print('Error: old_str not found in file') if cnt==0 else "
            "(print('Error: old_str appears %%d times, must be unique'%%cnt) if cnt>1 else "
            "(open(path,'w',encoding='utf-8',errors='surrogateescape').write(c.replace(o,n,1)), print('The file '+path+' has been edited. Here\\'s the result of running cat -n on a snippet:')))"
        ) % (path, old_b64, new_b64)
        output, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && PYTHONUTF8=1 python3 -c {repr(python_snippet)}")
        if "has been edited" in (output or ""):
            find_snippet = (  # noqa: UP031 — preserve generated Python snippets.
                "import base64; n=base64.b64decode(%r).decode(); lines=open(%r,encoding='utf-8',errors='surrogateescape').readlines(); "
                "start=next((i for i,l in enumerate(lines,1) if n.split(chr(10))[0] in l),1); print(start)"
            ) % (
                new_b64,
                path,
            )
            line_output, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && PYTHONUTF8=1 python3 -c {repr(find_snippet)}")
            try:
                center = int(line_output.strip())
            except (ValueError, AttributeError):
                center = 1
            lo = max(1, center - 4)
            hi = center + len(new_str.splitlines()) + 4
            snippet, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && awk 'NR>={lo} && NR<={hi} {{printf \"%6d\\t%s\\n\", NR, $0}}' {path}")
            output = (output or "") + "\n" + (snippet or "")
        return output or "(no output)"

    def _editor_create(self, path: str, params: dict) -> str:
        if not path:
            return "Error: path is required for create"
        file_text = params.get("file_text", "")
        check_output, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && test -e {path} && echo EXISTS || echo OK")
        if "EXISTS" in (check_output or ""):
            return f"Error: File already exists at {path}. Cannot overwrite with 'create'. Use 'str_replace' instead."
        text_b64 = base64.b64encode(file_text.encode()).decode()
        python_snippet = (  # noqa: UP031 — preserve generated Python snippets.
            "import base64,os; path=%r; t=base64.b64decode(%r).decode(); os.makedirs(os.path.dirname(path) or '.',exist_ok=True); "
            "open(path,'w',encoding='utf-8',errors='surrogateescape').write(t); print('File created successfully at '+path)"
        ) % (path, text_b64)
        output, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && PYTHONUTF8=1 python3 -c {repr(python_snippet)}")
        return output or "(no output)"

    def _editor_insert(self, path: str, params: dict) -> str:
        if not path:
            return "Error: path is required for insert"
        new_str = params.get("new_str", "")
        insert_line_raw = params.get("insert_line", "0")
        try:
            insert_line = int(insert_line_raw)
        except (ValueError, TypeError):
            return f"Error: insert_line must be an integer, got {insert_line_raw!r}"
        text_b64 = base64.b64encode(new_str.encode()).decode()
        python_snippet = (
            "import base64; path=%r; n=base64.b64decode(%r).decode(); line=%d; "  # noqa: UP031 — preserve generated Python snippets.
            "lines=open(path,encoding='utf-8',errors='surrogateescape').readlines(); lines.insert(line, n+'\\n'); "
            "open(path,'w',encoding='utf-8',errors='surrogateescape').writelines(lines); "
            "print('The file '+path+' has been edited. Here\\'s the result of running cat -n on a snippet:')"
        ) % (path, text_b64, insert_line)
        output, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && PYTHONUTF8=1 python3 -c {repr(python_snippet)}")
        if "has been edited" in (output or ""):
            lo = max(1, insert_line - 2)
            hi = insert_line + len(new_str.splitlines()) + 4
            snippet, _ = self._exec(f"cd {self.DOCKER_WORKDIR} && awk 'NR>={lo} && NR<={hi} {{printf \"%6d\\t%s\\n\", NR, $0}}' {path}")
            output = (output or "") + "\n" + (snippet or "")
        return output or "(no output)"

    def step(self, action: Any) -> tuple[str, float, bool, dict]:
        action_str = action.action if hasattr(action, "action") else str(action)
        fn_name, params = _parse_xml_action(action_str)
        reward = 0.0
        done = False
        observation = ""

        instance = self.task.get("instance", self.task)
        if fn_name == "submit":
            self._tool_stats["submit_calls"] += 1
            git_diff = self._get_git_diff()
            reward, reason = self._compute_reward()
            observation = f"Submitting solution. Episode complete. Result: {reason}"
            done = True
            info = {
                "instance_id": instance.get("instance_id"),
                "submitted": True,
                "reward_reason": reason,
                "git_diff": git_diff,
                "tests_status": getattr(self, "_last_tests_status", {}),
                "test_counts": getattr(self, "_last_test_counts", {}),
                "all_passed": reward >= 1.0,
            }
        elif fn_name == "execute_bash":
            self._tool_stats["bash_calls"] += 1
            command = params.get("command", "")
            if command:
                self._commands_history.append(command)
                output, code = self._exec(
                    f"cd {self.DOCKER_WORKDIR} && {command}",
                    timeout=self.step_timeout,
                )
                self._last_output = output
                observation = output if output else f"(exit {code})"
            else:
                observation = "No command provided."
            info = {"instance_id": instance.get("instance_id")}
        elif fn_name == "str_replace_editor":
            command_type = params.get("command", "view")
            path = params.get("path", "")
            if command_type == "view":
                self._tool_stats["view_calls"] += 1
                observation = self._editor_view(path, params)
            elif command_type == "str_replace":
                self._tool_stats["edit_calls"] += 1
                observation = self._editor_str_replace(path, params)
                if "has been edited" in observation:
                    self._tool_stats["edit_successes"] += 1
                elif "Error:" in observation:
                    self._tool_stats["edit_failures"] += 1
            elif command_type == "create":
                self._tool_stats["create_calls"] += 1
                observation = self._editor_create(path, params)
            elif command_type == "insert":
                self._tool_stats["insert_calls"] += 1
                observation = self._editor_insert(path, params)
            else:
                observation = f"str_replace_editor: unknown command '{command_type}'"
            info = {"instance_id": instance.get("instance_id")}
        else:
            self._tool_stats["unknown_action"] += 1
            observation = f"Unknown or empty action: {fn_name or '(none)'}"
            info = {"instance_id": instance.get("instance_id")}

        # Fail-fast on container-backend failure (Docker Hub rate limit, OCI
        # build error, out-of-space). Otherwise the model would receive dozens of
        # identical infra-error observations and the dead rollout would enter the
        # training batch as a spurious reward-0 attempt. Terminate now, reward 0,
        # and flag it so the trainer can drop the trajectory.
        if self._infra_failed:
            self.step_count += 1
            info["infra_failed"] = True
            info["max_steps"] = self.max_steps
            info["step"] = self.step_count
            self._cleanup_host_testbed()
            observation = _truncate("[environment error] container backend unavailable (image pull/build failed); terminating episode.\n\n" + observation)
            return observation, 0.0, True, info

        observation = _truncate(observation)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            if fn_name != "submit":
                git_diff = self._get_git_diff()
                info["git_diff"] = git_diff
                reward, reason = self._compute_reward()
                info["reward_reason"] = reason
                info["tests_status"] = getattr(self, "_last_tests_status", {})
                info["test_counts"] = getattr(self, "_last_test_counts", {})
                info["all_passed"] = reward >= 1.0

        info["max_steps"] = self.max_steps
        info["step"] = self.step_count
        if done:
            info["tool_stats"] = dict(self._tool_stats)
            self._cleanup_host_testbed()

        return observation, reward, done, info

    def close(self):
        self._cleanup_host_testbed()

    @staticmethod
    def from_dict(info: dict) -> MiniSWEnv:
        return MiniSWEnv(
            task=info.get("task", info),
            max_steps=info.get("max_steps", 30),
            step_timeout=info.get("step_timeout", 120),
            reward_timeout=info.get("reward_timeout", 300),
            scratch_root=info.get("scratch_root"),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    @staticmethod
    def is_retryable_error(exc: BaseException) -> bool:
        return is_transient_env_error(exc, include_runtime_error_messages=True)
