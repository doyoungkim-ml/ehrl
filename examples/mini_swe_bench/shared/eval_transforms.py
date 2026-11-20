"""Shared SWE evaluation-script transforms for Apptainer environments.

Used by the live MiniSWEnv grading path and the offline stock/transformed
A/B regrader in examples.mini_swe_bench.eval.regrade_ab_swegym.
"""

from __future__ import annotations

import json
import re
import shlex

_GIT_AUDIT_PREFIXES = (
    "git status",
    "git show",
    "git -c core.fileMode=false diff",
)


def strip_git_audit(eval_script: str) -> str:
    """Drop the informational `git status` / `git show` / `git diff` lines from
    the official eval_script. They print before START_TEST_OUTPUT and are never
    read by the grader. The agent's loop is already over when this runs, so
    these lines are not part of the agent's observed feedback. Saves ~0.4-2s
    per task.
    """
    return "\n".join(line for line in eval_script.splitlines() if not line.lstrip().startswith(_GIT_AUDIT_PREFIXES))


def strip_locale_gen(eval_script: str) -> str:
    """Drop the `sed -i ... /etc/locale.gen && locale-gen` line.

    The SWE-bench image's pre_install already ran `locale-gen en_US.UTF-8` at
    build time. A fresh container's `locale -a` shows en_US.utf8 and Python
    `setlocale('en_US.UTF-8')` succeeds without any runtime regeneration —
    verified via apptainer probe (job 9032765, 2026-05-19). Saves ~4.4s on
    Django tasks (the only repo whose `eval_commands` include this pair).
    """
    return re.sub(
        r"^[ \t]*sed -i '/en_US\.UTF-8/.*locale-gen[^\n]*$",
        "echo '[strip-locale-gen] sed+locale-gen skipped (image already has en_US.utf8)'",
        eval_script,
        flags=re.MULTILINE,
    )


def strip_all_conda(eval_script: str) -> str:
    """Remove ALL `source /opt/miniconda3/bin/activate` + `conda activate <env>`
    pairs from eval_script. The testbed conda env is still used end-to-end
    because our apptainer wrapper passes
    `--env PATH=/opt/miniconda3/envs/testbed/bin:...`, so `python`, `pytest`,
    `pip`, etc. still resolve to the env's binaries and find packages via the
    interpreter's built-in site-packages location.

    What's lost vs. running `conda activate`:
      - $CONDA_PREFIX, $CONDA_DEFAULT_ENV env vars unset
      - Activation hooks in `etc/conda/activate.d/` do not run (rare; most envs
        don't have any)

    Saves ~5-7s per task (two conda activates × ~3s each).
    """
    out: list[str] = []
    skip_next_conda_activate = False
    for line in eval_script.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("source /opt/miniconda3/bin/activate"):
            skip_next_conda_activate = True
            continue
        if skip_next_conda_activate and stripped.startswith("conda activate"):
            skip_next_conda_activate = False
            continue
        skip_next_conda_activate = False
        out.append(line)
    return "\n".join(out)


_REBUILD_TRIGGER_EXTS = (".c", ".pyx", ".pxd", ".pxi", ".h", ".cc", ".cpp", ".hpp")
# Makefile added 2026-08-03 for the SWE-Gym install idioms (moto's hook is
# `make init` — skipping it is only safe if the Makefile itself is untouched).
_REBUILD_TRIGGER_BASENAMES = ("setup.py", "pyproject.toml", "setup.cfg", "MANIFEST.in", "Makefile")


def agent_touched_buildable_files(agent_diff: str) -> bool:
    """True iff `agent_diff` includes any file that would invalidate a cached
    C extension build — i.e. C/Cython source or build configuration.

    Conservative-by-default: returns True for any unrecognized weirdness in
    the diff, so callers fall back to running the install hook normally.
    Only returns False when we're confident the diff is Python-only.
    """
    if not agent_diff:
        return False
    for line in agent_diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        m = re.match(r"diff --git a/(.*?) b/", line)
        if not m:
            return True
        path = m.group(1)
        basename = path.rsplit("/", 1)[-1]
        if any(path.endswith(ext) for ext in _REBUILD_TRIGGER_EXTS):
            return True
        if basename in _REBUILD_TRIGGER_BASENAMES:
            return True
        # requirements*.txt / test-requirements.txt etc. (2026-08-03): the
        # SWE-Gym install hooks pip-install these; edits to them require the
        # real install to run.
        if "requirements" in basename and basename.endswith(".txt"):
            return True
    return False


def maybe_strip_install(eval_script: str, agent_diff: str, instance_id: str | None = None) -> str:
    """If the agent's diff has no C source / build config, replace any
    `pip install -e .[test] --verbose` (or `python -m pip install -e .`) line
    in eval_script with a no-op echo. Saves ~30-45s on heavy-rebuild repos
    (astropy etc.).

    No-op (returns eval_script unchanged) when the agent touched buildable
    files — those need a real rebuild to be tested correctly.

    Original regex is intentionally conservative (requires `-e` immediately
    after `pip install`). Looser variants (intervening flags, non-editable
    `pip install .`) live in [[add_test_parallelism]]'s neighborhood and are
    NOT enabled by default — they failed verdict A/B on requests/sklearn
    (2026-05-19) and need image-by-image cwd-import verification.
    """
    if agent_touched_buildable_files(agent_diff):
        if instance_id:
            print(
                f"[skip-install] {instance_id}: agent diff touched buildable files; keeping install hook",
                flush=True,
            )
        return eval_script
    if instance_id:
        print(
            f"[skip-install] {instance_id}: SAFE (no C source / build config); replacing pip install with no-op",
            flush=True,
        )
    noop = "echo '[skip-install] install hook replaced with no-op; agent diff is Python-only'"
    out = re.sub(
        r"^[ \t]*(python -m )?pip install -e[ '\"][^\n]*$",
        noop,
        eval_script,
        flags=re.MULTILINE,
    )
    # SWE-Gym install idioms (2026-08-03, Phase-1 of the training-eval speedup
    # plan; VERDICT A/B PENDING — these only take effect once env.py ungates
    # transforms for the swegym flavor):
    #   moto:  the install hook is a bare `make init` line (pip install -e .
    #          + requirements; the image is already editable-installed).
    #   mypy:  one compound line `python -m pip install -r
    #          test-requirements.txt; python -m pip install -e .; [pytest
    #          xdist...]; hash -r` — pytest is preinstalled in the image.
    # Both are safe to no-op only under the buildable-files guard above
    # (which now covers Makefile + requirements*.txt).
    out = re.sub(r"^[ \t]*make init[ \t]*$", noop, out, flags=re.MULTILINE)
    out = re.sub(
        r"^[ \t]*python -m pip install -r test-requirements\.txt; python -m pip install -e \.[^\n]*$",
        noop,
        out,
        flags=re.MULTILINE,
    )
    return out


def add_test_parallelism(eval_script: str, instance: dict, n_workers: int = 4) -> str:
    """Inject test parallelism into the eval_script, repo-aware.

    Modes:
      - Django (test_cmd uses `--parallel 1`): change to `--parallel N`.
        Native runner; zero install cost.
      - Astropy pytest: add `-n N` to the pytest invocation. pytest-xdist is
        pre-installed in astropy docker images.
      - Other pytest-based repos: pip install pytest-xdist into the testbed
        env first (~1-2s, cached after first task), then add `-n N`.
      - Sympy (bin/test) / sphinx (tox): runner doesn't support parallel; left
        as-is.

    Shared with the offline regrader so the live environment evaluation
    benefits too. Verdict-safety A/B verified per repo before relying on it.
    """
    if n_workers is None or n_workers <= 1:
        return eval_script

    repo = (instance.get("repo") or "").lower()

    # Django: --parallel 1 -> --parallel N. Affects Django >=1.10
    # (Django 1.9 uses TEST_DJANGO_NO_PARALLEL which has no --parallel flag).
    eval_script = re.sub(r"--parallel\s+1\b", f"--parallel {n_workers}", eval_script)

    # pytest-based: add -n N; install xdist first if not astropy.
    if re.search(r"\bpytest\s+", eval_script):
        if "astropy" not in repo:
            eval_script = re.sub(
                r"^(:\s*'>>>>> Start Test Output')",
                r"pip install pytest-xdist -q --no-deps 2>&1 | tail -1\n\1",
                eval_script,
                flags=re.MULTILINE,
                count=1,
            )
        eval_script = re.sub(
            r"\bpytest\s+(?!-n\b)",
            f"pytest -n {n_workers} ",
            eval_script,
            count=1,
        )

    return eval_script


def restrict_to_graded_tests(eval_script: str, instance: dict, instance_id: str | None = None) -> str:
    """For pytest-based eval_scripts, replace the bare test-file argument with
    the explicit list of FAIL_TO_PASS + PASS_TO_PASS test IDs from the
    instance row. Skips ungraded tests in the same file — a 2-10x speedup on
    monolithic test files (matplotlib's test_axes.py, xarray's
    test_variable.py, seaborn's test_plot.py).

    Verdict-safe by construction: we still run every test that
    get_eval_report grades.

    No-op when:
      - instance lacks FTP/PTP fields, or both are empty
      - test IDs aren't pytest-style (e.g. Django dotted-name format, sympy
        bin/test paths) — detected by absence of `::`
      - test IDs span multiple files (rare for SWE-bench Verified; would need
        multi-file pytest invocation handling)
      - the eval_script's pytest line doesn't mention the file we're filtering
    """
    ftp = instance.get("FAIL_TO_PASS")
    ptp = instance.get("PASS_TO_PASS")
    if isinstance(ftp, str):
        try:
            ftp = json.loads(ftp)
        except json.JSONDecodeError:
            ftp = []
    if isinstance(ptp, str):
        try:
            ptp = json.loads(ptp)
        except json.JSONDecodeError:
            ptp = []
    test_ids = list(ftp or []) + list(ptp or [])
    if not test_ids:
        return eval_script

    # pytest-style IDs: <path>::<test_name> (possibly with class:: in between)
    test_files: set[str] = set()
    for tid in test_ids:
        if "::" not in tid:
            return eval_script
        test_files.add(tid.split("::", 1)[0])
    if len(test_files) != 1:
        return eval_script
    test_file = next(iter(test_files))

    quoted_ids = " ".join(shlex.quote(tid) for tid in test_ids)

    new_lines: list[str] = []
    found = False
    for line in eval_script.splitlines():
        stripped = line.lstrip()
        if (stripped.startswith("pytest ") or stripped.startswith("pytest\t")) and test_file in line:
            new_lines.append(line.replace(test_file, quoted_ids, 1))
            found = True
        else:
            new_lines.append(line)

    if not found:
        return eval_script

    if instance_id:
        print(
            f"[restrict-to-graded] {instance_id}: pytest restricted to {len(test_ids)} graded tests in {test_file}",
            flush=True,
        )
    return "\n".join(new_lines)


def apply_all_speed_transforms(
    eval_script: str,
    agent_diff: str,
    instance_id: str | None = None,
    *,
    instance: dict | None = None,
    n_workers: int = 4,
) -> str:
    """Apply all verdict-safe speed transformations in canonical order:
    strip_git_audit, strip_locale_gen, strip_all_conda, maybe_strip_install,
    add_test_parallelism.

    `instance` is needed to make parallelism repo-aware (django runner vs.
    pytest vs. sympy/sphinx no-op). Omit it (or set n_workers<=1) to skip
    parallelism.
    """
    eval_script = strip_git_audit(eval_script)
    eval_script = strip_locale_gen(eval_script)
    eval_script = strip_all_conda(eval_script)
    eval_script = maybe_strip_install(eval_script, agent_diff, instance_id)
    if instance is not None:
        eval_script = add_test_parallelism(eval_script, instance, n_workers)
        eval_script = restrict_to_graded_tests(eval_script, instance, instance_id)
    return eval_script
