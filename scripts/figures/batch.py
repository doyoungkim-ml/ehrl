"""Run the available plotting presets in isolated Python processes."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_NO_VALUE_ACTIONS = {"store_true", "store_false", "store_const", "append_const", "count", "help", "version"}


def _option_schema(path: Path) -> dict[str, int | str]:
    """Read option arity without importing plotting or benchmark dependencies."""
    schema = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        flags = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-")]
        keywords = {kw.arg: kw.value for kw in node.keywords}
        action = ast.literal_eval(keywords["action"]) if "action" in keywords else "store"
        arity = 0 if action in _NO_VALUE_ACTIONS else ast.literal_eval(keywords["nargs"]) if "nargs" in keywords else 1
        for flag in flags:
            schema[flag] = arity
    return schema


def _looks_like_option(value: str) -> bool:
    if not value.startswith("-") or value == "-":
        return False
    try:
        float(value)
        return False
    except ValueError:
        return True


def _argument_groups(arguments: list[str], schemas: list[dict[str, int | str]]) -> list[tuple[str, list[str]]]:
    """Keep complete repeated options, inline values, and negative values intact."""
    union = {}
    for schema in schemas:
        for flag, arity in schema.items():
            if flag in union and union[flag] != arity:
                raise ValueError(f"{flag} has incompatible argument counts across presets; run those presets separately")
            union[flag] = arity
    groups = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        flag, separator, _ = token.partition("=")
        if flag not in union:
            raise ValueError(f"unknown plotting option or unexpected value: {token}")
        if flag in {"--out", "--outdir", "--out_dir"}:
            raise ValueError(f"{flag} cannot be shared by all presets; use --output-dir for a base directory")
        arity = union[flag]
        group = [token]
        index += 1
        if separator:
            if arity == 0:
                raise ValueError(f"{flag} does not take a value")
            if isinstance(arity, int) and arity != 1:
                raise ValueError(f"{flag} expects {arity} values; use separate arguments")
        else:
            minimum = arity if isinstance(arity, int) else 1 if arity == "+" else 0
            maximum = arity if isinstance(arity, int) else 1 if arity == "?" else len(arguments)
            while len(group) - 1 < maximum and index < len(arguments) and not _looks_like_option(arguments[index]):
                group.append(arguments[index])
                index += 1
            if len(group) - 1 < minimum:
                raise ValueError(f"{flag} expects {minimum} value(s)")
        groups.append((flag, group))
    return groups


def _last_value(groups: list[tuple[str, list[str]]], flag: str) -> str | None:
    selected = [group for name, group in groups if name == flag]
    if not selected:
        return None
    group = selected[-1]
    return group[0].split("=", 1)[1] if "=" in group[0] else group[1]


def _evaluation_available(directory: Path) -> bool:
    """Only absent optional inputs may be skipped; malformed inputs must fail.

    Configuration without metrics is an incomplete evaluation, so it is sent
    to the normal validator as well.
    """
    paths = set(directory.glob("**/metrics_step*.json")) | set(directory.glob("**/evaluation_config.json"))
    return any(not any(part in {"archive", "historical"} for part in path.parts) for path in paths)


def _plans(benchmark, presets, *, run, config, output_dir, remaining):
    methods = {name: ROOT / path for name, path in presets.items()}
    schemas = {name: _option_schema(path) for name, path in methods.items()}
    groups = _argument_groups(remaining, list(schemas.values()))
    explicit_runs = _last_value(groups, "--runs-dir")
    runs_dir = Path(explicit_runs).expanduser().resolve() if explicit_runs else run / "eval" if run else None
    plans = []
    for name, method in methods.items():
        use_config = config if name.startswith(("comparison", "turns", "iterations")) else None
        routed = [token for flag, group in groups if flag in schemas[name] and not (use_config and flag == "--runs-dir") for token in group]
        destination = output_dir / name if output_dir else run / "figures" / name if run else ROOT / "examples" / benchmark / "figures" / "output" / name
        command = [sys.executable, str(ROOT / "plot.py"), benchmark, "--preset", name]
        if run:
            command += ["--run", str(run)]
        if use_config:
            command += ["--config", str(use_config)]
        if output_dir:
            command += ["--output-dir", str(destination)]
        command += routed
        reason = None
        if name in {"budget-study", "campaign-report"}:
            reason = f"{name} requires its own report config; invoke this preset explicitly"
        elif name == "training":
            if run is None:
                reason = "training requires --run"
            elif _last_value(groups, "--metrics-file") is None and not (run / "metrics.jsonl").exists():
                reason = "no training metrics.jsonl; use --metrics-file for an explicit export"
        elif name.startswith("comparison"):
            if use_config is None:
                reason = "comparison requires --config"
        elif name.startswith(("eval", "turns", "iterations")):
            if use_config is None:
                if runs_dir is None:
                    reason = "evaluation requires --run or --runs-dir"
                elif not _evaluation_available(runs_dir):
                    reason = "no evaluation artifacts"
        plans.append(
            {
                "benchmark": benchmark,
                "preset": name,
                "method": str(method),
                "run": str(run) if run else None,
                "config": str(use_config) if use_config else None,
                "output_dir": str(destination),
                "arguments": routed,
                "command": command,
                "status": "skipped" if reason else "selected",
                "reason": reason,
            }
        )
    return plans


def _execute(plan):
    try:
        return subprocess.run(plan["command"], text=True, capture_output=True)
    except OSError as error:
        return subprocess.CompletedProcess(plan["command"], 1, "", f"{error}\n")


def run_all(
    benchmark: str,
    presets: dict[str, str],
    *,
    run: Path | None,
    config: Path | None,
    output_dir: Path | None,
    remaining: list[str],
    jobs: int = 1,
    resolve: bool = False,
) -> int:
    """Run each applicable preset once, reporting skips and every child failure.

    ``output_dir`` is a base directory, with one subdirectory per preset. A
    comparison config is routed to comparisons and trajectory/iteration views;
    training and scalar evaluation retain their explicit run inputs. ``resolve``
    prints the entire plan as JSON without creating outputs or starting children.
    """
    try:
        if jobs < 1:
            raise ValueError("--jobs must be a positive integer")
        if not resolve and run is not None and not run.is_dir():
            raise ValueError(f"run directory does not exist: {run}")
        plans = _plans(benchmark, presets, run=run, config=config, output_dir=output_dir, remaining=remaining)
    except (OSError, SyntaxError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if resolve:
        print(json.dumps(plans, indent=2))
        return 0
    selected = [plan for plan in plans if plan["status"] == "selected"]
    for plan in plans:
        if plan["status"] == "skipped":
            print(f"[{plan['preset']}] SKIPPED: {plan['reason']}", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(_execute, plan): plan for plan in selected}
        for future in as_completed(futures):
            plan = futures[future]
            result = future.result()
            plan["status"] = "ok" if result.returncode == 0 else "failed"
            print(f"[{plan['preset']}] {plan['status'].upper()}", flush=True)
            if result.stdout:
                print(result.stdout.rstrip(), flush=True)
            if result.stderr:
                print(f"[{plan['preset']}] {result.stderr.rstrip()}", file=sys.stderr, flush=True)
    counts = {status: sum(plan["status"] == status for plan in plans) for status in ("ok", "failed", "skipped")}
    print(f"Preset results: {counts['ok']} succeeded, {counts['failed']} failed, {counts['skipped']} skipped.", flush=True)
    if not selected:
        print("No presets could run with these inputs.", file=sys.stderr)
    return int(bool(counts["failed"]) or not selected)
