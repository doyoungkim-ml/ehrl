"""One plotting command per benchmark, with explicit analysis presets."""

from __future__ import annotations

import argparse
import ast
import json
import os
import runpy
import sys
from pathlib import Path

from remote import cache_path, fetch_run, parse_remote

ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = Path(__file__).resolve().parent
PRESETS = json.loads((Path(__file__).with_name("presets.json")).read_text())
TASK_LABELS = {"mbpp": "100", "mini_swe_bench": "100", "mastermind": "0-all"}


def method_options(path: Path) -> tuple[str, list[tuple[list[str], str]]]:
    """Describe an analysis without importing its plotting or cloud dependencies."""
    tree = ast.parse(path.read_text())
    options = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            help_text = next((kw.value.value for kw in node.keywords if kw.arg == "help" and isinstance(kw.value, ast.Constant)), "")
            if flags:
                options.append((flags, str(help_text)))
    return ast.get_docstring(tree) or path.stem, options


def main(benchmark: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"{benchmark} figures. PRESET selects the same analysis as --preset.",
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--preset", default=os.environ.get("PRESET", "training"), help="Plot name, or all to run every applicable preset")
    parser.add_argument("--run", help="Training run: local directory or SSH HOST:/path/to/run (also HOST:~/path)")
    parser.add_argument("--runs-dir", help="Evaluation results: local directory or SSH HOST:/path/to/results")
    parser.add_argument("--config", type=Path, help="Explicit comparison configuration")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/figures/remote", help="Local directory for downloaded SSH plotting inputs")
    parser.add_argument("--offline", action="store_true", help="Use cached SSH inputs without connecting; default is to refresh")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent plot processes with --preset all (default: 1)")
    parser.add_argument("--output-dir", type=Path, help="Override the directory containing this preset's outputs")
    parser.add_argument("--out", type=Path, help="Output filename for methods supporting --out")
    parser.add_argument("--list", action="store_true", help="List maintained presets")
    parser.add_argument("--resolve", action="store_true", help="Print the selected method and paths without running it")
    parser.add_argument("--help", "-h", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    presets = PRESETS[benchmark]
    if args.list:
        print("\n".join(sorted([*presets, "all"])))
        return 0
    if args.preset != "all" and args.preset not in presets:
        parser.error(f"unknown preset {args.preset!r}; use --list")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.preset != "all" and args.jobs != 1:
        parser.error("--jobs requires --preset all")
    script = ROOT / presets[args.preset] if args.preset != "all" else None
    doc, options = method_options(script) if script else ("Run all applicable presets, reporting missing inputs and failures. --output-dir is the parent of one directory per preset.", [])
    flags = {flag for names, _ in options for flag in names}
    if args.help:
        parser.print_help()
        print(f"\nSelected preset: {args.preset}\nMethod: {script.relative_to(ROOT) if script else 'all presets'}\n\n{doc}")
        if options:
            print("\nMethod options:")
            for names, help_text in options:
                print(f"  {', '.join(names)}  {help_text}")
        return 0
    trajectories = args.preset in ("all", "turns")

    def local_input(value, *, evaluation_only=False):
        if value is None:
            return None
        if parse_remote(value):
            if args.resolve:
                return cache_path(value, args.cache_dir)
            return fetch_run(value, args.cache_dir, trajectories=trajectories, offline=args.offline, evaluation_only=evaluation_only)
        return Path(value).expanduser().resolve()

    try:
        run = local_input(args.run)
        runs_dir = local_input(args.runs_dir, evaluation_only=True)
    except (ValueError, OSError) as error:
        parser.exit(1, f"error: {error}\n")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (run / "figures" / args.preset if run else ROOT / "examples" / benchmark / "figures" / "output" / args.preset)
    config = args.config.expanduser().resolve() if args.config else None
    if args.preset == "all":
        from batch import run_all

        if args.out:
            parser.error("--preset all writes multiple files; use --output-dir instead of --out")
        batch_args = ["--runs-dir", str(runs_dir), *remaining] if runs_dir else remaining
        return run_all(
            benchmark, presets, run=run, config=config,
            output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
            remaining=batch_args, jobs=args.jobs, resolve=args.resolve,
        )
    if args.resolve:
        print(
            json.dumps(
                {
                    "benchmark": benchmark,
                    "preset": args.preset,
                    "method": str(script),
                    "run": str(run) if run else None,
                    "remote_run": args.run if args.run and parse_remote(args.run) else None,
                    "runs_dir": str(runs_dir) if runs_dir else None,
                    "remote_runs_dir": args.runs_dir if args.runs_dir and parse_remote(args.runs_dir) else None,
                    "config": str(config) if config else None,
                    "output_dir": str(output_dir),
                    "arguments": remaining,
                },
                indent=2,
            )
        )
        return 0
    if run and not run.is_dir():
        parser.error(f"run directory does not exist: {run}")
    method_args = []
    if args.preset == "training":
        if run is None:
            parser.error("training requires --run /path/to/experiment")
        method_args += ["--run", str(run)]
    elif args.preset in ("eval", "turns", "iterations"):
        default_label = "auto" if run else TASK_LABELS.get(benchmark, "all")
        method_args += ["--benchmark", benchmark, "--task-label", os.environ.get("TASK_LABEL", default_label)]
        if runs_dir is None and config is None:
            if run is None:
                parser.error(f"{args.preset} requires --run /path/to/experiment or --runs-dir /path/to/evaluation/results")
            method_args += ["--runs-dir", str(run / "eval")]
        if args.preset == "iterations":
            method_args += ["--axis", "iteration"]
        if args.preset == "eval" and not any(arg.split("=", 1)[0] == "--metric" for arg in remaining):
            method_args += ["--metric", "max_reward_mean"]
    elif args.preset == "comparison":
        method_args += ["--benchmark", benchmark]
    if runs_dir:
        if "--runs-dir" not in flags:
            parser.error(f"{args.preset} does not use --runs-dir")
        method_args += ["--runs-dir", str(runs_dir)]
    if config:
        if "--config" not in flags:
            parser.error(f"{args.preset} does not use --config; see --help for its explicit input options")
        method_args += ["--config", str(config)]
    if args.out and "--out" not in flags:
        parser.error(f"{args.preset} writes named files; use --output-dir instead of --out")
    if "--out" in flags:
        destination = args.out.expanduser().resolve() if args.out else output_dir / f"{args.preset}.pdf"
        # Data extraction methods retain a CSV extension.
        if args.preset in ("extract", "train") and not args.out:
            destination = destination.with_suffix(".csv")
        method_args += ["--out", str(destination)]
    elif "--outdir" in flags and "--outdir" not in remaining:
        method_args += ["--outdir", str(output_dir)]
    elif "--out_dir" in flags and "--out_dir" not in remaining:
        method_args += ["--out_dir", str(output_dir)]
    if remaining and not options:
        parser.error(f"{args.preset} has no method-specific arguments: {' '.join(remaining)}")
    os.environ["FIGURE_OUTPUT_DIR"] = str(output_dir)
    os.environ["FIG_OUT_DIR"] = str(output_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path[:0] = [str(script.parent), str(COMMON_ROOT)]
    sys.argv = [str(script), *method_args, *remaining]
    runpy.run_path(str(script), run_name="__main__")
    return 0
