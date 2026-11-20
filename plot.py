#!/usr/bin/env python3
"""Plot experiments in one command.

    python plot.py appworld                      # gallery of every local appworld run
    python plot.py --run RUN_DIR                 # training curve of one run
    python plot.py --run RUN_DIR --preset eval   # one run's checkpoint evaluations
    python plot.py appworld --preset comparison --config comparison.json

The gallery reads runs under examples/<benchmark>/train/, writes CSV tables to
examples/<benchmark>/figures/output/<model>/data/ and figures plus index.html
beside them. Single-run figures go to RUN_DIR/figures/<preset>/.
See docs/visualization.md.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCHMARKS = ("appworld", "mbpp", "mini_swe_bench", "mastermind")
plotting_python = ROOT / "environments/plotting/bin/python"
if (importlib.util.find_spec("matplotlib") is None and plotting_python.is_file()
        and Path(sys.executable).resolve() != plotting_python.resolve()):
    os.execv(str(plotting_python), [str(plotting_python), str(ROOT / "plot.py"), *sys.argv[1:]])
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

RUN_OPTIONS = ("--run", "--runs-dir", "--config", "--preset", "--list-presets")


def run_benchmark(run: str) -> str | None:
    """A run path names its benchmark: .../examples/<benchmark>/train/..."""
    parts = Path(run.split(":", 1)[-1]).parts
    return next((parts[index + 1] for index, part in enumerate(parts[:-1])
                 if part == "examples" and parts[index + 1] in BENCHMARKS), None)


def gallery(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Single-run options: python plot.py --run RUN_DIR --preset NAME --help")
    parser.add_argument("benchmark", nargs="?", choices=BENCHMARKS)
    parser.add_argument("--model", help="Exact run model directory; default: every model with runs")
    parser.add_argument("--plots", help="Comma-separated plot names; default: all")
    parser.add_argument("--methods", help="Comma-separated training methods; default: all")
    parser.add_argument("--source", default="local", help="local (default) or an SSH host holding a checkout")
    parser.add_argument("--source-root", help="Checkout root on --source; default: RLLM_PLOT_<HOST>_ROOT")
    parser.add_argument("--offline", action="store_true", help="Reuse cached SSH rows without connecting")
    parser.add_argument("--full-turn-check", action="store_true", help="Recheck every trajectory before reusing cached SSH turn curves")
    parser.add_argument("--append", action="store_true", help="Merge into existing extracted data instead of replacing it")
    parser.add_argument("--no-open", action="store_true", help="Do not open the gallery in a browser")
    parser.add_argument("--list", action="store_true", help="List gallery plot and method names")
    args = parser.parse_args(argv)
    if args.list:
        from gallery_schema import METHODS, PLOTS

        print("Plots: " + ", ".join(PLOTS))
        print("Methods: " + ", ".join(METHODS))
        return 0
    if args.benchmark is None:
        parser.error("name a benchmark, or use --run RUN_DIR; see --help")
    from gallery_render import main as render
    from quickplot import main as extract

    selection = ["--plots", args.plots] if args.plots else ["--all"]
    shared = [*(["--model", args.model] if args.model else []), *(["--methods", args.methods] if args.methods else [])]
    extract_args = ["--source", f"{args.source}:{args.benchmark}", *selection, *shared]
    extract_args += [*(["--source-root", args.source_root] if args.source_root else []),
                     *(["--offline"] if args.offline else []), *(["--full-turn-check"] if args.full_turn_check else []),
                     *(["--append"] if args.append else [])]
    status = extract(extract_args)
    render_args = ["--benchmark", args.benchmark, *(["--plots", args.plots] if args.plots else []), *shared]
    return render([*render_args, *(["--no-open"] if args.no_open else [])]) or status


def single_run(argv: list[str]) -> int:
    named = argv[0] if argv and argv[0] in BENCHMARKS else None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run")
    parser.add_argument("--list-presets", action="store_true")
    args, remaining = parser.parse_known_args(argv[1:] if named else argv)
    benchmark = named or (run_benchmark(args.run) if args.run else None)
    if benchmark is None:
        parser.exit(2, "error: name the benchmark, e.g. python plot.py appworld --preset comparison --config FILE\n")
    from entrypoint import main

    forwarded = [*(["--run", args.run] if args.run else []), *(["--list"] if args.list_presets else []), *remaining]
    return main(benchmark, forwarded)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if any(arg.split("=", 1)[0] in RUN_OPTIONS for arg in argv):
        raise SystemExit(single_run(argv))
    raise SystemExit(gallery(argv))
