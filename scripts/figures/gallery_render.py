"""Draw gallery figures from extracted CSV tables.

This module never reads training or evaluation artifacts. It reads the
``training.csv``, ``evaluation.csv``, ``turns.csv`` and ``manifest.json``
written by ``quickplot`` and only arranges those values on axes.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import textwrap
import webbrowser
from pathlib import Path

from gallery_schema import BENCHMARKS, METHODS, PLOT_METHODS, PLOTS, TOKEN_AXES, read_tables

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "base": "Base", "grpo": "GRPO", "ppo": "PPO", "joint_ppo": "Joint-outcome PPO",
    "erl": "ERL", "ace_rl": "ACE-RL", "delayed_joint": "Delayed-joint",
    "alternating": "Alternating", "ace_n_evals": "ACE-RL, N evals",
}
COLORS = {
    "base": "#9e9e9e", "grpo": "#1f77b4", "ppo": "#6baed6", "joint_ppo": "#ff7f0e",
    "erl": "#2ca02c", "ace_rl": "#d62728", "delayed_joint": "#e377c2",
    "alternating": "#8c564b", "ace_n_evals": "#f4a6a6",
}
GRPO_RSI_COLOR = "#d4a900"


def _select(tables: dict, table: str, **match) -> list[dict]:
    return [row for row in tables[table] if all(row[key] == value for key, value in match.items())]


def _evaluation(tables: dict, method: str, measure: str, k: int | None = None) -> list[dict]:
    match = {"method": method, "measure": measure, **({} if k is None else {"k": k})}
    return sorted(_select(tables, "evaluation", **match), key=lambda row: row["step"])


def _with_base(rows: list[dict], baseline: list[dict]) -> list[dict]:
    """Attach a measured Base result to a measured method series at step 0."""
    if not rows:
        return rows
    return sorted({row["step"]: row for row in (*baseline, *rows)}.values(), key=lambda row: row["step"])


def _series_xy(rows: list[dict], token_axis: str | None, *, training: bool = False,
               values: list[float] | None = None) -> tuple[list[float], list[float]]:
    pairs = []
    for row, value in zip(rows, values if values is not None else [row["value"] for row in rows], strict=True):
        x = row["step"] if token_axis is None else row[f"cumulative_{token_axis}"]
        if x is not None:
            pairs.append((x, value))
    if token_axis is not None and not training:
        pairs.sort(key=lambda pair: pair[0])
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _active_comparison_methods(loaded: set[str], methods: set[str]) -> list[str]:
    core = {"grpo", "ace_rl", "delayed_joint"}
    return [method for method in METHODS if method != "base" and method in methods
            and (method in core or method in loaded)]


def _figure_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "lines.linewidth": 1.4,
    })


def _save_figure(fig, axes, out: Path, title: str, missing: list[str], *, colorbar: bool = False) -> None:
    import matplotlib.pyplot as plt
    if not any(ax.lines or ax.collections for ax in axes):
        axes[-1].text(0.5, 0.5, "No measured data yet", ha="center", va="center",
                      transform=axes[-1].transAxes, color="#666666")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles = {}
    for ax in axes:
        artists, labels = ax.get_legend_handles_labels()
        handles.update(zip(labels, artists, strict=True))
    if handles and not colorbar:
        if len(handles) > 6:
            fig.legend(list(handles.values()), list(handles), loc="center left",
                       bbox_to_anchor=(0.77, 0.52), fontsize=7, frameon=False)
        else:
            fig.legend(list(handles.values()), list(handles), loc="lower center",
                       bbox_to_anchor=(0.5, 0.09 if missing else 0.01),
                       ncol=min(len(handles), 3), fontsize=8, frameon=False)
    fig.suptitle(title)
    if missing:
        note = textwrap.fill("Missing: " + ", ".join(dict.fromkeys(missing)), width=100)
        fig.text(0.5, 0.01, note, fontsize=7, color="#666666", ha="center", va="bottom")
    bottom = 0.21 if missing and handles and len(handles) <= 6 else 0.10 if missing else 0.08
    right = 0.76 if len(handles) > 6 and not colorbar else 1
    fig.tight_layout(rect=(0, bottom, right, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(out.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_curves(name: str, benchmark: str, model: str, tables: dict, manifest: dict, out: Path,
                methods: set[str], *, token_axis: str | None = None) -> None:
    _figure_style()
    import matplotlib.pyplot as plt
    loaded = set(manifest.get("loaded_methods", []))
    missing = []
    xlabel = "Training step" if token_axis is None else f"Cumulative rollout/{token_axis}"
    if name == "learning":
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.8), sharex=True)
        for method in PLOT_METHODS[name]:
            if method not in methods:
                continue
            for ax, key in zip(axes, ("training", "ace"), strict=True):
                rows = (sorted(_select(tables, "training", method=method, metric="critic/score/mean"),
                               key=lambda row: row["record"])
                        if key == "training" else _evaluation(tables, method, "reward_carried"))
                if not rows:
                    missing.append(f"{LABELS[method]} {key}")
                    continue
                segments = sorted({row["segment"] for row in rows}) if key == "training" else [None]
                for segment in segments:
                    subset = [row for row in rows if segment is None or row["segment"] == segment]
                    values = [row["value"] for row in subset]
                    if key == "training":
                        values = [sum(values[max(0, i - 2):i + 3]) / len(values[max(0, i - 2):i + 3])
                                  for i in range(len(values))]
                    x, y = _series_xy(subset, token_axis, training=key == "training", values=values)
                    if x:
                        ax.plot(x, y, color=COLORS[method],
                                label=LABELS[method] if segment == segments[0] else None,
                                linewidth=2 if method == "ace_rl" else 1.4)
                    else:
                        missing.append(f"{LABELS[method]} {key} ({xlabel})")
        axes[0].set_ylabel("Train reward (5-step mean)")
        axes[1].set(xlabel=xlabel, ylabel="Held-out reward (success carried)")
    elif name == "comparison":
        fig, ax = plt.subplots(figsize=(8, 4.2))
        base_pass = _evaluation(tables, "base", "pass", 1)
        base_ace = _evaluation(tables, "base", "rsi", 3)
        for method in _active_comparison_methods(loaded, methods):
            pass_rows = _evaluation(tables, method, "pass", 1)
            ace_rows = _evaluation(tables, method, "rsi", 3)
            if pass_rows and not base_pass:
                missing.append("Base pass@1 step 0")
            if ace_rows and not base_ace:
                missing.append("Base RSI@3 step 0")
            pass_rows = _with_base(pass_rows, base_pass)
            ace_rows = _with_base(ace_rows, base_ace)
            for label, rows, style in (("pass@1", pass_rows, "--"), ("RSI@3", ace_rows, "-")):
                if rows:
                    x, y = _series_xy(rows, token_axis)
                    if x:
                        color = GRPO_RSI_COLOR if method == "grpo" and label.startswith("RSI@") else COLORS[method]
                        ax.plot(x, y, color=color, linestyle=style, marker="o", markersize=2.5,
                                label=f"{LABELS[method]} {label}")
                    else:
                        missing.append(f"{LABELS[method]} {label} ({xlabel})")
                else:
                    missing.append(f"{LABELS[method]} {label}")
        ax.set(xlabel=xlabel, ylabel="Any-success rate", ylim=(0, 1))
        axes = [ax]
    elif name == "comparison_extended":
        fig, ax = plt.subplots(figsize=(9, 5))
        styles = {1: "-", 2: "--", 4: ":", 8: "-."}
        for method in _active_comparison_methods(loaded, methods):
            for measure, evaluation in (("pass", "pass"), ("rsi", "RSI")):
                group_missing = []
                for k in (1, 2, 4, 8):
                    rows = _evaluation(tables, method, measure, k)
                    baseline = _evaluation(tables, "base", measure, k)
                    if rows and not baseline:
                        missing.append(f"Base {evaluation}@{k} step 0")
                    rows = _with_base(rows, baseline)
                    if not rows:
                        group_missing.append(f"{LABELS[method]} {evaluation}@{k}")
                        continue
                    x, y = _series_xy(rows, token_axis)
                    if x:
                        color = GRPO_RSI_COLOR if method == "grpo" and evaluation == "RSI" else COLORS[method]
                        ax.plot(x, y, color=color, linestyle=styles[k], marker="o", markersize=2.5,
                                label=f"{LABELS[method]} {evaluation}@{k}")
                    else:
                        group_missing.append(f"{LABELS[method]} {evaluation}@{k} ({xlabel})")
                if len(group_missing) == 4:
                    missing.append(f"{LABELS[method]} {evaluation}@{{1,2,4,8}}")
                else:
                    missing.extend(group_missing)
        ax.set(xlabel=xlabel, ylabel="Any-success rate", ylim=(0, 1))
        axes = [ax]
    elif name == "variants":
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for method in PLOT_METHODS[name]:
            if method not in methods:
                continue
            rows = _evaluation(tables, method, "reward_carried")
            if rows:
                x, y = _series_xy(rows, token_axis)
                if x:
                    ax.plot(x, y, color=COLORS[method], marker="o", markersize=2.5, label=LABELS[method])
                else:
                    missing.append(f"{LABELS[method]} ({xlabel})")
            else:
                missing.append(LABELS[method])
        ax.set(xlabel=xlabel, ylabel="Held-out reward (success carried)")
        axes = [ax]
    elif name == "rollout":
        rollout_metrics = manifest.get("rollout_metrics", {})
        selected = [method for method in PLOT_METHODS[name] if method in methods and method in loaded]
        available = {metric for method in selected for metric in rollout_metrics.get(method, [])}
        preferred = [f"rollout/{axis}" for axis in TOKEN_AXES] + [
            "rollout/total_prompt_tokens", "rollout/total_trajectories",
        ]
        metrics = [metric for metric in preferred if metric in available]
        metrics += sorted(available - set(metrics))
        fig, panels = plt.subplots(max(1, len(metrics)), 1,
                                   figsize=(8.5, 2.8 * max(1, len(metrics))), squeeze=False)
        axes = list(panels[:, 0])
        if not metrics:
            missing.append("rollout/* metrics")
        missing.extend(f"{LABELS[method]} rollout/*" for method in selected
                       if not rollout_metrics.get(method))
        for ax, metric in zip(axes, metrics):
            for method in selected:
                rows = sorted(_select(tables, "training", method=method, metric=metric),
                              key=lambda row: row["record"])
                if not rows:
                    if rollout_metrics.get(method):
                        missing.append(f"{LABELS[method]} {metric}")
                    continue
                plotted = False
                for segment in sorted({row["segment"] for row in rows}):
                    subset = [row for row in rows if row["segment"] == segment]
                    x, y = _series_xy(subset, token_axis, training=True)
                    if x:
                        ax.plot(x, y, color=COLORS[method], linewidth=1.2,
                                label=LABELS[method] if not plotted else None)
                        plotted = True
                if not plotted:
                    missing.append(f"{LABELS[method]} {metric} ({xlabel})")
            ax.set(ylabel="Per step")
            ax.set_title(metric, loc="left", fontsize=9)
        axes[-1].set_xlabel(xlabel)
    else:
        raise ValueError(f"Unknown figure {name}")
    title = f"{benchmark} · {model} · {name}"
    if token_axis is not None:
        title += f" · cumulative {token_axis}"
    _save_figure(fig, axes, out, title, missing)


def plot_turns_method(benchmark: str, model: str, method: str, tables: dict,
                      out: Path, *, extended: bool, nopad: bool = False) -> None:
    _figure_style()
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    key = "turns_extended" if extended else "turns"
    if nopad:
        key += "_nopad"
    curves: dict[int, list[dict]] = {}
    for row in _select(tables, "turns", method=method, curve=key):
        curves.setdefault(row["step"], []).append(row)
    own = bool(curves)
    base = [row for row in _select(tables, "turns", method="base", curve=key) if row["step"] == 0]
    if base:
        curves[0] = base
    curves = {step: sorted(rows, key=lambda row: row["turn_global"]) for step, rows in curves.items()}
    missing = []
    if 0 not in curves:
        missing.append("Base step 0")
    if not own:
        missing.append(f"{LABELS[method]} checkpoint turns")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    norm = Normalize(vmin=0, vmax=max(1, max(curves, default=1)))
    cmap = plt.get_cmap("viridis")
    for step, rows in sorted(curves.items()):
        ax.plot([row["turn_global"] for row in rows], [row["value"] for row in rows],
                color=cmap(norm(step)), linewidth=1.8 if step == 0 else 1.1,
                label=f"step {step}" if len(curves) <= 8 else None)
    boundaries = sorted({row["turn_global"] - 0.5 for rows in curves.values() for row in rows
                         if row["turn"] == 1 and (row["local_epoch"] or 1) > 1})
    for boundary in boundaries:
        ax.axvline(boundary, color="#777777", linestyle=":", linewidth=0.7)
    if curves:
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Checkpoint step", pad=0.02)
    ax.set(xlabel="Turn across adaptation visits",
           ylabel="Average reward (fresh visit)" if nopad else "Average reward (success carried)")
    suffix = " · extended RSI@8" if extended else " · RSI@3"
    if nopad:
        suffix += " · no across-visit padding"
    _save_figure(fig, [ax], out, f"{benchmark} · {model} · {LABELS[method]}{suffix}",
                 missing, colorbar=bool(curves))


def gallery(directory: Path, rendered: list[str]) -> Path:
    cards = "\n".join(
        f'<section><h2>{html.escape(name.title())}</h2><a href="{name}.pdf"><img src="{name}.png" alt="{html.escape(name)}"></a></section>'
        for name in rendered
    )
    path = directory / "index.html"
    path.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Figures</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;color:#222}'
        'img{width:100%;border:1px solid #ddd}section{margin:32px 0}a{color:inherit}</style>'
        f'</head><body><h1>Figures</h1>{cards}</body></html>'
    )
    return path


def render(data_dir: Path, destination: Path, plots: set[str] | None, methods: set[str] | None) -> list[str]:
    tables, manifest = read_tables(data_dir)
    benchmark, model = manifest["benchmark"], manifest["model"]
    extracted = set(manifest["plots"])
    names = extracted if plots is None else plots
    if names - extracted:
        raise ValueError(f"{data_dir} has no extracted data for {', '.join(sorted(names - extracted))}; "
                         "extract those plots first")
    chosen_methods = set(manifest["methods"]) if methods is None else methods & set(manifest["methods"])
    selected_runs = set(manifest.get("selected_runs", {}))
    rendered = []
    for name in PLOTS:
        if name not in names:
            continue
        if name == "turns":
            turn_methods = [method for method in METHODS
                            if method != "base" and method in selected_runs and method in chosen_methods]
            if not turn_methods:
                turn_methods = [method for method in ("grpo", "ace_rl", "delayed_joint") if method in chosen_methods]
            if not turn_methods:
                turn_methods = [method for method in METHODS if method != "base" and method in chosen_methods]
            for method in turn_methods:
                for extended, nopad in ((False, False), (False, True), (True, False), (True, True)):
                    title = f"turns_{method}" + ("_extended" if extended else "") + ("_nopad" if nopad else "")
                    plot_turns_method(benchmark, model, method, tables, destination / title,
                                      extended=extended, nopad=nopad)
                    rendered.append(title)
        else:
            for token_axis in (None, *TOKEN_AXES):
                title = name if token_axis is None else f"{name}_{token_axis}"
                plot_curves(name, benchmark, model, tables, manifest, destination / title, chosen_methods,
                            token_axis=token_axis)
                rendered.append(title)
    for obsolete in ("iterations", "turns", "motivation", "handoff", "staleness",
                     "passk", "passk_total_gradient_update_tokens", "passk_total_completion_tokens"):
        for suffix in (".pdf", ".png"):
            (destination / f"{obsolete}{suffix}").unlink(missing_ok=True)
    if "turns" in names:
        for path in destination.glob("turns_*"):
            if path.suffix in {".pdf", ".png"} and path.stem not in rendered:
                path.unlink()
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS),
                        help="Read examples/<benchmark>/figures/output/<model>/data")
    parser.add_argument("--model", help="Exact model directory; default: every extracted model")
    parser.add_argument("--data", type=Path, action="append",
                        help="An extracted data directory; repeat for several. Overrides --benchmark/--model")
    parser.add_argument("--plots", help="Comma-separated plot names; default: every extracted plot")
    parser.add_argument("--methods", help="Comma-separated training methods; default: every extracted method")
    parser.add_argument("--output-dir", type=Path,
                        help="Figure directory for a single data directory; default: the data directory's parent")
    parser.add_argument("--no-open", action="store_true", help="Do not open the gallery on a desktop")
    parser.add_argument("--list", action="store_true", help="List figure and training method names")
    args = parser.parse_args(argv)
    try:
        if args.list:
            print("Plots: " + ", ".join(PLOTS))
            print("Methods: " + ", ".join(METHODS))
            return 0
        plots = None if not args.plots else set(args.plots.split(","))
        methods = None if not args.methods else set(args.methods.split(","))
        if plots is not None and (not plots or plots - set(PLOTS)):
            raise ValueError(f"Unknown plots: {', '.join(sorted(plots - set(PLOTS)))}; choices: {', '.join(PLOTS)}")
        if methods is not None and (not methods or methods - set(METHODS)):
            raise ValueError(f"Unknown methods: {', '.join(sorted(methods - set(METHODS)))}; choices: {', '.join(METHODS)}")
        if args.data:
            directories = args.data
        elif args.benchmark:
            parent = ROOT / "examples" / args.benchmark / "figures" / "output"
            directories = (sorted(path.parent for path in parent.glob("*/data/manifest.json"))
                           if args.model in (None, "all") else [parent / args.model / "data"])
        else:
            raise ValueError("--benchmark or --data is required")
        if not directories:
            raise ValueError("No extracted data directories found; run `python plot.py <benchmark>` first")
        if args.output_dir and len(directories) > 1:
            raise ValueError("--output-dir needs exactly one data directory")
        for directory in directories:
            destination = args.output_dir or directory.parent
            rendered = render(directory, destination, plots, methods)
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest.get("failures"):
                print(f"{manifest['model']}: extraction recorded {len(manifest['failures'])} failure(s); "
                      f"see {directory / 'manifest.json'}", file=sys.stderr)
            if rendered:
                page = gallery(destination, rendered)
                print(f"{manifest['model']}: {', '.join(rendered)} → {page}")
                if not args.no_open and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                    webbrowser.open(page.as_uri())
        return 0
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
