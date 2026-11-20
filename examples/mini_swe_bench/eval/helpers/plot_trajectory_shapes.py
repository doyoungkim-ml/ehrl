"""Plot trajectory-shape distributions from trajectory_distribution.csv.

Renders two figure variants from the same data:

  trajectory_distribution.pdf          — vector, print/paper quality
  trajectory_distribution.png          — 600 dpi raster, matches the PDF
  trajectory_distribution_sheets.png   — smaller dims + larger fonts, tuned
                                           for legibility when inserted into
                                           Google Sheets cells (heavy downscale)

Faceted grid: rows = experiment cells (auto-detected from summary CSV),
cols = max_turns. Per panel: success/failure histograms + KDE + three p90
dashed lines.

Shared across benchmarks: --exp-dir defaults to the mini-SWE experiments dir
(examples/mini_swe_bench/logs/experiments); point it at another benchmark's
experiments dir to plot that benchmark's CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[4]
EXP_DIR = REPO_ROOT / "examples" / "mini_swe_bench" / "logs" / "experiments"

DEFAULT_CELLS = ["A", "B", "C"]
MRL_BY_CELL = {"A": 16384, "B": 24576, "C": 32768}

COLOR_SUCCESS = "#1f77b4"  # matplotlib C0 blue
COLOR_FAILURE = "#d62728"  # matplotlib C3 red
COLOR_OVERALL = "#7f7f7f"  # gray

KDE_LW = 1.0
P90_LW = 0.8
P90_LS = (0, (4, 2))  # dashed: 4 on, 2 off


def plot_outcome(ax, turn_values, max_turns, color, label, zorder=2):
    """Histogram bars + KDE overlay. Always emits a legend handle (even N=0)."""
    n = len(turn_values)
    if n == 0:
        ax.plot([], [], color=color, linewidth=KDE_LW, label=f"{label} (N=0)")
        return
    bins = np.arange(1, max_turns + 2) - 0.5
    counts, edges = np.histogram(turn_values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, counts, width=0.8, color=color, alpha=0.35, edgecolor="none", zorder=zorder)
    if n >= 5 and np.var(turn_values) > 0.1:
        try:
            kde = gaussian_kde(turn_values, bw_method=0.3)
            x = np.linspace(1, max_turns, 200)
            y = kde(x) * n
            ax.plot(x, y, color=color, linewidth=KDE_LW, label=f"{label} (N={n})", zorder=zorder + 1)
            return
        except Exception:
            pass
    ax.plot([], [], color=color, linewidth=KDE_LW, label=f"{label} (N={n})")


def nice_xticks(mt):
    """Pick x-axis ticks that always include max_turns at the right edge."""
    if mt <= 20:
        step = 5
    elif mt <= 50:
        step = 10
    else:
        step = 15
    ticks = list(range(0, mt + 1, step))
    if mt not in ticks:
        ticks.append(mt)
    return ticks


def plot_p90_line(ax, values, color, label):
    """Always emits legend entry so legend size is consistent.

    When the outcome has N=0 trajectories, p90 is reported as 0 (consistent
    with the CSV) and no axvline is drawn (no data to anchor it).
    """
    if len(values) == 0:
        ax.plot([], [], color=color, linestyle=P90_LS, linewidth=P90_LW, alpha=0.9, label=f"{label} p90 (0)")
        return
    p90 = np.percentile(values, 90)
    ax.axvline(p90, color=color, linestyle=P90_LS, linewidth=P90_LW, alpha=0.9, label=f"{label} p90 ({int(round(p90))})")


def render_figure(
    df: pd.DataFrame,
    running_panels: set,
    present_cells: list,
    present_mts: list,
    *,
    panel_w: float,
    panel_h: float,
    fs_suptitle: int,
    fs_title: int,
    fs_label: int,
    fs_tick: int,
    fs_legend: int,
    fs_anno: int,
    fs_placeholder: int,
    out_pdf: Path | None,
    out_png: Path | None,
    dpi: int,
    title: str = "Trajectory Distribution (Mini-SWE vanilla)",
):
    """Build + save one figure variant. Same data, parameterized styling."""
    n_rows = max(1, len(present_cells))
    n_cols = max(1, len(present_mts))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * panel_w, n_rows * panel_h),
        sharex=False,
        sharey="row",
        squeeze=False,
        gridspec_kw={"wspace": 0.22, "hspace": 0.42},
    )

    for r, cell in enumerate(present_cells):
        for c, mt in enumerate(present_mts):
            ax = axes[r, c]

            if (cell, mt) in running_panels:
                mrl = MRL_BY_CELL.get(cell, "?")
                ax.text(0.5, 0.5, "running", ha="center", va="center", transform=ax.transAxes, fontsize=fs_placeholder, color="gray", style="italic")
                ax.set_title(f"{mrl}/{mt}/—", fontsize=fs_title)
                ax.text(0.5, -0.22, "avg tok/turn = —", transform=ax.transAxes, ha="center", va="top", fontsize=fs_anno, color="gray")
                ax.set_xlim(0.5, mt + 0.5)
                ax.set_xticks(nice_xticks(mt))
                ax.tick_params(axis="both", labelsize=fs_tick, left=True, labelleft=True)
                continue

            panel = df[(df.cell == cell) & (df.max_turns == mt)]
            if len(panel) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{cell}, MT={mt}", fontsize=fs_title)
                ax.set_xlim(0.5, mt + 0.5)
                continue

            successes = panel[panel.success == 1].num_turns.values
            failures = panel[panel.success == 0].num_turns.values

            plot_outcome(ax, successes, mt, COLOR_SUCCESS, "success", zorder=4)
            plot_outcome(ax, failures, mt, COLOR_FAILURE, "failure", zorder=2)
            plot_p90_line(ax, successes, COLOR_SUCCESS, "success")
            plot_p90_line(ax, failures, COLOR_FAILURE, "failure")
            plot_p90_line(ax, panel.num_turns.values, COLOR_OVERALL, "overall")

            mrl = int(panel.max_response_length.iloc[0])
            sr = len(successes) / len(panel) if len(panel) else 0.0
            ax.set_title(f"{mrl}/{mt}/{sr:.1%}", fontsize=fs_title)

            valid = panel[panel.num_turns > 0]
            panel_tpt = (valid.total_response_tokens / valid.num_turns).mean() if len(valid) else float("nan")
            tpt_str = f"{panel_tpt:.2f}" if not np.isnan(panel_tpt) else "—"
            ax.text(0.5, -0.22, f"avg tok/turn = {tpt_str}", transform=ax.transAxes, ha="center", va="top", fontsize=fs_anno, color="gray")

            ax.set_xlim(0.5, mt + 0.5)
            ax.set_xticks(nice_xticks(mt))
            ax.tick_params(axis="both", labelsize=fs_tick, left=True, labelleft=True)
            ax.legend(fontsize=fs_legend, loc="upper right", framealpha=0.85)

    for r in range(n_rows):
        axes[r, 0].set_ylabel("# trajectories", fontsize=fs_label)
    for c in range(n_cols):
        axes[-1, c].set_xlabel("# turns", fontsize=fs_label)

    fig.suptitle(title, fontsize=fs_suptitle)
    fig.subplots_adjust(top=1.0 - 0.8 / (n_rows * panel_h), bottom=0.09, left=0.05, right=0.97)

    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Wrote {out_pdf}")
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out_png} (dpi={dpi})")
    plt.close(fig)


def select_thinking_variant(frame, disable_thinking, accumulate_thinking):
    """Refuse to combine distinct thinking configurations into one panel."""
    columns = []
    for column, requested in (("disable_thinking", disable_thinking), ("accumulate_thinking", accumulate_thinking)):
        if column not in frame.columns:
            if requested is not None:
                raise ValueError(f"Cannot select {column}: input has no such column")
            continue
        columns.append(column)
        normalized = frame[column].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if normalized.isna().any():
            raise ValueError(f"Unrecognized values in {column}")
        if requested is not None:
            frame = frame[normalized == (requested == "true")]
    if columns and len(frame[columns].drop_duplicates()) > 1:
        raise ValueError("Multiple thinking configurations found; select --disable-thinking and/or --accumulate-thinking")
    return frame


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suffix",
        default="",
        help="Input/output CSV+figure suffix, e.g. 'gemini' → reads trajectory_distribution_gemini.csv and writes trajectory_distribution_gemini.{pdf,png}. Default '' → Qwen sweep.",
    )
    parser.add_argument(
        "--exp-dir",
        default=None,
        help="Experiments dir to read CSVs from and write figures to. Default: "
        "the mini-SWE experiments dir "
        "(examples/mini_swe_bench/logs/experiments). For AppWorld, point at "
        "that benchmark's experiments dir.",
    )
    parser.add_argument(
        "--title",
        default="Trajectory Distribution (Mini-SWE vanilla)",
        help="Figure suptitle (set per benchmark/model, e.g. 'Trajectory Distribution (AppWorld vanilla, Gemini-2.5-pro)').",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="Output figure base name (overrides the suffix-derived name), e.g. "
        "trajectory_distribution_appworld_qwen3-4b_vanilla_all. The input CSVs "
        "still use --suffix; only the .pdf/.png/_sheets.png names change.",
    )
    parser.add_argument("--disable-thinking", choices=["true", "false"], help="Select one generator thinking configuration")
    parser.add_argument("--accumulate-thinking", choices=["true", "false"], help="Select whether prior-turn reasoning is retained")
    args = parser.parse_args()
    suf = f"_{args.suffix}" if args.suffix else ""
    exp_dir = Path(args.exp_dir) if args.exp_dir else EXP_DIR

    traj_csv = exp_dir / f"trajectory_distribution{suf}.csv"
    summary_csv = exp_dir / f"summary{suf}.csv"
    variant_suffix = ""
    if args.disable_thinking is not None:
        variant_suffix += f"_dt{args.disable_thinking}"
    if args.accumulate_thinking is not None:
        variant_suffix += f"_at{args.accumulate_thinking}"
    out_base = args.out_name or f"trajectory_distribution{suf}{variant_suffix}"
    out_pdf = exp_dir / f"{out_base}.pdf"
    out_png = exp_dir / f"{out_base}.png"
    out_png_sheets = exp_dir / f"{out_base}_sheets.png"

    if not traj_csv.exists():
        print(f"[ERROR] {traj_csv} not found. Run aggregator first.")
        return 1

    try:
        df = select_thinking_variant(pd.read_csv(traj_csv), args.disable_thinking, args.accumulate_thinking)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Loaded {len(df)} trajectories from {traj_csv.name}")

    if summary_csv.exists():
        try:
            summary_df = select_thinking_variant(pd.read_csv(summary_csv), args.disable_thinking, args.accumulate_thinking)
        except ValueError as exc:
            parser.error(str(exc))
        # Summary column was renamed cell -> Group; fall back to cell for old CSVs.
        gcol = "Group" if "Group" in summary_df.columns else "cell"
        present_cells = [c for c in DEFAULT_CELLS if c in summary_df[gcol].unique()]
        present_mts = sorted(summary_df.max_turns.unique().tolist())
        running_panels = set()
        for _, row in summary_df.iterrows():
            if pd.isna(row.get("num_total")) or row.get("num_total") == "":
                running_panels.add((row[gcol], int(row["max_turns"])))
        print(f"Grid from summary: {len(present_cells)} cells × {len(present_mts)} max_turns; {len(running_panels)} running")
    else:
        present_cells = [c for c in DEFAULT_CELLS if c in df.cell.unique()]
        present_mts = sorted(df.max_turns.unique().tolist())
        running_panels = set()
        print(f"Grid auto-detected: {len(present_cells)} cells × {len(present_mts)} max_turns")

    if not present_cells or not present_mts:
        parser.error("The selected configuration contains no panels")
    if variant_suffix:
        labels = []
        if args.disable_thinking is not None:
            labels.append("thinking off" if args.disable_thinking == "true" else "thinking on")
        if args.accumulate_thinking is not None:
            labels.append("retain reasoning on" if args.accumulate_thinking == "true" else "retain reasoning off")
        args.title += "\n" + "; ".join(labels)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Print/PDF variant: 4.0 × 3.4 per panel, tighter inter-panel spacing,
    # larger fontsizes than before.
    render_figure(
        df,
        running_panels,
        present_cells,
        present_mts,
        panel_w=4.0,
        panel_h=3.4,
        fs_suptitle=14,
        fs_title=10,
        fs_label=10,
        fs_tick=9,
        fs_legend=8,
        fs_anno=11,
        fs_placeholder=14,
        out_pdf=out_pdf,
        out_png=out_png,
        dpi=600,
        title=args.title,
    )

    # Sheets variant: same visual proportions as print PNG, only lower DPI so
    # Sheets has fewer pixels to downsample.
    render_figure(
        df,
        running_panels,
        present_cells,
        present_mts,
        panel_w=4.0,
        panel_h=3.4,
        fs_suptitle=14,
        fs_title=10,
        fs_label=10,
        fs_tick=9,
        fs_legend=8,
        fs_anno=11,
        fs_placeholder=14,
        out_pdf=None,
        out_png=out_png_sheets,
        dpi=100,
        title=args.title,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
