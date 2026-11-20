"""Aggregate finished Mini-SWE vanilla trajectories for one checkpoint step.

Inputs are task artifacts plus W&B run-state metadata. Each response-budget,
turn-limit, and thinking configuration stays separate. Ambiguous checkpoint
steps and failed/incomplete W&B runs stop aggregation before CSV output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNS_DIR = REPO_ROOT / "examples" / "mini_swe_bench" / "logs" / "runs"
OUT_DIR = REPO_ROOT / "examples" / "mini_swe_bench" / "logs" / "experiments"

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mini_swe_bench")
MODEL_TAG = "Qwen3.5-4B_vanilla"

# Resubmit-command target (overridable for Gemini via --slurm / --model-name).
RESUBMIT_SLURM = "examples/mini_swe_bench/eval/vanilla/eval.slurm"
RESUBMIT_MODEL_NAME = "Qwen/Qwen3.5-4B"

# MRL buckets (relabeled cells for tooling parity with AppWorld). MML is fixed at
# Qwen3-4B native 40960 for mini-SWE and is cosmetic (the column label only).
MML_FIXED = 40960
CELLS = {
    "A": (16384, MML_FIXED),
    "B": (24576, MML_FIXED),
    "C": (32768, MML_FIXED),
}


def _compile_naming_re(model_tag: str) -> re.Pattern:
    # <tag>_step<N>_<MRL>_<MT>_dt<bool>_at<bool>_tasks<label>  (any step number).
    return re.compile(rf"{re.escape(model_tag)}_step(\d+)_(\d+)_(\d+)_dt(True|False)_at(True|False)_tasks(.+)")


_NAMING_RE = _compile_naming_re(MODEL_TAG)


def parse_run_dir(name: str):
    """Parse run-dir name → (step, mrl, mt, accumulate_thinking, disable_thinking, label) or None."""
    m = _NAMING_RE.fullmatch(name)
    if not m:
        return None
    return (
        int(m.group(1)),  # step
        int(m.group(2)),  # mrl
        int(m.group(3)),  # mt
        m.group(5) == "True",  # at
        m.group(4) == "True",  # dt
        m.group(6),  # label
    )


def cell_for_mrl(mrl: int):
    for name, (m, _) in CELLS.items():
        if m == mrl:
            return name
    return None


def read_wandb_id(run_dir: Path):
    path = run_dir / "wandb_run_id.txt"
    if not path.exists():
        return None
    return path.read_text().strip()


def fetch_wandb_state(api, wandb_id: str):
    try:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{wandb_id}")
    except Exception:
        return "missing", None
    return run.state, run.created_at


def load_trajectories(run_dir: Path) -> list[dict]:
    """Walk task_*/chat_completions/eval/policy_step*_iter_0.json → per-trajectory rows.

    Step-agnostic (globs the step), unlike the AppWorld loader which hardcodes
    step000000. Otherwise identical (the on-disk schema is shared).
    """
    rows: list[dict] = []
    for task_dir in sorted(run_dir.glob("task_*")):
        eval_dir = task_dir / "chat_completions" / "eval"
        if not eval_dir.is_dir():
            continue
        matches = sorted(eval_dir.glob("policy_step*_iter_0.json"))
        if not matches:
            continue
        json_path = matches[0]
        try:
            data = json.loads(json_path.read_text())
        except Exception as exc:
            print(f"  [warn] failed to parse {json_path}: {exc}")
            continue
        for traj in data.get("trajectories", []):
            steps = traj.get("steps") or []
            num_turns = len(steps)
            reward = traj.get("reward")
            reward_f = float(reward) if reward is not None else 0.0
            success = 1 if reward_f > 0 else 0
            info = traj.get("info") or {}
            termination = info.get("termination_reason")
            cur_tokens = 0
            last_info = {}
            if steps:
                last_info = steps[-1].get("info") or {}
                cur_tokens = int(last_info.get("cur_tokens", 0) or 0)
            # Submit signal: the mini-SWE env sets info["submitted"]=True ONLY in
            # the `submit` branch (env.py:418); the max-steps termination path
            # (env.py:468) never touches it and the env checks the turn cap AFTER
            # executing, so a real submit is never rejected. (Unlike AppWorld,
            # termination_reason can't be used here — max-step exhaustion is also
            # ENV_DONE.) Verified: submitted-flag == submit-in-action == submit_calls.
            submitted = 1 if last_info.get("submitted") is True else 0
            task_info = traj.get("task") or {}
            rows.append(
                {
                    "task_id": task_info.get("task_id") or task_info.get("instance_id") or task_dir.name.replace("task_", ""),
                    "task_dir_name": task_dir.name,
                    "num_turns": num_turns,
                    "success": success,
                    "reward": reward_f,
                    "termination_reason": termination,
                    "submitted": submitted,
                    "total_response_tokens": cur_tokens,
                    "avg_tokens_per_turn": round(cur_tokens / num_turns, 2) if num_turns > 0 else 0,
                }
            )
    return rows


def parse_csv_list(s, conv=str):
    if not s:
        return None
    return [conv(x.strip()) for x in s.split(",") if x.strip()]


def build_resubmit_cmd(mrl: int, mt: int, label: str, step: int, *, at: bool = False, dt: bool = False) -> str:
    env = [
        f"MAX_RESPONSE_LENGTH={mrl}",
        f"MAX_TURNS={mt}",
        f"MODEL_NAME={shlex.quote(RESUBMIT_MODEL_NAME)}",
        f"ACCUMULATE_THINKING={at}",
        f"DISABLE_THINKING={dt}",
        f"STEPS={step}",
    ]
    # label is "all", a count ("100"), or a range ("10-19").
    if label not in ("all",):
        if "-" in label:
            start, end = label.split("-", 1)
            offset = int(start)
            max_tasks = int(end) - offset + 1
            env.extend([f"TASK_OFFSET={offset}", f"MAX_TASKS={max_tasks}"])
        else:
            env.append(f"MAX_TASKS={label}")
    return " ".join(env) + f" \\\n      sbatch {shlex.quote(RESUBMIT_SLURM)}"


def main() -> int:
    global MODEL_TAG, _NAMING_RE, RESUBMIT_SLURM, RESUBMIT_MODEL_NAME, RUNS_DIR, OUT_DIR, WANDB_ENTITY, WANDB_PROJECT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tag", default=MODEL_TAG, help=f"Run-name model tag to aggregate (default: {MODEL_TAG}).")
    parser.add_argument("--out-suffix", default="", help="Suffix for output CSVs, e.g. 'qwen' -> summary_qwen.csv.")
    parser.add_argument("--cells", default=None, help="Comma-separated MRL-bucket cells to keep (A,B,C). Default: all found.")
    parser.add_argument("--max-turns", default=None, help="Comma-separated max_turns subset to keep. Default: all found.")
    parser.add_argument("--expected-tasks", type=int, default=None, help="Warn when a panel covers fewer than this many tasks (e.g. 100).")
    parser.add_argument("--include-running", action="store_true", help="Emit empty summary rows for panels with running chunks (still halts on failed/crashed chunks).")
    parser.add_argument("--slurm", default=RESUBMIT_SLURM, help="Slurm path used in printed resubmit commands.")
    parser.add_argument("--model-name", default=RESUBMIT_MODEL_NAME, help="MODEL_NAME used in printed resubmit commands.")
    parser.add_argument("--generator", default=None, help="Generator-model label for the summary 'Generator' column (e.g. 'gemini-2.5-pro' or 'Qwen 3-4B'). Default: derived from --model-tag.")
    parser.add_argument("--reflector", default="-", help="'Reflector/Curator' column label ('-' for vanilla; ACE model otherwise).")
    parser.add_argument("--experiment", default=None, help="'Experiment' column label. Default: '<generator> vanilla inference'.")
    parser.add_argument("--dataset", default="unspecified", help="'dataset' column value (subset/size), e.g. 'prop/100' or 'fullval/500'.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY, help="W&B team/user; defaults to WANDB_ENTITY")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--step", type=int, help="Checkpoint step; required when multiple steps match")
    args = parser.parse_args()
    if not args.wandb_entity:
        parser.error("Supply --wandb-entity or WANDB_ENTITY for run-state verification")
    RUNS_DIR, OUT_DIR = args.runs_dir, args.out_dir
    WANDB_ENTITY, WANDB_PROJECT = args.wandb_entity, args.wandb_project

    MODEL_TAG = args.model_tag
    _NAMING_RE = _compile_naming_re(MODEL_TAG)
    RESUBMIT_SLURM = args.slurm
    RESUBMIT_MODEL_NAME = args.model_name
    out_suffix = f"_{args.out_suffix}" if args.out_suffix else ""

    # Labels for the unified summary sheet (Experiment/Generator/Reflector columns).
    generator = args.generator or MODEL_TAG.replace("_vanilla", "")
    reflector = args.reflector
    experiment = args.experiment or f"{generator} vanilla inference"
    dataset = args.dataset

    cells_filter = parse_csv_list(args.cells)
    mt_filter = parse_csv_list(args.max_turns, int)

    if not RUNS_DIR.is_dir():
        print(f"[ERROR] runs dir not found: {RUNS_DIR}")
        return 1

    # ─── Phase 1: discover all matching run dirs, group into (cell, mt) panels ───
    # panel key = (cell, mt); value = list of (label, step, run_dir)
    panels: dict[tuple, list] = defaultdict(list)
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_run_dir(d.name)
        if parsed is None:
            continue
        step, mrl, mt, at, dt, label = parsed
        if args.step is not None and step != args.step:
            continue
        cell = cell_for_mrl(mrl)
        if cell is None:
            print(f"  [skip] {d.name}: MRL {mrl} not in CELLS")
            continue
        if cells_filter and cell not in cells_filter:
            continue
        if mt_filter and mt not in mt_filter:
            continue
        panels[(cell, mt, at, dt)].append((label, step, d))

    if not panels:
        print(f"[ERROR] no run dirs matched model-tag '{MODEL_TAG}' in {RUNS_DIR}")
        return 1

    matched_steps = {step for chunks in panels.values() for _, step, _ in chunks}
    if len(matched_steps) > 1:
        print(f"[ERROR] multiple checkpoint steps matched: {sorted(matched_steps)}; select --step to avoid merging different weights")
        return 2
    import pandas as pd
    import wandb

    api = wandb.Api()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Phase 2: verify each discovered chunk's wandb state ───
    failed: list[str] = []
    running: list[tuple] = []
    chosen: dict[tuple, list] = defaultdict(list)  # (cell,mt,at,dt) -> [(label, run_dir, wb_id)]

    for (cell, mt, at, dt), chunks in sorted(panels.items()):
        # de-dup labels: keep latest mtime per label
        by_label: dict[str, Path] = {}
        for label, step, d in chunks:
            if label not in by_label or d.stat().st_mtime > by_label[label].stat().st_mtime:
                by_label[label] = d
        for label, d in sorted(by_label.items()):
            desc = f"cell={cell}, max_turns={mt}, at={at}, dt={dt}, tasks{label}"
            step, mrl = parse_run_dir(d.name)[0], CELLS[cell][0]
            wb_id = read_wandb_id(d)
            if wb_id is None:
                failed.append(f"{desc}:  [wandb_run_id.txt not in {d.name}]  (step={step})")
                continue
            state, _ = fetch_wandb_state(api, wb_id)
            if state == "running":
                if args.include_running:
                    running.append((cell, mt, at, dt, label, wb_id))
                else:
                    failed.append(f"{desc}:  wandb {wb_id} state=running (use --include-running)")
                continue
            if state != "finished":
                failed.append(f"{desc}:  wandb {wb_id} state={state}  (step={step})")
                continue
            chosen[(cell, mt, at, dt)].append((label, d, wb_id))

    # ─── Phase 3: report + strict halt on failures ───
    print("=== Mini-SWE Trajectory Shape Aggregator (discovery mode) ===")
    n_panels = len(panels)
    n_chunks_ok = sum(len(v) for v in chosen.values())
    print(f"Discovered {n_panels} panels (cell × max_turns), {n_chunks_ok} finished chunks")
    if running:
        print(f"[RUNNING] {len(running)} chunks still in flight (empty summary rows emitted):")
        for cell, mt, at, dt, label, wb_id in running:
            print(f"  cell={cell}, max_turns={mt}, tasks{label} -> wandb {wb_id}")

    if failed:
        print(f"\n[FAILED] {len(failed)} chunks need attention:")
        for f in failed:
            print(f"  {f}")
        print(f"\nHALT: {len(failed)} chunks failed/incomplete. CSVs not written.")
        print("\nResubmit examples:")
        for entry in failed:
            m = re.match(
                r"cell=(\w+), max_turns=(\d+), at=(True|False), dt=(True|False), tasks([^\s:]+):.*step=(\d+)",
                entry,
            )
            if not m:
                continue
            cell, mt_int, label, step = m.group(1), int(m.group(2)), m.group(5), int(m.group(6))
            mrl = CELLS[cell][0]
            print(f"\n  # {entry}")
            print("  " + build_resubmit_cmd(mrl, mt_int, label, step, at=m.group(3) == "True", dt=m.group(4) == "True"))
        return 1

    # ─── Phase 4: load trajectories + emit CSVs ───
    print(f"\n=== Loading trajectories from {n_chunks_ok} run dirs ===")
    all_rows: list[dict] = []
    panel_running: dict[tuple, list] = defaultdict(list)
    for cell, mt, at, dt, label, wb_id in running:
        panel_running[(cell, mt, at, dt)].append(wb_id)
    panel_ids: dict[tuple, list] = defaultdict(list)

    for (cell, mt, at, dt), entries in chosen.items():
        mrl, mml = CELLS[cell]
        for label, run_dir, wb_id in entries:
            panel_ids[(cell, mt, at, dt)].append(wb_id)
            rows = load_trajectories(run_dir)
            for r in rows:
                r.update(
                    {
                        "cell": cell,
                        "max_response_length": mrl,
                        "max_model_length": mml,
                        "max_turns": mt,
                        "disable_thinking": dt,
                        "accumulate_thinking": at,
                        "task_label": label,
                        "wandb_id": wb_id,
                        "run_dir": run_dir.name,
                    }
                )
            all_rows.extend(rows)

    if not all_rows and not panel_running:
        print("[ERROR] no trajectories loaded.")
        return 1

    cols = [
        "cell",
        "max_response_length",
        "max_model_length",
        "max_turns",
        "disable_thinking",
        "accumulate_thinking",
        "task_label",
        "task_id",
        "wandb_id",
        "num_turns",
        "success",
        "reward",
        "termination_reason",
        "submitted",
        "total_response_tokens",
        "avg_tokens_per_turn",
    ]
    traj_df = pd.DataFrame(all_rows)[cols] if all_rows else pd.DataFrame(columns=cols)
    traj_csv = OUT_DIR / f"trajectory_distribution{out_suffix}.csv"
    traj_df.to_csv(traj_csv, index=False)
    print(f"=== Wrote {traj_csv.name} ({len(traj_df)} rows) ===")

    # ─── Summary CSV (per panel) ───
    timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%dT%H:%M:%S%z")
    timestamp = timestamp[:-2] + ":" + timestamp[-2:]
    summary_rows = []
    no_counter = 1
    all_keys = sorted(set(list(chosen.keys()) + list(panel_running.keys())))
    for cell, mt, at, dt in all_keys:
        mrl, mml = CELLS[cell]
        running_ids = panel_running.get((cell, mt, at, dt), [])
        finished_ids = panel_ids.get((cell, mt, at, dt), [])
        base = {
            "No.": no_counter,
            "Experiment": experiment,
            "Generator": generator,
            "Reflector/Curator": reflector,
            "dataset": dataset,
            "Group": cell,
            "wandb_id": ";".join(finished_ids + running_ids),
            "time": timestamp,
            "max_response_length": mrl,
            "max_model_length": mml,
            "max_turns": mt,
            "disable_thinking": dt,
            "accumulate_thinking": at,
        }
        if running_ids and not finished_ids:
            summary_rows.append(
                {
                    **base,
                    "num_total": "",
                    "num_success": "",
                    "num_fail": "",
                    "success_rate": "",
                    "turns_p90_success": "",
                    "turns_p90_failure": "",
                    "turns_p90_overall": "",
                    "tokens_per_turn_mean": "",
                    "submit_rate_overall": "",
                    "submit_rate_success": "",
                    "submit_rate_failed": "",
                }
            )
            no_counter += 1
            continue
        panel = traj_df[(traj_df.cell == cell) & (traj_df.max_turns == mt) & (traj_df.accumulate_thinking == at) & (traj_df.disable_thinking == dt)]
        if len(panel) == 0:
            continue
        successes = panel[panel.success == 1].num_turns
        failures = panel[panel.success == 0].num_turns
        n_total = len(panel)
        n_success = int(panel.success.sum())
        n_fail = n_total - n_success
        valid = panel[panel.num_turns > 0]
        tpt = (valid.total_response_tokens / valid.num_turns).mean() if len(valid) else 0.0
        if args.expected_tasks and n_total < args.expected_tasks:
            print(f"[WARN] panel cell={cell} mt={mt}: {n_total} tasks < expected {args.expected_tasks}")
        summary_rows.append(
            {
                **base,
                "num_total": n_total,
                "num_success": n_success,
                "num_fail": n_fail,
                "success_rate": round(n_success / n_total, 4) if n_total else 0.0,
                "turns_p90_success": round(successes.quantile(0.90), 2) if len(successes) else 0,
                "turns_p90_failure": round(failures.quantile(0.90), 2) if len(failures) else 0,
                "turns_p90_overall": round(panel.num_turns.quantile(0.90), 2),
                "tokens_per_turn_mean": round(tpt, 2),
                # Submit rates (see load_trajectories: submitted==1 iff info["submitted"]).
                "submit_rate_overall": round(panel.submitted.mean(), 4) if n_total else 0.0,
                "submit_rate_success": round(panel[panel.success == 1].submitted.mean(), 4) if n_success else 0.0,
                "submit_rate_failed": round(panel[panel.success == 0].submitted.mean(), 4) if n_fail else 0.0,
            }
        )
        no_counter += 1

    summary_csv = OUT_DIR / f"summary{out_suffix}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"=== Wrote {summary_csv.name} ({len(summary_rows)} rows) ===")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
