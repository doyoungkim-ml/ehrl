#!/usr/bin/env python3
"""Phase-2 verdict A/B for the SWE-Gym eval-script speed transforms (2026-08-03).

For N sampled graded trajectories from episode logs: reconstruct the patched
testbed (fresh base + git apply of the recorded agent diff), grade it TWICE —
stock eval_script vs apply_all_speed_transforms — and compare verdicts.

GATE (what must be 100%): stock-vs-transformed agreement on
    (resolved, patch_successfully_applied, FTP success/failure sets,
     PTP success/failure sets)
per sample. Both sides grade the IDENTICAL reconstructed state, so this is
internally valid even where reconstruction differs from the live rollout
(untracked created files are absent from `git diff HEAD`); agreement of the
stock verdict with the ORIGINAL training outcome is reported as a bonus
diagnostic only, NOT gated.

Usage (inside the slurm wrapper, from RLLM_DIR):
  python examples/mini_swe_bench/eval/helpers/regrade_ab_swegym.py \
      --n 100 --workers 12 --out /scratch/$USER/evalab_phase2
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import shlex
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from examples.mini_swe_bench.shared.eval_transforms import (
    apply_all_speed_transforms,
)

CLASS_QUOTA = {"resolved": 40, "applied": 50, "patch_not_applied": 10}


def classify(wf_reason: str, env_reason: str) -> str:
    if wf_reason == "resolved":
        return "resolved"
    if wf_reason == "tests_failed":
        return "patch_not_applied" if "patch_not_applied" in (env_reason or "") else "applied"
    return "other"


def collect_samples(n_target: int, experiment_dirs: list[str], seed: int = 7) -> list[dict]:
    pools: dict[str, list[dict]] = defaultdict(list)
    for exp in experiment_dirs:
        for f in glob.glob(f"{exp}/episodes/train_step_*/episode_*.json"):
            try:
                e = json.load(open(f))
            except Exception:
                continue
            t = (e.get("trajectories") or [{}])[0]
            steps = t.get("steps") or []
            if not steps:
                continue
            li = steps[-1].get("info") or {}
            diff = (li.get("git_diff") or "").strip()
            info = t.get("info") or {}
            cls = classify(info.get("reward_reason", ""), info.get("env_reward_reason", ""))
            if cls == "other" or not diff:
                continue
            inst = (e.get("task") or {}).get("instance") or e.get("task") or {}
            if not inst.get("instance_id"):
                continue
            pools[cls].append(
                {
                    "file": f,
                    "cls": cls,
                    "instance": dict(inst),
                    "agent_diff": diff,
                    "orig_reward": t.get("reward"),
                    "orig_reason": info.get("reward_reason"),
                }
            )
    rng = random.Random(seed)
    picked: list[dict] = []
    scale = n_target / sum(CLASS_QUOTA.values())
    for cls, quota in CLASS_QUOTA.items():
        pool = pools[cls]
        rng.shuffle(pool)
        # dedup by instance_id within a class to broaden coverage
        seen, uniq = set(), []
        for s in pool:
            iid = s["instance"]["instance_id"]
            if iid in seen:
                continue
            seen.add(iid)
            uniq.append(s)
        take = min(len(uniq), max(1, round(quota * scale)))
        picked.extend(uniq[:take])
        print(f"[sample] {cls}: {take} of {len(uniq)} unique (pool {len(pool)})", flush=True)
    return picked


def verdict_from_report(report: dict, iid: str) -> dict:
    inst = report.get(iid, {})
    ts = inst.get("tests_status") or {}
    ftp = ts.get("FAIL_TO_PASS") or {}
    ptp = ts.get("PASS_TO_PASS") or {}
    return {
        "resolved": bool(inst.get("resolved")),
        "applied": bool(inst.get("patch_successfully_applied")),
        "ftp_success": sorted(ftp.get("success", [])),
        "ftp_failure": sorted(ftp.get("failure", [])),
        "ptp_success": sorted(ptp.get("success", [])),
        "ptp_failure": sorted(ptp.get("failure", [])),
    }


def grade_once(sample: dict, variant: str) -> dict:
    from swegym.harness.grading import get_eval_report
    from swegym.harness.test_spec import make_test_spec

    from examples.mini_swe_bench.env import MiniSWEnv

    instance = sample["instance"]
    iid = instance["instance_id"]
    diff = sample["agent_diff"]
    env = MiniSWEnv(task={"instance": instance, "data_source": "SumanthRH/SWE-Gym-Subset"}, reward_timeout=600)
    try:
        env._init_host_testbed()
        if not env._host_testbed:
            return {"error": "testbed_init_failed"}
        Path(env._host_testbed, ".agent.patch").write_text(diff + "\n")
        out, code = env._exec(
            f"cd {env.DOCKER_WORKDIR} && git apply .agent.patch && rm .agent.patch",
            timeout=120,
        )
        if code != 0:
            return {"error": "agent_diff_apply_failed", "detail": (out or "")[-300:]}
        spec = make_test_spec(instance)
        script = spec.eval_script
        if variant == "transformed":
            script = apply_all_speed_transforms(script, diff, iid)
        full = 'export HOME=/tmp\nmkdir -p "$HOME"\n' + script
        t0 = time.time()
        out, code = env._exec(f"bash -c {shlex.quote(full)}", timeout=600)
        elapsed = time.time() - t0
        grade_root = Path(env._overlay_dir or "/tmp") / f"abgrade_{iid}_{variant}"
        log_dir = grade_root / iid
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "test_output.txt").write_text(out or "")
        report = get_eval_report(
            test_spec=spec,
            prediction={"instance_id": iid, "model_name_or_path": "ab_regrade", "model_patch": diff},
            include_tests_status=True,
            log_path=str(log_dir / "test_output.txt"),
        )
        v = verdict_from_report(report, iid)
        v["elapsed"] = round(elapsed, 1)
        v["exit"] = code
        return v
    except Exception as exc:  # noqa: BLE001 — record, don't kill the batch
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        env._cleanup_host_testbed()


def run_sample(sample: dict) -> dict:
    iid = sample["instance"]["instance_id"]
    stock = grade_once(sample, "stock")
    trans = grade_once(sample, "transformed")
    keys = ("resolved", "applied", "ftp_success", "ftp_failure", "ptp_success", "ptp_failure")
    agree = "error" not in stock and "error" not in trans and all(stock[k] == trans[k] for k in keys)
    return {
        "instance_id": iid,
        "cls": sample["cls"],
        "orig_reason": sample["orig_reason"],
        "stock": stock,
        "transformed": trans,
        "agree": agree,
        "stock_matches_original": ("error" not in stock and stock["resolved"] == (sample["orig_reason"] == "resolved")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiments", nargs="+", required=True, help="Training experiment directories containing episodes/")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.n, args.experiments)
    print(f"[ab] grading {len(samples)} samples x2 with {args.workers} workers", flush=True)
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_sample, s): s for s in samples}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            print(f"[ab] {i}/{len(samples)} {r['instance_id']} cls={r['cls']} agree={r['agree']} stock={r['stock'].get('elapsed')}s trans={r['transformed'].get('elapsed')}s", flush=True)

    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in results if "error" not in r["stock"] and "error" not in r["transformed"]]
    errs = Counter()
    for r in results:
        for side in ("stock", "transformed"):
            if "error" in r[side]:
                errs[f"{side}:{r[side]['error'].split(':')[0]}"] += 1
    n_agree = sum(r["agree"] for r in ok)
    gate_passed = len(results) == args.n and len(ok) == len(results) and n_agree == len(ok) and bool(ok)
    st = [r["stock"]["elapsed"] for r in ok]
    tr = [r["transformed"]["elapsed"] for r in ok]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    print("\n========== PHASE 2 SUMMARY ==========", flush=True)
    print(f"samples graded both sides: {len(ok)}/{len(results)}  errors: {dict(errs)}")
    print(f"VERDICT AGREEMENT: {n_agree}/{len(ok)}  ({'100% - GATE PASSED' if gate_passed else 'GATE FAILED'})")
    for r in ok:
        if not r["agree"]:
            print(f"  DISAGREE {r['instance_id']}: stock={r['stock']} vs trans={r['transformed']}")
    print(f"grading time: stock median {med(st):.1f}s -> transformed median {med(tr):.1f}s  (speedup {med(st) / max(med(tr), 0.1):.2f}x)")
    print(f"stock-vs-original-training-verdict (diagnostic): {sum(r['stock_matches_original'] for r in ok)}/{len(ok)}")
    print(f"wall time: {(time.time() - t0) / 60:.1f} min")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
