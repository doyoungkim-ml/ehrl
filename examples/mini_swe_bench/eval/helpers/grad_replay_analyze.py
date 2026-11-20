"""Analyse grad-replay outputs.

Reads one or more replay_<tag>.npz (from grad_replay.py) and prints /
saves per-step statistics of h_i = A_i g_i:
  * validation: token lengths vs logged num_steps, |g| distribution, and
    ||G|| = ||sum_i A_i g_i / T_batch|| over the gate-passing groups (16-of-pool
    random downselect unknown -> report the pool value and a 200-draw range of
    random 4-group selections) next to the logged actor/grad_norm if given
  * dominance:   max_i |h_i| / sum_j |h_j|   (token-mean weighting: |h_i| = |A_i| ||g_i||)
  * coherence:   mean pairwise cos(h_i, h_j) within group / within step, split
                 pos-pos / neg-neg / pos-neg (cosines from sketches)
  * alignment:   cos(h_i, G) per trajectory (A != 0), and cos(g_i, G) for A == 0
                 (untrained: transfer signal)
Writes a CSV of per-trajectory rows with the derived columns next to the npz.

Usage: python grad_replay_analyze.py replay_a.npz [replay_b.npz ...] [--logged-gradnorm STEP=VAL ...]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def load(path):
    z = np.load(path, allow_pickle=False)
    rows = pd.DataFrame(json.loads(str(z["rows"])))
    S = z["sketches"].astype(np.float32)  # [n, m]
    return rows, S


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--logged-gradnorm", nargs="*", default=[], help="STEP=VALUE pairs for the check")
    ap.add_argument("--task-features", help="Optional CSV with instance_id and ftp_tests columns")
    args = ap.parse_args()
    logged = {int(k): float(v) for k, v in (s.split("=") for s in args.logged_gradnorm)}
    feat = None
    if args.task_features:
        feat = pd.read_csv(args.task_features).set_index("instance_id").ftp_tests.to_dict()
    stats = []
    for path in args.npz:
        rows, S = load(path)
        print(f"\n===== {os.path.basename(path)}: {len(rows)} trajectories, sketch dim {S.shape[1]}")
        # sketch-vs-full norm calibration: E[|sketch|^2] = (m/N)|g|^2 -> check ratio is stable
        rows["sk_norm"] = np.linalg.norm(S, axis=1)
        rows["calib"] = rows.sk_norm / rows.gnorm
        print(f"sketch/full norm ratio: median {rows.calib.median():.4g}, IQR {rows.calib.quantile(0.25):.4g}-{rows.calib.quantile(0.75):.4g} (should be ~constant = sqrt(m/N))")
        rows["absh"] = rows.adv.abs() * rows.gnorm
        rows["ftp_tests"] = rows.task.map(feat) if feat else np.nan
        tag = str(rows.tag.iloc[0]) if "tag" in rows else os.path.basename(path)
        rows["gnorm_per_tok"] = rows.gnorm / rows.L.clip(lower=1)
        for step, d in rows.groupby("step"):
            trained = d[d.adv != 0]
            print(
                f"\n--- step {step}: {len(d)} traj, {d.task.nunique()} tasks, {trained.task.nunique()} gate-passing groups, "
                f"L median {d.L.median():.0f} (max {d.L.max()}), T median {d['T'].median():.0f}, truncated {int(d.truncated.sum())}, "
                f"|g| median {d.gnorm.median():.3g} [{d.gnorm.min():.3g}, {d.gnorm.max():.3g}], "
                f"|g|/L median {d.gnorm_per_tok.median():.3g}"
            )
            print(f"    reward_reason: {d.reward_reason.value_counts().to_dict()}")
            if len(trained):
                # ||G|| pool estimate over ALL gate-passing groups (token-mean over their tokens)
                Lt = trained.L.sum()
                Gs = (trained.adv.values[:, None] * S[trained.index.values]).sum(0)
                scale = 1.0 / rows.calib.median()  # sketch -> full-norm units
                G_norm_pool = np.linalg.norm(Gs) * scale / Lt
                # training: 4 groups per step, split into 2 mini-batches of 2 groups (8 traj); grad_norm is logged per
                # mini-batch (pre-clip) and averaged -> emulate with random 2-group draws, 200 draws
                groups = trained.task.unique()
                draws = []
                rng = np.random.default_rng(0)
                for _ in range(200):
                    sel = rng.choice(groups, size=min(2, len(groups)), replace=False)
                    dd = trained[trained.task.isin(sel)]
                    G = (dd.adv.values[:, None] * S[dd.index.values]).sum(0)
                    draws.append(np.linalg.norm(G) * scale / dd.L.sum())
                lo, hi = np.percentile(draws, [5, 95])
                msg = f"    ||G||/L_resp: pool(all {len(groups)} groups)={G_norm_pool:.3f}; 2-group (8-traj mini-batch) draws 5-95%: {lo:.3f}-{hi:.3f} (median {np.median(draws):.3f})"
                if step in logged:
                    msg += f"   | logged actor/grad_norm@{step} = {logged[step]:.3f}"
                print(msg)
                # dominance & alignment
                dom = trained.absh.max() / trained.absh.sum()
                Gfull = (trained.adv.values[:, None] * S[trained.index.values]).sum(0)
                cosG = [cos(a * S[i], Gfull) for i, a in zip(trained.index.values, trained.adv.values, strict=True)]
                trained = trained.assign(cosG=cosG)
                print(f"    dominance max|h|/sum|h| = {dom:.3f}; |h| = |A|*|g|: top-3 {np.sort(trained.absh.values)[-3:][::-1].round(3)}")
                print(f"    cos(h_i,G): pos-A mean {trained[trained.adv > 0].cosG.mean():.3f}, neg-A mean {trained[trained.adv < 0].cosG.mean():.3f}, share<0: {(trained.cosG < 0).mean():.2f}")
                # pairwise coherence within step
                H = trained.adv.values[:, None] * S[trained.index.values]
                Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
                C = Hn @ Hn.T
                sign = np.sign(trained.adv.values)
                iu = np.triu_indices(len(trained), 1)
                pp = C[iu][(sign[iu[0]] > 0) & (sign[iu[1]] > 0)]
                nn = C[iu][(sign[iu[0]] < 0) & (sign[iu[1]] < 0)]
                pn = C[iu][sign[iu[0]] * sign[iu[1]] < 0]
                print(
                    f"    coherence cos(h_i,h_j) within step: all {C[iu].mean():.3f} | pos-pos {pp.mean() if len(pp) else float('nan'):.3f} "
                    f"| neg-neg {nn.mean() if len(nn) else float('nan'):.3f} | pos-neg {pn.mean() if len(pn) else float('nan'):.3f} | share<0 {(C[iu] < 0).mean():.2f}"
                )
                # raw-gradient coherence (no advantage sign): cos(g_i,g_j)
                Gn = S[trained.index.values] / np.linalg.norm(S[trained.index.values], axis=1, keepdims=True)
                Cg = Gn @ Gn.T
                same_task_cosines = [Cg[a, b] for a, b in zip(*iu, strict=True) if trained.task.values[a] == trained.task.values[b]]
                same_task_mean = np.mean(same_task_cosines) if len(iu[0]) else float("nan")
                print(f"    raw cos(g_i,g_j) within step: mean {Cg[iu].mean():.3f}; same-task pairs {same_task_mean:.3f}")
                # untrained rollouts: transfer
                un = d[d.adv == 0]
                if len(un):
                    cg = [cos(S[i], Gfull) for i in un.index.values]
                    print(f"    untrained (A=0) rollouts n={len(un)}: cos(g_i, G) mean {np.mean(cg):.3f}, share>0 {np.mean(np.array(cg) > 0):.2f}")
                rows.loc[trained.index, "cosG"] = trained.cosG.values
                rec = dict(
                    tag=tag,
                    step=step,
                    n_traj=len(d),
                    n_trained=len(trained),
                    n_groups=len(groups),
                    L_med=d.L.median(),
                    gnorm_med=d.gnorm.median(),
                    gnorm_per_tok_med=d.gnorm_per_tok.median(),
                    G_pool=G_norm_pool,
                    G_draw_med=float(np.median(draws)),
                    G_draw_lo=lo,
                    G_draw_hi=hi,
                    dominance=dom,
                    absh_max=trained.absh.max(),
                    absh_mean=trained.absh.mean(),
                    cosG_pos=trained[trained.adv > 0].cosG.mean(),
                    cosG_neg=trained[trained.adv < 0].cosG.mean(),
                    coh_all=C[iu].mean(),
                    coh_pp=pp.mean() if len(pp) else np.nan,
                    coh_nn=nn.mean() if len(nn) else np.nan,
                    coh_pn=pn.mean() if len(pn) else np.nan,
                    coh_share_neg=(C[iu] < 0).mean(),
                    rawcos_all=Cg[iu].mean(),
                    rawcos_same_task=np.mean([Cg[a, b] for a, b in zip(*iu, strict=True) if trained.task.values[a] == trained.task.values[b]]) if len(iu[0]) else np.nan,
                    untrained_n=len(un),
                    untrained_cosG=np.mean(cg) if len(un) else np.nan,
                    untrained_share_pos=np.mean(np.array(cg) > 0) if len(un) else np.nan,
                    resolved_share=(d.reward_reason == "resolved").mean(),
                    nopatch_share=(d.reward_reason == "no_valid_patch").mean(),
                )
                # FTP splits of coherence (pos-pos within FTP class) and dominance
                for lab, sel in (("ftp1", trained.ftp_tests == 1), ("ftp2p", trained.ftp_tests >= 2)):
                    ii = np.where(sel.values)[0]
                    if len(ii) >= 2:
                        sub = C[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)]
                        rec[f"coh_{lab}"] = sub.mean()
                        rec[f"n_{lab}"] = len(ii)
                        rec[f"cosG_{lab}"] = trained.cosG.values[ii].mean()
                stats.append(rec)
        out = path.replace(".npz", "_rows.csv")
        rows.drop(columns=[c for c in ("file",) if c in rows]).to_csv(out, index=False)
        print("wrote", out)
    if stats:
        so = os.path.join(os.path.dirname(args.npz[0]), "grad_replay_stats.csv")
        pd.DataFrame(stats).to_csv(so, index=False)
        print("wrote", so)


if __name__ == "__main__":
    main()
