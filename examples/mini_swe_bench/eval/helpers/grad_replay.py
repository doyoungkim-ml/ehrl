"""Per-trajectory policy-gradient replay for mini-SWE checkpoints.

For a (merged HF) checkpoint and one or more episode-log step dirs, rebuild every
logged trajectory from its final-step `chat_completions`, run ONE backward pass
per trajectory of  L_i = -sum_{assistant tokens} log pi_theta(y_t | y_<t, x)  and
record, per trajectory:
  * ||g_i||           full-parameter gradient norm (fp32 accumulate)
  * sketch_i          gradient at a FIXED random coordinate subset (n_sketch
                      coords, stratified over parameters, seeded) -> fp16.
                      Inner products / cosines / PCA are computed offline from
                      the sketches (unbiased for dense grads).
  * L_i               number of assistant (response) tokens
  * A_i               GRPO advantage recomputed from logged rewards within the
                      task group of that step: (r - mean)/(std_ddof1 + 1e-6);
                      0 if the group has no variance (would not pass the gate)
  * bookkeeping       step, epoch, uid, task, reward, reward_reason, ftp/ptp,
                      truncated flag, mean per-token logprob

Design notes
  * token-mean training loss => trajectory i's contribution to the update is
    A_i * g_i / T_minibatch; g_i here is the UNnormalised sum over tokens, so
    downstream code divides by token totals when reproducing ||G||.
  * on-policy iff the episodes were rolled out by the loaded checkpoint
    (ckpt s <-> episodes of step s+1). Episodes of step s (rolled by policy
    s-1) are one update off-policy; ratio ~ 1, fine for direction analysis.
  * fp32 master params + bf16 autocast; gradient checkpointing; the LM head
    log-softmax is computed in position chunks under torch.utils.checkpoint so
    the [T, vocab] logits never exist at once (40k x 152k fp32 = 24 GB).
  * one trajectory per backward; grads zeroed after each.

Usage (inside the slurm job):
  python grad_replay.py --model <hf_dir> --episodes <dir> [<dir> ...]
      --out <npz> [--max-tokens 40960] [--n-sketch 2000000] [--limit N]
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_sketch_index(model, n_sketch: int, seed: int = 0):
    """Fixed coordinate subset: for each parameter, k_p ~ n_sketch * numel_p / total
    random flat indices (seeded by parameter order). Returns list of (name, idx LongTensor)."""
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in params)
    g = torch.Generator().manual_seed(seed)
    out = []
    for n, p in params:
        k = max(1, int(round(n_sketch * p.numel() / total)))
        k = min(k, p.numel())
        idx = torch.randperm(p.numel(), generator=g)[:k].sort().values
        out.append((n, idx))
    return out


def _ids(x):
    """apply_chat_template(tokenize=True) -> list[int] across transformers versions
    (list, BatchEncoding/UserDict with input_ids, or nested [[ids]])."""
    if hasattr(x, "keys"):
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    x = list(x)
    if x and isinstance(x[0], list | tuple):
        x = list(x[0])
    return [int(t) for t in x]


def rebuild_tokens(tok, messages, max_tokens: int):
    """Tokenise the full conversation with the model's chat template and mark
    assistant tokens via prefix differences. Returns (input_ids [T], mask [T] bool, truncated)."""
    full = _ids(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    mask = np.zeros(len(full), dtype=bool)
    for k, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        # tokens of the assistant message = template(messages[:k] + gen prompt) .. template(messages[:k+1])
        pre = _ids(tok.apply_chat_template(messages[:k], tokenize=True, add_generation_prompt=True))
        post = _ids(tok.apply_chat_template(messages[: k + 1], tokenize=True, add_generation_prompt=False))
        a, b = len(pre), min(len(post), len(full))
        if a < b:
            mask[a:b] = True
    truncated = False
    if len(full) > max_tokens:
        full, mask, truncated = full[:max_tokens], mask[:max_tokens], True
    return torch.tensor(full, dtype=torch.long), torch.tensor(mask), truncated


def traj_logprob_sum(model, input_ids, mask, chunk: int = 2048):
    """-sum of log pi over masked (assistant) target positions, memory-chunked over the LM head."""
    dev = model.device
    ids = input_ids.unsqueeze(0).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(input_ids=ids).last_hidden_state[0]  # [T, H]
    # targets: position t predicts token t+1
    tgt = input_ids[1:].to(dev)
    tmask = mask[1:].to(dev)
    h = hidden[:-1]
    lm = model.lm_head
    total = torch.zeros((), device=dev, dtype=torch.float32)
    n_tok = int(tmask.sum().item())

    def chunk_nll(h_c, t_c, m_c):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = lm(h_c)
        lp = torch.log_softmax(logits.float(), dim=-1)
        picked = lp.gather(-1, t_c.unsqueeze(-1)).squeeze(-1)
        return -(picked * m_c).sum()

    for s in range(0, h.shape[0], chunk):
        e = min(s + chunk, h.shape[0])
        if not bool(tmask[s:e].any()):
            continue
        total = total + checkpoint(chunk_nll, h[s:e], tgt[s:e], tmask[s:e].float(), use_reentrant=False)
    return total, n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", nargs="+", required=True, help="episode step dirs (train_step_*_epoch_*)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=40960)
    ap.add_argument("--n-sketch", type=int, default=2_000_000)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N episodes per dir")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.cuda()
    model.train()  # train() only for grad; no dropout in Qwen3
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    sk_index = build_sketch_index(model, args.n_sketch)
    n_sk = sum(len(i) for _, i in sk_index)
    print(f"[replay] model loaded {n_params / 1e9:.2f}B params, sketch coords {n_sk}, {time.time() - t0:.0f}s", flush=True)

    rows, sketches = [], []
    for d in args.episodes:
        m = re.search(r"train_step_(\d+)_epoch_(\d+)", d)
        step, epoch = int(m.group(1)), int(m.group(2))
        files = sorted(glob.glob(os.path.join(d, "episode_*.json")))
        if args.limit:
            files = files[: args.limit]
        # first pass: rewards per group for advantages
        meta = {}
        for f in files:
            ep = json.load(open(f))
            tr = ep["trajectories"][0]
            meta[f] = (ep["task"]["instance_id"], float(tr["reward"]))
        by_task = {}
        for f, (tid, r) in meta.items():
            by_task.setdefault(tid, []).append(r)
        adv = {}
        for tid, rs in by_task.items():
            rs_t = torch.tensor(rs)
            mu = rs_t.mean()
            sd = rs_t.std(unbiased=True) if len(rs) > 1 else torch.tensor(0.0)
            adv[tid] = ((rs_t - mu) / (sd + 1e-6)).tolist() if float(sd) > 0 else [0.0] * len(rs)
        seen = {tid: 0 for tid in by_task}
        print(f"[replay] {os.path.basename(d)}: {len(files)} episodes, {len(by_task)} tasks, {sum(1 for v in adv.values() if any(a != 0 for a in v))} groups with variance", flush=True)
        for f in files:
            ep = json.load(open(f))
            tr = ep["trajectories"][0]
            info = tr["info"]
            tid, r = meta[f]
            a_i = adv[tid][seen[tid]]
            seen[tid] += 1
            msgs = tr["steps"][-1]["chat_completions"]
            # the last step's chat_completions ends with the final assistant message
            ids, mask, trunc = rebuild_tokens(tok, msgs, args.max_tokens)
            L = int(mask.sum().item())
            t1 = time.time()
            model.zero_grad(set_to_none=True)
            nll, n_tok = traj_logprob_sum(model, ids, mask)
            nll.backward()
            # collect: full norm + sketch
            sq = torch.zeros((), device="cuda", dtype=torch.float64)
            sk_parts = []
            pmap = dict(model.named_parameters())
            for name, idx in sk_index:
                g = pmap[name].grad
                if g is None:
                    sk_parts.append(torch.zeros(len(idx), dtype=torch.float16))
                    continue
                gf = g.detach().reshape(-1)
                sq += (gf.double() ** 2).sum()
                sk_parts.append(gf[idx.to(gf.device)].to(torch.float16).cpu())
            gnorm = float(sq.sqrt().item())
            sketches.append(torch.cat(sk_parts).numpy())
            tc = info.get("test_counts") or {}
            rows.append(
                dict(
                    step=step,
                    epoch=epoch,
                    uid=tr["uid"],
                    task=tid,
                    reward=r,
                    adv=a_i,
                    L=L,
                    T=int(ids.numel()),
                    gnorm=gnorm,
                    nll=float(nll.item()),
                    truncated=int(trunc),
                    n_turns=int(tr["num_steps"]),
                    reward_reason=info.get("reward_reason", ""),
                    ftp_frac=tc and (tc.get("ftp_passed", 0) / max(tc.get("ftp_total", 1), 1)) or float("nan"),
                    ptp_frac=tc and ((tc.get("ptp_passed", 0) / tc["ptp_total"]) if tc.get("ptp_total") else 1.0) or float("nan"),
                    term=ep.get("termination_reason", ""),
                    file=os.path.basename(f),
                    tag=args.tag,
                )
            )
            print(
                f"  {os.path.basename(f)[:40]} task={tid} r={r:.2f} A={a_i:+.2f} T={ids.numel()} L={L} |g|={gnorm:.3f} nll/tok={float(nll.item()) / max(L, 1):.3f} {time.time() - t1:.1f}s", flush=True
            )
            del nll
            torch.cuda.empty_cache()
    keys = list(rows[0].keys())
    np.savez_compressed(
        args.out, sketches=np.stack(sketches), rows=json.dumps(rows), keys=json.dumps(keys), sketch_names=json.dumps([n for n, _ in sk_index]), sketch_sizes=np.array([len(i) for _, i in sk_index])
    )
    print(f"[replay] wrote {args.out}: {len(rows)} trajectories, sketch dim {sketches[0].shape[0]}, total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
