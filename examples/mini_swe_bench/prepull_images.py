#!/usr/bin/env python3
"""Eagerly pre-pull all SWE-bench container images to local .sif files (Kailin).

WHY
---
At rollout time the env execs one container per agent command; with a docker://
uri that re-resolves `:latest` against Docker Hub every exec, exhausting the
anonymous pull-rate limit (TOOMANYREQUESTS). The fix is `ensure_image_available`
returning a local .sif (pulled once). This script pulls every image the run will
need UP FRONT — before the trainer starts — so:
  * step 0 isn't a thundering herd of N workers each blocking on a cold image,
  * an unreachable registry / bad image fails the job in minutes, not hours in.

It is an OPTIMISATION, not a correctness requirement: `ensure_image_available`
still lazily pulls (under a lock) anything missed here. So this script is
deliberately best-effort and exits 0 even if some images fail — the lazy path
retries them, and the fail-fast guard handles any that stay broken.

The instance set mirrors the trainer EXACTLY (same parquet, same
MAX_TRAIN_TASKS / TRAIN_TASK_OFFSET / MAX_VAL_TASKS slicing), so we never pull
images the run won't use.

USAGE
    export APPTAINER_SIF_DIR=/scratch/$USER/sif
    python examples/mini_swe_bench/prepull_images.py [--max-parallel 4] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.mini_swe_bench.preprocess_swegym import get_default_data_dir, preprocess_swegym  # noqa: E402
from examples.mini_swe_bench.shared.apptainer_utils import (  # noqa: E402
    build_sif,
    sif_path_for,
)
from examples.mini_swe_bench.shared.skyrl_utils import get_docker_image_name  # noqa: E402
from rllm.data import Dataset  # noqa: E402


def _resolve_uri(example: dict) -> str | None:
    """Mirror MiniSWEnv._get_image_uri()."""
    instance = example.get("instance", example)
    image = instance.get("image_name") or example.get("image_uri") or example.get("instance_image_key")
    if image:
        return image if image.startswith("docker://") else f"docker://{image}"
    data_source = example.get("data_source", "SumanthRH/SWE-Gym-Subset")
    try:
        name = get_docker_image_name(instance, data_source).replace("docker.io/", "")
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] cannot resolve image for {instance.get('instance_id')}: {e}", flush=True)
        return None
    return f"docker://{name}"


def _sliced(examples: list[dict], offset: int, max_n: int | None) -> list[dict]:
    return examples[offset : offset + max_n] if max_n is not None else examples[offset:]


def _collect_uris(splits: tuple[str, ...] = ("train",)) -> list[str]:
    data_dir = get_default_data_dir()
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "validation.parquet"
    if not train_path.exists() or not val_path.exists():
        print(f"[prepull] preprocessing SWE-Gym subset under {data_dir} ...", flush=True)
        preprocess_swegym(output_dir=data_dir)

    train_offset = int(os.environ.get("TRAIN_TASK_OFFSET", "0") or "0")
    max_train = os.environ.get("MAX_TRAIN_TASKS", "").strip()
    max_val = os.environ.get("MAX_VAL_TASKS", "").strip()

    examples: list[dict] = []
    if "train" in splits:
        train = _sliced(Dataset.load_data(str(train_path)).get_data(), train_offset, int(max_train) if max_train else None)
        print(f"[prepull] train tasks={len(train)} (offset={train_offset}, cap={max_train or 'all'})", flush=True)
        examples += train
    if "validation" in splits:
        # NB: SWE-bench_Verified validation is 500 tasks (=500 images). Training
        # uses test_freq=-1 (no periodic val), so these are only needed at a final
        # validation — include them only when you actually run it.
        val = _sliced(Dataset.load_data(str(val_path)).get_data(), 0, int(max_val) if max_val else None)
        print(f"[prepull] val tasks={len(val)} (cap={max_val or 'all'})", flush=True)
        examples += val

    uris: list[str] = []
    seen: set[str] = set()
    for ex in examples:
        uri = _resolve_uri(ex)
        if uri and uri not in seen:
            seen.add(uri)
            uris.append(uri)
    return uris


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-parallel", type=int, default=4, help="concurrent pulls (default 4)")
    ap.add_argument("--limit", type=int, default=0, help="only pull the first N unique images (0 = all)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-image pull timeout seconds")
    ap.add_argument(
        "--splits",
        default="train",
        help="comma-separated splits to prepull: train,validation (default: train). "
        "Training uses test_freq=-1, so 'train' alone covers the rollout images; "
        "add 'validation' (500 SWE-bench_Verified images) only for a final eval.",
    )
    args = ap.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    sif_dir = os.environ.get("APPTAINER_SIF_DIR", str(REPO_ROOT / ".cache/mini_swe_bench/sif"))
    uris = _collect_uris(splits)
    if args.limit:
        uris = uris[: args.limit]
    n = len(uris)
    have = sum(1 for u in uris if sif_path_for(u).exists())
    print(f"[prepull] {n} unique images -> {sif_dir} (one .sif each)  ({have} already present, {n - have} to build, {args.max_parallel}-way parallel)", flush=True)

    t0 = time.time()
    ok = skipped = failed = 0

    def _pull(uri: str) -> tuple[str, str, float]:
        s = time.time()
        if sif_path_for(uri).exists():
            return uri, "have", 0.0
        try:
            build_sif(uri, timeout=args.timeout)
            return uri, "ok", time.time() - s
        except Exception as e:  # noqa: BLE001
            return uri, f"FAIL: {e}", time.time() - s

    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(_pull, u): u for u in uris}
        for i, fut in enumerate(as_completed(futs), 1):
            uri, status, dt = fut.result()
            short = uri.split("/")[-1]
            if status == "ok":
                ok += 1
                print(f"  [{i}/{n}] ok   {short} ({dt:.0f}s)", flush=True)
            elif status == "have":
                skipped += 1
                print(f"  [{i}/{n}] have {short}", flush=True)
            else:
                failed += 1
                print(f"  [{i}/{n}] {status}  {short}", flush=True)

    dt = time.time() - t0
    print(f"\n[prepull] done in {dt / 60:.1f} min: {ok} pulled, {skipped} already had, {failed} failed.", flush=True)
    if failed:
        print("[prepull] NOTE: failures will be retried lazily at rollout time (ensure_image_available) under a per-image lock.", flush=True)
    # Best-effort: never block the job on prepull failures.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
