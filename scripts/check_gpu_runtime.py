#!/usr/bin/env python3
"""Exercise the selected GPU runtime inside a single-node Slurm allocation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import socket
import tempfile
from datetime import timedelta
from pathlib import Path


def check_worker(rank: int, world_size: int, rendezvous: str) -> None:
    import torch
    import torch.distributed as dist
    from flash_attn import flash_attn_func

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=90))
    try:
        value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
        dist.all_reduce(value)
        expected = world_size * (world_size + 1) / 2
        if value.item() != expected:
            raise RuntimeError(f"NCCL all-reduce returned {value.item()}, expected {expected}")
        query = torch.randn(1, 16, 4, 32, device=f"cuda:{rank}", dtype=torch.bfloat16, requires_grad=True)
        output = flash_attn_func(query, query, query, causal=True)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        if query.grad is None or not torch.isfinite(query.grad).all().item():
            raise RuntimeError(f"FlashAttention backward produced invalid gradients on GPU {rank}")
        print(f"[gpu-check] rank={rank} NCCL all-reduce and FlashAttention forward/backward passed", flush=True)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--expected-gpu-name", default="H200")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence file")
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("--gpus must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run inside a Slurm allocation; do not test GPUs on a login node")

    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() != args.gpus:
        raise RuntimeError(f"Expected {args.gpus} allocated GPUs, found {torch.cuda.device_count()}")
    devices = [torch.cuda.get_device_name(index) for index in range(args.gpus)]
    if any(args.expected_gpu_name not in name for name in devices):
        raise RuntimeError(f"Expected {args.expected_gpu_name} GPUs, found {devices}")
    modules = {}
    for name in (
        "vllm",
        "vllm._C",
        "flash_attn",
        "verl.experimental.agent_loop.agent_loop",
        "verl.workers.rollout.vllm_rollout.vllm_rollout",
        "rllm.engine.rollout.verl_engine",
        "rllm.trainer.verl.train_agent_ppo",
        "rllm.algorithms.ace.trainer",
    ):
        module = importlib.import_module(name)
        modules[name] = module.__file__
        print(f"[gpu-check] import {name}: {module.__file__}", flush=True)
    with tempfile.TemporaryDirectory(prefix="rllm-gpu-check-") as directory:
        rendezvous = (Path(directory) / "nccl-rendezvous").as_uri()
        mp.spawn(check_worker, args=(args.gpus, rendezvous), nprocs=args.gpus, join=True)
    report = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(),
        "devices": devices,
        "torch_cuda_version": torch.version.cuda,
        "cuda_home": os.environ.get("CUDA_HOME"),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "flash-attn", "ray", "verl")},
        "modules": modules,
        "nccl_all_reduce": "passed",
        "flash_attention_forward_backward": "passed",
        "scope": "GPU kernels, collectives and imports; not model rollout, optimization, or checkpoint validation",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
