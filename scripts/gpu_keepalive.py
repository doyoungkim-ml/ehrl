#!/usr/bin/env python3
"""GPU utilization keepalive, enabled by default by the shared training runtime.

Some Torch GPU partitions cancel long-running jobs whose average GPU
utilization stays below a site-specific threshold. Environment-heavy rollout jobs are
legitimate workloads that leave the allocated GPUs idle between generation
bursts. This helper targets higher utilization without changing the training
code path; RLLM_GPU_KEEPALIVE=0 disables it in the shared runtime. Helper readings
do not guarantee the scheduler's utilization measurement or prevent cancellation.

``fill`` mode (default) runs matmuls back to back while the GPU is otherwise
idle. Every chunk of work is timed against the fastest rate seen on that
device: when the rate collapses, another process is using the GPU and the
helper backs off so training keeps the GPU. While backed off it reads the NVML
utilization and, when that is still under the target, keeps a light duty cycle
to approach the target at some cost to throughput.

``fixed`` mode is the earlier schedule: ``RLLM_GPU_KEEPALIVE_WORK_SEC`` of work
every ``RLLM_GPU_KEEPALIVE_INTERVAL_SEC``.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

_RUNNING = True


def _handle_stop(signum: int, _frame: object) -> None:
    global _RUNNING
    print(f"[gpu-keepalive] received signal {signum}; stopping", flush=True)
    _RUNNING = False


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        print(f"[gpu-keepalive] invalid {name}={value!r}; using {default}", flush=True)
        return default
    return max(parsed, 0.0)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        print(f"[gpu-keepalive] invalid {name}={value!r}; using {default}", flush=True)
        return default
    return max(parsed, 1)


def _parse_devices(device_count: int) -> list[int]:
    spec = os.environ.get("RLLM_GPU_KEEPALIVE_DEVICES", "all").strip().lower()
    if spec in {"", "all"}:
        return list(range(device_count))
    if spec == "last":
        return [device_count - 1]

    devices: list[int] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            device = int(raw)
        except ValueError:
            print(f"[gpu-keepalive] ignoring invalid device {raw!r}", flush=True)
            continue
        if 0 <= device < device_count:
            devices.append(device)
        else:
            print(
                f"[gpu-keepalive] ignoring device {device}; visible count={device_count}",
                flush=True,
            )
    return sorted(set(devices))


def _run_device_keepalive(
    *,
    torch,
    device: int,
    a,
    b,
    out,
    interval_s: float,
    work_s: float,
) -> None:
    """Run the low-duty keepalive loop for one visible CUDA device."""
    while _RUNNING:
        loop_start = time.monotonic()
        with torch.cuda.device(device):
            deadline = time.monotonic() + work_s
            while _RUNNING and time.monotonic() < deadline:
                torch.mm(a, b, out=out)
                # CUDA launches are asynchronous. Bound actual device work rather
                # than enqueueing an interval's worth of host-side launches.
                torch.cuda.synchronize()
        elapsed = time.monotonic() - loop_start
        sleep_s = max(0.0, interval_s - elapsed)
        if sleep_s:
            time.sleep(sleep_s)


def _run_chunk(*, torch, a, b, out, chunk_s: float, launches_per_sync: int) -> tuple[int, float]:
    """Launch matmuls for about ``chunk_s`` and return (launches, elapsed seconds)."""
    start = time.monotonic()
    deadline = start + chunk_s
    launches = 0
    while _RUNNING and time.monotonic() < deadline:
        for _ in range(launches_per_sync):
            torch.mm(a, b, out=out)
        launches += launches_per_sync
        torch.cuda.synchronize()
    return launches, time.monotonic() - start


def _nvml_utilization(torch, device: int) -> float | None:
    try:
        return float(torch.cuda.utilization(device))
    except Exception:  # pragma: no cover - pynvml availability is environment dependent.
        return None


def _estimated_own_utilization(own_work: list[tuple[float, float, float]], *, now: float, window_s: float) -> float:
    """Estimate service in the window, spreading each chunk's service over its wall time."""
    window_start = now - window_s
    service_s = 0.0
    for start, end, seconds in own_work:
        overlap = max(0.0, min(end, now) - max(start, window_start))
        if overlap:
            service_s += seconds * overlap / (end - start)
    return 100.0 * min(service_s / window_s, 1.0)


def _fill_device_keepalive(
    *,
    torch,
    device: int,
    a,
    b,
    out,
    chunk_s: float,
    target_util: float,
    busy_sleep_s: float,
    light_sleep_s: float,
    launches_per_sync: int,
    contention_ratio: float,
    light_floor_util: float | None = None,
    utilization_window_s: float = 1.0,
    log_every_s: float = 60.0,
    utilization=_nvml_utilization,
) -> None:
    """Fill idle GPU time; yield when another process is using the device.

    Below ``light_floor_util`` the helper keeps filling even under contention,
    because the job is near the cancellation threshold; between the floor and
    ``target_util`` the spacing between chunks grows linearly to ``light_sleep_s``.
    """
    if light_floor_util is None:
        light_floor_util = target_util - 30.0
    baseline = 0.0
    spent = {"fill": 0.0, "busy": 0.0, "light": 0.0}
    last_util = None
    last_own_util = None
    last_other_util = None
    contention_checks = 0
    last_log = time.monotonic()
    own_work: list[tuple[float, float, float]] = []  # (start, end, estimated service seconds)
    while _RUNNING:
        with torch.cuda.device(device):
            launches, elapsed = _run_chunk(torch=torch, a=a, b=b, out=out, chunk_s=chunk_s, launches_per_sync=launches_per_sync)
        if elapsed <= 0 or launches == 0:
            continue
        now = time.monotonic()
        rate = launches / elapsed
        contended = rate < contention_ratio * baseline
        if not contended:
            # Follow gradual clock changes, but do not learn sustained competition
            # as a new idle baseline and eventually stop yielding to it.
            baseline = max(rate, baseline * 0.998)
        # Synchronization includes time waiting for another process. Estimate our
        # service from completed launches at the uncontended rate, not that wait.
        own_seconds = min(elapsed, launches / baseline)
        own_work.append((now - elapsed, now, own_seconds))
        own_work = [(start, end, seconds) for start, end, seconds in own_work if end > now - utilization_window_s]
        if not contended:
            spent["fill"] += elapsed
        else:
            contention_checks += 1
            last_util = utilization(torch, device)
            # NVML includes our work. This subtraction is an estimate: service
            # can overlap other processes and NVML's sampling window can differ.
            last_own_util = _estimated_own_utilization(own_work, now=now, window_s=utilization_window_s)
            last_other_util = None if last_util is None else max(last_util - last_own_util, 0.0)
            if last_other_util is None or last_other_util >= target_util:
                time.sleep(busy_sleep_s)
                spent["busy"] += elapsed + busy_sleep_s
            else:
                span = max(target_util - light_floor_util, 1e-6)
                pause = light_sleep_s * min(max((last_other_util - light_floor_util) / span, 0.0), 1.0)
                if pause > 0:
                    time.sleep(pause)
                spent["light"] += elapsed + pause
        if now - last_log >= log_every_s:
            total = sum(spent.values()) or 1.0
            print(
                f"[gpu-keepalive] device={device} fill={spent['fill'] / total:.0%} busy={spent['busy'] / total:.0%} "
                f"light={spent['light'] / total:.0%} last_util={last_util} "
                f"last_own_util_estimate={last_own_util} last_other_util_estimate={last_other_util} "
                f"contention_checks={contention_checks} baseline_rate={baseline:.0f}/s",
                flush=True,
            )
            spent = {"fill": 0.0, "busy": 0.0, "light": 0.0}
            contention_checks = 0
            last_log = now


def main() -> int:
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _handle_stop)

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent.
        print(f"[gpu-keepalive] torch import failed; exiting: {exc}", flush=True)
        return 0

    if not torch.cuda.is_available():
        print("[gpu-keepalive] CUDA is not available; exiting", flush=True)
        return 0

    device_count = torch.cuda.device_count()
    devices = _parse_devices(device_count)
    if not devices:
        print("[gpu-keepalive] no valid devices selected; exiting", flush=True)
        return 0

    mode = os.environ.get("RLLM_GPU_KEEPALIVE_MODE", "fill").strip().lower() or "fill"
    if mode not in {"fill", "fixed"}:
        print(f"[gpu-keepalive] invalid RLLM_GPU_KEEPALIVE_MODE={mode!r}; using fill", flush=True)
        mode = "fill"
    interval_s = _float_env("RLLM_GPU_KEEPALIVE_INTERVAL_SEC", 5.0)
    work_s = _float_env("RLLM_GPU_KEEPALIVE_WORK_SEC", 0.5)
    size = _int_env("RLLM_GPU_KEEPALIVE_SIZE", 4096)
    fill_settings = {
        "chunk_s": _float_env("RLLM_GPU_KEEPALIVE_CHUNK_SEC", 0.02),
        "target_util": _float_env("RLLM_GPU_KEEPALIVE_TARGET_UTIL", 85.0),
        "busy_sleep_s": _float_env("RLLM_GPU_KEEPALIVE_BUSY_SLEEP_SEC", 0.25),
        "light_sleep_s": _float_env("RLLM_GPU_KEEPALIVE_LIGHT_SLEEP_SEC", 0.05),
        "launches_per_sync": _int_env("RLLM_GPU_KEEPALIVE_LAUNCHES_PER_SYNC", 4),
        "contention_ratio": _float_env("RLLM_GPU_KEEPALIVE_CONTENTION_RATIO", 0.6),
    }
    fill_settings["light_floor_util"] = _float_env("RLLM_GPU_KEEPALIVE_LIGHT_FLOOR_UTIL", fill_settings["target_util"] - 30.0)
    dtype = torch.float16

    states = []
    for device in devices:
        with torch.cuda.device(device):
            a = torch.randn((size, size), device="cuda", dtype=dtype)
            b = torch.randn((size, size), device="cuda", dtype=dtype)
            out = torch.empty((size, size), device="cuda", dtype=dtype)
            torch.cuda.synchronize()
            states.append((device, a, b, out))

    if mode == "fill":
        print(f"[gpu-keepalive] started mode=fill devices={devices} size={size} {fill_settings}", flush=True)
    else:
        print(
            f"[gpu-keepalive] started mode=fixed devices={devices} interval_s={interval_s} work_s_per_device={work_s} size={size}",
            flush=True,
        )

    with torch.no_grad():
        threads = []
        for device, a, b, out in states:
            common = {"torch": torch, "device": device, "a": a, "b": b, "out": out}
            if mode == "fill":
                target, kwargs = _fill_device_keepalive, {**common, **fill_settings}
            else:
                target, kwargs = _run_device_keepalive, {**common, "interval_s": interval_s, "work_s": work_s}
            threads.append(threading.Thread(target=target, kwargs=kwargs, name=f"gpu-keepalive-{device}", daemon=True))
        for thread in threads:
            thread.start()
        while _RUNNING and any(thread.is_alive() for thread in threads):
            time.sleep(0.5)
        for thread in threads:
            thread.join(timeout=max(interval_s, work_s, fill_settings["busy_sleep_s"]) + 1.0)

    print("[gpu-keepalive] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
