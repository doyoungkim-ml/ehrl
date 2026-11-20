"""The keepalive bounds actual asynchronous GPU work and tolerates CPU hosts."""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location("gpu_keepalive_test", ROOT / "scripts/gpu_keepalive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_launch_is_synchronized_so_work_budget_measures_gpu_work(helper, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    events = []

    def mm(*args, **kwargs):
        events.append("mm")

    def synchronize():
        events.append("sync")
        clock.now += 0.1

    def sleep(seconds):
        events.append(("sleep", seconds))
        helper._RUNNING = False

    monkeypatch.setattr(helper.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(helper.time, "sleep", sleep)
    torch = SimpleNamespace(mm=mm, cuda=SimpleNamespace(device=lambda device: nullcontext(), synchronize=synchronize))
    helper._run_device_keepalive(torch=torch, device=0, a=None, b=None, out=None, interval_s=5.0, work_s=0.5)
    assert events[:-1] == ["mm", "sync"] * 5
    assert events[-1] == ("sleep", pytest.approx(4.5))


def test_cpu_only_host_exits_without_allocating_or_starting_threads(helper, monkeypatch, capsys):
    monkeypatch.setattr(helper.signal, "signal", lambda *args: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    assert helper.main() == 0
    assert "CUDA is not available; exiting" in capsys.readouterr().out


def test_devices_are_indices_within_the_visible_allocation(helper, monkeypatch):
    monkeypatch.setenv("RLLM_GPU_KEEPALIVE_DEVICES", "0,1,1,2,-1,invalid")
    assert helper._parse_devices(2) == [0, 1]
    monkeypatch.setenv("RLLM_GPU_KEEPALIVE_DEVICES", "last")
    assert helper._parse_devices(2) == [1]


def _fill_torch(helper, clock, *, kernel_s, stop_after_launches=None, launches=None):
    launches = launches if launches is not None else []

    def mm(*args, **kwargs):
        launches.append(clock.now)
        if stop_after_launches is not None and len(launches) >= stop_after_launches:
            helper._RUNNING = False

    def synchronize():
        clock.now += kernel_s()

    return SimpleNamespace(mm=mm, cuda=SimpleNamespace(device=lambda device: nullcontext(), synchronize=synchronize))


def _fill_settings():
    return {"chunk_s": 0.02, "target_util": 85.0, "busy_sleep_s": 0.5, "light_sleep_s": 0.05, "launches_per_sync": 4, "contention_ratio": 0.6}


def test_fill_mode_runs_back_to_back_while_the_gpu_is_idle(helper, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    sleeps = []
    monkeypatch.setattr(helper.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(helper.time, "sleep", lambda seconds: sleeps.append(seconds))
    helper._RUNNING = True
    torch = _fill_torch(helper, clock, kernel_s=lambda: 0.001, stop_after_launches=400)
    helper._fill_device_keepalive(torch=torch, device=0, a=None, b=None, out=None, utilization=lambda *_: 0.0, **_fill_settings())
    assert sleeps == []


# About 40 ms of the helper's own work precedes the check, so roughly 4 points are its own share.
@pytest.mark.parametrize("util,expected_sleep", [(95.0, 0.5), (90.0, 0.5), (74.0, 0.025), (None, 0.5)])
def test_fill_mode_yields_when_another_process_slows_the_chunk(helper, monkeypatch, util, expected_sleep):
    clock = SimpleNamespace(now=0.0)
    sleeps = []
    state = {"contended": False}

    def sleep(seconds):
        sleeps.append(seconds)
        helper._RUNNING = False

    monkeypatch.setattr(helper.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(helper.time, "sleep", sleep)
    helper._RUNNING = True
    launches = []

    def kernel_s():
        # Two uncontended chunks establish the baseline; the rest run at a third of that rate.
        if len(launches) >= 40:
            state["contended"] = True
        return 0.003 if state["contended"] else 0.001

    torch = _fill_torch(helper, clock, kernel_s=kernel_s, launches=launches)
    helper._fill_device_keepalive(torch=torch, device=0, a=None, b=None, out=None, utilization=lambda *_: util, **_fill_settings())
    assert sleeps == [pytest.approx(expected_sleep, abs=0.002)]
    assert state["contended"]


def test_fill_mode_keeps_filling_under_contention_near_the_cancellation_threshold(helper, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    sleeps = []
    monkeypatch.setattr(helper.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(helper.time, "sleep", lambda seconds: sleeps.append(seconds))
    helper._RUNNING = True
    launches = []
    torch = _fill_torch(helper, clock, kernel_s=lambda: 0.001 if len(launches) < 40 else 0.003, stop_after_launches=400, launches=launches)
    helper._fill_device_keepalive(torch=torch, device=0, a=None, b=None, out=None, utilization=lambda *_: 50.0, **_fill_settings())
    assert sleeps == []
    assert len(launches) >= 400


def _run_fill_chunks(helper, monkeypatch, chunks, *, util=100.0, log_every_s=1e9):
    """Replay completed chunks while advancing both work and sleep on one clock."""
    clock = SimpleNamespace(now=0.0, chunk=-1)
    sleeps = []

    def run_chunk(**kwargs):
        clock.chunk += 1
        launches, elapsed = chunks[clock.chunk]
        clock.now += elapsed
        if clock.chunk == len(chunks) - 1:
            helper._RUNNING = False
        return launches, elapsed

    def sleep(seconds):
        sleeps.append((clock.chunk, clock.now, seconds))
        clock.now += seconds

    monkeypatch.setattr(helper.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(helper.time, "sleep", sleep)
    monkeypatch.setattr(helper, "_run_chunk", run_chunk)
    helper._RUNNING = True
    torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda device: nullcontext()))
    helper._fill_device_keepalive(
        torch=torch,
        device=0,
        a=None,
        b=None,
        out=None,
        utilization=lambda *_: util,
        log_every_s=log_every_s,
        **_fill_settings(),
    )
    return sleeps


def test_fill_yields_through_sustained_contention_after_a_full_idle_window(helper, monkeypatch):
    # Two idle seconds saturate own-work history. The next 2,000 chunks run at
    # one tenth of that rate; an adapting baseline would eventually hide this.
    warmup_chunks = 80
    busy_chunks = 2000
    chunks = [(100, 0.025)] * warmup_chunks + [(10, 0.025)] * busy_chunks
    sleeps = _run_fill_chunks(helper, monkeypatch, chunks)

    assert sleeps
    assert sleeps[0][0] >= warmup_chunks
    assert sleeps[0][1] <= 2.0 + 1.0 + 0.025
    busy_sleep_chunks = {chunk for chunk, _, seconds in sleeps if seconds == 0.5}
    # After the old window has expired, every contended chunk yields, including
    # the end of a long workload. This catches both accounting and baseline drift.
    assert set(range(warmup_chunks + 40, len(chunks))) <= busy_sleep_chunks


def test_fill_resumes_without_pauses_when_uncontended_throughput_returns(helper, monkeypatch):
    warmup_chunks = 80
    busy_chunks = 80
    chunks = [(100, 0.025)] * warmup_chunks + [(10, 0.025)] * busy_chunks + [(100, 0.025)] * 80
    sleeps = _run_fill_chunks(helper, monkeypatch, chunks)

    assert sleeps
    assert all(warmup_chunks <= chunk < warmup_chunks + busy_chunks for chunk, _, _ in sleeps)


def test_fill_does_not_count_a_long_synchronization_wait_as_own_work(helper, monkeypatch):
    chunks = [(100, 0.025)] * 80 + [(10, 3.0)]
    sleeps = _run_fill_chunks(helper, monkeypatch, chunks)

    assert sleeps == [(80, pytest.approx(5.0), 0.5)]


@pytest.mark.parametrize(
    "own_work,now,expected_util",
    [
        ([], 1.0, 0.0),
        # Only half a second of the first chunk overlaps: 0.1 s service from
        # that chunk, plus 0.1 s from the second, yields 20 percent utilization.
        ([(-1.0, 0.5, 0.3), (0.5, 1.0, 0.1)], 1.0, 20.0),
        ([(-3.0, 0.0, 0.2)], 1.0, 0.0),
        # A three-second blocked chunk contributes only its overlapping third.
        ([(-1.0, 2.0, 0.3)], 2.0, 10.0),
    ],
)
def test_own_service_estimate_counts_only_the_window_overlap(helper, own_work, now, expected_util):
    assert helper._estimated_own_utilization(own_work, now=now, window_s=1.0) == pytest.approx(expected_util)


def test_fill_logs_contention_and_corresponding_utilization_estimates(helper, monkeypatch, capsys):
    _run_fill_chunks(helper, monkeypatch, [(100, 0.025)] * 80 + [(10, 0.025)] * 80, log_every_s=1.0)
    last_line = capsys.readouterr().out.splitlines()[-1]
    fields = dict(field.split("=", 1) for field in last_line.split() if "=" in field)

    assert float(fields["last_util"]) == 100.0
    assert 0.0 < float(fields["last_own_util_estimate"]) < 15.0
    assert float(fields["last_other_util_estimate"]) >= 85.0
    assert int(fields["contention_checks"]) > 0
    assert fields["baseline_rate"] == "4000/s"
