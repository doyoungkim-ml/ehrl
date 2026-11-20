"""CPU checks that the opt-in MiniSWE preflight fails before runtime startup."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.mini_swe_bench.eval.helpers import verify_apptainer_sif as verifier
from examples.mini_swe_bench.shared import apptainer_utils

REPO = Path(__file__).resolve().parents[2]
GIB = 1024**3


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    image = tmp_path / "prepared.sif"
    image.write_bytes(b"prepared image must stay unchanged")
    source = tmp_path / "tasks.csv"
    source.write_text("instance_id\nexample__repo-1\n")
    manifest = {
        "schema_version": 1,
        "scratch_gate": {
            "literal_path": str(scratch),
            "required_free_bytes": 8 * GIB,
            "required_free_inodes": 1000,
        },
        "prepared_container_images": [{
            "task_id": "example__repo-1",
            "path": str(image),
            "size": image.stat().st_size,
            "mtime_ns": image.stat().st_mtime_ns,
            "base_commit": "0123456789abcdef",
        }],
        "input_files": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
    }
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("MINISW_SCRATCH_ROOT", str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(verifier, "_mount_for", lambda path: {"filesystem": "ext4", "mountpoint": "/tmp", "source": "/dev/test"})
    capacity = SimpleNamespace(f_frsize=4096, f_bavail=16 * GIB // 4096, f_favail=2000)
    monkeypatch.setattr(verifier.os, "statvfs", lambda path: capacity)
    probe = Mock(return_value={"task_id": "example__repo-1", "host_bash_env_absent": True})
    monkeypatch.setattr(verifier, "_probe_container", probe)
    build = Mock(side_effect=AssertionError("Preflight must not prepare images"))
    monkeypatch.setattr(verifier, "ensure_image_available", build)
    return SimpleNamespace(
        scratch=scratch, image=image, source=source, manifest=manifest, capacity=capacity,
        path=tmp_path / "manifest.json", receipt=tmp_path / "receipts" / "runtime.json",
        probe=probe, build=build,
    )


def run_preflight(prepared):
    prepared.path.write_text(json.dumps(prepared.manifest))
    return verifier.runtime_preflight(prepared.path, prepared.receipt)


def assert_rejected(prepared, message):
    with pytest.raises(RuntimeError, match=message):
        run_preflight(prepared)
    receipt = json.loads(prepared.receipt.read_text())
    assert receipt["passed"] is False
    assert message in receipt["error"]
    assert receipt["finished_at"]
    prepared.probe.assert_not_called()
    prepared.build.assert_not_called()


def test_preflight_receipt_records_capacity_and_preserves_prepared_image(prepared):
    before = (prepared.image.read_bytes(), prepared.image.stat().st_mtime_ns)
    result = run_preflight(prepared)
    assert result == json.loads(prepared.receipt.read_text())
    assert result["passed"] is True
    assert result["manifest_sha256"] == hashlib.sha256(prepared.path.read_bytes()).hexdigest()
    assert result["scratch"]["tempfile_parent"] == str(prepared.scratch)
    assert result["scratch"]["write_fsync_passed"] is True
    assert result["scratch"]["capacity"]["free_bytes"] == 16 * GIB
    assert result["prepared_sifs"] == {"count": 1, "metadata_matched": True}
    prepared.probe.assert_called_once_with(prepared.manifest["prepared_container_images"][0], prepared.scratch)
    prepared.build.assert_not_called()
    assert (prepared.image.read_bytes(), prepared.image.stat().st_mtime_ns) == before
    assert list(prepared.scratch.iterdir()) == []


@pytest.mark.parametrize(("field", "value", "error"), [
    ("f_bavail", 1, "Insufficient free scratch bytes"),
    ("f_favail", 1, "Insufficient free scratch inodes"),
    ("f_frsize", 8192, "Scratch blocks exceed 4KiB"),
])
def test_insufficient_scratch_fails_before_container_probe(prepared, field, value, error):
    setattr(prepared.capacity, field, value)
    assert_rejected(prepared, error)


@pytest.mark.parametrize("filesystem", ["tmpfs", "nfs4", "ceph"])
def test_nonlocal_scratch_is_rejected_even_if_manifest_allows_it(prepared, monkeypatch, filesystem):
    prepared.manifest["scratch_gate"]["allowed_local_disk_filesystems"] = [filesystem]
    monkeypatch.setattr(verifier, "_mount_for", lambda path: {"filesystem": filesystem})
    assert_rejected(prepared, "Scratch is not a verified local disk filesystem")


def test_cached_python_tempfile_must_resolve_to_registered_scratch(prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert_rejected(prepared, "Python tempfile resolved outside registered scratch")


def test_inherited_environment_must_match_manifest(prepared, monkeypatch):
    prepared.manifest["expected_environment"] = {"APPTAINER_CACHEDIR": "/registered/cache"}
    monkeypatch.setenv("APPTAINER_CACHEDIR", "/inherited/cache")
    assert_rejected(prepared, "Registered environment differs: APPTAINER_CACHEDIR")


@pytest.mark.parametrize("change", ["missing", "size", "mtime"])
def test_every_image_is_checked_before_any_container_probe(prepared, change):
    second = dict(prepared.manifest["prepared_container_images"][0], task_id="example__repo-2")
    prepared.manifest["prepared_container_images"].append(second)
    if change == "missing":
        second["path"] = str(prepared.image.with_name("missing.sif"))
        error = "Prepared SIF missing"
    elif change == "size":
        second["size"] += 1
        error = "Prepared SIF size differs"
    else:
        second["mtime_ns"] += 1
        error = "Prepared SIF timestamp differs"
    before = prepared.image.read_bytes()
    assert_rejected(prepared, error)
    assert prepared.image.read_bytes() == before


def test_changed_input_file_is_rejected_before_container_probe(prepared):
    prepared.source.write_text("different input\n")
    assert_rejected(prepared, "Input file hash differs")


@pytest.mark.parametrize(("stdout", "stderr", "error"), [
    ("MINISW_PREFLIGHT_OK\nBASH_ENV=/host/init\nBASE_COMMIT=abc\n", "", "Host BASH_ENV leaked"),
    ("MINISW_PREFLIGHT_OK\nBASH_ENV=\nBASE_COMMIT=wrong\n", "", "Container base commit differs"),
    ("MINISW_PREFLIGHT_OK\nBASH_ENV=\nBASE_COMMIT=abc\n", "FATAL: failed", "Container reported a build or runtime failure"),
    ("BASH_ENV=\nBASE_COMMIT=abc\n", "", "Container did not execute the readiness sentinel"),
])
def test_container_output_must_prove_readiness(tmp_path, monkeypatch, stdout, stderr, error):
    image = tmp_path / "prepared.sif"
    image.write_bytes(b"SIF")
    monkeypatch.setattr(apptainer_utils, "_apptainer_or_singularity", lambda: "apptainer")
    monkeypatch.setattr(verifier.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=stdout, stderr=stderr)))
    with pytest.raises(RuntimeError, match=error):
        verifier._probe_container({"task_id": "task", "path": str(image), "base_commit": "abc"}, tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["prepared.sif"]


def test_container_probe_uses_local_image_clean_environment_and_owned_overlay(tmp_path, monkeypatch):
    image = tmp_path / "prepared.sif"
    image.write_bytes(b"SIF")
    monkeypatch.setattr(apptainer_utils, "_apptainer_or_singularity", lambda: "apptainer")
    command = Mock(return_value=SimpleNamespace(stdout="MINISW_PREFLIGHT_OK\nBASH_ENV=\nBASE_COMMIT=abc\n", stderr=""))
    monkeypatch.setattr(verifier.subprocess, "run", command)
    result = verifier._probe_container({"task_id": "task", "path": str(image), "base_commit": "abc"}, tmp_path)
    argv = command.call_args.args[0]
    assert result["host_bash_env_absent"] is True
    assert str(image) in argv
    assert all(not item.startswith("docker://") for item in argv)
    assert {"--containall", "--cleanenv", "--no-home", "--no-eval"}.issubset(argv)
    overlay = Path(argv[argv.index("--overlay") + 1])
    assert overlay.parent == tmp_path and not overlay.exists()
    assert "--writable-tmpfs" not in argv


@pytest.mark.parametrize("mode", ["ace", "vanilla"])
@pytest.mark.parametrize("receipt_override", [False, True])
def test_failed_preflight_stops_maintained_launcher_before_model_server(tmp_path, mode, receipt_override):
    binary = tmp_path / "bin"
    binary.mkdir()
    fake_python = binary / "python"
    fake_python.write_text(f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args and args[0].endswith('/gpu_runtime.py'):
    sys.exit(0)
phase = None
if len(args) > 1 and args[0] == '-c' and 'swegym.harness.grading' in args[1]:
    phase = 'grader'
elif args[:2] == ['-m', 'examples.mini_swe_bench.eval.helpers.verify_apptainer_sif']:
    phase = 'preflight'
elif args[:2] == ['-m', 'vllm.entrypoints.openai.api_server']:
    phase = 'server'
if phase:
    with Path(os.environ['TEST_EVENTS']).open('a') as handle:
        handle.write(json.dumps(dict(phase=phase, tmpdir=os.environ.get('TMPDIR'), argv=args)) + '\\n')
    sys.exit(37 if phase == 'preflight' else 0)
os.execv({sys.executable!r}, [{sys.executable!r}, *args])
""")
    fake_python.chmod(0o755)
    apptainer = binary / "apptainer"
    apptainer.write_text("#!/bin/sh\nexit 0\n")
    apptainer.chmod(0o755)
    scratch = tmp_path / "scratch"
    events = tmp_path / "events.jsonl"
    output = tmp_path / "output"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    env = {
        "PATH": str(binary) + os.pathsep + os.environ["PATH"], "HOME": str(tmp_path),
        "USER": "test", "RLLM_DIR": str(REPO), "MODEL_SOURCE": "hub", "MODEL_NAME": "Qwen/test",
        "PYTHON_BIN": str(fake_python), "OUTPUT_DIR": str(output), "TEST_EVENTS": str(events),
        "MINISW_PREFLIGHT_MANIFEST": str(manifest), "MINISW_SCRATCH_ROOT": str(scratch),
        "APPTAINER_TMPDIR": str(scratch / "apptainer"), "APPTAINER_CACHEDIR": str(tmp_path / "cache"),
        "APPTAINER_SIF_DIR": str(tmp_path / "sif"), "TMPDIR": "/wrong/inherited/tmpdir",
    }
    expected_receipt = tmp_path / "output.runtime-preflight.json"
    if receipt_override:
        expected_receipt = tmp_path / "receipts" / "custom-preflight.json"
        env["MINISW_PREFLIGHT_RECEIPT"] = str(expected_receipt)
    result = subprocess.run(["bash", str(REPO / "examples/mini_swe_bench/eval" / mode / "eval.slurm")], env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 37, result.stdout + result.stderr
    observed = [json.loads(line) for line in events.read_text().splitlines()]
    assert [entry["phase"] for entry in observed] == ["grader", "preflight"]
    assert all(entry["tmpdir"] == str(scratch) for entry in observed)
    argv = observed[-1]["argv"]
    assert argv[argv.index("--runtime-preflight") + 1] == str(manifest)
    actual_receipt = Path(argv[argv.index("--preflight-receipt") + 1])
    assert actual_receipt == expected_receipt
    assert not actual_receipt.is_relative_to(output)


@pytest.mark.parametrize("mode", ["ace", "vanilla"])
def test_opt_in_keeps_dry_run_commands_and_creates_no_runtime_files(tmp_path, mode):
    script = REPO / "examples/mini_swe_bench/eval" / mode / "eval.slurm"
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": "test", "RLLM_DIR": str(REPO), "RUNS_BASE": str(tmp_path / "runs")}
    baseline = subprocess.run(["bash", str(script), "--dry-run"], env=env, text=True, capture_output=True, timeout=30)
    env.update(MINISW_PREFLIGHT_MANIFEST=str(tmp_path / "missing-manifest.json"), MINISW_SCRATCH_ROOT=str(tmp_path / "scratch"))
    opted_in = subprocess.run(["bash", str(script), "--dry-run"], env=env, text=True, capture_output=True, timeout=30)
    assert baseline.returncode == opted_in.returncode == 0, baseline.stderr + opted_in.stderr

    def evaluator_args(result):
        return [token for token in shlex.split(result.stdout.splitlines()[-1]) if not token.startswith("http://127.0.0.1:")]

    assert evaluator_args(baseline) == evaluator_args(opted_in)
    assert list(tmp_path.iterdir()) == []
