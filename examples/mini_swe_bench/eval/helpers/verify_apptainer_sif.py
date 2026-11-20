#!/usr/bin/env python3
"""Fast verifier for the mini_swe_bench Apptainer/Docker image-source fix.

WHY THIS EXISTS
---------------
The training run produced thousands of identical observations of the form

    INFO: Fetching OCI image...
    FATAL: ... GET .../manifests/latest: TOOMANYREQUESTS (anonymous pull rate limit)

because the env launches `apptainer exec docker://<image>:latest` for EVERY agent
command, so each command re-resolves the `:latest` tag against Docker Hub. Under
RL (many rollouts x many steps x parallel workers) that blows past the anonymous
pull limit and every container build fails — the failure text becomes the model's
observation.

The fix routes execs through a LOCAL `.sif` file (pulled once) so Docker Hub is
never contacted at rollout time. This script verifies that WITHOUT a 6-hour
training run: the broken behaviour lives entirely in the env/image layer, which
needs no model, GPU, Ray, vLLM or wandb.

WHAT IT CHECKS  (run the same script before and after the fix to see it flip)
    [A] image resolution + ensure_image_available() returns a local .sif (post-fix)
        vs the raw docker:// uri (pre-fix).
    [B] a real command runs inside the container and returns REAL output
        (a `MARKER_<n>` sentinel + a /testbed listing), with NONE of the
        Fetching-OCI / Converting-OCI / TOOMANYREQUESTS / FATAL signatures.
    [C] a second exec is fast (warm) and equally clean.
    [D] (optional, --netns) the local .sif execs with the network severed
        (`unshare -rn`) -> undeniable proof nothing contacts Docker Hub.
    [E] (optional, --concurrency N) N processes asking for the SAME cold image
        trigger exactly ONE `apptainer pull` (the per-image lock works) -> the
        8-parallel-worker scenario, validated in seconds.

Part A/B/C import ONLY shared/apptainer_utils.py + shared/skyrl_utils.py (stdlib
only — no rllm/torch), so it runs on any CPU node with apptainer on PATH.
Part F (--full-env) additionally drives MiniSWEnv.reset()/step() for an
end-to-end integration check and is skipped gracefully if rllm can't import.

USAGE (a CPU node is enough)
    export APPTAINER_SIF_DIR=/scratch/$USER/sif
    export APPTAINER_CACHEDIR=/scratch/$USER/apptainer-cache
    export APPTAINER_TMPDIR=/scratch/$USER/apptainer_tmp
    python examples/mini_swe_bench/eval/helpers/verify_apptainer_sif.py \
        --instance getmoto__moto-5601 [--netns] [--concurrency 4] [--full-env]

Exit code is 0 unless --strict is passed (then non-zero if any [FAIL]).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Import the exact code paths the trainer uses. stdlib-only modules, so this is
# safe on a bare CPU node without the heavy ML venv.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.mini_swe_bench.shared.apptainer_utils import (  # noqa: E402
    _apptainer_or_singularity,
    build_apptainer_exec_command,
    ensure_image_available,
    run_apptainer_exec,
)
from examples.mini_swe_bench.shared.skyrl_utils import get_docker_image_name  # noqa: E402

# Signatures that mean "we touched Docker Hub / the build path" — must be ABSENT
# once execs run against a local .sif.
NETWORK_SIGNATURES = (
    "TOOMANYREQUESTS",
    "pull rate limit",
    "Fetching OCI image",
    "Converting OCI blobs",
    "FATAL:",
    "Unable to handle docker",
)

_RESULTS: list[tuple[str, str]] = []  # (level, "PASS"/"FAIL"/"SKIP"/"INFO")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _mount_for(path: Path) -> dict:
    """Return the deepest Linux mount containing an existing scratch path."""
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields, tail = left.split(), right.split()
        mount = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), fields[4])
        if path == Path(mount) or Path(mount) in path.parents:
            matches.append({"mountpoint": mount, "filesystem": tail[0], "source": tail[1]})
    _require(bool(matches), f"No mount record for scratch: {path}")
    return max(matches, key=lambda row: len(row["mountpoint"]))


def _scratch_preflight(gate: dict) -> dict:
    scratch = Path(gate["literal_path"])
    _require(scratch.is_absolute() and scratch.is_dir() and not scratch.is_symlink(), "Scratch must be an existing absolute directory, not a symlink")
    _require(scratch.stat().st_uid == os.getuid(), "Scratch is not owned by the current user")
    for key in ("TMPDIR", "MINISW_SCRATCH_ROOT"):
        _require(os.environ.get(key) == str(scratch), f"{key} differs from registered scratch")
    _require(Path(tempfile.gettempdir()).resolve() == scratch.resolve(), "Python tempfile resolved outside registered scratch")
    filesystem = _mount_for(scratch.resolve())
    local_filesystems = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs"}
    allowed = set(gate.get("allowed_local_disk_filesystems", local_filesystems)) & local_filesystems
    _require(filesystem["filesystem"] in allowed, "Scratch is not a verified local disk filesystem")
    stats = os.statvfs(scratch)
    required_bytes = int(gate["required_free_bytes"])
    required_inodes = int(gate["required_free_inodes"])
    _require(required_bytes > 0 and required_inodes > 0, "Scratch byte and inode floors must be positive")
    # Optional metadata-derived allowance for filesystems allocating above 4KiB.
    if stats.f_frsize > 4096:
        fields = ("concurrency", "max_full_image_entries", "max_testbed_entries")
        _require(all(key in gate for key in fields), "Scratch blocks exceed 4KiB; supply concurrency and image/testbed entry counts")
        extra = int(gate["concurrency"]) * (int(gate["max_full_image_entries"]) + int(gate["max_testbed_entries"])) * (stats.f_frsize - 4096)
        required_bytes = math.ceil((required_bytes + extra) / (1024**3)) * 1024**3
    capacity = {
        "free_bytes": stats.f_bavail * stats.f_frsize,
        "free_inodes": stats.f_favail,
        "filesystem_fragment_bytes": stats.f_frsize,
        "required_free_bytes": required_bytes,
        "required_free_inodes": required_inodes,
    }
    _require(capacity["free_bytes"] >= required_bytes, "Insufficient free scratch bytes")
    _require(capacity["free_inodes"] >= required_inodes, "Insufficient free scratch inodes")
    # Create only a temporary child; never clean or reuse task-owned directories.
    with tempfile.TemporaryDirectory(prefix="miniswe-preflight-", dir=scratch) as child:
        with (Path(child) / "write-probe").open("xb") as handle:
            handle.write(b"\0" * 4096)
            handle.flush()
            os.fsync(handle.fileno())
    return {"path": str(scratch), "tempfile_parent": tempfile.gettempdir(), "filesystem": filesystem, "capacity": capacity, "write_fsync_passed": True}


def _probe_container(image: dict, scratch: Path) -> dict:
    """Check a prepared local SIF using the same flags and overlay as rollouts."""
    path = Path(image["path"])
    _require(path.is_absolute() and path.is_file() and path.suffix == ".sif", "Container probe requires an existing local SIF")
    command = (
        "printf 'MINISW_PREFLIGHT_OK\\n'; "
        "printf 'BASH_ENV=%s\\n' \"${BASH_ENV-}\"; "
        "printf 'BASE_COMMIT='; git -C /testbed rev-parse HEAD"
    )
    with tempfile.TemporaryDirectory(prefix="miniswe-preflight-overlay-", dir=scratch) as overlay:
        # An absolute local path cannot invoke the docker:// image-build branch.
        argv = build_apptainer_exec_command(str(path), command, overlay_path=overlay)
        result = subprocess.run(argv, text=True, capture_output=True, timeout=120, check=True)
    _require("MINISW_PREFLIGHT_OK" in result.stdout.splitlines(), "Container did not execute the readiness sentinel")
    _require(not _scan_signatures(result.stdout + result.stderr), "Container reported a build or runtime failure")
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    _require(values.get("BASH_ENV") == "", "Host BASH_ENV leaked into the clean container")
    if image.get("base_commit"):
        _require(values.get("BASE_COMMIT") == image["base_commit"], f"Container base commit differs for {image['task_id']}")
    return {"task_id": image["task_id"], "path": str(path), "base_commit": values.get("BASE_COMMIT"), "host_bash_env_absent": True, "stdout": result.stdout, "stderr": result.stderr}


def runtime_preflight(manifest_path: Path, receipt_path: Path) -> dict:
    """Fail closed before model startup; never prepare, replace, or delete images."""
    manifest_path, receipt_path = Path(manifest_path), Path(receipt_path)
    result = {
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "passed": False,
        "manifest": str(manifest_path),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    try:
        data = manifest_path.read_bytes()
        result["manifest_sha256"] = hashlib.sha256(data).hexdigest()
        manifest = json.loads(data)
        _require(manifest.get("schema_version") == 1, "Unsupported runtime preflight manifest version")
        expected = manifest.get("expected_environment", {})
        result["environment"] = {key: os.environ.get(key) for key in set(expected) | {"TMPDIR", "MINISW_SCRATCH_ROOT", "APPTAINER_TMPDIR", "APPTAINER_CACHEDIR", "APPTAINER_SIF_DIR", "BASH_ENV"}}
        for key, value in expected.items():
            _require(os.environ.get(key) == value, f"Registered environment differs: {key}")
        result["scratch"] = _scratch_preflight(manifest["scratch_gate"])
        for item in manifest.get("input_files", []):
            _require(hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"], f"Input file hash differs: {item['path']}")
        images = manifest["prepared_container_images"]
        _require(bool(images), "Runtime preflight requires prepared container images")
        by_id = {}
        for image in images:
            path = Path(image["path"])
            _require(path.is_absolute() and path.suffix == ".sif" and path.is_file(), f"Prepared SIF missing: {image['task_id']}")
            _require(path.stat().st_size == image["size"], f"Prepared SIF size differs: {image['task_id']}")
            if "mtime_ns" in image:
                _require(path.stat().st_mtime_ns == image["mtime_ns"], f"Prepared SIF timestamp differs: {image['task_id']}")
            _require(image["task_id"] not in by_id, "Duplicate prepared image task ID")
            by_id[image["task_id"]] = image
        result["prepared_sifs"] = {"count": len(images), "metadata_matched": True}
        probe_ids = manifest.get("probe_task_ids", [images[0]["task_id"]])
        _require(bool(probe_ids) and all(task_id in by_id for task_id in probe_ids), "Container probes must select prepared task IDs")
        result["container_probes"] = [_probe_container(by_id[task_id], Path(result["scratch"]["path"])) for task_id in probe_ids]
        result["passed"] = True
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        # Only the explicitly requested receipt is replaced, via an owned temp file.
        with tempfile.NamedTemporaryFile(mode="w", dir=receipt_path.parent, prefix=receipt_path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(receipt_path)


def _log(level: str, status: str, msg: str) -> None:
    _RESULTS.append((level, status))
    print(f"  [{status:4}] {level}: {msg}", flush=True)


def _scan_signatures(text: str) -> list[str]:
    return [s for s in NETWORK_SIGNATURES if s in text]


def _image_uri_for(instance_id: str, data_source: str) -> str:
    """Mirror MiniSWEnv._get_image_uri() exactly."""
    instance = {"instance_id": instance_id}
    name = get_docker_image_name(instance, data_source).replace("docker.io/", "")
    return name if name.startswith("docker://") else f"docker://{name}"


def _default_apptainer_dirs() -> None:
    """Cache + .sif on /scratch (persistent, big); build temp on NODE-LOCAL disk.

    The .sif output and OCI cache can live on /scratch, but the SIF *build* temp
    must be a local POSIX filesystem: unpacking an OCI layer on NFS/Ceph /scratch
    fails with `unpriv.lclearxattrs ... system.nfs4_dacl: invalid argument`. The
    training slurm mirrors this (APPTAINER_TMPDIR under local /tmp).
    """
    user = os.environ.get("USER", "unknown")
    scratch = Path("/scratch") / user
    os.environ.setdefault("APPTAINER_SIF_DIR", str(scratch / "sif"))
    os.environ.setdefault("APPTAINER_CACHEDIR", str(scratch / "apptainer-cache"))
    local_tmp = os.environ.get("SLURM_TMPDIR", "/tmp")
    os.environ.setdefault("APPTAINER_TMPDIR", f"{local_tmp}/apptainer_tmp_{user}")
    # Keep singularity-prefixed vars aligned so apptainer doesn't warn/diverge.
    os.environ["SINGULARITY_CACHEDIR"] = os.environ["APPTAINER_CACHEDIR"]
    os.environ["SINGULARITY_TMPDIR"] = os.environ["APPTAINER_TMPDIR"]
    for key in ("APPTAINER_SIF_DIR", "APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Level A — image resolution + ensure_image_available()
# --------------------------------------------------------------------------- #
def level_a(uri: str) -> str:
    print("\n[A] Resolve image + ensure_image_available()")
    print(f"      docker uri: {uri}")

    t0 = time.time()
    target = ensure_image_available(uri)
    dt0 = time.time() - t0
    print(f"      ensure_image_available() -> {target}  ({dt0:.1f}s)")

    # Post-fix target is a LOCAL sandbox dir (or a .sif on clusters where that builds).
    p = Path(target)
    is_local_target = (not str(target).startswith("docker://")) and (str(target).endswith(".sif") or str(target).endswith(".sandbox") or p.is_dir())
    if is_local_target and p.exists():
        if p.is_dir():
            ready = (p / ".ready").exists()
            _log("A.local_target", "PASS" if ready else "FAIL", f"returned a local sandbox dir ({'complete: .ready present' if ready else 'MISSING .ready marker'})")
        else:
            _log("A.local_target", "PASS", f"returned an existing local .sif ({p.stat().st_size / 1e9:.2f} GB)")
    elif str(target).startswith("docker://"):
        _log("A.local_target", "FAIL", "returned a docker:// uri (PRE-FIX behaviour — exec will hit Docker Hub each call)")
    else:
        _log("A.local_target", "FAIL", f"returned a local target that does NOT exist: {target}")

    # Second call should be a no-op (already local / already recorded).
    t1 = time.time()
    target2 = ensure_image_available(uri)
    dt1 = time.time() - t1
    if target2 == target and dt1 < 2.0:
        _log("A.warm_call", "PASS", f"2nd ensure_image_available() instant ({dt1:.2f}s), same target")
    else:
        _log("A.warm_call", "INFO", f"2nd call {dt1:.2f}s -> {target2}")

    # Show the actual exec command the env would build — the clearest before/after tell.
    try:
        cmd = build_apptainer_exec_command(uri, "true")
        # Built command tail is [..., <image>, "bash", "-c", <command>] -> image is cmd[-4].
        exec_target = cmd[-4] if len(cmd) >= 4 else "?"
        print(f"      exec target in built command: {exec_target}")
        if str(exec_target).startswith("docker://"):
            _log("A.exec_target", "FAIL", "built command execs docker:// (registry round-trip every call)")
        elif str(exec_target).endswith((".sif", ".sandbox")) or Path(exec_target).is_dir():
            _log("A.exec_target", "PASS", "built command execs a local target (no registry round-trip)")
    except Exception as e:  # noqa: BLE001
        _log("A.exec_target", "INFO", f"could not build command: {e}")

    return target


# --------------------------------------------------------------------------- #
# Level B/C — real command output, clean of network signatures
# --------------------------------------------------------------------------- #
def _run_probe(uri: str, label: str) -> str:
    sentinel = "MARKER_$((2+2))"  # bash arithmetic -> MARKER_4 proves a real shell ran
    cmd = f"echo {sentinel}; echo '--- /testbed ---'; ls -1 /testbed 2>&1 | head -n 5"
    t0 = time.time()
    output, code = run_apptainer_exec(uri, cmd, timeout=600)
    dt = time.time() - t0
    print(f"\n[{label}] exec `ls /testbed` ({dt:.1f}s, exit={code})")
    preview = "\n".join(("        " + ln) for ln in output.splitlines()[:8])
    print(preview or "        (no output)")
    return output


def level_b(uri: str) -> None:
    output = _run_probe(uri, "B")
    hits = _scan_signatures(output)
    if hits:
        _log("B.clean", "FAIL", f"observation still contains network/build errors: {hits}")
    else:
        _log("B.clean", "PASS", "no Fetching-OCI / TOOMANYREQUESTS / FATAL signatures")
    if "MARKER_4" in output:
        _log("B.real_shell", "PASS", "sentinel MARKER_4 present — a real shell ran the command")
    else:
        _log("B.real_shell", "FAIL", "sentinel MARKER_4 absent — command did not really execute")


def level_c(uri: str) -> None:
    t0 = time.time()
    output = _run_probe(uri, "C (warm)")
    dt = time.time() - t0
    hits = _scan_signatures(output)
    if not hits and dt < 30:
        _log("C.warm_fast", "PASS", f"2nd exec clean and fast ({dt:.1f}s)")
    elif not hits:
        _log("C.warm_fast", "INFO", f"2nd exec clean but slow ({dt:.1f}s)")
    else:
        _log("C.warm_fast", "FAIL", f"2nd exec dirty: {hits}")


# --------------------------------------------------------------------------- #
# Level D — provably no network (optional)
# --------------------------------------------------------------------------- #
def level_d(target: str) -> None:
    print("\n[D] network-severed exec (unshare -rn)")
    if str(target).startswith("docker://"):
        _log("D.netns", "SKIP", "target is docker:// (pre-fix) — a no-network test would trivially fail")
        return
    apptainer = _apptainer_or_singularity()
    if apptainer is None:
        _log("D.netns", "SKIP", "no apptainer/singularity on PATH")
        return
    inner = f"{apptainer} exec --containall --cleanenv {target} bash -c 'echo NETLESS_OK'"
    try:
        res = subprocess.run(["unshare", "-rn", "bash", "-c", inner], capture_output=True, text=True, timeout=120)
        if "NETLESS_OK" in (res.stdout + res.stderr):
            _log("D.netns", "PASS", "local .sif executed with the network namespace severed")
        else:
            _log("D.netns", "INFO", f"unshare ran but no sentinel (rc={res.returncode}); likely userns/apptainer interaction, not network")
    except FileNotFoundError:
        _log("D.netns", "SKIP", "`unshare` not available")
    except subprocess.TimeoutExpired:
        _log("D.netns", "INFO", "unshare exec timed out")


# --------------------------------------------------------------------------- #
# Level E — concurrency / single-pull lock (optional, destructive: re-pulls cold)
# --------------------------------------------------------------------------- #
def level_e(uri: str, n: int) -> None:
    print(f"\n[E] concurrency: {n} processes request the SAME cold image")
    target = ensure_image_available(uri)
    if str(target).startswith("docker://"):
        _log("E.single_pull", "SKIP", "pre-fix (no local target) — lock test is meaningless")
        return
    tgt = Path(target)
    if tgt.exists():
        if tgt.is_dir():
            shutil.rmtree(tgt, ignore_errors=True)  # force a cold cache so the lock must arbitrate
        else:
            tgt.unlink()
        print(f"      removed {tgt} to force a cold build")
    worker = (
        "import time,os,sys;"
        "sys.path.insert(0, os.environ['REPO_ROOT']);"
        "from examples.mini_swe_bench.shared.apptainer_utils import ensure_image_available;"
        "t=time.time();"
        f"p=ensure_image_available({uri!r});"
        "print(f'{os.getpid()} {time.time()-t:.1f}s -> {p}')"
    )
    env = {**os.environ, "REPO_ROOT": str(REPO_ROOT)}
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, "-c", worker], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) for _ in range(n)]
    outs = [p.communicate()[0].decode().strip() for p in procs]
    dt = time.time() - t0
    for o in outs:
        print(f"        {o}")
    # If the lock works, exactly one process pays the long build; the rest return
    # quickly after it lands.
    final = Path(target)
    ok = (final / ".ready").exists() if final.is_dir() else final.exists()
    if ok:
        _log("E.single_pull", "PASS", f"all {n} workers resolved to a valid target in {dt:.1f}s (one build, shared)")
    else:
        _log("E.single_pull", "FAIL", "after concurrent requests the target still does not exist")


# --------------------------------------------------------------------------- #
# Level F — full MiniSWEnv integration (optional; needs rllm venv)
# --------------------------------------------------------------------------- #
def level_f(instance_id: str, data_source: str) -> None:
    print("\n[F] full MiniSWEnv.reset()/step() integration")
    try:
        from examples.mini_swe_bench.env import MiniSWEnv  # noqa: E402
    except Exception as e:  # noqa: BLE001
        _log("F.integration", "SKIP", f"rllm/env import unavailable on this venv: {e}")
        return
    task = {"instance": {"instance_id": instance_id, "problem_statement": "(verify)"}, "data_source": data_source}
    env = MiniSWEnv(task=task, max_steps=5, step_timeout=120)
    obs, info = env.reset(task)
    print(f"      reset() ok, instance={info.get('instance_id')}")
    action = "<function=execute_bash>\n<parameter=command>echo MARKER_$((6*7)); ls /testbed | head -3</parameter>\n</function>"
    obs, reward, done, info = env.step(action)
    preview = "\n".join("        " + ln for ln in str(obs).splitlines()[:6])
    print(preview)
    hits = _scan_signatures(str(obs))
    if not hits and "MARKER_42" in str(obs):
        _log("F.integration", "PASS", "env.step() returned real /testbed output, no Docker errors")
    elif hits:
        _log("F.integration", "FAIL", f"env.step() observation still shows: {hits}")
    else:
        _log("F.integration", "INFO", "env.step() ran but sentinel absent — inspect output above")
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="getmoto__moto-5601", help="SWE-Gym instance_id whose image to test (default: getmoto__moto-5601)")
    ap.add_argument("--data-source", default="SumanthRH/SWE-Gym-Subset")
    ap.add_argument("--netns", action="store_true", help="run Level D (no-network proof)")
    ap.add_argument("--concurrency", type=int, default=0, help="run Level E with N processes (re-pulls cold!)")
    ap.add_argument("--full-env", action="store_true", help="run Level F (MiniSWEnv integration; needs rllm)")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any [FAIL]")
    ap.add_argument("--runtime-preflight", type=Path, help="check a registered runtime manifest without preparing images")
    ap.add_argument("--preflight-receipt", type=Path, help="JSON receipt for --runtime-preflight")
    args = ap.parse_args()

    if args.runtime_preflight is not None:
        if args.preflight_receipt is None:
            ap.error("--runtime-preflight requires --preflight-receipt")
        runtime_preflight(args.runtime_preflight, args.preflight_receipt)
        return 0
    if args.preflight_receipt is not None:
        ap.error("--preflight-receipt requires --runtime-preflight")

    _default_apptainer_dirs()
    if _apptainer_or_singularity() is None:
        print("ERROR: neither apptainer nor singularity is on PATH — run on a node that has it.", file=sys.stderr)
        return 2

    print("=" * 72)
    print("mini_swe_bench Apptainer/Docker image-source verifier")
    print(f"  instance     : {args.instance}")
    print(f"  data_source  : {args.data_source}")
    print(f"  SIF_DIR      : {os.environ['APPTAINER_SIF_DIR']}")
    print(f"  CACHEDIR     : {os.environ['APPTAINER_CACHEDIR']}")
    print("=" * 72)

    uri = _image_uri_for(args.instance, args.data_source)
    target = level_a(uri)
    level_b(uri)
    level_c(uri)
    if args.netns:
        level_d(target)
    if args.concurrency:
        level_e(uri, args.concurrency)
    if args.full_env:
        level_f(args.instance, args.data_source)

    # ---- summary ----
    print("\n" + "=" * 72)
    n_pass = sum(1 for _, s in _RESULTS if s == "PASS")
    n_fail = sum(1 for _, s in _RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s in _RESULTS if s == "SKIP")
    print(f"SUMMARY: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    verdict = (
        "FIX VERIFIED"
        if (n_fail == 0 and n_pass > 0)
        else "PRE-FIX / BROKEN"
        if any(label.endswith("exec_target") and status == "FAIL" for label, status in _RESULTS)
        else "INCONCLUSIVE — see lines above"
    )
    print(f"VERDICT: {verdict}")
    print("=" * 72)
    return 1 if (args.strict and n_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
