"""Shared Apptainer/Singularity execution helpers for mini_swe_bench."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from filelock import FileLock

_PULLED_IMAGES: set[str] = set()


def _apptainer_or_singularity() -> str | None:
    for cmd in ["apptainer", "singularity"]:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True, timeout=5)
            return cmd
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _ensure_tmpdir() -> None:
    if "APPTAINER_TMPDIR" not in os.environ:
        base = os.environ.get("SLURM_TMPDIR", "/tmp")
        os.environ["APPTAINER_TMPDIR"] = f"{base}/apptainer_tmp_{os.environ.get('USER', 'unknown')}"
    Path(os.environ["APPTAINER_TMPDIR"]).mkdir(parents=True, exist_ok=True)


def _sif_dir() -> Path:
    return Path(os.environ.get("APPTAINER_SIF_DIR", Path(__file__).resolve().parents[3] / ".cache/mini_swe_bench/sif"))


def _image_key(image_uri: str) -> str:
    body = image_uri[len("docker://") :] if image_uri.startswith("docker://") else image_uri
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", body)[:120]
    digest = hashlib.sha1(image_uri.encode()).hexdigest()[:10]
    return f"{safe}__{digest}"


def sif_path_for(image_uri: str) -> Path:
    """Deterministic /scratch path for an image's .sif (ONE compressed file).

    The cluster's bundled `mksquashfs` segfaults, so apptainer's own `.sif` build
    is broken — but a user-installed (conda) `mksquashfs` works fine. So we build
    the .sif ourselves (see build_sif) and exec it DIRECTLY from /scratch: one
    file per image (file-quota-friendly), read-only mounted (no extraction, no
    node-local disk, no LRU). Published atomically (write .tmp, rename).
    """
    return _sif_dir() / f"{_image_key(image_uri)}.sif"


def _mksquashfs_bin() -> str | None:
    """A WORKING mksquashfs. The bundled apptainer one segfaults on large images,
    so we use a user-installed one (conda squashfs-tools). Resolution order:
    MKSQUASHFS_BIN env -> `mksquashfs` on PATH."""
    p = os.environ.get("MKSQUASHFS_BIN")
    if p and Path(p).exists():
        return p
    return shutil.which("mksquashfs")


def build_sif(image_uri: str, timeout: int = 2400) -> str:
    """PREPULL step: build an image to a single .sif on /scratch, once.

    Pipeline (all user-space, no admin): (1) `apptainer build --sandbox` unpacks
    the OCI layers to node-local disk (the unpack works; only the bundled
    mksquashfs is broken); (2) a WORKING mksquashfs (MKSQUASHFS_BIN / conda
    squashfs-tools) packs that sandbox into a squashfs; (3) `apptainer sif new` +
    `sif add` wrap the squashfs into a .sif (these do NOT call mksquashfs);
    (4) atomic publish to /scratch. Idempotent; per-image lock. Non-docker uris
    pass through.
    """
    if not image_uri.startswith("docker://"):
        return image_uri
    sif = sif_path_for(image_uri)
    if sif.exists():
        return str(sif)
    apptainer = _apptainer_or_singularity()
    if apptainer is None:
        raise RuntimeError("Neither apptainer nor singularity found on PATH")
    mksq = _mksquashfs_bin()
    if mksq is None:
        raise RuntimeError(
            "No working mksquashfs found. Install one (conda install -c conda-forge squashfs-tools) and set MKSQUASHFS_BIN to it — the bundled apptainer mksquashfs segfaults on these images."
        )
    sif.parent.mkdir(parents=True, exist_ok=True)
    _ensure_tmpdir()
    tmp_root = Path(os.environ["APPTAINER_TMPDIR"])
    with FileLock(str(sif) + ".lock", timeout=timeout + 900):
        if sif.exists():
            return str(sif)
        # Per-image name (not just pid) so concurrent builds of DIFFERENT images in
        # the same process (prepull thread pool) never share a build dir.
        local_build = tmp_root / f"sbxbuild_{sif.stem}.{os.getpid()}"
        sqsh = tmp_root / f"{sif.stem}.{os.getpid()}.sqsh"
        sif_tmp = sif.parent / f"{sif.name}.tmp.{os.getpid()}"
        shutil.rmtree(local_build, ignore_errors=True)
        sqsh.unlink(missing_ok=True)
        sif_tmp.unlink(missing_ok=True)
        try:
            # (1) unpack OCI -> node-local sandbox
            subprocess.run(
                [apptainer, "build", "--force", "--sandbox", str(local_build), image_uri],
                check=True,
                timeout=timeout,
            )
            # (2) pack with a WORKING mksquashfs -> node-local squashfs
            subprocess.run(
                [mksq, str(local_build), str(sqsh), "-noappend", "-no-progress"],
                check=True,
                timeout=timeout,
            )
            # (3) wrap squashfs into a SIF (no mksquashfs involved)
            subprocess.run([apptainer, "sif", "new", str(sif_tmp)], check=True, timeout=300)
            subprocess.run(
                [
                    apptainer,
                    "sif",
                    "add",
                    "--datatype",
                    "4",  # Partition
                    "--partfs",
                    "1",  # Squashfs
                    "--parttype",
                    "2",  # PrimSys (primary rootfs)
                    "--partarch",
                    "2",  # amd64
                    "--groupid",
                    "1",
                    str(sif_tmp),
                    str(sqsh),
                ],
                check=True,
                timeout=600,
            )
            os.replace(str(sif_tmp), str(sif))  # atomic publish (one file)
        finally:
            shutil.rmtree(local_build, ignore_errors=True)
            sqsh.unlink(missing_ok=True)
            sif_tmp.unlink(missing_ok=True)
    return str(sif)


def ensure_image_available(image_uri: str, timeout: int = 2400) -> str:
    """RUNTIME: resolve a docker:// image to its /scratch .sif, ready to exec.

    The .sif is built once by the prepull (build_sif) and execs DIRECTLY from
    /scratch — no extraction, no node-local disk, no Docker Hub at rollout time.
    So this is just: return the .sif path (lazy-building it if the prepull missed
    this image). Non-docker uris (already a local path/.sif) pass through.
    """
    if not image_uri.startswith("docker://"):
        if Path(image_uri).exists():
            _PULLED_IMAGES.add(image_uri)
        return image_uri

    sif = sif_path_for(image_uri)
    if not sif.exists():
        build_sif(image_uri, timeout=timeout)  # prepull missed it -> build now (under lock)
    _PULLED_IMAGES.add(image_uri)
    return str(sif)


def build_apptainer_exec_command(
    image_uri: str,
    command: str,
    bind_dirs: list[tuple[Path | str, str]] | None = None,
    env_vars: dict[str, str] | None = None,
    use_torch_flags: bool = True,
    overlay_path: Path | str | None = None,
) -> list[str]:
    cmd_name = _apptainer_or_singularity()
    if cmd_name is None:
        raise RuntimeError("Neither apptainer nor singularity found on PATH")

    bind_dirs = bind_dirs or []
    env_vars = env_vars or {}
    # Replaces the previous `--compat` (which implies --writable-tmpfs and is
    # capped by /etc/apptainer/apptainer.conf `sessiondir max size = 64` MiB).
    # We use a caller-provided directory overlay on /scratch instead so the
    # writable layer is effectively unlimited. The flags below are the rest of
    # --compat (containall + no-init + no-umask + no-eval) without writable-tmpfs.
    # `--no-mount tmp` skips the default /tmp+/var/tmp tmpfs mount so those paths
    # fall through to the overlay too — otherwise pip's build dirs in /tmp hit
    # the same 64 MiB cap and `pip install -e .` dies with ENOSPC.
    full_cmd = [cmd_name, "exec", "--containall", "--no-init", "--no-umask", "--no-eval", "--no-mount", "tmp"]

    if overlay_path is not None:
        full_cmd.extend(["--overlay", str(overlay_path)])

    if use_torch_flags:
        full_cmd.extend(["--contain", "--cleanenv", "--no-home"])

    # SWE-bench images ship a pre-built `testbed` conda env at
    # /opt/miniconda3/envs/testbed/ with numpy/astropy/pytest/etc., but the
    # image's default PATH starts with the base conda (no scientific stack).
    # Prepend the testbed env so `python`/`pip`/`pytest` resolve to the
    # pre-installed binaries — otherwise the agent burns turns reinstalling
    # packages it already has, and the eval script fails on missing pytest.
    full_cmd.extend(
        [
            "--env",
            "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        ]
    )

    binds = []
    for host_path, container_path in bind_dirs:
        resolved = Path(host_path)
        if resolved.resolve().exists():
            binds.append(f"{resolved.resolve()}:{container_path}")
    if binds:
        full_cmd.extend(["--bind", ",".join(binds)])

    for key, value in env_vars.items():
        full_cmd.extend(["--env", f"{key}={value}"])

    full_cmd.extend([ensure_image_available(image_uri), "bash", "-c", command])
    return full_cmd


def run_apptainer_exec(
    image_uri: str,
    command: str,
    bind_dirs: list[tuple[Path | str, str]] | None = None,
    env_vars: dict[str, str] | None = None,
    timeout: int | None = 3600,
    use_torch_flags: bool = True,
    overlay_path: Path | str | None = None,
) -> tuple[str, int]:
    _ensure_tmpdir()
    cmd = build_apptainer_exec_command(
        image_uri=image_uri,
        command=command,
        bind_dirs=bind_dirs,
        env_vars=env_vars,
        use_torch_flags=use_torch_flags,
        overlay_path=overlay_path,
    )
    try:
        # errors="replace" so binary content in stdout/stderr (e.g. a `git diff HEAD`
        # spanning a binary file) doesn't crash with UnicodeDecodeError. Invalid bytes
        # become U+FFFD; reward computation only cares whether the output is non-empty.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        output = (result.stdout or "") + (result.stderr or "")
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return "", -1


def get_scratch_root(netid: str | None = None) -> Path:
    uid = netid or os.environ.get("USER", "unknown")
    # Clusters without a /scratch filesystem use the node's temporary directory.
    return Path("/scratch") / uid if Path("/scratch").is_dir() else Path(tempfile.gettempdir()) / f"rllm-{uid}"
