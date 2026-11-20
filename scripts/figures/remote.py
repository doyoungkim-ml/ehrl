"""Retrieve only plotting inputs from an explicit SSH run into a local cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REMOTE = re.compile(r"^(?P<host>(?:[\w.][\w.-]*@)?[a-zA-Z0-9_][\w.-]*):(?P<path>.+)$", re.ASCII)
MARKER = ".remote-source.json"
EXCLUDED = ("archive", "historical", "checkpoints", "checkpoints_merged", "wandb", "figures")


def parse_remote(value: str) -> tuple[str, str] | None:
    """Use scp-style host:path syntax; ./host:path still names a local path."""
    if value.startswith(("/", "./", "../", "~/")) or ":" not in value:
        return None
    match = REMOTE.fullmatch(value)
    if not match or not match["path"].strip() or any(char in value for char in "\n\r\0"):
        raise ValueError("Remote runs use HOST:/absolute/path, HOST:~/path, or HOST:path (relative to the remote home)")
    path = match["path"]
    if any(part in ("archive", "historical") for part in PurePosixPath(path).parts):
        raise ValueError("Archived sources are not maintained plot inputs")
    return match["host"], path


def cache_path(value: str, cache_dir: Path) -> Path:
    host, path = parse_remote(value)
    label = re.sub(r"[^a-zA-Z0-9_.-]", "_", PurePosixPath(path).name) or "run"
    digest = hashlib.sha256(value.encode()).hexdigest()[:16]
    return cache_dir.expanduser().resolve() / host / f"{label[:80]}-{digest}"


def checked_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise ValueError(f"SSH retrieval requires local ssh and rsync, and python3 and rsync on the remote host: {error}") from error
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise ValueError(f"Remote retrieval failed ({command[0]}): {detail}")
    return result.stdout


def remote_directory(host: str, path: str) -> str:
    # Pass the literal path as a quoted argument; never interpolate it as shell code.
    code = (
        "import json,pathlib,sys; "
        "p=pathlib.Path(sys.argv[1]).expanduser().resolve(strict=True); "
        "assert p.is_dir(), 'Remote run is not a directory'; "
        "print(json.dumps(str(p)))"
    )
    output = checked_command(["ssh", host, shlex.join(["python3", "-c", code, path])])
    try:
        directory = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError(f"ssh {host}: expected the resolved run path; check remote shell startup output") from error
    if not isinstance(directory, str) or not directory.startswith("/"):
        raise ValueError(f"ssh {host}: invalid resolved run path")
    if any(part in ("archive", "historical") for part in PurePosixPath(directory).parts):
        raise ValueError("Archived sources are not maintained plot inputs")
    return directory


def rsync_command(host: str, directory: str, destination: Path, *, trajectories: bool, evaluation_only: bool = False) -> list[str]:
    # Excluded files are protected from deletion. Only this cache's included
    # input files are refreshed/deleted; local figures and other files survive.
    command = ["rsync", "--recursive", "--times", "--checksum", "--delete", "--prune-empty-dirs", "--protect-args", "--rsh=ssh"]
    command += [f"--exclude={name}/" for name in EXCLUDED]
    if trajectories:
        # Do not enumerate AppWorld databases or other per-task runtime trees.
        # Only chat_completions/eval contains the acting trajectories we plot.
        command += [
            "--include=task_*/chat_completions/",
            "--include=task_*/chat_completions/eval/",
            "--include=task_*/chat_completions/eval/policy_step*.json",
            "--exclude=task_*/chat_completions/eval/*",
            "--exclude=task_*/chat_completions/*",
            "--exclude=task_*/*",
        ]
    else:
        command += ["--exclude=task_*/", "--exclude=playbooks/"]
    command += ["--include=/metrics.jsonl", "--include=/run.json"] if not evaluation_only else []
    prefix = "/" if evaluation_only else "/eval/**/"
    command += ["--include=*/"] if evaluation_only else ["--include=/eval/", "--include=/eval/**/"]
    for name in ("metrics_step*.json", "evaluation_config.json", "evaluation_complete.json"):
        command.append(f"--include={prefix}{name}" if not evaluation_only else f"--include={name}")
    if trajectories:
        command += ["--include=policy_step*.json", "--include=trajectories.json"]
    command += ["--exclude=*", "--", f"{host}:{directory.rstrip('/')}/", str(destination) + "/"]
    return command


def write_marker(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def scalar_fingerprint(directory: Path) -> str:
    """Fingerprint completed evaluation inputs that identify cached turn trajectories."""
    digest = hashlib.sha256()
    for root, directories, files in os.walk(directory):
        directories[:] = sorted(name for name in directories
                                if name not in EXCLUDED and name != "playbooks" and not name.startswith("task_"))
        metrics = sorted(name for name in files if name.startswith("metrics_step") and name.endswith(".json"))
        for name in (*metrics, *(name for name in ("evaluation_config.json", "evaluation_complete.json")
                                  if metrics and name in files)):
            path = Path(root) / name
            digest.update(str(path.relative_to(directory)).encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def fetch_run(value: str, cache_dir: Path, *, trajectories: bool = False, offline: bool = False, evaluation_only: bool = False) -> Path:
    """Refresh a dedicated artifact cache; never write back to the remote host."""
    host, path = parse_remote(value)
    destination = cache_path(value, cache_dir)
    marker = destination / MARKER
    if offline:
        if not marker.is_file():
            raise ValueError(f"No cached remote run at {destination}; omit --offline to retrieve it")
        metadata = json.loads(marker.read_text())
        if metadata.get("source") != value or metadata.get("evaluation_only") != evaluation_only or not metadata.get("complete"):
            raise ValueError(f"Remote cache is incomplete or belongs to another input: {destination}; refresh without --offline")
        if trajectories and not metadata.get("trajectories"):
            raise ValueError(f"Remote cache has no refreshed trajectories: {destination}; omit --offline for turn plots")
        print(f"Using cached {value}: {destination}", file=sys.stderr)
        return destination
    directory = remote_directory(host, path)
    destination.mkdir(parents=True, exist_ok=True)
    previous = json.loads(marker.read_text()) if marker.is_file() else {}
    prior_fingerprint = (scalar_fingerprint(destination)
                         if previous.get("complete") and previous.get("trajectories")
                         and previous.get("source") == value else None)
    metadata = {"source": value, "host": host, "remote_dir": directory, "evaluation_only": evaluation_only, "trajectories": trajectories, "complete": False}
    write_marker(marker, metadata)
    print(f"Retrieving plotting inputs from {value} into {destination}", file=sys.stderr)
    checked_command(rsync_command(host, directory, destination, trajectories=trajectories, evaluation_only=evaluation_only))
    if not trajectories and prior_fingerprint is not None:
        metadata["trajectories"] = scalar_fingerprint(destination) == prior_fingerprint
    metadata.update(complete=True, retrieved_at=datetime.now(timezone.utc).isoformat())
    write_marker(marker, metadata)
    return destination


def source_directory_name(directory: Path) -> str:
    """Retain legacy step/task identifiers when a result root is cached by hash."""
    marker = directory / MARKER
    if marker.is_file():
        return PurePosixPath(json.loads(marker.read_text())["remote_dir"]).name
    return directory.name
