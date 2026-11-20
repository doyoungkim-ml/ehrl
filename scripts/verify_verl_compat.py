#!/usr/bin/env python3
"""Verify VeRL source or installed files, without importing VeRL or using the network."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "compat/verl/manifest.json"


class CompatibilityError(ValueError):
    """The catalog or source checkout does not match the reviewed patch contract."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CompatibilityError(f"Invalid relative path: {value!r}")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise CompatibilityError(f"Invalid relative path: {value!r}")
    return path


def _regular_path(root: Path, relative: str) -> Path:
    path = root / _relative_path(relative)
    # Reject symlinks in every component, including symlinks that remain inside root.
    for component in (path, *path.parents):
        if component == root:
            break
        if component.is_symlink():
            raise CompatibilityError(f"Symlink in patch path: {relative}")
    if path.exists() and not path.is_file():
        raise CompatibilityError(f"Patch path is not a regular file: {relative}")
    return path


def load_manifest(path: Path) -> tuple[dict, bytes]:
    """Load one reviewed patch contract; never execute catalog contents."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1:
            raise CompatibilityError("Unsupported compatibility manifest schema")
        if not re.fullmatch(r"[0-9a-f]{40}", manifest["base_revision"]):
            raise CompatibilityError("base_revision must be a full Git commit SHA")
        patch = manifest["patch"]
        if not patch["files"]:
            raise CompatibilityError("Patch file inventory must not be empty")
        for relative, expected in patch["files"].items():
            _relative_path(relative)
            if expected["git_mode"] not in {"100644", "100755"}:
                raise CompatibilityError(f"Unsupported Git file mode for {relative}")
            for key in ("base_sha256", "patched_sha256"):
                digest = expected[key]
                if digest is None and key == "base_sha256":
                    continue
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise CompatibilityError(f"Invalid {key} for {relative}")
        patch_path = _regular_path(path.resolve().parent, patch["path"])
        patch_bytes = patch_path.read_bytes()
        if sha256(patch_bytes) != patch["sha256"]:
            raise CompatibilityError("Patch checksum mismatch; restore the reviewed artifact")
    except (KeyError, TypeError, AttributeError, OSError, json.JSONDecodeError) as exc:
        raise CompatibilityError(f"Invalid compatibility manifest or artifact: {exc}") from exc
    # The inventory must match the patch, so an accidental additional diff cannot be applied.
    headers = re.findall(rb"^diff --git a/(\S+) b/(\S+)$", patch_bytes, flags=re.MULTILINE)
    try:
        paths = [right.decode("utf-8") for left, right in headers if left == right]
    except UnicodeDecodeError as exc:
        raise CompatibilityError("Patch paths must be UTF-8") from exc
    if len(paths) != len(headers) or len(paths) != len(set(paths)) or set(paths) != set(patch["files"]):
        raise CompatibilityError("Patch paths do not match the manifest inventory")
    return manifest, patch_bytes


def _git(source: Path, *args: str, data: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "--no-optional-locks", "-C", str(source), *args], input=data, capture_output=True)
    if check and result.returncode:
        reason = result.stderr.decode("utf-8", errors="replace").strip()
        raise CompatibilityError(f"Git {' '.join(args)} failed: {reason}")
    return result


def inspect_source(source: Path, manifest: dict, patch_bytes: bytes) -> str:
    """Return 'unpatched' or 'applied'; reject wrong pins, partial patches, and drift."""
    source = source.resolve()
    if {"site-packages", "dist-packages"}.intersection(source.parts):
        raise CompatibilityError("Installed packages are unsupported; select a dedicated VeRL source checkout")
    if not (source / ".git").exists() or not (source / "verl").is_dir():
        raise CompatibilityError("--source must be the root of a VeRL Git source checkout")
    root = Path(_git(source, "rev-parse", "--show-toplevel").stdout.decode().strip()).resolve()
    if root != source:
        raise CompatibilityError("--source must be the Git checkout root")
    revision = _git(source, "rev-parse", "HEAD").stdout.decode().strip()
    if revision != manifest["base_revision"]:
        raise CompatibilityError(f"Wrong VeRL pin: expected {manifest['base_revision']}, found {revision}")
    if _git(source, "diff", "--cached", "--quiet", "HEAD", "--", check=False).returncode:
        raise CompatibilityError("Staged changes found; use a clean source checkout at the exact pin")
    tracked = _git(source, "diff", "HEAD", "--name-only", "-z", "--").stdout
    untracked = _git(source, "ls-files", "--others", "--exclude-standard", "-z").stdout
    changed = {part.decode("utf-8") for part in (tracked + untracked).split(b"\0") if part}
    inventory = manifest["patch"]["files"]
    extra = changed - inventory.keys()
    if extra:
        raise CompatibilityError(f"Unrelated source drift: {', '.join(sorted(extra))}")
    states = []
    for relative, expected in inventory.items():
        path = _regular_path(source, relative)
        if path.exists():
            executable = bool(path.stat().st_mode & 0o111)
            if executable != (expected["git_mode"] == "100755"):
                raise CompatibilityError(f"Source executable-mode drift: {relative}")
        actual = sha256(path.read_bytes()) if path.exists() else None
        if actual == expected["base_sha256"]:
            states.append("unpatched")
        elif actual == expected["patched_sha256"]:
            states.append("applied")
        else:
            raise CompatibilityError(f"Source checksum drift: {relative}")
    if len(set(states)) != 1:
        raise CompatibilityError("Partial patch detected; use a clean source checkout")
    state = states[0]
    if state == "unpatched" and changed:
        raise CompatibilityError("Source is not clean at the reviewed base revision")
    args = ["apply", "--check", "--whitespace=error-all"]
    if state == "applied":
        args.append("--reverse")
    _git(source, *args, "-", data=patch_bytes)
    return state


def verify(source: Path, manifest_path: Path = DEFAULT_MANIFEST, *, apply: bool = False) -> dict:
    manifest, patch_bytes = load_manifest(manifest_path)
    state = inspect_source(source, manifest, patch_bytes)
    changed = False
    if apply and state == "unpatched":
        # git apply validates the whole patch before writing and leaves the index untouched.
        _git(source, "apply", "--whitespace=error-all", "-", data=patch_bytes)
        state = inspect_source(source, manifest, patch_bytes)
        changed = True
    return {
        "state": state,
        "changed": changed,
        "source": str(source.resolve()),
        "base_revision": manifest["base_revision"],
        "patch": manifest["patch"]["id"],
        "patch_sha256": manifest["patch"]["sha256"],
        "scope": "source checksums and patch applicability; no runtime validation",
    }


def verify_installed(manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """Read only the protected production files in the interpreter-selected package."""
    manifest, _ = load_manifest(manifest_path)
    try:
        spec = importlib.util.find_spec("verl")
    except (ImportError, ValueError) as exc:
        raise CompatibilityError(f"Could not resolve the installed VeRL package: {exc}") from exc
    if spec is None or spec.origin is None or spec.submodule_search_locations is None:
        raise CompatibilityError("Could not resolve an ordinary installed VeRL package")
    package_root = Path(spec.origin).resolve().parent
    locations = [Path(location).resolve() for location in spec.submodule_search_locations]
    if locations != [package_root] or not package_root.is_dir():
        raise CompatibilityError("Installed VeRL package resolution is ambiguous")
    protected = {relative: expected for relative, expected in manifest["patch"]["files"].items() if relative.startswith("verl/")}
    if not protected:
        raise CompatibilityError("Manifest contains no protected VeRL production files")
    for relative, expected in protected.items():
        path = _regular_path(package_root, relative.removeprefix("verl/"))
        if not path.is_file():
            raise CompatibilityError(f"Missing protected installed VeRL file: {path}")
        if sha256(path.read_bytes()) != expected["patched_sha256"]:
            raise CompatibilityError(f"Installed VeRL checksum mismatch: {path}; reinstall from the verified patched source")
    return {
        "state": "verified",
        "changed": False,
        "package_root": str(package_root),
        "protected_files": sorted(protected),
        "expected_base_revision": manifest["base_revision"],
        "patch": manifest["patch"]["id"],
        "patch_sha256": manifest["patch"]["sha256"],
        "scope": "protected installed source hashes, not full runtime or GPU validation",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--source", type=Path, help="Dedicated VeRL Git checkout root")
    target.add_argument("--installed", action="store_true", help="Read-only check of protected files in the interpreter-selected package")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Reviewed compatibility manifest")
    parser.add_argument("--apply", action="store_true", help="Apply to clean exact-pin source; never installs packages")
    args = parser.parse_args(argv)
    if args.installed and args.apply:
        parser.error("--apply requires --source; installed-package verification is read-only")
    try:
        result = verify_installed(args.manifest) if args.installed else verify(args.source, args.manifest, apply=args.apply)
    except (CompatibilityError, OSError) as exc:
        print(json.dumps({"state": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    # A clean but unpatched checkout is inspectable, yet not ready for exact resume.
    return 0 if result["state"] in {"applied", "verified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
