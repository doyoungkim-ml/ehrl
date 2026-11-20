"""Keep each training run's prepared dataset files with the run.

Launchers prepare VeRL parquet files in a job-owned registry that is deleted
when the job ends, while the run's recorded identity hashes those bytes. A
later job regenerates the rows from the same source but not necessarily the
same bytes (a different pandas/pyarrow, another machine), so a resume would be
rejected with no original to fall back on. The first job therefore copies the
effective train/validation files under ``RUN_DIR/data/`` and later jobs use
those copies. The copies are the run's dataset; only a distinct ``RUN_DIR``
changes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

RUN_DATA_DIRNAME = "data"
MANIFEST_NAME = "manifest.json"
SPLITS = ("train", "val")


def _sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(files) -> list[str]:
    if isinstance(files, str | os.PathLike):
        return [str(files)]
    return [str(item) for item in files]


def _is_job_prepared(path: str) -> bool:
    """Files under the job registry are regenerated per job; anything else is explicit."""
    registry = os.environ.get("RLLM_HOME")
    if not registry:
        return False
    try:
        Path(path).resolve().relative_to(Path(registry).expanduser().resolve())
    except ValueError:
        return False
    return True


def _atomic_copy(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        json.dump(manifest, temporary, indent=2, sort_keys=True)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def persist_run_datasets(config, run_dir: str | os.PathLike | None = None) -> dict | None:
    """Store the configured dataset files with the run, or route to the stored copies.

    Returns the per-split summary, or ``None`` when no run directory is selected.
    A split that already has stored copies keeps them: a regenerated job file is
    replaced silently, while an explicit file with different bytes is rejected
    because it conflicts with the run's identity.
    """
    run_dir = run_dir or os.environ.get("RLLM_RUN_DIR")
    if not run_dir:
        return None
    data_dir = Path(run_dir).expanduser().resolve() / RUN_DATA_DIRNAME
    manifest_path = data_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema_version": 1, "splits": {}}
    summary: dict[str, dict] = {}
    changed = False
    for split in SPLITS:
        key = f"{split}_files"
        files = config.data.get(key)
        if not files:
            continue
        sources = _sources(files)
        stored = manifest["splits"].get(split)
        if stored is None:
            data_dir.mkdir(parents=True, exist_ok=True)
            entries = []
            for index, source in enumerate(sources):
                name = f"{split}_{index}_{Path(source).name}" if len(sources) > 1 else f"{split}_{Path(source).name}"
                _atomic_copy(Path(source), data_dir / name)
                entries.append({"name": name, "sha256": _sha256(data_dir / name), "bytes": (data_dir / name).stat().st_size, "source": str(source)})
            manifest["splits"][split] = entries
            stored = entries
            changed = True
            action = "stored"
        else:
            copies = [data_dir / entry["name"] for entry in stored]
            for entry, copy in zip(stored, copies, strict=True):
                if not copy.is_file() or _sha256(copy) != entry["sha256"]:
                    raise RuntimeError(f"Stored run dataset {copy} is missing or altered; restore it from a backup or select a distinct RUN_DIR")
            configured = [Path(source).resolve() for source in sources]
            if configured != [copy.resolve() for copy in copies]:
                explicit = [source for source in sources if not _is_job_prepared(source)]
                if explicit and (len(sources) != len(stored) or any(_sha256(source) != entry["sha256"] for source, entry in zip(sources, stored, strict=True))):
                    raise RuntimeError(f"Explicit data.{key} differs from this run's stored dataset under {data_dir}; the run keeps its dataset, so use a distinct RUN_DIR for a different one")
            action = "reused"
        copies = [str(data_dir / entry["name"]) for entry in stored]
        config.data[key] = copies[0] if isinstance(files, str | os.PathLike) and len(copies) == 1 else copies
        summary[split] = {"action": action, "files": copies, "sha256": [entry["sha256"] for entry in stored]}
        print(f"[run-data] {split}: {action} {', '.join(copies)}")
    if changed:
        _write_manifest(manifest_path, manifest)
    return summary
