"""Select training rows while preserving explicit VeRL parquet inputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def _task_limit(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    limit = int(value)
    if limit < 1:
        raise ValueError(f"{name} must be positive")
    return limit


def _select_parquet_rows(files, *, offset: int, limit: int | None, subset_tag: str):
    """Slice the ordered input files into a job-local parquet; never rewrite inputs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    sources = [str(files)] if isinstance(files, str | os.PathLike) else list(files)
    tables = [pq.read_table(source) for source in sources]
    table = pa.concat_tables(tables, promote_options="default")
    selected = table.slice(offset, limit)
    if selected.num_rows == 0:
        raise ValueError(f"{subset_tag} task selection is empty (offset={offset}, limit={limit})")
    if offset == 0 and selected.num_rows == table.num_rows:
        return files, table.num_rows, table.num_rows

    subset_dir = Path(os.environ.get("RLLM_HOME", "~/.rllm")).expanduser() / "datasets" / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps([sources, offset, limit]).encode()).hexdigest()[:16]
    tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in subset_tag)
    destination = subset_dir / f"{tag}_{identity}.parquet"
    with tempfile.NamedTemporaryFile(dir=subset_dir, suffix=".parquet", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        pq.write_table(selected, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(destination), table.num_rows, selected.num_rows


def configure_training_dataset_paths(config, *, train_dataset=None, val_dataset=None, subset_tag: str) -> dict:
    """Prefer explicit files, then apply task caps/offsets to the selected sources.

    Explicit files without a cap or offset pass through unchanged, including file
    lists. The provided datasets are fallbacks for unset train/validation paths.
    Environment task caps and data.*_max_samples combine by taking the smaller
    positive limit. Generated parquet subsets retain the input schema.
    """
    offset = int(os.environ.get("TRAIN_TASK_OFFSET", "0") or "0")
    if offset < 0:
        raise ValueError("TRAIN_TASK_OFFSET must be nonnegative")
    task_limits = {"train": _task_limit("MAX_TRAIN_TASKS"), "val": _task_limit("MAX_VAL_TASKS")}
    result = {}
    for split, dataset in (("train", train_dataset), ("val", val_dataset)):
        files = config.data.get(f"{split}_files")
        total = None
        if not files and dataset is not None:
            files = dataset.get_verl_data_path()
            total = len(dataset.get_data())
        if not files:
            raise ValueError(f"No {split} dataset files were configured or prepared")

        configured_limit = int(config.data.get(f"{split}_max_samples", -1) or -1)
        limits = [value for value in (task_limits[split], configured_limit) if value is not None and value > 0]
        limit = min(limits) if limits else None
        split_offset = offset if split == "train" else 0
        effective = total
        if split_offset or limit is not None:
            files, total, effective = _select_parquet_rows(files, offset=split_offset, limit=limit, subset_tag=f"{subset_tag}_{split}")
        config.data[f"{split}_files"] = files
        result[f"{split}_verl_path"] = files
        result[f"total_{split}_examples"] = total
        result[f"effective_{split}_examples"] = effective
    return result
