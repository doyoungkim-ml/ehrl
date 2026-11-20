"""Utilities for initializing Ray consistently.

Issue #166 reports that in some environments (notably Docker), a child process may
call `ray.init(namespace=...)` and accidentally start a fresh local Ray cluster
instead of attaching to the already-running one. This can lead to confusing
failures where named actors appear to be missing.

This module centralizes the logic for selecting Ray init parameters.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def _ray_current_cluster_path(temp_dir: str | None = None) -> Path:
    # Match Ray's address-file lookup without importing Ray. An explicit
    # _temp_dir is already the Ray root; environment temp roots get /ray added.
    if temp_dir is not None:
        return Path(temp_dir) / "ray_current_cluster"
    if "RAY_TMPDIR" in os.environ:
        root = os.environ["RAY_TMPDIR"]
    elif sys.platform.startswith("linux") and "TMPDIR" in os.environ:
        root = os.environ["TMPDIR"]
    elif sys.platform.startswith(("linux", "darwin")):
        root = "/tmp"
    else:
        root = tempfile.gettempdir()
    return Path(root) / "ray" / "ray_current_cluster"


def should_attach_to_existing_ray_cluster(temp_dir: str | None = None) -> bool:
    """Whether we should attempt to attach to an existing Ray cluster."""

    # Explicitly configured by the user or environment.
    if os.getenv("RAY_ADDRESS"):
        return True

    # Only consult the selected Ray directory. A marker in /tmp/ray may belong
    # to an unrelated or stopped job when this launcher uses its own _temp_dir.
    try:
        return _ray_current_cluster_path(temp_dir).exists()
    except Exception:
        return False


def get_ray_init_settings(config: Any | None = None) -> dict[str, Any]:
    """Build kwargs for `ray.init(...)` from config + environment.

    Notes:
    - If `config.ray_init.address` is set, we pass it through verbatim.
    - Otherwise, if the selected Ray directory has an address marker (or
      RAY_ADDRESS is set), we use `address="auto"` to attach. Marker presence
      does not itself prove that the cluster is still running.
    - If none of the above applies, we return no `address`, so Ray will start a
      local cluster.
    """

    settings: dict[str, Any] = {}

    if config is not None and hasattr(config, "ray_init"):
        for k, v in config.ray_init.items():
            if v is not None:
                settings[k] = v

    # Prefer explicit address from config.
    if "address" in settings:
        return settings

    if should_attach_to_existing_ray_cluster(settings.get("_temp_dir")):
        settings["address"] = "auto"

    return settings
