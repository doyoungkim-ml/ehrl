from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def get_example_name(script_path: str) -> str:
    if "/" not in script_path and "\\" not in script_path:
        return script_path

    path = Path(script_path).resolve()
    parts = path.parts
    if "examples" not in parts:
        return path.parent.name

    idx = parts.index("examples")
    if idx + 1 >= len(parts):
        return path.parent.name

    example_name = parts[idx + 1]
    if example_name == "sdk" and idx + 2 < len(parts):
        return parts[idx + 2]
    return example_name


def get_default_run_name(script_path: str, now: datetime | None = None) -> str:
    return get_default_run_name_with_model(script_path, None, now)


def normalize_model_name(model_name: str | None) -> str:
    if not model_name:
        return "unknown-model"
    normalized = model_name.split("/")[-1]
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    return normalized or "unknown-model"


def get_default_run_name_with_model(
    script_path: str,
    model_name: str | None,
    now: datetime | None = None,
) -> str:
    if now is None:
        now = datetime.now()
    return f"{get_example_name(script_path)}_{normalize_model_name(model_name)}_{now.strftime('%m%d%H')}"
