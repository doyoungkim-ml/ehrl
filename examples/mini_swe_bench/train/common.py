"""Shared MiniSWEBench dataset and training validation helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from omegaconf import OmegaConf

from examples.mini_swe_bench.preprocess_swegym import get_default_data_dir, preprocess_swegym
from rllm.data import Dataset, DatasetRegistry

MODEL_META: dict[str, tuple[int, int]] = {
    "Qwen/Qwen3.5-4B": (248320, 2560),
    "Qwen/Qwen3.5-2B": (151936, 1536),
    "Qwen/Qwen3.5-8B": (248320, 3584),
}


def preflight_checks(config) -> None:
    errors = []
    max_prompt = config.data.max_prompt_length
    max_response = config.data.max_response_length
    max_seq = max_prompt + max_response

    ppo_max_tok = OmegaConf.select(config, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu")
    if ppo_max_tok is not None and ppo_max_tok < max_seq:
        errors.append(f"ppo_max_token_len_per_gpu ({ppo_max_tok}) < max_prompt + max_response ({max_prompt}+{max_response}={max_seq})")

    lp_max_tok = OmegaConf.select(config, "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu")
    if lp_max_tok is not None and lp_max_tok < max_seq:
        errors.append(f"log_prob_max_token_len_per_gpu ({lp_max_tok}) < max_seq ({max_seq})")

    max_model_len = OmegaConf.select(config, "actor_rollout_ref.rollout.max_model_len")
    if max_model_len is not None and max_model_len < max_seq:
        errors.append(f"max_model_len ({max_model_len}) < max_seq ({max_seq})")

    model_path = OmegaConf.select(config, "actor_rollout_ref.model.path")
    bucket_mb = OmegaConf.select(
        config,
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes",
        default=2048,
    )
    if model_path and model_path in MODEL_META:
        vocab, hidden = MODEL_META[model_path]
        embed_mb = vocab * hidden * 4 / (1024 * 1024)
        if bucket_mb < embed_mb:
            errors.append(f"update_weights_bucket_megabytes ({bucket_mb} MB) too small for {model_path} embed_tokens ({embed_mb:.0f} MB)")

    if errors:
        formatted = "\n".join(f"  [{idx}] {error}" for idx, error in enumerate(errors, 1))
        raise ValueError(f"\n{'=' * 60}\nPREFLIGHT CHECK FAILED — fix config before GPU init:\n{'=' * 60}\n{formatted}\n{'=' * 60}")
    print("[preflight] All config checks passed.")


def select_rows(rows, *, split: str, environ=None):
    """Apply explicit task limits, including an offset without a task cap."""
    environ = os.environ if environ is None else environ
    cap_key = "MAX_TRAIN_TASKS" if split == "train" else "MAX_VAL_TASKS"
    cap_text = environ.get(cap_key, "").strip()
    cap = int(cap_text) if cap_text else None
    offset = int(environ.get("TRAIN_TASK_OFFSET", "0") or "0") if split == "train" else 0
    if offset < 0 or (cap is not None and cap <= 0):
        raise ValueError(f"{split}: task offset must be nonnegative and task cap positive")
    selected = rows[offset : None if cap is None else offset + cap]
    if not selected:
        raise ValueError(f"{split}: task selection is empty (offset={offset}, cap={cap})")
    return selected


def load_datasets():
    data_dir = Path(os.environ.get("MINISW_DATA_DIR") or get_default_data_dir())
    paths = [data_dir / "train.parquet", data_dir / "validation.parquet"]
    if not all(path.exists() for path in paths):
        preprocess_swegym(output_dir=data_dir)
    result = []
    for split, path in zip(("train", "validation"), paths, strict=True):
        rows = select_rows(Dataset.load_data(str(path)).get_data(), split=split)
        # File content plus selection prevents same-length replacements from
        # silently using an earlier registry snapshot.
        selection = repr((len(rows), os.environ.get("TRAIN_TASK_OFFSET", "0") if split == "train" else "0"))
        with path.open("rb") as source:
            fingerprint = hashlib.file_digest(source, "sha256")
        fingerprint.update(selection.encode())
        digest = fingerprint.hexdigest()[:16]
        name = f"minisw_{split}_{digest}"
        if not DatasetRegistry.dataset_exists(name, split):
            DatasetRegistry.register_dataset(name, rows, split)
        dataset = DatasetRegistry.load_dataset(name, split)
        if dataset is None:
            raise RuntimeError(f"Failed to load MiniSWEBench {split} dataset")
        print(f"[data] {split}: {len(rows)} tasks from {path}", flush=True)
        result.append(dataset)
    return tuple(result)
