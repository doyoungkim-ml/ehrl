#!/usr/bin/env python3
"""Merge a verl FSDP checkpoint without implicitly downcasting weights.

The upstream verl FSDP merger used by this repository casts every shard to
``bfloat16`` before concatenating and then builds the HuggingFace shell model as
``bfloat16``.  That is convenient for inference-sized exports, but it silently
loses precision for training checkpoints saved in fp32.  This wrapper keeps the
checkpoint tensor dtype by default and can optionally force a target floating
dtype via ``--merge-dtype``.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from accelerate import init_empty_weights
from torch.distributed._tensor import Placement, Shard
from tqdm import tqdm

try:
    # torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.utils import hf_processor, hf_tokenizer

DTYPE_CHOICES = ("preserve", "fp32", "float32", "bf16", "bfloat16", "fp16", "float16")
DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge one verl FSDP actor checkpoint to HuggingFace format while preserving fp32 by default.")
    parser.add_argument("--actor-dir", required=True, help="Path to the checkpoint actor/ directory.")
    parser.add_argument("--target-dir", required=True, help="Directory where the merged HF checkpoint is written.")
    parser.add_argument(
        "--hf-config",
        required=True,
        help="Path to actor/huggingface with the source HuggingFace config/tokenizer files.",
    )
    parser.add_argument(
        "--merge-dtype",
        default=os.environ.get("MERGE_DTYPE", "preserve"),
        choices=DTYPE_CHOICES,
        help=("Floating tensor dtype for the merged checkpoint. 'preserve' keeps the shard dtype; use 'fp32' to force float32. Default: %(default)s."),
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Forward trust_remote_code to transformers.")
    return parser.parse_args()


def _target_dtype(dtype_name: str) -> torch.dtype | None:
    return DTYPE_ALIASES.get(dtype_name.lower())


def _infer_model_dtype(state_dict: dict[str, torch.Tensor]) -> torch.dtype | None:
    counts = Counter(tensor.dtype for tensor in state_dict.values() if torch.is_tensor(tensor) and tensor.is_floating_point())
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _format_dtype_counts(state_dict: dict[str, torch.Tensor]) -> str:
    counts = Counter(str(tensor.dtype) for tensor in state_dict.values() if torch.is_tensor(tensor) and tensor.is_floating_point())
    if not counts:
        return "none"
    return ", ".join(f"{dtype}={count}" for dtype, count in sorted(counts.items()))


class PrecisionAwareFSDPModelMerger(FSDPModelMerger):
    """verl FSDP merger variant that does not hard-code bfloat16."""

    def __init__(self, config: ModelMergerConfig, requested_dtype: torch.dtype | None):
        super().__init__(config)
        self.requested_dtype = requested_dtype

    def _maybe_cast_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.requested_dtype is None or not tensor.is_floating_point():
            return tensor
        return tensor.to(dtype=self.requested_dtype)

    def _merge_by_placement(self, tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
        """Merges a list of tensors based on their DTensor placement."""
        if placement.is_replicate():
            return tensors[0]
        if placement.is_partial():
            raise NotImplementedError("Partial placement is not supported yet")
        if placement.is_shard():
            return torch.cat(tensors, dim=placement.dim).contiguous()

        raise NotImplementedError(f"Unsupported placement: {placement}")

    def _load_and_merge_state_dicts(self, world_size: int, total_shards: int, mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
        model_state_dict_lst = [None] * total_shards

        def process_one_shard(rank: int, model_state_dict_lst: list):
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state_dict_lst[rank] = state_dict
            return state_dict

        max_workers = min(32, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                future.result()

        state_dict = {}
        param_placements: dict[str, tuple[Shard, ...]] = {}

        for key in set(model_state_dict_lst[0].keys()):
            state_dict[key] = []
            for model_state_shard in model_state_dict_lst:
                tensor = model_state_shard.pop(key)
                if isinstance(tensor, DTensor):
                    state_dict[key].append(self._maybe_cast_tensor(tensor._local_tensor))

                    placements = tuple(tensor.placements)
                    # Replicated placement at the dp/ddp dimension can be discarded.
                    if mesh_dim_names[0] in ("dp", "ddp"):
                        placements = placements[1:]

                    if key not in param_placements:
                        param_placements[key] = placements
                    else:
                        assert param_placements[key] == placements
                else:
                    state_dict[key].append(self._maybe_cast_tensor(tensor))

        del model_state_dict_lst

        for key in sorted(state_dict):
            if not isinstance(state_dict[key], list):
                print(f"No need to merge key {key}")
                continue
            if key in param_placements:
                placements = param_placements[key]
                if len(mesh_shape) == 1:
                    assert len(placements) == 1
                    shards = state_dict[key]
                    state_dict[key] = self._merge_by_placement(shards, placements[0])
                else:
                    raise NotImplementedError("FSDP + TP is not supported yet")
            else:
                state_dict[key] = torch.cat(state_dict[key], dim=0)

        return state_dict

    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()
        model_dtype = self.requested_dtype or _infer_model_dtype(state_dict)
        print(f"Merged floating tensor dtypes: {_format_dtype_counts(state_dict)}")
        print(f"Initializing HuggingFace model shell with dtype: {model_dtype or 'default'}")

        model_kwargs = {"trust_remote_code": self.config.trust_remote_code}
        if model_dtype is not None:
            model_kwargs["torch_dtype"] = model_dtype

        with init_empty_weights():
            model = auto_model_class.from_config(self.model_config, **model_kwargs)
        model.to_empty(device="cpu")
        model = self.patch_model_generation_config(model)

        lora_path = self.save_lora_adapter(state_dict)
        if lora_path:
            print(f"Saving lora adapter to {lora_path}")

        print(f"Saving model to {self.config.target_dir}")
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)
        del state_dict
        del model

        processor = hf_processor(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
        tokenizer = hf_tokenizer(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
        if processor is not None:
            print(f"Saving processor to {self.config.target_dir}")
            processor.save_pretrained(self.config.target_dir)
        if tokenizer is not None:
            print(f"Saving tokenizer to {self.config.target_dir}")
            tokenizer.save_pretrained(self.config.target_dir)


def main() -> None:
    args = _parse_args()
    requested_dtype = _target_dtype(args.merge_dtype)
    dtype_label = str(requested_dtype) if requested_dtype is not None else "preserve source tensor dtypes"
    print(f"Merge dtype mode: {args.merge_dtype} ({dtype_label})")

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=args.actor_dir,
        target_dir=args.target_dir,
        hf_model_config_path=args.hf_config,
        trust_remote_code=args.trust_remote_code,
    )
    merger = PrecisionAwareFSDPModelMerger(config, requested_dtype=requested_dtype)
    merger.merge_and_save()


if __name__ == "__main__":
    main()
