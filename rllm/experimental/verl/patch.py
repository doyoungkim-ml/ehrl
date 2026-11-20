"""Monkey-patches for Verl and vLLM within the rLLM unified trainer.

All patches are applied lazily (on first call) and are idempotent - calling
them multiple times is safe.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_VERL_DYNAMIC_BATCH_PATCHED = False
_VERL_TRAIN_OUTPUT_PATCHED = False
_VLLM_SDK_PATCHED = False


# ---------------------------------------------------------------------------
# Verl dynamic batch: sync micro-batch counts across DP ranks
# ---------------------------------------------------------------------------


def patch_verl_dynamic_batch_sync() -> None:
    """Patch ``prepare_dynamic_batch`` to sync micro-batch counts across DP ranks.

    Fixes `verl#5750 <https://github.com/verl-project/verl/issues/5750>`_:
    when ``use_dynamic_bsz=True``, each DP rank independently calculates
    ``num_micro_batches`` based on its local sequence lengths.  Different
    ranks can end up with different counts, causing NCCL collective
    operations (AllGather/ReduceScatter in FSDP) to deadlock.

    The fix defaults ``dp_group`` to ``torch.distributed.group.WORLD`` so
    that ``prepare_dynamic_batch`` performs an ``all_reduce(MAX)`` across
    ranks, forcing every rank to iterate through the same number of
    micro-batches.  This is the same approach as verl PR #5591.
    """
    global _VERL_DYNAMIC_BATCH_PATCHED
    if _VERL_DYNAMIC_BATCH_PATCHED:
        return

    import verl.utils.seqlen_balancing as sbl

    _original_prepare = sbl.prepare_dynamic_batch

    def _patched_prepare(data, max_token_len, dp_group=None, **kwargs):
        if dp_group is None:
            import torch.distributed

            if torch.distributed.is_initialized():
                dp_group = torch.distributed.group.WORLD
        return _original_prepare(data, max_token_len, dp_group=dp_group, **kwargs)

    sbl.prepare_dynamic_batch = _patched_prepare

    # Also patch the already-imported reference in dp_actor so both
    # compute_log_prob and update_policy use the patched version.
    try:
        from verl.workers.actor import dp_actor

        dp_actor.prepare_dynamic_batch = _patched_prepare
    except (ImportError, AttributeError):
        pass  # dp_actor may not be importable outside GPU workers

    _VERL_DYNAMIC_BATCH_PATCHED = True
    logger.info("Patched prepare_dynamic_batch to sync micro-batch counts across DP ranks (verl#5750)")


def patch_verl_training_output_postprocess() -> None:
    """Skip training-only model-output reconstruction in the new verl worker path.

    The new EngineWorker path always reconstructs ``model_output`` from nested
    tensors inside ``postprocess_batch_func()``, even for training batches where
    callers immediately discard that field and only keep metrics. With some
    model/runtime combinations, notably the Qwen3.5-0.8B MBPP GRPO run, PyTorch
    can raise inside ``NestedTensor.unbind()`` while rebuilding those outputs.

    We patch two pieces together:

    1. Mark all training batches with ``return_model_output=False``.
    2. Teach ``postprocess_batch_func()`` to skip nested output reconstruction
       when that flag is present.

    This keeps inference/log-prob paths unchanged while avoiding the failing
    training-only code path.
    """

    global _VERL_TRAIN_OUTPUT_PATCHED
    if _VERL_TRAIN_OUTPUT_PATCHED:
        return

    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.utils.py_functional import append_to_dict
    from verl.workers.engine import base as engine_base
    from verl.workers.engine import utils as engine_utils

    original_train_batch = engine_base.BaseEngine.train_batch
    original_postprocess = engine_utils.postprocess_batch_func

    def _patched_train_batch(self, data, loss_function):
        tu.assign_non_tensor(data, return_model_output=False)
        return original_train_batch(self, data, loss_function)

    def _patched_postprocess_batch_func(output_lst, indices, data):
        return_model_output = tu.get_non_tensor_data(data=data, key="return_model_output", default=True)
        if return_model_output:
            return original_postprocess(output_lst=output_lst, indices=indices, data=data)

        use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
        pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, "postprocess_batch_func only support NO_PADDING pad_mode"

        losses = []
        aggregated_metrics = {}

        for output in output_lst:
            if "loss" in output:
                losses.append(output["loss"])
            if "metrics" in output:
                append_to_dict(aggregated_metrics, output["metrics"])

        # Keep the shape of the original return value so downstream callers do
        # not need any changes. Dynamic-batch restoration is only relevant for
        # model outputs, which are intentionally skipped here.
        output = {
            "model_output": {},
            "loss": losses,
            "metrics": aggregated_metrics,
        }

        if use_dynamic_bsz:
            logger.debug("Skipped training-time nested model_output reconstruction for dynamic batch")

        return output

    engine_base.BaseEngine.train_batch = _patched_train_batch
    engine_utils.postprocess_batch_func = _patched_postprocess_batch_func

    for module_name in [
        "verl.workers.engine.automodel.transformer_impl",
        "verl.workers.engine.fsdp.transformer_impl",
        "verl.workers.engine.megatron.transformer_impl",
        "verl.workers.engine.torchtitan.transformer_impl",
        "verl.workers.engine.veomni.transformer_impl",
    ]:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "postprocess_batch_func"):
            module.postprocess_batch_func = _patched_postprocess_batch_func

    _VERL_TRAIN_OUTPUT_PATCHED = True
    logger.info("Patched verl training postprocess to skip unused nested model outputs")


# ---------------------------------------------------------------------------
# vLLM SDK instrumentation
# ---------------------------------------------------------------------------


def patch_vllm_for_sdk() -> None:
    """Patch vLLM replicas to add logprob/token-id instrumentation for SDK traces.

    Creates an ``InstrumentedvLLMHttpServer`` Ray actor that loads
    ``rllm/patches/vllm_instrumentation.py`` in an isolated module namespace
    and patches ``vLLMReplica.__init__`` to use it.
    """
    global _VLLM_SDK_PATCHED
    if _VLLM_SDK_PATCHED:
        return

    import ray
    from verl.workers.rollout.vllm_rollout.vllm_async_server import (
        vLLMHttpServer,
        vLLMReplica,
    )

    @ray.remote(num_cpus=1)
    class InstrumentedvLLMHttpServer(vLLMHttpServer):
        """vLLM HTTP server with automatic vLLM instrumentation in Ray worker."""

        def __init__(self, *args, **kwargs):
            import importlib.util
            import sys
            from pathlib import Path

            instrumentation_path = Path(__file__).parent.parent.parent / "patches" / "vllm_instrumentation.py"

            spec = importlib.util.spec_from_file_location("rllm_vllm_instrumentation_isolated", str(instrumentation_path))
            vllm_instrumentation = importlib.util.module_from_spec(spec)
            sys.modules["rllm_vllm_instrumentation_isolated"] = vllm_instrumentation
            spec.loader.exec_module(vllm_instrumentation)

            vllm_instrumentation.instrument_vllm(add_response_logprobs=True)
            super().__init__(*args, **kwargs)

    _original_init = vLLMReplica.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.server_class = InstrumentedvLLMHttpServer

    vLLMReplica.__init__ = _patched_init
    _VLLM_SDK_PATCHED = True
    logger.info("Patched vLLMReplica for SDK instrumentation")
