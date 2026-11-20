"""Exercise retained patch behavior against backend doubles, without GPU or ML imports.

These tests guard routing and output contracts. They do not establish distributed
collective correctness or GPU numerical equivalence.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

PATCH_SOURCE = Path(__file__).resolve().parents[2] / "rllm/experimental/verl/patch.py"


def load_patch_module():
    spec = importlib.util.spec_from_file_location("isolated_verl_runtime_patch", PATCH_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_tree(*names):
    """Make independent package trees so patch.dict cannot mutate installed modules."""
    modules = {}
    for name in names:
        parts = name.split(".")
        for depth in range(1, len(parts) + 1):
            full = ".".join(parts[:depth])
            if full not in modules:
                modules[full] = ModuleType(full)
                modules[full].__path__ = []
            if depth > 1:
                parent = modules[".".join(parts[: depth - 1])]
                setattr(parent, parts[depth - 1], modules[full])
    return modules


class DynamicBatchPatchTests(unittest.TestCase):
    def setUp(self):
        self.runtime = load_patch_module()
        self.modules = module_tree(
            "torch.distributed",
            "verl.utils.seqlen_balancing",
            "verl.workers.actor.dp_actor",
        )
        self.distributed = self.modules["torch.distributed"]
        self.world = object()
        self.distributed.group = SimpleNamespace(WORLD=self.world)
        self.distributed.is_initialized = Mock(return_value=True)
        self.prepare = Mock(return_value=object())
        self.balancing = self.modules["verl.utils.seqlen_balancing"]
        self.balancing.prepare_dynamic_batch = self.prepare
        self.actor = self.modules["verl.workers.actor.dp_actor"]
        self.actor.prepare_dynamic_batch = self.prepare
        modules_patch = patch.dict(sys.modules, self.modules)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def test_initialized_distributed_defaults_to_world_and_updates_actor_alias(self):
        self.runtime.patch_verl_dynamic_batch_sync()
        data = object()
        result = self.balancing.prepare_dynamic_batch(data, 1024, use_dynamic_bsz=True)
        self.prepare.assert_called_once_with(data, 1024, dp_group=self.world, use_dynamic_bsz=True)
        self.assertIs(result, self.prepare.return_value)
        self.assertIs(self.actor.prepare_dynamic_batch, self.balancing.prepare_dynamic_batch)

    def test_explicit_data_parallel_group_is_preserved(self):
        self.runtime.patch_verl_dynamic_batch_sync()
        group, data = object(), object()
        self.actor.prepare_dynamic_batch(data, 512, dp_group=group)
        self.prepare.assert_called_once_with(data, 512, dp_group=group)
        self.distributed.is_initialized.assert_not_called()

    def test_non_distributed_call_preserves_none_group(self):
        self.distributed.is_initialized.return_value = False
        self.runtime.patch_verl_dynamic_batch_sync()
        data = object()
        self.actor.prepare_dynamic_batch(data, 512)
        self.prepare.assert_called_once_with(data, 512, dp_group=None)

    def test_repeated_patch_does_not_nest_wrappers(self):
        self.runtime.patch_verl_dynamic_batch_sync()
        first = self.balancing.prepare_dynamic_batch
        self.runtime.patch_verl_dynamic_batch_sync()
        self.assertIs(self.balancing.prepare_dynamic_batch, first)
        self.assertIs(self.actor.prepare_dynamic_batch, first)
        first("batch", 512)
        self.prepare.assert_called_once()

    def test_missing_optional_actor_import_does_not_disable_balancing_fix(self):
        del sys.modules["verl.workers.actor.dp_actor"]
        del self.modules["verl.workers.actor"].dp_actor
        self.runtime.patch_verl_dynamic_batch_sync()
        self.balancing.prepare_dynamic_batch("batch", 512)
        self.prepare.assert_called_once_with("batch", 512, dp_group=self.world)


class TrainingOutputPatchTests(unittest.TestCase):
    def setUp(self):
        self.runtime = load_patch_module()
        self.modules = module_tree(
            "verl.utils.tensordict_utils",
            "verl.utils.dataset.dataset_utils",
            "verl.utils.py_functional",
            "verl.workers.engine.base",
            "verl.workers.engine.utils",
            "verl.workers.engine.fsdp.transformer_impl",
        )
        self.tu = self.modules["verl.utils.tensordict_utils"]

        def assign_non_tensor(data, **values):
            data.update(values)

        self.tu.assign_non_tensor = assign_non_tensor
        self.tu.get_non_tensor_data = lambda data, key, default: data.get(key, default)
        self.modules["verl.utils.dataset.dataset_utils"].DatasetPadMode = SimpleNamespace(NO_PADDING="no_padding")

        def append_to_dict(destination, values):
            for key, value in values.items():
                destination.setdefault(key, []).append(value)

        self.modules["verl.utils.py_functional"].append_to_dict = append_to_dict
        self.train_calls = []
        self.train_result = object()
        calls, result = self.train_calls, self.train_result

        class Engine:
            def train_batch(self, data, loss_function):
                calls.append((self, dict(data), loss_function))
                return result

        self.engine = Engine
        self.modules["verl.workers.engine.base"].BaseEngine = Engine
        self.utils = self.modules["verl.workers.engine.utils"]
        self.original_postprocess = Mock(return_value=object())
        self.utils.postprocess_batch_func = self.original_postprocess
        self.fsdp = self.modules["verl.workers.engine.fsdp.transformer_impl"]
        self.fsdp.postprocess_batch_func = self.original_postprocess
        modules_patch = patch.dict(sys.modules, self.modules)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def test_train_flag_is_set_before_original_method_and_output_is_preserved(self):
        self.runtime.patch_verl_training_output_postprocess()
        engine = self.engine()
        data = {"use_dynamic_bsz": True, "return_model_output": True}
        loss = object()
        result = engine.train_batch(data, loss)
        self.assertIs(result, self.train_result)
        self.assertEqual(self.train_calls, [(engine, {"use_dynamic_bsz": True, "return_model_output": False}, loss)])

    def test_inference_without_training_flag_calls_original_unchanged(self):
        self.runtime.patch_verl_training_output_postprocess()
        outputs, indices, data = [object()], [2, 1], {"use_dynamic_bsz": True}
        result = self.utils.postprocess_batch_func(outputs, indices, data)
        self.original_postprocess.assert_called_once_with(output_lst=outputs, indices=indices, data=data)
        self.assertIs(result, self.original_postprocess.return_value)

    def test_explicit_model_output_flag_preserves_original_path(self):
        self.runtime.patch_verl_training_output_postprocess()
        data = {"return_model_output": True}
        self.utils.postprocess_batch_func([], [], data)
        self.original_postprocess.assert_called_once_with(output_lst=[], indices=[], data=data)

    def test_training_keeps_losses_and_metrics_without_touching_model_outputs(self):
        self.runtime.patch_verl_training_output_postprocess()
        loss_a, loss_b, model_output = object(), object(), object()
        outputs = [
            {"loss": loss_a, "metrics": {"grad_norm": 1.0}, "model_output": model_output},
            {"metrics": {"grad_norm": 2.0, "entropy": 0.5}},
            {"loss": loss_b, "model_output": model_output},
        ]
        for dynamic in (True, False):
            with self.subTest(dynamic=dynamic):
                result = self.fsdp.postprocess_batch_func(outputs, [2, 0, 1], {"return_model_output": False, "use_dynamic_bsz": dynamic})
                self.assertEqual(result, {"model_output": {}, "loss": [loss_a, loss_b], "metrics": {"grad_norm": [1.0, 2.0], "entropy": [0.5]}})
                self.assertIs(outputs[0]["model_output"], model_output)
        self.original_postprocess.assert_not_called()
        self.assertIs(self.fsdp.postprocess_batch_func, self.utils.postprocess_batch_func)

    def test_empty_training_outputs_preserve_return_shape(self):
        self.runtime.patch_verl_training_output_postprocess()
        result = self.utils.postprocess_batch_func([], [], {"return_model_output": False})
        self.assertEqual(result, {"model_output": {}, "loss": [], "metrics": {}})

    def test_unsupported_padding_fails_instead_of_silently_changing_layout(self):
        self.runtime.patch_verl_training_output_postprocess()
        with self.assertRaisesRegex(AssertionError, "NO_PADDING"):
            self.utils.postprocess_batch_func([], [], {"return_model_output": False, "pad_mode": "padded"})

    def test_repeated_patch_keeps_original_single_training_call(self):
        self.runtime.patch_verl_training_output_postprocess()
        train, postprocess = self.engine.train_batch, self.utils.postprocess_batch_func
        self.runtime.patch_verl_training_output_postprocess()
        self.assertIs(self.engine.train_batch, train)
        self.assertIs(self.utils.postprocess_batch_func, postprocess)
        self.engine().train_batch({}, object())
        self.assertEqual(len(self.train_calls), 1)


if __name__ == "__main__":
    unittest.main()
