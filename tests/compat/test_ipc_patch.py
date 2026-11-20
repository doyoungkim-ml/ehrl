"""Hermetic source fixtures guard IPC patch preparation; no installed VeRL is touched."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/patch_verl_zmq_ipc.py"
spec = importlib.util.spec_from_file_location("isolated_ipc_patch", SCRIPT)
ipc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ipc)

# Reduced, syntactically valid forms of the supported VeRL source. Repeated
# worker methods are deliberate: accepting just one patched occurrence caused
# a previous false success when the other extension remained unpatched.
UTILS = """logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class Worker:
    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        return receive(
            zmq_handle=self._get_zmq_handle(),
        )

class OmniWorker:
    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        return receive(
            zmq_handle=self._get_zmq_handle(),
        )
"""

ROLLOUT = '''logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

class Adapter:
    async def _execute_method(
        self, method, non_block=False, kwargs=None,
    ):
        if self.rollout_rank != 0:
            return None

        # Lazy init http server adapter because http server is launched after hybrid engine.
        if self.server_handle is None:
            prefix = self._get_server_name_prefix()
            self.server_handle = ray.get_actor(f"{prefix}server_{self.replica_rank}_{self.node_rank}")

        return await self.server_handle.collective_rpc.remote(method, kwargs=kwargs)

    async def update_weights(self, **kwargs):
        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )
        return send(
            zmq_handle=self.zmq_handle,
        )

    async def resume(self, tags=None):
        if self.config.free_cache_engine:
            await self._execute_method("wake_up", kwargs={"tags": tags})

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            await self._execute_method("sleep", kwargs={"level": self.sleep_level})
'''

SERVER = '''logger.setLevel(logging.INFO)

class Server:
    async def wake_up(self):
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is wake up in `update_weights`
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            await self.engine.wake_up(tags=self._get_wake_up_tags())
            await self.engine.reset_prefix_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def _sleep_hybrid(self):
        """HYBRID sleep: lora adapters only need level=1; full weights need level=2."""
        # Don't use engine.sleep(level=2) here
        # lora only update adapter weights, so set sleep level to 1
        if self.lora_as_adapter:
            sleep_level = 1
        else:
            sleep_level = 2
        await self.engine.collective_rpc("sleep", kwargs={"level": sleep_level})
        if _VLLM_VERSION >= version.parse("0.17.0"):
            await self.engine.reset_encoder_cache()
'''


class IpcPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name)
        self.package = self.source / "verl"
        self.directory = self.package / "workers/rollout/vllm_rollout"
        self.directory.mkdir(parents=True)
        self.paths = {name: self.directory / name for name in ("utils.py", "vllm_rollout.py", "vllm_async_server.py")}
        for name, text in zip(self.paths, (UTILS, ROLLOUT, SERVER), strict=True):
            self.paths[name].write_text(text)
        self.original = self.contents()

    def contents(self):
        return {name: path.read_bytes() for name, path in self.paths.items() if path.exists()}

    def apply(self):
        return ipc.apply_plan(ipc.plan_patch(self.package))

    def test_plan_is_read_only_and_all_results_compile(self):
        plan = ipc.plan_patch(self.package)
        self.assertEqual(self.contents(), self.original)
        self.assertEqual(len(plan), 3)
        for path, (_, result) in plan.items():
            compile(result, str(path), "exec")

    def test_apply_updates_both_extensions_and_lifecycle_then_is_idempotent(self):
        self.paths["utils.py"].chmod(0o640)
        self.assertEqual(self.apply(), 3)
        content = self.contents()
        utils = content["utils.py"].decode()
        rollout = content["vllm_rollout.py"].decode()
        server = content["vllm_async_server.py"].decode()
        self.assertEqual(utils.count("zmq_handle=zmq_handle or self._get_zmq_handle()"), 2)
        self.assertIn('"zmq_handle": zmq_handle', rollout)
        self.assertIn("await self.server_handle.wake_up.remote(tags=tags)", rollout)
        self.assertIn("await self.server_handle.sleep.remote()", rollout)
        self.assertIn("await self.engine.sleep(level=sleep_level)", server)
        self.assertIn("await self.engine.wake_up(tags=tags or self._get_wake_up_tags())", server)
        self.assertEqual(self.paths["utils.py"].stat().st_mode & 0o777, 0o640)
        self.assertEqual(self.apply(), 0)
        self.assertEqual(self.contents(), content)

    def test_later_transform_failure_leaves_every_file_unchanged(self):
        path = self.paths["vllm_async_server.py"]
        path.write_text(SERVER.replace("async def wake_up(self):", "async def incompatible_wake_up(self):"))
        before = self.contents()
        with self.assertRaisesRegex(RuntimeError, "unsupported source"):
            self.apply()
        self.assertEqual(self.contents(), before)

    def test_later_compilation_failure_leaves_every_file_unchanged(self):
        self.paths["vllm_async_server.py"].write_text(SERVER + "def incomplete(\n")
        before = self.contents()
        with self.assertRaises(SyntaxError):
            self.apply()
        self.assertEqual(self.contents(), before)

    def test_all_files_are_read_before_first_transformation(self):
        self.paths["vllm_async_server.py"].unlink()
        before = self.contents()
        with patch.object(ipc, "_patch_utils") as transform:
            with self.assertRaises(FileNotFoundError):
                self.apply()
            transform.assert_not_called()
        self.assertEqual(self.contents(), before)

    def test_one_patched_and_one_old_worker_is_rejected(self):
        path = self.paths["utils.py"]
        result = ipc.plan_patch(self.package)[path][1].decode()
        result = result.replace("zmq_handle=zmq_handle or self._get_zmq_handle()", "zmq_handle=self._get_zmq_handle()", 1)
        path.write_text(result)
        before = self.contents()
        with self.assertRaisesRegex(RuntimeError, "old=1, patched=1"):
            self.apply()
        self.assertEqual(self.contents(), before)

    def test_partial_transform_with_both_signatures_patched_is_rejected(self):
        path = self.paths["utils.py"]
        result = ipc.plan_patch(self.package)[path][1].decode()
        result = result.replace("zmq_handle=zmq_handle or self._get_zmq_handle()", "zmq_handle=self._get_zmq_handle()")
        path.write_text(result)
        before = self.contents()
        with self.assertRaisesRegex(RuntimeError, "mixed old and patched steps"):
            self.apply()
        self.assertEqual(self.contents(), before)

    def test_marker_alone_does_not_prove_patch_success(self):
        marker = f"# {ipc.PATCH_MARKER}: update_weights_from_ipc accepts caller-provided IPC handles.\n"
        self.paths["utils.py"].write_text(UTILS + marker)
        before = self.contents()
        with self.assertRaisesRegex(RuntimeError, "marker contradicts"):
            self.apply()
        self.assertEqual(self.contents(), before)

    def test_missing_marker_anchor_cannot_report_false_success(self):
        self.paths["utils.py"].write_text(UTILS.replace('logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))', "# no logging anchor"))
        before = self.contents()
        with self.assertRaisesRegex(RuntimeError, "unsupported source"):
            self.apply()
        self.assertEqual(self.contents(), before)

    def test_additional_old_worker_occurrence_is_rejected(self):
        self.paths["utils.py"].write_text(UTILS + UTILS[UTILS.index("class Worker:") : UTILS.index("class OmniWorker:")].replace("class Worker:", "class ExtraWorker:"))
        before = self.contents()
        with self.assertRaisesRegex(RuntimeError, "old=3, patched=0"):
            self.apply()
        self.assertEqual(self.contents(), before)

    def test_complete_previously_patched_file_is_valid_with_other_unpatched_files(self):
        path = self.paths["utils.py"]
        path.write_bytes(ipc.plan_patch(self.package)[path][1])
        self.assertEqual(self.apply(), 2)
        self.assertEqual(self.apply(), 0)

    def test_existing_ipc_patch_can_receive_the_later_lifecycle_backport(self):
        path = self.paths["vllm_rollout.py"]
        path.write_text(ipc._patch_vllm_rollout(ROLLOUT, path))
        self.assertEqual(self.apply(), 3)
        self.assertEqual(self.apply(), 0)

    def test_source_change_after_plan_aborts_before_any_writes(self):
        plan = ipc.plan_patch(self.package)
        path = self.paths["vllm_async_server.py"]
        path.write_bytes(path.read_bytes() + b"# another writer\n")
        before = self.contents()
        with self.assertRaisesRegex(RuntimeError, "source changed after planning"):
            ipc.apply_plan(plan)
        self.assertEqual(self.contents(), before)

    def test_symlink_target_is_rejected_before_writes(self):
        path = self.paths["vllm_async_server.py"]
        external = self.source / "outside.py"
        path.rename(external)
        path.symlink_to(external)
        before = self.contents()
        with self.assertRaisesRegex(RuntimeError, "symlink patch targets"):
            self.apply()
        self.assertEqual(self.contents(), before)
        self.assertEqual(external.read_text(), SERVER)

    def test_cli_check_is_read_only_and_source_bypasses_package_resolution(self):
        with patch.object(ipc, "_package_root") as resolve, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ipc.main(["--check", "--source", str(self.source)]), 1)
            resolve.assert_not_called()
        self.assertEqual(self.contents(), self.original)
        self.apply()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ipc.main(["--check", "--source", str(self.source)]), 0)

    def test_no_argument_launcher_applies_to_resolved_package(self):
        # All paths, including the selected package, are temporary test fixtures.
        with patch.object(ipc, "_package_root", return_value=self.package) as resolve, patch.object(ipc, "LOCK_PATH", self.source / ".patch.lock"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ipc.main([]), 0)
            resolve.assert_called_once_with()
        self.assertEqual(self.apply(), 0)


class PackageResolutionTests(unittest.TestCase):
    def test_find_spec_uses_import_precedence_without_importing_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, selected = root / "first-site", root / "selected-site"
            for parent in (first, selected):
                package = parent / "verl"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text('raise AssertionError("VeRL must not be imported")\n')
            with patch.dict(sys.modules), patch.object(sys, "path", [str(selected), str(first), *sys.path]):
                sys.modules.pop("verl", None)
                self.assertEqual(ipc._package_root(), selected / "verl")
                self.assertNotIn("verl", sys.modules)

    def test_missing_or_namespace_package_is_rejected(self):
        for spec in (None, SimpleNamespace(origin=None, submodule_search_locations=["/one", "/two"])):
            with self.subTest(spec=spec), patch.object(ipc.importlib.util, "find_spec", return_value=spec):
                with self.assertRaises(FileNotFoundError):
                    ipc._package_root()

    def test_conflicting_origin_and_search_location_is_rejected(self):
        spec = SimpleNamespace(origin="/one/verl/__init__.py", submodule_search_locations=["/two/verl"])
        with patch.object(ipc.importlib.util, "find_spec", return_value=spec):
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                ipc._package_root()


if __name__ == "__main__":
    unittest.main()
