#!/usr/bin/env python3
"""Patch local VERL/vLLM integration issues before multi-GPU Slurm jobs.

The pinned VeRL version names weight-sync ZMQ IPC sockets only by GPU
UUID (``/tmp/rl-colocate-zmq-<uuid>.sock``).  That collides when multiple
rollout ranks on a node update weights concurrently and can also collide with
stale sockets left by failed jobs.  This patch makes each update call allocate a
unique, job/rank-scoped socket path and passes that path explicitly to the vLLM
worker extension.

It also backports VERL PR #6456's coordinator-safe HYBRID sleep/wake route.
Older VERL releases use ``collective_rpc("sleep"/"wake_up")`` directly,
which bypasses vLLM's EngineCore lifecycle bookkeeping.  Calling the dedicated
server methods routes through ``AsyncLLM.sleep()`` / ``AsyncLLM.wake_up()``
instead, while retaining level-2 cache freeing and CUDA graphs.

The existing no-argument launcher invocation applies the patch in the selected
Python environment. Use --check for read-only validation or --source to select a
dedicated VeRL source checkout. Every target is validated and compiled before
any file is replaced; individual replacements are atomic, not a multi-file
transaction.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import os
import stat
import sys
import tempfile
from pathlib import Path

PATCH_MARKER = "rllm_unique_verl_zmq_ipc_patch"
SLEEP_WAKE_PATCH_MARKER = "rllm_verl_engine_sleep_wake_patch"
LOCK_PATH = Path("/tmp/rllm-patch-verl-zmq-ipc.lock")


def _package_root() -> Path:
    """Resolve the package this interpreter would import, without importing VeRL."""
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None or spec.submodule_search_locations is None:
        raise FileNotFoundError("Could not resolve an ordinary VeRL package in this Python environment")
    locations = [Path(location).resolve() for location in spec.submodule_search_locations]
    root = Path(spec.origin).resolve().parent
    if locations != [root] or not root.is_dir():
        raise RuntimeError("VeRL package resolution is ambiguous; select a dedicated checkout with --source")
    return root


def _write_atomic(path: Path, content: bytes) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as tmp:
            os.fchmod(tmp.fileno(), mode)
            tmp.write(content)
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _replace_or_already(text: str, old: str, new: str, path: Path, *, count: int = 1) -> tuple[str, bool]:
    new_count = text.count(new)
    # Some new snippets intentionally include the old method header. Count only
    # old occurrences outside complete new snippets, never infer state from a marker.
    old_count = text.replace(new, "").count(old)
    if new_count == count and old_count == 0:
        return text, False
    if old_count == count and new_count == 0:
        return text.replace(old, new), True
    raise RuntimeError(f"{path}: expected {count} old or patched occurrences, found old={old_count}, patched={new_count}; mixed, partial, or unsupported source")


def _transform(text: str, replacements: list[tuple[str, str, int]], path: Path) -> tuple[str, bool]:
    states = []
    for old, new, count in replacements:
        text, changed = _replace_or_already(text, old, new, path, count=count)
        states.append(changed)
    if any(states) and not all(states):
        raise RuntimeError(f"{path}: mixed old and patched steps in one transform")
    return text, all(states)


def _add_marker(text: str, marker: str, message: str, anchor: str, path: Path, *, transformed: bool) -> str:
    line = f"# {marker}: {message}\n"
    if marker in text:
        if text.count(marker) != 1 or text.count(line) != 1 or transformed:
            raise RuntimeError(f"{path}: patch marker contradicts validated source")
        return text
    text, _ = _replace_or_already(text, anchor, anchor + line, path)
    return text


def _patch_utils(text: str, path: Path) -> str:
    replacements = [
        (
            "    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):\n",
            "    def update_weights_from_ipc(\n"
            "        self,\n"
            "        peft_config: dict = None,\n"
            "        base_sync_done=False,\n"
            "        use_shm: bool = False,\n"
            "        zmq_handle: str | None = None,\n"
            "    ):\n",
        ),
        (
            "            zmq_handle=self._get_zmq_handle(),\n",
            "            zmq_handle=zmq_handle or self._get_zmq_handle(),\n",
        ),
    ]
    # Both supported worker extensions must agree on their patch state.
    text, changed = _transform(text, [(old, new, 2) for old, new in replacements], path)
    return _add_marker(
        text,
        PATCH_MARKER,
        "update_weights_from_ipc accepts caller-provided IPC handles.",
        'logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))\n',
        path,
        transformed=changed,
    )


def _patch_vllm_rollout(text: str, path: Path) -> str:
    old_future = (
        '        future = await self._execute_method(\n            "update_weights_from_ipc",\n            non_block=True,\n            kwargs={**kwargs, "use_shm": self.use_shm},\n        )\n'
    )
    new_future = (
        '        zmq_rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "norank"\n'
        '        zmq_job = os.environ.get("SLURM_JOB_ID") or os.environ.get("RAY_JOB_ID") or "nojid"\n'
        "        zmq_handle = (\n"
        '            f"ipc:///tmp/rl-colocate-zmq-{zmq_job}-{zmq_rank}-{self.device_uuid}-"\n'
        '            f"{os.getpid()}-{time.time_ns()}.sock"\n'
        "        )\n"
        "        future = await self._execute_method(\n"
        '            "update_weights_from_ipc",\n'
        "            non_block=True,\n"
        '            kwargs={**kwargs, "use_shm": self.use_shm, "zmq_handle": zmq_handle},\n'
        "        )\n"
    )

    old_sender = "            zmq_handle=self.zmq_handle,\n"
    new_sender = "            zmq_handle=zmq_handle,\n"
    text, changed = _transform(text, [(old_future, new_future, 1), (old_sender, new_sender, 1)], path)
    return _add_marker(
        text,
        PATCH_MARKER,
        "use per-update IPC handles to avoid ZMQ bind collisions.",
        'logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))\n',
        path,
        transformed=changed,
    )


def _patch_sleep_wake_vllm_rollout(text: str, path: Path) -> str:
    """Backport the ServerAdapter portion of VERL PR #6456."""

    old_execute_header = "    async def _execute_method(\n"
    new_execute_header = (
        "    def _ensure_server_handle(self) -> bool:\n"
        '        """Lazy-init server handle. Returns False if this rank should not proceed."""\n'
        "        if self.rollout_rank != 0:\n"
        "            return False\n"
        "        # Lazy init http server adapter because http server is launched after hybrid engine.\n"
        "        if self.server_handle is None:\n"
        "            prefix = self._get_server_name_prefix()\n"
        '            self.server_handle = ray.get_actor(f"{prefix}server_{self.replica_rank}_{self.node_rank}")\n'
        "        return True\n"
        "\n"
        "    async def _execute_method(\n"
    )

    old_execute_init = (
        "        if self.rollout_rank != 0:\n"
        "            return None\n"
        "\n"
        "        # Lazy init http server adapter because http server is launched after hybrid engine.\n"
        "        if self.server_handle is None:\n"
        "            prefix = self._get_server_name_prefix()\n"
        '            self.server_handle = ray.get_actor(f"{prefix}server_{self.replica_rank}_{self.node_rank}")\n'
        "\n"
    )
    new_execute_init = "        if not self._ensure_server_handle():\n            return None\n\n"

    old_resume_release = (
        "        if self.config.free_cache_engine:\n"
        '            await self._execute_method("wake_up", kwargs={"tags": tags})\n'
        "\n"
        "    async def release(self):\n"
        '        """Release weights and kv cache in GPU memory."""\n'
        "        if self.config.free_cache_engine:\n"
        '            await self._execute_method("sleep", kwargs={"level": self.sleep_level})\n'
    )
    new_resume_release = (
        "        if self.config.free_cache_engine and self._ensure_server_handle():\n"
        "            await self.server_handle.wake_up.remote(tags=tags)\n"
        "\n"
        "    async def release(self):\n"
        '        """Release weights and kv cache in GPU memory."""\n'
        "        if self.config.free_cache_engine and self._ensure_server_handle():\n"
        "            await self.server_handle.sleep.remote()\n"
    )
    text, changed = _transform(
        text,
        [
            (old_execute_header, new_execute_header, 1),
            (old_execute_init, new_execute_init, 1),
            (old_resume_release, new_resume_release, 1),
        ],
        path,
    )
    return _add_marker(
        text,
        SLEEP_WAKE_PATCH_MARKER,
        "route HYBRID sleep/wake through EngineCore.",
        'logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))\n',
        path,
        transformed=changed,
    )


def _patch_vllm_async_server(text: str, path: Path) -> str:
    """Backport the vLLMHttpServer portion of VERL PR #6456."""

    old_wake_up = (
        "    async def wake_up(self):\n"
        "        if self.node_rank != 0:\n"
        "            return\n"
        "\n"
        "        if self.rollout_mode == RolloutMode.HYBRID:\n"
        "            # In hybrid mode, rollout is wake up in `update_weights`\n"
        '            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")\n'
        "        elif self.rollout_mode == RolloutMode.COLOCATED:\n"
        "            # Directly call engine to wake up without sync weights.\n"
        "            await self.engine.wake_up(tags=self._get_wake_up_tags())\n"
        "            await self.engine.reset_prefix_cache()\n"
        "        elif self.rollout_mode == RolloutMode.STANDALONE:\n"
        '            logger.info("skip wake_up in standalone mode")\n'
    )
    new_wake_up = (
        "    async def wake_up(self, tags: list[str] | None = None):\n"
        "        if self.node_rank != 0:\n"
        "            return\n"
        "\n"
        "        if self.rollout_mode == RolloutMode.HYBRID:\n"
        "            # Propagate to all data-parallel EngineCore processes.\n"
        "            await self.engine.wake_up(tags=tags or self._get_wake_up_tags())\n"
        "            await self.engine.reset_prefix_cache()\n"
        "        elif self.rollout_mode == RolloutMode.COLOCATED:\n"
        "            # Directly call engine to wake up without sync weights.\n"
        "            await self.engine.wake_up(tags=self._get_wake_up_tags())\n"
        "            await self.engine.reset_prefix_cache()\n"
        "        elif self.rollout_mode == RolloutMode.STANDALONE:\n"
        '            logger.info("skip wake_up in standalone mode")\n'
    )

    old_sleep_hybrid = (
        "    async def _sleep_hybrid(self):\n"
        '        """HYBRID sleep: lora adapters only need level=1; full weights need level=2."""\n'
        "        # Don't use engine.sleep(level=2) here\n"
        "        # lora only update adapter weights, so set sleep level to 1\n"
        "        if self.lora_as_adapter:\n"
        "            sleep_level = 1\n"
        "        else:\n"
        "            sleep_level = 2\n"
        '        await self.engine.collective_rpc("sleep", kwargs={"level": sleep_level})\n'
        '        if _VLLM_VERSION >= version.parse("0.17.0"):\n'
        "            await self.engine.reset_encoder_cache()\n"
    )
    new_sleep_hybrid = (
        "    async def _sleep_hybrid(self):\n"
        '        """HYBRID sleep: lora adapters only need level=1; full weights need level=2.\n'
        "\n"
        "        Route through AsyncLLM.sleep() so EngineCore coordinates the full\n"
        "        lifecycle instead of bypassing it through collective_rpc.\n"
        '        """\n'
        "        # lora only update adapter weights, so set sleep level to 1\n"
        "        if self.lora_as_adapter:\n"
        "            sleep_level = 1\n"
        "        else:\n"
        "            sleep_level = 2\n"
        "        await self.engine.sleep(level=sleep_level)\n"
        '        if _VLLM_VERSION >= version.parse("0.17.0"):\n'
        "            await self.engine.reset_encoder_cache()\n"
    )
    text, changed = _transform(text, [(old_wake_up, new_wake_up, 1), (old_sleep_hybrid, new_sleep_hybrid, 1)], path)
    return _add_marker(
        text,
        SLEEP_WAKE_PATCH_MARKER,
        "backport VERL PR #6456 lifecycle routing.",
        "logger.setLevel(logging.INFO)\n",
        path,
        transformed=changed,
    )


def plan_patch(package_root: Path) -> dict[Path, tuple[bytes, bytes]]:
    """Read, transform, and compile every target before returning any write plan."""
    package_root = package_root.resolve()
    directory = package_root / "workers/rollout/vllm_rollout"
    transforms = {
        directory / "utils.py": (_patch_utils,),
        directory / "vllm_rollout.py": (_patch_vllm_rollout, _patch_sleep_wake_vllm_rollout),
        directory / "vllm_async_server.py": (_patch_vllm_async_server,),
    }
    original = {}
    for path in transforms:
        for component in (path, *path.parents):
            if component == package_root:
                break
            if component.is_symlink():
                raise RuntimeError(f"{path}: symlink patch targets are unsupported")
        original[path] = path.read_bytes()
    plan = {}
    for path, functions in transforms.items():
        text = original[path].decode("utf-8")
        for transform in functions:
            text = transform(text, path)
        compile(text, str(path), "exec")
        plan[path] = (original[path], text.encode("utf-8"))
    return plan


def apply_plan(plan: dict[Path, tuple[bytes, bytes]]) -> int:
    """Apply a validated plan; an I/O failure can still leave a partial file set."""
    for path, (original, _) in plan.items():
        if path.read_bytes() != original:
            raise RuntimeError(f"{path}: source changed after planning; no files were written")
    changed = 0
    for path, (original, result) in plan.items():
        if original != result:
            _write_atomic(path, result)
            changed += 1
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate only; exit 1 if a patch is needed")
    parser.add_argument("--source", type=Path, help="Explicit VeRL checkout root instead of the selected Python package")
    args = parser.parse_args(argv)
    package_root = args.source.resolve() / "verl" if args.source else _package_root()
    if args.check:
        plan = plan_patch(package_root)
        pending = any(original != result for original, result in plan.values())
        status = "patch required" if pending else "already patched"
        print(f"[check] VERL IPC/lifecycle patches {status}: {package_root}")
        return int(pending)
    with LOCK_PATH.open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        changed = apply_plan(plan_patch(package_root))
    status = "patched" if changed else "already patched"
    print(f"[launcher] VERL IPC/lifecycle patches {status}: {package_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[launcher] failed to patch VERL IPC/lifecycle paths: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
