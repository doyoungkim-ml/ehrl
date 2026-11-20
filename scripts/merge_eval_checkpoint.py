#!/usr/bin/env python3
"""Publish one CPU-merged global_step_N checkpoint without replacing any target.

--manifest accepts either one checkpoint entry or the campaign inventory's
{"runs": {"name": {"checkpoints": [...]}}} document. A checkpoint entry contains
"path", "actor_files" ([{"name", "bytes", "mtime_ns"}] for both model ranks),
and "hf_hashes" (relative actor/huggingface asset names mapped to SHA-256).
Manifest hashes may be a subset; every source HF asset is hashed before/after.
Source weights are identified by size and mtime_ns, not cryptographic hashes.

The existing merge_fsdp_checkpoint.py runs in --python with GPUs hidden and
--merge-dtype preserve. Original tokenizer/template/generation assets are
copied byte-for-byte into staging; the merger's config.json retains its dtype.
A persistent sibling lock coordinates helpers. Linux renameat2(NOREPLACE)
publishes a complete directory atomically, refusing even an empty target.
When that operation is unsupported, an atomic, no-clobber relative symlink
publishes the validated retained attempt/model directory. Reuse accepts only
that exact sibling layout with matching target/attempt provenance and unchanged
validated output. Merger formatting changes can reuse a recorded matching AST
fingerprint; original source/output identity and merger byte hashes are preserved.
Referenced attempt directories must be retained with their published symlinks. Attempts and logs survive both success and failure.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MERGER = Path(__file__).with_name("merge_fsdp_checkpoint.py")
COMPLETE = "merge_complete.json"
SHARD_PATTERN = re.compile(r"model_world_size_([1-9][0-9]*)_rank_([0-9]+)\.pt")


def shard_names(world_size: int) -> list[str]:
    return [f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]


def complete_shard_set(names: list[str]) -> list[str] | None:
    """The sorted names if they are exactly ranks 0..W-1 of one world size W, else None."""
    matches = [SHARD_PATTERN.fullmatch(name) for name in names]
    if not names or not all(matches) or len({m.group(1) for m in matches}) != 1:
        return None
    expected = shard_names(int(matches[0].group(1)))
    return expected if sorted(names) == sorted(expected) else None


class MergeError(RuntimeError):
    pass


def write_json(path: Path, payload: dict) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def regular_stat(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise MergeError(f"Expected a regular, nonsymlink file: {path}")
    return info


def sha256(path: Path) -> str:
    before = regular_stat(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = regular_stat(path)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise MergeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def merger_fingerprint(path: Path) -> dict[str, str]:
    before = regular_stat(path)
    source = path.read_bytes()
    after = regular_stat(path)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise MergeError(f"Merger changed while fingerprinting: {path}")
    tree = ast.dump(ast.parse(source), include_attributes=False)
    return {"merger_sha256": hashlib.sha256(source).hexdigest(), "merger_ast_sha256": hashlib.sha256(tree.encode()).hexdigest()}


def matching_provenance(recorded: dict | None, current: dict) -> bool:
    if not isinstance(recorded, dict) or re.fullmatch(r"[0-9a-f]{64}", str(recorded.get("merger_sha256", ""))) is None:
        return False
    recorded_ast = recorded.get("merger_ast_sha256")
    if recorded["merger_sha256"] != current["merger_sha256"] and recorded_ast != current["merger_ast_sha256"]:
        return False
    # Legacy records without an AST fingerprint require an unchanged byte hash.
    # All other provenance fields, including source files and Python, stay exact.
    return {**recorded, **{key: current[key] for key in ("merger_sha256", "merger_ast_sha256")}} == current


def files_under(directory: Path) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise MergeError(f"Expected a nonsymlink directory: {directory}")
    files = []
    for parent, names, filenames in os.walk(directory, followlinks=False):
        for name in names:
            child = Path(parent) / name
            if child.is_symlink():
                raise MergeError(f"Symlink directory is unsupported: {child}")
        for name in filenames:
            child = Path(parent) / name
            regular_stat(child)
            files.append(child)
    return sorted(files)


def asset_hashes(directory: Path, *, output: bool = False) -> dict[str, str]:
    hashes = {}
    for path in files_under(directory):
        name = path.relative_to(directory).as_posix()
        if output and (path.suffix == ".safetensors" or name == COMPLETE):
            continue
        if not output and (path.suffix in {".safetensors", ".pt", ".bin"} or name == COMPLETE):
            raise MergeError(f"Expected HF config/tokenizer assets, found a weight or reserved file: {path}")
        hashes[name] = sha256(path)
    return hashes


def read_object(path: Path) -> dict:
    regular_stat(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise MergeError(f"Expected a JSON object: {path}")
    return value


def validate_tokenizer(directory: Path) -> None:
    read_object(directory / "tokenizer_config.json")
    if (directory / "tokenizer.json").exists():
        read_object(directory / "tokenizer.json")
    elif any((directory / name).is_file() for name in ("tokenizer.model", "spiece.model")):
        for name in ("tokenizer.model", "spiece.model"):
            if (directory / name).exists() and regular_stat(directory / name).st_size == 0:
                raise MergeError(f"Empty tokenizer asset: {directory / name}")
    elif (directory / "vocab.json").exists() and (directory / "merges.txt").exists():
        read_object(directory / "vocab.json")
        if regular_stat(directory / "merges.txt").st_size == 0:
            raise MergeError(f"Empty tokenizer merges: {directory}")
    else:
        raise MergeError(f"Missing tokenizer vocabulary/model assets: {directory}")


def source_snapshot(source: Path) -> dict:
    match = re.fullmatch(r"global_step_([0-9]+)", source.name)
    if not match or source.is_symlink() or not source.is_dir():
        raise MergeError(f"Source must be one global_step_N directory: {source}")
    actor = source / "actor"
    if actor.is_symlink() or not actor.is_dir():
        raise MergeError(f"Missing nonsymlink actor directory: {actor}")
    expected_shards = complete_shard_set([path.name for path in actor.glob("model_world_size_*_rank_*.pt")])
    if expected_shards is None:
        raise MergeError(f"Expected exactly one complete set of model_world_size_W rank shards: {actor}")
    shards = []
    for name in sorted(expected_shards):
        info = regular_stat(actor / name)
        if info.st_size <= 0:
            raise MergeError(f"Empty source model shard: {actor / name}")
        shards.append({"name": name, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns})
    hf = actor / "huggingface"
    read_object(hf / "config.json")
    validate_tokenizer(hf)
    return {
        "path": str(source),
        "experiment_dir": str(source.parent),
        "step": int(match.group(1)),
        "actor_files": shards,
        "hf_hashes": asset_hashes(hf),
    }


def expected_checkpoint(manifest: Path | None, source: Path) -> dict | None:
    if manifest is None:
        return None
    payload = read_object(manifest)
    if "runs" in payload:
        entries = [checkpoint for run in payload["runs"].values() for checkpoint in run["checkpoints"]]
    else:
        entries = [payload]
    matches = [entry for entry in entries if Path(entry.get("path", "")).expanduser().resolve() == source]
    if len(matches) != 1:
        raise MergeError(f"Manifest must identify this exact source once: {source}")
    expected = matches[0]
    shards = expected.get("actor_files", [])
    if complete_shard_set([shard.get("name", "") for shard in shards]) is None:
        raise MergeError("Manifest must fingerprint every model_world_size_W rank shard")
    for shard in shards:
        if any(type(shard.get(key)) is not int or shard[key] < 0 for key in ("bytes", "mtime_ns")):
            raise MergeError("Manifest shard bytes and mtime_ns must be nonnegative integers")
    hashes = expected.get("hf_hashes")
    if not isinstance(hashes, dict) or not {"config.json", "tokenizer_config.json"} <= hashes.keys():
        raise MergeError("Manifest hf_hashes must include config.json and tokenizer_config.json")
    for name, digest in hashes.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise MergeError("Manifest hf_hashes must contain relative asset names and lowercase SHA-256 hashes")
    return expected


def verify_expected(snapshot: dict, expected: dict | None) -> None:
    if expected is None:
        return
    expected_shards = [{key: shard[key] for key in ("name", "bytes", "mtime_ns")} for shard in expected["actor_files"]]
    if snapshot["actor_files"] != sorted(expected_shards, key=lambda shard: shard["name"]):
        raise MergeError("Source shard metadata differs from the expected manifest")
    if any(snapshot["hf_hashes"].get(name) != digest for name, digest in expected["hf_hashes"].items()):
        raise MergeError("Source HF asset hashes differ from the expected manifest")
    if "step" in expected and expected["step"] != snapshot["step"]:
        raise MergeError("Source step differs from the expected manifest")


def safe_weight_name(name: str) -> bool:
    return isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors")


def safetensor_header(path: Path) -> tuple[dict, dict]:
    info = regular_stat(path)
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise MergeError(f"Truncated safetensors header: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= length <= min(64 * 1024 * 1024, info.st_size - 8):
            raise MergeError(f"Invalid safetensors header size: {path}")
        raw = stream.read(length)
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise MergeError(f"Invalid safetensors header object: {path}")
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise MergeError(f"No tensors in merged weight file: {path}")
    width = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    intervals = []
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict) or tensor.get("dtype") not in width:
            raise MergeError(f"Invalid tensor descriptor for {name}: {path}")
        shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
        invalid_shape = not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape)
        invalid_offsets = not isinstance(offsets, list) or len(offsets) != 2 or any(type(offset) is not int for offset in offsets)
        if invalid_shape or invalid_offsets:
            raise MergeError(f"Invalid tensor shape/offsets for {name}: {path}")
        size = width[tensor["dtype"]]
        for dimension in shape:
            size *= dimension
        begin, end = offsets
        if begin < 0 or end - begin != size:
            raise MergeError(f"Invalid tensor payload size for {name}: {path}")
        intervals.append((begin, end))
    end = 0
    for begin, next_end in sorted(intervals):
        if begin != end:
            raise MergeError(f"Missing or overlapping tensor payload: {path}")
        end = next_end
    if end != info.st_size - length - 8:
        raise MergeError(f"Truncated or extra safetensors payload: {path}")
    return tensors, {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "header_sha256": hashlib.sha256(raw).hexdigest()}


def output_snapshot(directory: Path) -> dict:
    read_object(directory / "config.json")
    validate_tokenizer(directory)
    index_path = directory / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        weight_map = read_object(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map or any(not safe_weight_name(name) for name in weight_map.values()):
            raise MergeError(f"Invalid safetensors shard index: {index_path}")
        names = sorted(set(weight_map.values()))
    else:
        names = ["model.safetensors"]
    if sorted(path.name for path in directory.glob("*.safetensors")) != names:
        raise MergeError(f"Missing or unexpected merged weight shards: {directory}")
    weights, actual_map = {}, {}
    for name in names:
        tensors, weights[name] = safetensor_header(directory / name)
        for tensor in tensors:
            if tensor in actual_map:
                raise MergeError(f"Repeated tensor across merged shards: {tensor}")
            actual_map[tensor] = name
    if weight_map is not None and actual_map != weight_map:
        raise MergeError(f"Merged weight index does not match tensor headers: {directory}")
    return {"hf_hashes": asset_hashes(directory, output=True), "weight_files": weights}


@contextlib.contextmanager
def target_lock(target: Path):
    lock_path = target.with_name(target.name + ".merge.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise MergeError(f"Merge lock must be a regular file: {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MergeError(f"Another merger owns the target lock: {lock_path}") from exc
        yield
    finally:
        os.close(descriptor)
    # Keep the sibling lock inode: unlinking it would permit competing locks.


def retained_model(target: Path, link: str) -> Path:
    """Validate the sole permitted symlink layout and its publication identity."""
    if re.fullmatch(rf"\.{re.escape(target.name)}\.merge-[A-Za-z0-9_-]+/model", link) is None:
        raise MergeError(f"Existing target is an unrelated symlink: {target}")
    model = target.parent / link
    attempt = model.parent
    for path in (attempt, model):
        if path.is_symlink() or not path.is_dir():
            raise MergeError(f"Published symlink requires retained nonsymlink directories: {path}")
    completed = read_object(model / COMPLETE)
    if completed.get("attempt_dir") != str(attempt) or completed.get("target_dir") != str(target):
        raise MergeError(f"Published symlink has stale or unrelated target/attempt provenance: {target}")
    return model


def published_model(target: Path) -> Path:
    if target.is_symlink():
        return retained_model(target, os.readlink(target))
    if not target.is_dir() or not (target / COMPLETE).is_file():
        raise MergeError(f"Existing target is partial or unrelated; it will not be overwritten: {target}")
    return target


def rename_noreplace(staging: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, "Linux renameat2 is unavailable", str(target)) from exc
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def publish_noreplace(staging: Path, target: Path) -> str:
    """Publish complete output atomically; never replace even an empty target."""
    mode = "directory"
    try:
        rename_noreplace(staging, target)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise MergeError(f"Target appeared during merge; refusing to replace it: {target}") from exc
        if exc.errno not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
        link = staging.relative_to(target.parent).as_posix()
        retained_model(target, link)
        # The retained model's directory entry belongs to the attempt. Persist
        # it before a durable public symlink can refer to that backing path.
        fsync_directory(staging.parent)
        # symlink(2) creates one directory entry and fails if ANY entry exists,
        # including an empty directory or a dangling symlink. The validated
        # model stays in its unique attempt directory; no rename fallback can
        # overwrite a competitor between an existence check and publication.
        try:
            os.symlink(link, target, target_is_directory=True)
        except FileExistsError as race:
            raise MergeError(f"Target appeared during merge; refusing to replace it: {target}") from race
        mode = "symlink"
    fsync_directory(target.parent)
    return mode


def run_child(command: list[str], env: dict[str, str], log: Path) -> int:
    with log.open("x") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def merge_checkpoint(source: Path, target: Path, manifest: Path | None, python: str) -> dict:
    source = source.expanduser().resolve(strict=True)
    target = target.expanduser().absolute()
    target = target.parent.resolve() / target.name
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise MergeError("Source and target must be separate, non-nested directories")
    interpreter = shutil.which(python)
    if interpreter is None:
        raise MergeError(f"Python interpreter not found: {python}")
    expected = expected_checkpoint(manifest, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target_lock(target):
        before = source_snapshot(source)
        verify_expected(before, expected)
        provenance = {"format_version": 1, "merge_dtype": "preserve", "source": before, **merger_fingerprint(MERGER), "python": str(Path(interpreter).absolute())}
        if target.exists() or target.is_symlink():
            model = published_model(target)
            completed = read_object(model / COMPLETE)
            if not matching_provenance(completed.get("provenance"), provenance) or completed.get("output") != output_snapshot(model):
                raise MergeError(f"Existing target has stale provenance or changed output files: {target}")
            if source_snapshot(source) != before:
                raise MergeError("Source changed while verifying the existing target")
            if published_model(target) != model:
                raise MergeError("Published target changed while verifying the existing output")
            return {"status": "reused", "target_dir": str(target)}

        attempt = Path(tempfile.mkdtemp(prefix=f".{target.name}.merge-", dir=target.parent))
        staging = attempt / "model"
        staging.mkdir()
        command = [interpreter, str(MERGER), "--actor-dir", str(source / "actor"), "--target-dir", str(staging), "--hf-config", str(source / "actor" / "huggingface"), "--merge-dtype", "preserve"]
        write_json(attempt / "source_before.json", before)
        write_json(attempt / "command.json", {"command": command, "cpu_only": True})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", MERGE_DTYPE="preserve", PYTHONUNBUFFERED="1")
        try:
            returncode = run_child(command, env, attempt / "merge.log")
            after = source_snapshot(source)
            write_json(attempt / "source_after.json", after)
            if returncode:
                raise MergeError(f"Merger exited with status {returncode}")
            if after != before:
                raise MergeError("Source changed during the merge")
            if sha256(MERGER) != provenance["merger_sha256"]:
                raise MergeError("Merger source changed during the merge")
            # Reject staged symlink parents before copying any protected asset.
            files_under(staging)
            hf = source / "actor" / "huggingface"
            for name, digest in before["hf_hashes"].items():
                if name == "config.json":
                    continue
                destination = staging / name
                if destination.is_symlink() or (destination.exists() and not destination.is_file()):
                    raise MergeError(f"Unexpected staged HF asset: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(hf / name, destination)
                if sha256(destination) != digest:
                    raise MergeError(f"Source asset changed while copying: {name}")
            output = output_snapshot(staging)
            if source_snapshot(source) != before:
                raise MergeError("Source changed before publishing")
            for path in files_under(staging):
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            write_json(
                staging / COMPLETE, {"provenance": provenance, "output": output, "completed_utc": datetime.now(timezone.utc).isoformat(), "attempt_dir": str(attempt), "target_dir": str(target)}
            )
            fsync_directory(staging)
            publication_mode = publish_noreplace(staging, target)
        except BaseException as exc:
            if not (attempt / "source_after.json").exists():
                try:
                    after_failure = source_snapshot(source)
                except (MergeError, OSError, ValueError) as snapshot_error:
                    after_failure = {"snapshot_error": str(snapshot_error)}
                with contextlib.suppress(OSError):
                    write_json(attempt / "source_after.json", after_failure)
            with contextlib.suppress(OSError):
                write_json(attempt / "failure.json", {"error": str(exc), "staging_retained": str(attempt)})
            print(f"[merge-eval] Failed attempt retained at {attempt}; log: {attempt / 'merge.log'}", file=sys.stderr)
            raise
        return {"status": "merged", "target_dir": str(target), "attempt_dir": str(attempt), "log": str(attempt / "merge.log"), "publication_mode": publication_mode}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="Expected single-checkpoint entry or campaign inventory JSON")
    parser.add_argument("--python", required=True, help="Existing interpreter containing the merger's torch/VeRL dependencies")
    args = parser.parse_args(argv)
    try:
        result = merge_checkpoint(args.source_dir, args.target_dir, args.manifest, args.python)
    except (MergeError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[merge-eval] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
