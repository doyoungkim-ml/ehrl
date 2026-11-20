"""Real subprocess and filesystem coverage for CPU merge orchestration only."""

import errno
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("merge_eval_checkpoint", ROOT / "scripts/merge_eval_checkpoint.py")
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "training-run" / "global_step_5"
    hf = source / "actor" / "huggingface"
    hf.mkdir(parents=True)
    for rank in range(2):
        (source / "actor" / f"model_world_size_2_rank_{rank}.pt").write_bytes(f"original rank {rank}".encode())
    (hf / "config.json").write_text('{"dtype":"float32","model_type":"qwen3"}\n')
    (hf / "tokenizer_config.json").write_text('{"chat_template":"original prompt"}\n')
    (hf / "tokenizer.json").write_text('{"model":{"type":"BPE"}}\n')
    (hf / "generation_config.json").write_text('{"temperature":0.6,"top_p":0.95}\n')
    (hf / "chat_template.jinja").write_text("original {{ messages }} prompt\n")
    (hf / "custom").mkdir()
    (hf / "custom" / "asset.json").write_text('{"original":true}\n')
    original = merge.source_snapshot(source)
    expected = {key: original[key] for key in ("path", "step", "actor_files")}
    expected["hf_hashes"] = {name: original["hf_hashes"][name] for name in ("config.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")}
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps({"runs": {"ace_rl": {"checkpoints": [expected]}}}))
    child = tmp_path / "fake_merge.py"
    child.write_text(
        r"""
import argparse, json, os, struct
from pathlib import Path
parser = argparse.ArgumentParser()
for option in ("actor-dir", "target-dir", "hf-config", "merge-dtype"):
    parser.add_argument("--" + option, required=True)
args = parser.parse_args()
assert args.merge_dtype == "preserve"
assert os.environ["MERGE_DTYPE"] == "preserve"
for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    assert os.environ[name] == ""
output = Path(args.target_dir)
(output.parent / "child.json").write_text(json.dumps(vars(args)))
(output / "config.json").write_text('{"dtype":"float32"}')
(output / "tokenizer_config.json").write_text('{"normalized":"by fake merger"}')
(output / "generation_config.json").write_text('{"temperature":0.99}')
header = json.dumps({"weight": {"dtype":"F32","shape":[2],"data_offsets":[0,8]}}).encode()
(output / "model.safetensors").write_bytes(struct.pack("<Q",len(header)) + header + struct.pack("<ff",1.125,2.5))
print("merge child evidence", flush=True)
mode = os.environ.get("FAKE_MERGE_MODE", "success")
if mode == "fail":
    raise SystemExit(7)
if mode == "source_changed":
    path = Path(args.actor_dir) / "model_world_size_2_rank_0.pt"
    path.write_bytes(path.read_bytes() + b"changed during merge")
if mode == "asset_changed":
    (Path(args.hf_config) / "chat_template.jinja").write_text("changed prompt")
if mode == "partial":
    (output / "model.safetensors").write_bytes(b"incomplete")
if mode == "missing_index_shard":
    (output / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{"weight":"missing.safetensors"}}))
if mode == "target_race":
    Path(os.environ["FAKE_FINAL_TARGET"]).mkdir()
if mode == "symlink_output":
    (output / "custom").symlink_to(os.environ["FAKE_VICTIM"], target_is_directory=True)
"""
    )
    monkeypatch.setattr(merge, "MERGER", child)
    monkeypatch.setenv("MERGE_DTYPE", "bf16")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    target = tmp_path / "exports" / "global_step_5"
    return source, target, manifest, original


def run_merge(checkpoint):
    source, target, manifest, _ = checkpoint
    return merge.merge_checkpoint(source, target, manifest, sys.executable)


def test_preserves_dtype_assets_source_and_reuses_verified_target(checkpoint):
    source, target, _, original = checkpoint
    result = run_merge(checkpoint)
    assert result["status"] == "merged"
    attempt = Path(result["attempt_dir"])
    command = json.loads((attempt / "command.json").read_text())["command"]
    assert command[0] == sys.executable
    assert command[-2:] == ["--merge-dtype", "preserve"]
    assert command[command.index("--actor-dir") + 1] == str(source / "actor")
    assert command[command.index("--hf-config") + 1] == str(source / "actor" / "huggingface")
    assert merge.source_snapshot(source) == original
    for name, digest in original["hf_hashes"].items():
        if name != "config.json":
            assert merge.sha256(target / name) == digest
    assert json.loads((target / "config.json").read_text())["dtype"] == "float32"
    assert (target / "model.safetensors").read_bytes()[-8:] == struct.pack("<ff", 1.125, 2.5)
    assert (attempt / "merge.log").read_text() == "merge child evidence\n"
    assert json.loads((attempt / "source_before.json").read_text()) == original
    assert json.loads((attempt / "source_after.json").read_text()) == original
    assert (target.parent / (target.name + ".merge.lock")).is_file()
    assert not (attempt / "model").exists()
    completion = json.loads((target / merge.COMPLETE).read_text())
    assert completion["provenance"]["source"]["experiment_dir"] == str(source.parent)
    assert completion["provenance"]["merge_dtype"] == "preserve"
    assert run_merge(checkpoint) == {"status": "reused", "target_dir": str(target)}
    assert len(list(target.parent.glob(".*.merge-*"))) == 1


@pytest.mark.parametrize(
    "mode,message",
    (("fail", "status 7"), ("source_changed", "Source changed"), ("asset_changed", "Source changed"), ("partial", "header"), ("missing_index_shard", "weight shards")),
)
def test_failed_or_changed_merge_never_publishes_and_retains_attempt(checkpoint, monkeypatch, mode, message):
    monkeypatch.setenv("FAKE_MERGE_MODE", mode)
    with pytest.raises(merge.MergeError, match=message):
        run_merge(checkpoint)
    _, target, _, _ = checkpoint
    assert not target.exists()
    attempts = list(target.parent.glob(".*.merge-*"))
    assert len(attempts) == 1
    attempt = attempts[0]
    assert (attempt / "model" / "model.safetensors").is_file()
    assert (attempt / "merge.log").read_text() == "merge child evidence\n"
    assert (attempt / "source_before.json").is_file()
    assert (attempt / "source_after.json").is_file()
    assert message.lower() in json.loads((attempt / "failure.json").read_text())["error"].lower()


@pytest.mark.parametrize("change", ("shard", "asset", "unknown_source"))
def test_expected_manifest_rejects_stale_or_unselected_source_before_child(checkpoint, change):
    source, target, manifest, _ = checkpoint
    if change == "shard":
        shard = source / "actor" / merge.shard_names(2)[0]
        shard.write_bytes(shard.read_bytes() + b"new checkpoint bytes")
    elif change == "asset":
        (source / "actor" / "huggingface" / "chat_template.jinja").write_text("different prompt")
    else:
        payload = json.loads(manifest.read_text())
        payload["runs"]["ace_rl"]["checkpoints"][0]["path"] = str(source.parent / "global_step_10")
        manifest.write_text(json.dumps(payload))
    with pytest.raises(merge.MergeError, match="manifest|exact source"):
        run_merge(checkpoint)
    assert not target.exists()
    assert not list(target.parent.glob(".*.merge-*"))


@pytest.mark.parametrize("kind", ("empty", "partial", "file", "symlink"))
def test_existing_unowned_target_is_never_clobbered(checkpoint, kind, tmp_path):
    _, target, _, _ = checkpoint
    target.parent.mkdir()
    if kind == "file":
        target.write_text("unrelated file")
    elif kind == "symlink":
        victim = tmp_path / "unrelated"
        victim.mkdir()
        (victim / "keep.txt").write_text("keep")
        target.symlink_to(victim, target_is_directory=True)
    else:
        target.mkdir()
        if kind == "partial":
            (target / "config.json").write_text("partial merge evidence")
    before = target.lstat()
    with pytest.raises(merge.MergeError, match="partial|unrelated|symlink"):
        run_merge(checkpoint)
    assert target.lstat().st_ino == before.st_ino
    assert not list(target.parent.glob(".*.merge-*"))
    if kind == "partial":
        assert (target / "config.json").read_text() == "partial merge evidence"
    elif kind == "file":
        assert target.read_text() == "unrelated file"
    elif kind == "symlink":
        assert (target / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("change", ("asset", "weight", "provenance", "source"))
def test_success_marker_does_not_allow_stale_target_reuse(checkpoint, change):
    source, target, manifest, _ = checkpoint
    run_merge(checkpoint)
    if change == "asset":
        (target / "chat_template.jinja").write_text("new output prompt")
    elif change == "weight":
        weight = target / "model.safetensors"
        weight.write_bytes(weight.read_bytes()[:-1])
    elif change == "provenance":
        marker = target / merge.COMPLETE
        record = json.loads(marker.read_text())
        record["provenance"]["source"]["experiment_dir"] = "/unrelated/run"
        marker.write_text(json.dumps(record))
    else:
        source_asset = source / "actor" / "huggingface" / "tokenizer.json"
        source_asset.write_text('{"changed":true}')
        # This asset is deliberately outside the inventory's four expected
        # hashes: the helper's full snapshot must still prevent stale reuse.
    with pytest.raises(merge.MergeError, match="stale|changed|Truncated"):
        merge.merge_checkpoint(source, target, manifest, sys.executable)
    assert len(list(target.parent.glob(".*.merge-*"))) == 1


def test_publication_refuses_target_created_after_preflight(checkpoint, monkeypatch):
    _, target, _, _ = checkpoint
    monkeypatch.setenv("FAKE_MERGE_MODE", "target_race")
    monkeypatch.setenv("FAKE_FINAL_TARGET", str(target))
    with pytest.raises(merge.MergeError, match="Target appeared"):
        run_merge(checkpoint)
    assert target.is_dir() and list(target.iterdir()) == []
    attempt = next(target.parent.glob(".*.merge-*"))
    assert (attempt / "model" / merge.COMPLETE).is_file()


def test_staged_symlink_cannot_redirect_protected_asset_copy(checkpoint, monkeypatch, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "asset.json").write_text("unrelated original")
    monkeypatch.setenv("FAKE_MERGE_MODE", "symlink_output")
    monkeypatch.setenv("FAKE_VICTIM", str(victim))
    with pytest.raises(merge.MergeError, match="Symlink directory"):
        run_merge(checkpoint)
    assert (victim / "asset.json").read_text() == "unrelated original"
    assert not checkpoint[1].exists()


def test_external_lock_refuses_another_merger_before_child(checkpoint):
    _, target, _, _ = checkpoint
    target.parent.mkdir()
    with merge.target_lock(target):
        with pytest.raises(merge.MergeError, match="Another merger owns"):
            run_merge(checkpoint)
    assert not target.exists()
    assert not list(target.parent.glob(".*.merge-*"))


def test_single_checkpoint_manifest_format_is_supported(checkpoint):
    _, _, manifest, _ = checkpoint
    payload = json.loads(manifest.read_text())["runs"]["ace_rl"]["checkpoints"][0]
    manifest.write_text(json.dumps(payload))
    assert run_merge(checkpoint)["status"] == "merged"


def test_cli_returns_nonzero_without_touching_partial_target(checkpoint, monkeypatch):
    source, target, manifest, _ = checkpoint
    monkeypatch.setenv("FAKE_MERGE_MODE", "fail")
    assert merge.main(["--source-dir", str(source), "--target-dir", str(target), "--manifest", str(manifest), "--python", sys.executable]) == 1
    assert not target.exists()


@pytest.fixture
def unsupported_rename(monkeypatch):
    def fail(_staging, target):
        raise OSError(errno.EINVAL, "Filesystem does not support RENAME_NOREPLACE", str(target))

    monkeypatch.setattr(merge, "rename_noreplace", fail)


@pytest.mark.parametrize("error", sorted({errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}))
def test_unsupported_rename_publishes_complete_retained_model_and_reuses(checkpoint, monkeypatch, error):
    def fail(_staging, target):
        raise OSError(error, "unsupported", str(target))

    monkeypatch.setattr(merge, "rename_noreplace", fail)
    _, target, _, original = checkpoint
    result = run_merge(checkpoint)
    attempt = Path(result["attempt_dir"])
    assert result["publication_mode"] == "symlink"
    assert target.is_symlink()
    assert os.readlink(target) == f"{attempt.name}/model"
    assert target.resolve() == attempt / "model"
    assert merge.source_snapshot(checkpoint[0]) == original
    record = merge.read_object(target / merge.COMPLETE)
    assert record["target_dir"] == str(target)
    assert record["attempt_dir"] == str(attempt)
    assert record["output"] == merge.output_snapshot(attempt / "model")
    assert (attempt / "merge.log").read_text() == "merge child evidence\n"
    assert run_merge(checkpoint) == {"status": "reused", "target_dir": str(target)}
    assert len(list(target.parent.glob(".*.merge-*"))) == 1


def test_missing_libc_renameat2_uses_no_clobber_symlink(checkpoint, monkeypatch):
    monkeypatch.setattr(merge.ctypes, "CDLL", lambda *args, **kwargs: object())
    result = run_merge(checkpoint)
    assert result["publication_mode"] == "symlink"
    assert checkpoint[1].is_symlink()


@pytest.mark.parametrize("kind", ("empty_directory", "file", "dangling_symlink"))
def test_fallback_refuses_every_target_created_after_preflight(checkpoint, monkeypatch, kind):
    _, target, _, _ = checkpoint

    def competing_target(_staging, destination):
        if kind == "empty_directory":
            destination.mkdir()
        elif kind == "file":
            destination.write_text("competitor output")
        else:
            destination.symlink_to("missing-competitor-output")
        raise OSError(errno.EINVAL, "unsupported", str(destination))

    monkeypatch.setattr(merge, "rename_noreplace", competing_target)
    with pytest.raises(merge.MergeError, match="Target appeared"):
        run_merge(checkpoint)
    if kind == "empty_directory":
        assert target.is_dir() and list(target.iterdir()) == []
    elif kind == "file":
        assert target.read_text() == "competitor output"
    else:
        assert target.is_symlink() and os.readlink(target) == "missing-competitor-output"
    attempt = next(target.parent.glob(".*.merge-*"))
    assert (attempt / "model" / merge.COMPLETE).is_file()
    assert (attempt / "failure.json").is_file()


def test_permission_failure_does_not_enable_publication_fallback(checkpoint, monkeypatch):
    def denied(_staging, target):
        raise PermissionError(errno.EACCES, "publication denied", str(target))

    monkeypatch.setattr(merge, "rename_noreplace", denied)
    with pytest.raises(PermissionError, match="publication denied"):
        run_merge(checkpoint)
    assert not checkpoint[1].exists() and not checkpoint[1].is_symlink()
    attempt = next(checkpoint[1].parent.glob(".*.merge-*"))
    assert (attempt / "model" / merge.COMPLETE).is_file()


@pytest.mark.parametrize("change", ("asset", "weight", "provenance", "source", "target_identity", "attempt_identity"))
def test_symlink_publication_requires_unchanged_output_and_exact_provenance(checkpoint, unsupported_rename, change):
    source, target, _, _ = checkpoint
    run_merge(checkpoint)
    if change == "asset":
        (target / "chat_template.jinja").write_text("modified prompt")
    elif change == "weight":
        weight = target / "model.safetensors"
        weight.write_bytes(weight.read_bytes()[:-1])
    elif change == "source":
        (source / "actor" / "huggingface" / "tokenizer.json").write_text('{"changed":true}')
    else:
        marker = target / merge.COMPLETE
        record = json.loads(marker.read_text())
        if change == "provenance":
            record["provenance"]["source"]["experiment_dir"] = "/unrelated/run"
        elif change == "target_identity":
            record["target_dir"] = str(target.with_name("global_step_999"))
        else:
            record["attempt_dir"] = str(target.parent / ".global_step_5.merge-unrelated")
        marker.write_text(json.dumps(record))
    with pytest.raises(merge.MergeError, match="stale|changed|Truncated|unrelated"):
        run_merge(checkpoint)
    assert target.is_symlink()
    assert len(list(target.parent.glob(".*.merge-*"))) == 1


@pytest.mark.parametrize("redirect", ("absolute", "traversal", "attempt_symlink", "model_symlink"))
def test_symlink_reuse_rejects_redirected_backing_paths(checkpoint, unsupported_rename, redirect):
    _, target, _, _ = checkpoint
    result = run_merge(checkpoint)
    attempt = Path(result["attempt_dir"])
    if redirect in {"absolute", "traversal"}:
        target.unlink()
        link = str(attempt / "model") if redirect == "absolute" else f"../{target.parent.name}/{attempt.name}/model"
        target.symlink_to(link, target_is_directory=True)
    elif redirect == "attempt_symlink":
        moved = attempt.with_name(attempt.name + "-moved")
        attempt.rename(moved)
        attempt.symlink_to(moved, target_is_directory=True)
    else:
        moved = attempt / "moved-model"
        (attempt / "model").rename(moved)
        (attempt / "model").symlink_to(moved, target_is_directory=True)
    with pytest.raises(merge.MergeError, match="symlink"):
        run_merge(checkpoint)
    assert target.is_symlink()


def test_symlink_publication_syncs_backing_entry_before_public_name(checkpoint, unsupported_rename, monkeypatch):
    events = []
    sync = merge.fsync_directory
    symlink = merge.os.symlink

    def record_sync(directory):
        sync(directory)
        events.append(("sync", directory))

    def record_symlink(link, target, **kwargs):
        events.append(("publish", target))
        symlink(link, target, **kwargs)

    monkeypatch.setattr(merge, "fsync_directory", record_sync)
    monkeypatch.setattr(merge.os, "symlink", record_symlink)
    result = run_merge(checkpoint)
    attempt = Path(result["attempt_dir"])
    target = checkpoint[1]
    assert events.index(("sync", attempt / "model")) < events.index(("sync", attempt))
    assert events.index(("sync", attempt)) < events.index(("publish", target))
    assert events.index(("publish", target)) < events.index(("sync", target.parent))


def test_run_layout_selected_checkpoint_can_be_merged_without_nesting(checkpoint, tmp_path):
    runtime_spec = importlib.util.spec_from_file_location("evaluation_runtime", ROOT / "scripts/evaluation_runtime.py")
    runtime = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(runtime)
    source, _, _, original = checkpoint
    run_dir = tmp_path / "canonical run"
    (run_dir / "checkpoints").mkdir(parents=True)
    canonical_source = run_dir / "checkpoints/global_step_5"
    source.rename(canonical_source)
    layout = runtime.evaluation_paths("mini_swe_bench", "ace", {"RUN_DIR": str(run_dir), "STEP": "5"}, ROOT)
    item = layout["evaluations"][0]
    result = merge.merge_checkpoint(Path(item["checkpoint_source"]), Path(item["model"]), None, sys.executable)
    assert result["status"] == "merged"
    assert Path(item["model"]).parent == run_dir / "checkpoints_merged"
    assert merge.source_snapshot(canonical_source)["actor_files"] == original["actor_files"]
    assert merge.source_snapshot(canonical_source)["hf_hashes"] == original["hf_hashes"]


@pytest.mark.parametrize("recorded_ast", [None, "0" * 64])
def test_same_merger_hash_reuses_across_optional_ast_fingerprints(checkpoint, recorded_ast):
    run_merge(checkpoint)
    marker = checkpoint[1] / merge.COMPLETE
    saved = json.loads(marker.read_text())
    if recorded_ast is None:
        saved["provenance"].pop("merger_ast_sha256")
    else:
        saved["provenance"]["merger_ast_sha256"] = recorded_ast
    marker.write_text(json.dumps(saved))
    before = marker.read_bytes()
    assert run_merge(checkpoint)["status"] == "reused"
    assert marker.read_bytes() == before


def test_formatting_only_merger_change_reuses_recorded_ast_without_rewriting_provenance(checkpoint):
    run_merge(checkpoint)
    marker = checkpoint[1] / merge.COMPLETE
    before = marker.read_bytes()
    original = json.loads(before)["provenance"]
    merge.MERGER.write_text("# Reformat the same merger.\n\n" + merge.MERGER.read_text().replace("import argparse, json, os, struct", "import argparse,  json,  os,  struct"))
    current = merge.merger_fingerprint(merge.MERGER)
    assert current["merger_sha256"] != original["merger_sha256"]
    assert current["merger_ast_sha256"] == original["merger_ast_sha256"]
    assert run_merge(checkpoint)["status"] == "reused"
    assert marker.read_bytes() == before


@pytest.mark.parametrize("change", ("no_ast", "wrong_ast", "semantic_merger", "source", "output", "python", "unknown_field"))
def test_ast_reuse_cannot_bypass_source_output_or_semantic_identity(checkpoint, change):
    run_merge(checkpoint)
    source, target, manifest, _ = checkpoint
    marker = target / merge.COMPLETE
    saved = json.loads(marker.read_text())
    merge.MERGER.write_text("# Formatting change.\n" + merge.MERGER.read_text())
    if change == "no_ast":
        saved["provenance"].pop("merger_ast_sha256")
    elif change == "wrong_ast":
        saved["provenance"]["merger_ast_sha256"] = "0" * 64
    elif change == "semantic_merger":
        merge.MERGER.write_text(merge.MERGER.read_text() + "\nraise RuntimeError('new merger behavior')\n")
    elif change == "source":
        (source / "actor/huggingface/tokenizer.json").write_text('{"changed": true}')
    elif change == "output":
        (target / "chat_template.jinja").write_text("changed output prompt")
    elif change == "python":
        saved["provenance"]["python"] = "/different/python"
    else:
        saved["provenance"]["unexpected"] = True
    marker.write_text(json.dumps(saved))
    with pytest.raises(merge.MergeError, match="stale|changed"):
        merge.merge_checkpoint(source, target, manifest, sys.executable)
    assert len(list(target.parent.glob(".*.merge-*"))) == 1


def test_shard_sets_accept_any_complete_world_size():
    assert merge.complete_shard_set(["model_world_size_1_rank_0.pt"]) == ["model_world_size_1_rank_0.pt"]
    assert merge.complete_shard_set(["model_world_size_2_rank_1.pt", "model_world_size_2_rank_0.pt"]) == merge.shard_names(2)
    assert merge.complete_shard_set(["model_world_size_2_rank_0.pt"]) is None
    assert merge.complete_shard_set(["model_world_size_2_rank_0.pt", "model_world_size_1_rank_0.pt"]) is None
    assert merge.complete_shard_set([]) is None
