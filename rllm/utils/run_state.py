"""Small, local run record and checkpoint checks for the maintained launchers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_checkpoint(path: Path) -> None:
    """Check full FSDP continuation files without loading trusted pickle data."""
    if not path.is_dir() or not (path / "data.pt").is_file() or (path / "data.pt").stat().st_size == 0:
        raise ValueError(f"Incomplete checkpoint (missing directory or data.pt): {path}")
    actor = path / "actor"
    models = list(actor.glob("model_world_size_*_rank_*.pt"))
    if not models:
        raise ValueError(f"No resumable actor shards in {actor}; merged weights are not training state")
    matches = [re.fullmatch(r"model_world_size_(\d+)_rank_\d+\.pt", p.name) for p in models]
    if any(match is None for match in matches):
        raise ValueError(f"Malformed actor shard filename in {actor}")
    sizes = {int(match[1]) for match in matches}
    if len(sizes) != 1:
        raise ValueError(f"Mixed actor world sizes in {actor}")
    world_size = sizes.pop()
    if world_size < 1:
        raise ValueError(f"Invalid actor world size in {actor}")
    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            shard = actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
            if not shard.is_file() or shard.stat().st_size == 0:
                raise ValueError(f"Incomplete checkpoint: missing or empty {shard}")
    if not (path / "ace_state.json").is_file():
        print(f"[run] No ACE sidecar in {path}; GRPO needs none; historical ACE state cannot be reconstructed.", flush=True)


def checkpoint_preflight(checkpoint_dir, resume_mode="auto", resume_from_path=None) -> Path | None:
    root = Path(checkpoint_dir).expanduser().absolute()
    marker = root / "latest_checkpointed_iteration.txt"
    artifacts = root.exists() and any(root.iterdir())
    if resume_mode == "disable":
        if artifacts:
            raise ValueError(f"Refusing a fresh run over existing checkpoint artifacts: {root}")
        return None
    if resume_mode == "resume_path":
        if not resume_from_path:
            raise ValueError("resume_path requires trainer.resume_from_path")
        selected = Path(resume_from_path).expanduser().absolute()
    elif resume_mode == "auto":
        if not marker.is_file():
            if artifacts:
                raise ValueError(f"Checkpoint artifacts exist but {marker} is missing. Repair the marker before resuming.")
            return None
        step = marker.read_text().strip()
        if not step.isdecimal():
            raise ValueError(f"Invalid checkpoint marker: {marker}")
        selected = root / f"global_step_{int(step)}"
    else:
        raise ValueError(f"Unknown resume mode: {resume_mode}")
    if not re.fullmatch(r"global_step_\d+", selected.name):
        raise ValueError(f"Expected a global_step_<N> checkpoint directory: {selected}")
    validate_checkpoint(selected)
    print(f"[run] Resume checkpoint: {selected}", flush=True)
    return selected


def _normalize_training_identity(identity: dict) -> dict:
    """Exclude run labels from new and previously recorded scientific identities."""
    identity = copy.deepcopy(identity)
    for path in (("sampling", "trace"), ("rllm", "env", "env_args")):
        section = identity
        for key in path:
            section = section.get(key) if isinstance(section, dict) else None
        if isinstance(section, dict):
            section.pop("experiment_name", None)
    for section in ("optimizer", "critic"):
        optim = identity.get(section, {}).get("optim") if isinstance(identity.get(section), dict) else None
        # The step budget only shapes a schedule that decays or derives its warmup from it;
        # a constant rate with no ratio warmup trains identically, so it may be extended.
        if (
            isinstance(optim, dict)
            and optim.get("lr_scheduler_type", "constant") == "constant"
            and optim.get("warmup_style") in (None, "constant")
            and not optim.get("lr_warmup_steps_ratio")
        ):
            optim.pop("total_training_steps", None)
    agent = identity.get("rllm", {}).get("agent")
    if isinstance(agent, dict):
        agent.setdefault("preserve_terminal_feedback", False)
    ace = identity.get("rllm", {}).get("ace")
    if isinstance(ace, dict):
        # Old runs predate auxiliary admission gating. An explicit legacy value
        # is equivalent to that missing field; the new objective is not.
        ace.setdefault("auxiliary_update_gate", "none")
        ace.setdefault("handoff_protocol", "legacy")
        ace["handoff_protocol"] = str(ace["handoff_protocol"]).strip().lower()
        ace.setdefault("log_role_credit", False)
        ace.setdefault("controller_seed", None)
        seed = ace["controller_seed"]
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 32)):
            raise ValueError("ace.controller_seed must be null or an integer in [0, 2**32 - 1]")
        if ace.get("training_mode") in ("delayed_joint", "joint_ppo"):
            ace["auxiliary_update_gate"] = "none"
    return identity


def training_identity(config: dict) -> dict:
    """Compare scientific settings, excluding paths used only for placement/logging."""

    def clean(value):
        if isinstance(value, dict):
            placement = ("log_dir", "artifact", "cache", "output_dir", "output_root", "tmpdir", "temp_dir", "_port", "_url")
            return {k: clean(v) for k, v in value.items() if k not in {"port", "url", "api_base"} and not any(x in k for x in placement)}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    def dataset_identity(value):
        if isinstance(value, list):
            return [dataset_identity(item) for item in value]
        if isinstance(value, str) and Path(value).is_file():
            digest = hashlib.sha256()
            with Path(value).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            # The registry's job-local path changes; the actual dataset must not.
            return {"sha256": digest.hexdigest(), "bytes": Path(value).stat().st_size}
        return value

    actor = config.get("actor_rollout_ref", {})
    data = clean(config.get("data", {}))
    for key in ("train_files", "val_files"):
        if key in data:
            data[key] = dataset_identity(data[key])
    rllm = clean(config.get("rllm", {}))
    ace = rllm.get("ace", {})
    if ace.get("prompts_dir"):
        prompts = Path(ace["prompts_dir"])
        ace["prompts_dir"] = {name: dataset_identity(str(prompts / name)) for name in ("base_system_prompt.txt", "reflector_prompt.txt", "curator_prompt.txt")}
        if str(ace.get("handoff_protocol", "legacy")).strip().lower() == "feedback_v1":
            ace["prompts_dir"]["curator_prompt.txt"] = dataset_identity(str(prompts / "curator_prompt_handoff_v1.txt"))
            handoff_reflector = prompts / "reflector_prompt_handoff_v1.txt"
            if handoff_reflector.is_file():
                ace["prompts_dir"]["reflector_prompt.txt"] = dataset_identity(str(handoff_reflector))
    if ace.get("initial_playbook"):
        ace["initial_playbook"] = dataset_identity(ace["initial_playbook"])
    return _normalize_training_identity(
        {
            "model": clean(actor.get("model", {})),
            "sampling": clean(actor.get("rollout", {})),
            "data": data,
            "algorithm": clean(config.get("algorithm", {})),
            "optimizer": clean(actor.get("actor", {})),
            "critic": clean(config.get("critic", {})),
            "reward_model": clean(config.get("reward_model", {})),
            "update_schedule": {key: config.get("trainer", {}).get(key) for key in ("critic_warmup", "balance_batch")},
            "rllm": rllm,
            "world_size": [config.get("trainer", {}).get(key) for key in ("nnodes", "n_gpus_per_node")],
        }
    )


def prepare_training_run(config) -> None:
    """Validate resolved Hydra overrides before the trainer loads any checkpoint."""
    root = os.environ.get("RLLM_RUN_DIR")
    if not root:
        return
    trainer = config.trainer
    expected = Path(root).resolve() / "checkpoints"
    if Path(trainer.default_local_dir).resolve() != expected:
        raise ValueError(f"trainer.default_local_dir must be {expected}; select a different experiment with RUN_DIR")
    if trainer.get("del_local_ckpt_after_load", False):
        raise ValueError("The maintained launchers preserve checkpoints; del_local_ckpt_after_load must be false")
    selected = checkpoint_preflight(expected, trainer.get("resume_mode", "auto"), trainer.get("resume_from_path"))
    os.environ["RLLM_RESUME_STEP"] = selected.name.removeprefix("global_step_") if selected else "0"
    from omegaconf import OmegaConf

    RunState(root, trainer.project_name, trainer.experiment_name, OmegaConf.to_container(config, resolve=True))


def finish_data_exhausted(trainer) -> None:
    """Keep the last update when the epochs run out before the step target.

    A skipped batch consumes data without an optimizer step, so total_epochs
    times the batches per epoch can fall short of trainer.total_training_steps.
    Save the last completed step if save_freq skipped it, then fail when the
    step target was explicit, so the run cannot look complete.
    """
    last = trainer.global_steps - 1
    save_freq = trainer.config.trainer.save_freq
    saved = (save_freq > 0 and last % save_freq == 0) or (Path(trainer.config.trainer.default_local_dir) / f"global_step_{last}").is_dir()
    if last > 0 and not saved:
        trainer.global_steps = last
        try:
            trainer._save_checkpoint()
        finally:
            trainer.global_steps = last + 1
    target = trainer.config.trainer.get("total_training_steps")
    if target is not None and last + 1 < int(target):
        raise RuntimeError(
            f"Training data ended after step {last}, before trainer.total_training_steps={target}: "
            f"total_epochs={trainer.config.trainer.total_epochs} supplies too few batches once skipped batches "
            "are counted. Raise trainer.total_epochs (TRAINER_TOTAL_EPOCHS) and resume."
        )


class RunState:
    """Persistent identity; the launcher's flock owns the experiment writer lock."""

    def __init__(self, root: str, project: str, name: str, config: dict | None = None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "run.json"
        self.record = json.loads(self.path.read_text()) if self.path.exists() else {"schema_version": 1}
        self.record.setdefault("name", name)
        source = os.environ.get("RLLM_DIR")
        if source:
            revision = subprocess.run(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            self.record["source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
        self.record["preset"] = os.environ.get("PRESET", self.record.get("preset"))
        self.record["model_variant"] = os.environ.get("MODEL_VARIANT", self.record.get("model_variant"))
        if config is not None:
            identity = training_identity(config)
            previous = self.record.get("training_identity")
            if previous is not None:
                previous = _normalize_training_identity(previous)
            if previous is not None and previous != identity:
                changed = [key for key in identity if identity[key] != previous.get(key)]
                raise ValueError(f"Experiment settings changed ({', '.join(changed)}) in {self.root}; use a distinct RUN_DIR for an ablation")
            self.record["training_identity"] = identity
        self.project = project
        self.resume_step = int(os.environ.get("RLLM_RESUME_STEP", "0"))
        self.index = int(self.record.get("logging_index", -1))
        journal = self.root / "metrics.jsonl"
        if journal.exists() and journal.stat().st_size:
            original = journal.read_bytes()
            rows = original.splitlines(keepends=True)
            try:
                last = json.loads(rows[-1])
            except (ValueError, UnicodeDecodeError):
                # Preserve the complete original before recovering an interrupted
                # final append. Earlier malformed rows remain errors for readers.
                archive = self.root / "archive" / "metrics-recovery"
                archive.mkdir(parents=True, exist_ok=True)
                backup = archive / (hashlib.sha256(original).hexdigest() + ".jsonl")
                if backup.exists():
                    raise ValueError(f"Journal recovery already exists: {backup}") from None
                journal.rename(backup)
                journal.write_bytes(b"".join(rows[:-1]))
                last = json.loads(rows[-2]) if len(rows) > 1 else {}
            else:
                if not original.endswith(b"\n"):
                    with journal.open("ab") as handle:
                        handle.write(b"\n")
            self.index = max(self.index, int(last.get("logging/index", -1)))
            self.record["logging_index"] = self.index
        atomic_json(self.path, self.record)

    def wandb_options(self) -> dict:
        # An unset WANDB_ENTITY resumes the recorded entity, or lets a new run use
        # the logged-in account's default entity, which wandb_started records.
        entity = os.environ.get("WANDB_ENTITY") or None
        previous = self.record.get("wandb")
        requested = os.environ.get("WANDB_RUN_ID")
        if previous:
            entity = entity or previous["entity"]
            if previous["entity"] not in (None, entity) or previous["project"] != self.project or (requested and requested != previous["id"]):
                raise ValueError(f"W&B identity does not match {self.path}; reconcile the recorded cloud run before resuming")
        else:
            if self.resume_step > 0 and not self.record.get("wandb_migration", {}).get("start_new_team_segment"):
                raise ValueError(f"Existing checkpoint has no recorded W&B identity in {self.path}; adopt a verified run before resuming")
            previous = {"entity": entity, "project": self.project, "id": requested or uuid.uuid4().hex[:8], "initialized": False}
            self.record["wandb"] = previous
            atomic_json(self.path, self.record)
        return {"entity": entity, "project": self.project, "id": previous["id"], "resume": "must" if previous.get("initialized") else "allow", "dir": str(self.root)}

    def wandb_started(self, run) -> None:
        mode = getattr(getattr(run, "settings", None), "mode", os.environ.get("WANDB_MODE", "online"))
        if mode == "disabled":
            return
        expected = self.record["wandb"]
        if expected.get("entity") is None:
            expected["entity"] = run.entity
        for key in ("entity", "project", "id"):
            if getattr(run, key) != expected[key]:
                raise ValueError(f"W&B returned an unexpected {key}; expected {expected[key]!r}")
        # Offline files do not establish that this ID exists on the server.
        if mode != "offline":
            expected["initialized"] = True
        atomic_json(self.path, self.record)

    def log(self, data: dict, step: int) -> dict:
        self.index += 1
        scalar = {key: value.item() if hasattr(value, "item") else value for key, value in data.items() if isinstance(value, (int, float, bool)) or (hasattr(value, "ndim") and value.ndim == 0)}
        scalar.update({"training/global_step": step, "training/resume_from_step": self.resume_step, "logging/index": self.index})
        with (self.root / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(scalar) + "\n")
        self.record["logging_index"] = self.index
        atomic_json(self.path, self.record)
        return {**data, "training/global_step": step, "training/resume_from_step": self.resume_step, "logging/index": self.index}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume-mode", default="auto")
    parser.add_argument("--resume-from-path")
    args = parser.parse_args()
    try:
        checkpoint_preflight(args.checkpoint_dir, args.resume_mode, args.resume_from_path)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
