"""Run identity, incomplete-checkpoint rejection, and rollback logging."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_state", ROOT / "rllm/utils/run_state.py")
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


def checkpoint(root, step=5):
    selected = root / f"global_step_{step}"
    (selected / "actor").mkdir(parents=True)
    (selected / "data.pt").write_bytes(b"saved dataloader")
    for rank in range(2):
        for kind in ("model", "optim", "extra_state"):
            (selected / "actor" / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"saved shard")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    return selected


def test_fresh_and_resume(tmp_path):
    root = tmp_path / "checkpoints"
    assert state.checkpoint_preflight(root) is None
    selected = checkpoint(root)
    assert state.checkpoint_preflight(root) == selected
    with pytest.raises(ValueError, match="fresh run"):
        state.checkpoint_preflight(root, "disable")


@pytest.mark.parametrize("damage", ["marker", "missing_target", "invalid_marker", "data", "optimizer", "rng", "model_rank"])
def test_partial_checkpoint_fails_without_mutating(tmp_path, damage):
    selected = checkpoint(tmp_path)
    marker = tmp_path / "latest_checkpointed_iteration.txt"
    if damage == "marker":
        marker.unlink()
    elif damage == "missing_target":
        marker.write_text("10")
    elif damage == "invalid_marker":
        marker.write_text("../5")
    elif damage == "data":
        (selected / "data.pt").unlink()
    else:
        kind = {"optimizer": "optim", "rng": "extra_state", "model_rank": "model"}[damage]
        (selected / "actor" / f"{kind}_world_size_2_rank_1.pt").unlink()
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    with pytest.raises(ValueError):
        state.checkpoint_preflight(tmp_path)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_identity_and_wandb_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "example-team")
    run = state.RunState(str(tmp_path), "appworld", "test", {"algorithm": {"adv_estimator": "grpo"}})
    first = run.wandb_options()
    assert first["entity"] == "example-team" and first["resume"] == "allow" and first["dir"] == str(tmp_path)
    run.wandb_started(SimpleNamespace(**{key: first[key] for key in ("id", "entity", "project")}))
    resumed = state.RunState(str(tmp_path), "appworld", "test", {"algorithm": {"adv_estimator": "grpo"}})
    assert resumed.wandb_options() == {**first, "resume": "must"}
    monkeypatch.setenv("WANDB_ENTITY", "personal")
    with pytest.raises(ValueError, match="identity"):
        resumed.wandb_options()
    with pytest.raises(ValueError, match="settings changed"):
        state.RunState(str(tmp_path), "appworld", "test", {"algorithm": {"adv_estimator": "different"}})


def test_unset_wandb_entity_uses_account_default_then_resumes_recorded_entity(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    run = state.RunState(str(tmp_path), "appworld", "test", {"algorithm": {"adv_estimator": "grpo"}})
    first = run.wandb_options()
    assert first["entity"] is None
    run.wandb_started(SimpleNamespace(id=first["id"], entity="team", project=first["project"]))
    resumed = state.RunState(str(tmp_path), "appworld", "test", {"algorithm": {"adv_estimator": "grpo"}})
    assert resumed.wandb_options()["entity"] == "team"
    monkeypatch.setenv("WANDB_ENTITY", "team")
    assert resumed.wandb_options()["entity"] == "team"
    monkeypatch.setenv("WANDB_ENTITY", "other")
    with pytest.raises(ValueError, match="identity"):
        resumed.wandb_options()


@pytest.mark.parametrize(
    "before,after",
    [
        ({"data": {"train_files": "a.parquet"}}, {"data": {"train_files": "b.parquet"}}),
        ({"data": {"dataloader_num_workers": 0}}, {"data": {"dataloader_num_workers": 2}}),
        ({"actor_rollout_ref": {"rollout": {"max_model_len": 8192}}}, {"actor_rollout_ref": {"rollout": {"max_model_len": 16384}}}),
        ({"actor_rollout_ref": {"model": {"path": "model-a"}}}, {"actor_rollout_ref": {"model": {"path": "model-b"}}}),
        ({"actor_rollout_ref": {"rollout": {"temperature": 0.7}}}, {"actor_rollout_ref": {"rollout": {"temperature": 0.8}}}),
        ({"actor_rollout_ref": {"actor": {"optim": {"lr": 1e-6}}}}, {"actor_rollout_ref": {"actor": {"optim": {"lr": 2e-6}}}}),
    ],
)
def test_identity_rejects_changed_scientific_inputs(tmp_path, before, after):
    state.RunState(str(tmp_path), "appworld", "test", before)
    with pytest.raises(ValueError, match="settings changed"):
        state.RunState(str(tmp_path), "appworld", "test", after)


def test_identity_allows_new_job_ports_and_identical_relocated_data(tmp_path):
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    first.write_bytes(b"identical training examples")
    second.write_bytes(first.read_bytes())
    before = {"data": {"train_files": [str(first)]}, "rllm": {"env": {"env_args": {"remote_environment_url": "http://host:1000", "base_port": 1000}}}}
    after = {"data": {"train_files": [str(second)]}, "rllm": {"env": {"env_args": {"remote_environment_url": "http://host:2000", "base_port": 2000}}}}
    assert state.training_identity(before) == state.training_identity(after)


def test_constant_schedule_run_may_extend_its_step_budget(tmp_path):
    def config(steps, **schedule):
        optim = {"lr": 1e-6, "lr_scheduler_type": "constant", "warmup_style": None, "lr_warmup_steps": -1, "lr_warmup_steps_ratio": 0.0, "total_training_steps": steps, **schedule}
        return {"actor_rollout_ref": {"actor": {"optim": optim}}, "critic": {"optim": dict(optim)}}

    state.RunState(str(tmp_path / "constant"), "appworld", "test", config(301))
    state.RunState(str(tmp_path / "constant"), "appworld", "test", config(501))
    for name, schedule in (("cosine", {"lr_scheduler_type": "cosine"}), ("ratio", {"lr_warmup_steps_ratio": 0.1})):
        state.RunState(str(tmp_path / name), "appworld", "test", config(301, **schedule))
        with pytest.raises(ValueError, match="settings changed.*optimizer"):
            state.RunState(str(tmp_path / name), "appworld", "test", config(501, **schedule))


def test_auxiliary_gate_identity_preserves_legacy_and_rejects_changed_objective(tmp_path):
    legacy = {"rllm": {"ace": {"training_mode": "standard"}}}
    explicit_legacy = {"rllm": {"ace": {"training_mode": "standard", "auxiliary_update_gate": "none"}}}
    gated = {"rllm": {"ace": {"training_mode": "standard", "auxiliary_update_gate": "strict_improvement"}}}
    assert state.training_identity(legacy) == state.training_identity(explicit_legacy)
    state.RunState(str(tmp_path), "mbpp", "legacy", legacy)
    state.RunState(str(tmp_path), "mbpp", "legacy", explicit_legacy)
    with pytest.raises(ValueError, match="settings changed.*rllm"):
        state.RunState(str(tmp_path), "mbpp", "gated", gated)


def test_handoff_identity_normalizes_missing_legacy_fields_and_preserves_saved_run(tmp_path):
    legacy = {"rllm": {"ace": {"training_mode": "standard"}}}
    explicit = {"rllm": {"ace": {"training_mode": "standard", "handoff_protocol": "legacy", "log_role_credit": False}}}
    assert state.training_identity(legacy) == state.training_identity(explicit)
    saved_identity = state.training_identity(legacy)
    saved_identity["rllm"]["ace"].pop("handoff_protocol")
    saved_identity["rllm"]["ace"].pop("log_role_credit")
    (tmp_path / "run.json").write_text(json.dumps({"schema_version": 1, "name": "legacy", "training_identity": saved_identity}))
    resumed = state.RunState(str(tmp_path), "appworld", "legacy", explicit)
    assert resumed.record["training_identity"] == state.training_identity(explicit)
    with pytest.raises(ValueError, match="settings changed"):
        state.RunState(str(tmp_path), "appworld", "repaired", {"rllm": {"ace": {"training_mode": "standard", "handoff_protocol": "feedback_v1"}}})


def test_terminal_feedback_identity_preserves_explicit_legacy_false():
    implicit = {"rllm": {"agent": {"max_steps": 20}}}
    legacy = {"rllm": {"agent": {"max_steps": 20, "preserve_terminal_feedback": False}}}
    enabled = {"rllm": {"agent": {"max_steps": 20, "preserve_terminal_feedback": True}}}
    assert state.training_identity(implicit) == state.training_identity(legacy)
    assert state.training_identity(enabled) != state.training_identity(legacy)


def test_controller_seed_identity_preserves_missing_none_and_rejects_changed_seed(tmp_path):
    missing = {"rllm": {"ace": {"training_mode": "standard"}}}
    explicit_none = {"rllm": {"ace": {"training_mode": "standard", "controller_seed": None}}}
    seeded = {"rllm": {"ace": {"training_mode": "standard", "controller_seed": 42}}}
    assert state.training_identity(missing) == state.training_identity(explicit_none)
    old_identity = state.training_identity(missing)
    old_identity["rllm"]["ace"].pop("controller_seed")
    (tmp_path / "run.json").write_text(json.dumps({"schema_version": 1, "name": "legacy", "training_identity": old_identity}))
    state.RunState(str(tmp_path), "appworld", "legacy", explicit_none)
    with pytest.raises(ValueError, match="settings changed"):
        state.RunState(str(tmp_path), "appworld", "seeded", seeded)
    seeded_path = tmp_path / "seeded"
    state.RunState(str(seeded_path), "appworld", "seeded", seeded)
    state.RunState(str(seeded_path), "appworld", "seeded", seeded)
    changed = {"rllm": {"ace": {"training_mode": "standard", "controller_seed": 43}}}
    with pytest.raises(ValueError, match="settings changed"):
        state.RunState(str(seeded_path), "appworld", "changed", changed)


@pytest.mark.parametrize("seed", [True, False, -1, 1.5, "42", 2**32])
def test_invalid_controller_seed_cannot_claim_run_identity(seed):
    with pytest.raises(ValueError, match="ace.controller_seed"):
        state.training_identity({"rllm": {"ace": {"controller_seed": seed}}})


def test_handoff_identity_hashes_only_the_selected_curator_asset(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in ("base_system_prompt.txt", "reflector_prompt.txt", "curator_prompt.txt", "curator_prompt_handoff_v1.txt"):
        (prompts / name).write_text(name)
    legacy = {"rllm": {"ace": {"prompts_dir": str(prompts)}}}
    repaired = {"rllm": {"ace": {"prompts_dir": str(prompts), "handoff_protocol": "feedback_v1"}}}
    before_legacy = state.training_identity(legacy)
    before_repaired = state.training_identity(repaired)
    (prompts / "curator_prompt_handoff_v1.txt").write_text("changed repaired prompt")
    assert state.training_identity(legacy) == before_legacy
    after_repaired = state.training_identity(repaired)
    assert after_repaired != before_repaired
    (prompts / "curator_prompt.txt").write_text("changed legacy prompt")
    assert state.training_identity(repaired) == after_repaired
    assert state.training_identity(legacy) != before_legacy


def test_handoff_identity_uses_the_same_protocol_normalization_as_the_trainer():
    canonical = {"rllm": {"ace": {"handoff_protocol": "feedback_v1"}}}
    padded = {"rllm": {"ace": {"handoff_protocol": " Feedback_V1 "}}}
    assert state.training_identity(canonical) == state.training_identity(padded)


def test_handoff_identity_hashes_optional_selected_reflector_variant(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in ("base_system_prompt.txt", "reflector_prompt.txt", "curator_prompt.txt", "curator_prompt_handoff_v1.txt"):
        (prompts / name).write_text(name)
    legacy = {"rllm": {"ace": {"prompts_dir": str(prompts)}}}
    repaired = {"rllm": {"ace": {"prompts_dir": str(prompts), "handoff_protocol": "feedback_v1"}}}
    before_legacy = state.training_identity(legacy)
    before_repaired = state.training_identity(repaired)
    (prompts / "reflector_prompt_handoff_v1.txt").write_text("repaired reflector")
    after_repaired = state.training_identity(repaired)
    assert after_repaired != before_repaired
    assert state.training_identity(legacy) == before_legacy
    (prompts / "reflector_prompt.txt").write_text("changed legacy reflector")
    assert state.training_identity(repaired) == after_repaired
    assert state.training_identity(legacy) != before_legacy


@pytest.mark.parametrize("mode", ["delayed_joint", "joint_ppo"])
def test_delayed_auxiliary_gate_identity_uses_effective_legacy_objective(mode):
    legacy = {"rllm": {"ace": {"training_mode": mode}}}
    ignored = {"rllm": {"ace": {"training_mode": mode, "auxiliary_update_gate": "strict_improvement"}}}
    assert state.training_identity(legacy) == state.training_identity(ignored)


def test_canonical_run_label_preserves_existing_identity_and_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "example-team")
    config = {
        "actor_rollout_ref": {"model": {"path": "Qwen/Qwen3-4B"}, "rollout": {"trace": {"experiment_name": "grpo-thinking"}, "n": 8}},
        "rllm": {"env": {"env_args": {"experiment_name": "grpo-thinking", "max_turns": 30}}},
    }
    previous = state.training_identity(config)
    # Reproduce the records written before run labels were excluded.
    previous["sampling"]["trace"]["experiment_name"] = "previous-trace-label"
    previous["rllm"]["env"]["env_args"]["experiment_name"] = "previous-environment-label"
    wandb = {"entity": "example-team", "project": "appworld", "id": "saved-run", "initialized": True}
    record = {"schema_version": 1, "name": "original-run-name", "training_identity": previous, "wandb": wandb}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(record))

    resumed = state.RunState(str(tmp_path), "appworld", "grpo-thinking", config)
    assert resumed.record["name"] == "original-run-name"
    assert resumed.record["wandb"] == wandb
    assert resumed.wandb_options()["resume"] == "must"
    assert "experiment_name" not in resumed.record["training_identity"]["sampling"]["trace"]
    assert "experiment_name" not in resumed.record["training_identity"]["rllm"]["env"]["env_args"]
    assert config["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"] == "grpo-thinking"

    preserved = path.read_bytes()
    config["actor_rollout_ref"]["model"]["path"] = "different-model"
    with pytest.raises(ValueError, match="settings changed.*model"):
        state.RunState(str(tmp_path), "appworld", "another-label", config)
    assert path.read_bytes() == preserved


def test_local_metrics_retain_replayed_training_steps(tmp_path, monkeypatch):
    first = state.RunState(str(tmp_path), "appworld", "test")
    first.log({"reward": 0.5}, 7)
    monkeypatch.setenv("RLLM_RESUME_STEP", "5")
    resumed = state.RunState(str(tmp_path), "appworld", "test")
    resumed.log({"reward": 0.6}, 6)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["training/global_step"] for row in rows] == [7, 6]
    assert [row["logging/index"] for row in rows] == [0, 1]
    assert rows[1]["training/resume_from_step"] == 5


def test_resumed_checkpoint_cannot_silently_fork_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("RLLM_RESUME_STEP", "5")
    monkeypatch.setenv("WANDB_ENTITY", "example-team")
    resumed = state.RunState(str(tmp_path), "appworld", "test")
    with pytest.raises(ValueError, match="no recorded W&B identity"):
        resumed.wandb_options()
    resumed.record["wandb_migration"] = {"start_new_team_segment": True, "reason": "Recorded migration from a preserved historical segment"}
    options = resumed.wandb_options()
    assert options["entity"] == "example-team" and options["resume"] == "allow"


def test_offline_and_disabled_do_not_claim_a_cloud_run(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "example-team")
    record = state.RunState(str(tmp_path), "appworld", "test")
    options = record.wandb_options()
    offline = SimpleNamespace(**{key: options[key] for key in ("entity", "project", "id")}, settings=SimpleNamespace(mode="offline"))
    record.wandb_started(offline)
    assert record.wandb_options()["resume"] == "allow"
    record.wandb_started(SimpleNamespace(entity="", project="", id="", settings=SimpleNamespace(mode="disabled")))
    assert not record.record["wandb"]["initialized"]


def test_update_schedule_is_part_of_identity():
    assert state.training_identity({"trainer": {"critic_warmup": 0}}) != state.training_identity({"trainer": {"critic_warmup": 100}})


def test_prompt_content_change_rejects_resume(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in ("base_system_prompt.txt", "reflector_prompt.txt", "curator_prompt.txt"):
        (prompts / name).write_text("original prompt")
    config = {"rllm": {"ace": {"prompts_dir": str(prompts), "reflector_system_prompt": "reflect"}}}
    state.RunState(str(tmp_path / "run"), "appworld", "test", config)
    (prompts / "reflector_prompt.txt").write_text("changed prompt")
    with pytest.raises(ValueError, match="settings changed"):
        state.RunState(str(tmp_path / "run"), "appworld", "test", config)


def test_interrupted_journal_commit_and_partial_tail_are_recoverable(tmp_path):
    run = state.RunState(str(tmp_path), "appworld", "test")
    old_record = (tmp_path / "run.json").read_bytes()
    run.log({"reward": 1.0}, 5)
    (tmp_path / "run.json").write_bytes(old_record)
    with (tmp_path / "metrics.jsonl").open("ab") as handle:
        handle.write(b'{"reward":')
    original = (tmp_path / "metrics.jsonl").read_bytes()
    resumed = state.RunState(str(tmp_path), "appworld", "test")
    resumed.log({"reward": 2.0}, 6)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["logging/index"] for row in rows] == [0, 1]
    assert next((tmp_path / "archive/metrics-recovery").glob("*.jsonl")).read_bytes() == original


def test_wandb_adapter_uses_automatic_cloud_index():
    tracking_spec = importlib.util.spec_from_file_location("tracking", ROOT / "rllm/utils/tracking.py")
    tracking = importlib.util.module_from_spec(tracking_spec)
    sys.modules["tracking"] = tracking
    tracking_spec.loader.exec_module(tracking)
    calls = []
    adapter = tracking._WandbAdapter(SimpleNamespace(log=lambda *args, **kw: calls.append((args, kw))))
    adapter.log({"reward": 1}, step=7)
    adapter.log({"reward": 2}, step=6)
    assert [args[0]["training/global_step"] for args, _ in calls] == [7, 6]
    assert all("step" not in kw for _, kw in calls)


def test_tracking_resumes_same_team_run_and_writes_local_metrics(tmp_path, monkeypatch):
    tracking_spec = importlib.util.spec_from_file_location("tracking", ROOT / "rllm/utils/tracking.py")
    tracking = importlib.util.module_from_spec(tracking_spec)
    sys.modules["tracking"] = tracking
    tracking_spec.loader.exec_module(tracking)
    calls = []
    logs = []

    def initialize(**options):
        calls.append(options)
        return SimpleNamespace(**{key: options[key] for key in ("entity", "project", "id")})

    fake = SimpleNamespace(init=initialize, define_metric=lambda *a, **kw: None, log=lambda data, **kw: logs.append((data, kw)), finish=lambda **kw: None)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "rllm.utils.run_state", state)
    monkeypatch.setenv("RLLM_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_ENTITY", "example-team")
    for step in (7, 6):
        logger = tracking.Tracking("appworld", "test", default_backend=["wandb"], config={})
        logger.log({"reward": 0.5}, step=step)
        logger.finish()
    assert calls[0]["entity"] == calls[1]["entity"] == "example-team"
    assert calls[0]["id"] == calls[1]["id"]
    assert calls[1]["resume"] == "must"
    assert [data["training/global_step"] for data, _ in logs] == [7, 6]
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2


def exhausted_trainer(tmp_path, global_steps, target, save_freq=10):
    trainer = SimpleNamespace(global_steps=global_steps, saved=[])
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            save_freq=save_freq,
            total_epochs=100,
            default_local_dir=str(tmp_path),
            get=lambda key, default=None: {"total_training_steps": target}.get(key, default),
        )
    )
    trainer._save_checkpoint = lambda: trainer.saved.append(trainer.global_steps)
    return trainer


def test_exhausted_data_saves_last_step_and_fails_short_of_target(tmp_path):
    # AppWorld GRPO: 100 epochs x 5 batches with one skipped batch ended at 499.
    trainer = exhausted_trainer(tmp_path, global_steps=500, target=501)
    with pytest.raises(RuntimeError, match="ended after step 499.*TRAINER_TOTAL_EPOCHS"):
        state.finish_data_exhausted(trainer)
    assert trainer.saved == [499] and trainer.global_steps == 500


@pytest.mark.parametrize("global_steps,existing", [(491, False), (500, True)])
def test_exhausted_data_does_not_resave_a_checkpoint(tmp_path, global_steps, existing):
    if existing:
        (tmp_path / f"global_step_{global_steps - 1}").mkdir()
    trainer = exhausted_trainer(tmp_path, global_steps=global_steps, target=None)
    state.finish_data_exhausted(trainer)
    assert trainer.saved == []


def test_both_fit_loops_finish_exhausted_data_after_the_epoch_loop():
    import ast

    for name in ("agent_ppo_trainer.py", "agent_workflow_trainer.py"):
        tree = ast.parse((ROOT / "rllm/trainer/verl" / name).read_text())
        fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fit_agent")
        loop = next(i for i, n in enumerate(fit.body) if isinstance(n, ast.For) and ast.unparse(n.target) == "epoch")
        assert ast.unparse(fit.body[loop + 1]) == "finish_data_exhausted(self)", name
