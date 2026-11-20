"""Exercise maintained launcher commands without starting GPU runtimes."""

import hashlib
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("mbpp", "mastermind", "appworld", "mini_swe_bench")
# (standard preset, its policy rollouts, delayed preset, its policy rollouts)
ACE_PRESETS = {benchmark: ("oneeval_rollout8", "8", "delayed_oneeval_rollout8", "8") for benchmark in BENCHMARKS}
MAX_TURNS = {"mbpp": "6", "mastermind": "5", "appworld": "10", "mini_swe_bench": "10"}
STANDARD_ACE_MAX_TURNS = {benchmark: "6" if benchmark == "mbpp" else "20" for benchmark in BENCHMARKS}
TURN_OVERRIDES = {"mbpp": "MBPP_MAX_TURNS", "mastermind": "MASTERMIND_MAX_TURNS", "appworld": "APPWORLD_MAX_TURNS", "mini_swe_bench": "MINISW_MAX_STEPS"}
MATRIX = [
    (benchmark, method, variant)
    for benchmark in BENCHMARKS
    for method in ("grpo", "ace_rl")
    for variant in (("nonthinking", "thinking", "instruct", "thinking_2507") if benchmark in {"appworld", "mini_swe_bench"} else ("nonthinking", "thinking"))
]


def command(benchmark, method, variant, *overrides, **env):
    filename = "train.slurm"
    launcher = ROOT / "examples" / benchmark / "train" / method / filename
    # Avoid inheriting any experiment settings from the caller.
    clean_env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    clean_env.update(RLLM_DIR=str(ROOT), RLLM_DRY_RUN="1", MODEL_VARIANT=variant, PYTHON_BIN=sys.executable)
    clean_env.update(env)
    result = subprocess.run(["bash", str(launcher), *overrides], cwd=ROOT, env=clean_env, text=True, capture_output=True, check=True)
    argv = json.loads(result.stdout)
    return argv, {arg.lstrip("+").split("=", 1)[0]: arg.split("=", 1)[1] for arg in argv if "=" in arg}


@pytest.mark.parametrize("benchmark,method,variant", MATRIX)
def test_training_matrix_resolves_model_resources_and_method(benchmark, method, variant):
    argv, settings = command(benchmark, method, variant)
    size = "4B" if benchmark in {"appworld", "mini_swe_bench"} else "0.6B"
    suffix = {"instruct": "-Instruct-2507", "thinking_2507": "-Thinking-2507"}.get(variant, "")
    assert settings["actor_rollout_ref.model.path"] == f"Qwen/Qwen3-{size}{suffix}"
    assert settings["trainer.n_gpus_per_node"] == "2"
    assert settings["rllm.disable_thinking"] == ("False" if variant in {"thinking", "thinking_2507"} else "True")
    assert settings["actor_rollout_ref.rollout.n"] == ("8" if method == "grpo" else ACE_PRESETS[benchmark][1])
    assert settings["actor_rollout_ref.rollout.val_kwargs.n"] == "8"
    assert settings["data.train_batch_size"] == "8"
    assert settings["rllm.rejection_sample.multiplier"] == "2"
    assert settings["trainer.total_training_steps"] == "301"
    assert settings["trainer.save_freq"] == "10"
    assert "trainer.max_actor_ckpt_to_keep" not in settings
    turns = settings.get("rllm.agent.max_steps") or settings["rllm.workflow.workflow_args.max_steps"]
    expected_turns = STANDARD_ACE_MAX_TURNS[benchmark] if method == "ace_rl" else MAX_TURNS[benchmark]
    assert turns == expected_turns
    assert settings["actor_rollout_ref.actor.loss_agg_mode"] == "token-mean"
    module = argv[argv.index("-m") + 1]
    assert module == f"examples.{benchmark}.train.{method}.train"
    assert (ROOT / (module.replace(".", "/") + ".py")).is_file()
    if method == "ace_rl":
        expected = {
            "g_ref": "4",
            "g_cur": "4",
            "singlemode": "True",
            "singlemode_curator_one_group": "True",
            "policy_train_source": "seed",
            "selection_mode": "best_rejection_update",
            "candidate_eval_rollouts": "1",
            "reflection_source_selection": "random",
            "max_turns": expected_turns,
            "auxiliary_update_gate": "strict_improvement",
        }
        assert {key: settings[f"rllm.ace.{key}"] for key in expected} == expected


@pytest.mark.parametrize("benchmark,method,variant", MATRIX)
def test_training_matrix_composes_against_pinned_verl(benchmark, method, variant):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("Set VERL_RESUME_SOURCE to validate the actual pinned Hydra schema")
    argv, _ = command(benchmark, method, variant)
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    search = f"hydra.searchpath=[file://{source}/verl/trainer/config]"
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        cfg = compose(config_name="agent_ppo_trainer", overrides=argv[argv.index("-m") + 2 :] + [search])
    assert cfg.trainer.n_gpus_per_node == 2
    assert cfg.data.gen_batch_size >= cfg.data.train_batch_size
    assert list(cfg.trainer.logger) == ["console", "wandb"]


@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("method", ("grpo", "ppo", "ace_rl", "delayed_ace_rl", "erl", "delayed_joint_ppo"))
def test_all_maintained_methods_compose_with_eight_actor_rollouts(benchmark, method):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("Set VERL_RESUME_SOURCE for exact-pin Hydra schema coverage")
    env = {"PRESET": "delayed_oneeval_rollout8"} if method == "delayed_ace_rl" else {}
    launcher = "ace_rl" if method == "delayed_ace_rl" else method
    argv, _ = command(benchmark, launcher, "thinking", **env)
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        cfg = compose(config_name="agent_ppo_trainer", overrides=argv[argv.index("-m") + 2 :] + [f"hydra.searchpath=[file://{source}/verl/trainer/config]"])
    assert cfg.actor_rollout_ref.rollout.n == 8
    batch = {"mbpp": 6, "mastermind": 8, "appworld": 8, "mini_swe_bench": 8}[benchmark] if method == "delayed_joint_ppo" else 8
    assert cfg.data.train_batch_size == batch
    if method in {"ace_rl", "delayed_ace_rl", "delayed_joint_ppo"}:
        auxiliary_group = 4 if method == "ace_rl" else 1
        assert cfg.rllm.ace.g_ref == cfg.rllm.ace.g_cur == auxiliary_group
    if method == "delayed_joint_ppo":
        assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == batch
        assert f"/batch{batch}_rollout8/" in cfg.trainer.default_local_dir


def test_model_modes_have_distinct_default_checkpoint_identity():
    names = []
    for benchmark, method, variant in MATRIX:
        _, settings = command(benchmark, method, variant)
        names.append((settings["trainer.project_name"], settings["trainer.experiment_name"]))
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("benchmark,method,variant", MATRIX)
def test_same_preset_resolves_one_visible_run_across_jobs(benchmark, method, variant):
    _, first = command(benchmark, method, variant, SLURM_JOB_ID="123")
    _, second = command(benchmark, method, variant, SLURM_JOB_ID="999")
    model = first["actor_rollout_ref.model.path"].split("/")[-1].lower()
    tag = model if variant in {"instruct", "thinking_2507"} else f"{model}-{variant}"
    preset = ACE_PRESETS[benchmark][0] if method == "ace_rl" else "grpo"
    run = ROOT / "examples" / benchmark / "train" / method / "runs" / preset / tag
    assert first["trainer.default_local_dir"] == second["trainer.default_local_dir"] == str(run / "checkpoints")
    assert first["trainer.resume_mode"] == second["trainer.resume_mode"] == "auto"
    assert first["trainer.episode_log_dir"] == str(run / "episodes")
    if method == "ace_rl":
        assert first["rllm.ace.local_task_artifact_root"] == str(run / "task_artifacts")


@pytest.mark.parametrize("benchmark,variant", [(b, v) for b, m, v in MATRIX if m == "ace_rl"])
def test_delayed_oneeval_changes_only_aligned_schedule_and_identity(benchmark, variant):
    standard_preset, standard_n, delayed_preset, delayed_n = ACE_PRESETS[benchmark]
    turns = {TURN_OVERRIDES[benchmark]: STANDARD_ACE_MAX_TURNS[benchmark]}
    _, standard = command(benchmark, "ace_rl", variant, PRESET=standard_preset, **turns)
    _, delayed = command(benchmark, "ace_rl", variant, PRESET=delayed_preset, **turns)
    assert standard["rllm.ace.training_mode"] == "standard"
    assert delayed["rllm.ace.training_mode"] == "delayed_joint"
    assert delayed["rllm.ace.g_ref"] == delayed["rllm.ace.g_cur"] == "1"
    assert (standard["actor_rollout_ref.rollout.n"], delayed["actor_rollout_ref.rollout.n"]) == (standard_n, delayed_n)
    assert f"/runs/{delayed_preset}/" in delayed["trainer.default_local_dir"]
    assert f"rollout{delayed_n}" in delayed["trainer.experiment_name"]
    assert "rollout8" not in delayed["trainer.experiment_name"] or delayed_n == "8"
    assert delayed["trainer.experiment_name"] != standard["trainer.experiment_name"]
    changed = {
        "rllm.ace.training_mode", "rllm.ace.g_ref", "rllm.ace.g_cur", "actor_rollout_ref.rollout.n",
        "trainer.experiment_name", "trainer.default_local_dir", "trainer.episode_log_dir",
        "rllm.ace.local_task_artifact_root", "rllm.env.env_args.experiment_name",
    }
    changed.update(("rllm.ace.reflection_source_selection", "rllm.ace.auxiliary_update_gate"))
    assert standard["rllm.ace.reflection_source_selection"] == "random"
    assert delayed["rllm.ace.reflection_source_selection"] == "min_reward"
    assert standard["rllm.ace.auxiliary_update_gate"] == "strict_improvement"
    assert delayed["rllm.ace.auxiliary_update_gate"] == "none"
    assert {k: v for k, v in standard.items() if k not in changed} == {k: v for k, v in delayed.items() if k not in changed}


def test_explicit_run_dir_is_used_without_creating_outputs_in_dry_run(tmp_path):
    run = tmp_path / "a deliberate fork"
    _, settings = command("appworld", "ace_rl", "thinking", RUN_DIR=str(run))
    assert settings["trainer.default_local_dir"] == str(run / "checkpoints")
    assert not run.exists()


def test_explicit_miniswe_experiment_name_is_not_suffixed_twice():
    _, settings = command("mini_swe_bench", "ace_rl", "thinking", EXPERIMENT_NAME="reviewed-existing-name")
    assert settings["trainer.experiment_name"] == "reviewed-existing-name"


def test_checkpoint_root_override_requires_explicit_run_dir():
    with pytest.raises(subprocess.CalledProcessError) as error:
        command("appworld", "ace_rl", "thinking", CHECKPOINT_DIR="/elsewhere/checkpoints")
    assert "set RUN_DIR" in error.value.stderr


@pytest.mark.parametrize("benchmark", ("mbpp", "mastermind"))
def test_small_model_ace_rejects_conflicting_training_mode(benchmark):
    with pytest.raises(subprocess.CalledProcessError) as error:
        command(benchmark, "ace_rl", "thinking", PRESET=ACE_PRESETS[benchmark][0], ACE_TRAINING_MODE="delayed_joint")
    assert "conflicts with PRESET" in error.value.stderr


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_ace_presets_reject_group_sizes_other_than_the_benchmark_setting(benchmark):
    standard_preset = ACE_PRESETS[benchmark][0]
    for preset in ("oneeval_rollout4", "delayed_oneeval_rollout2", "delayed_oneeval_rollout5", "delayed_oneeval"):
        with pytest.raises(subprocess.CalledProcessError) as error:
            command(benchmark, "ace_rl", "thinking", PRESET=preset)
        assert "PRESET" in error.value.stderr
    with pytest.raises(subprocess.CalledProcessError) as error:
        command(benchmark, "ace_rl", "thinking", PRESET=standard_preset, POLICY_ROLLOUTS="3")
    assert "conflicts with PRESET" in error.value.stderr


def test_miniswe_shared_rollout8_matches_appworld_budget_and_keeps_oneeval_identity():
    _, settings = command("mini_swe_bench", "ace_rl", "instruct", PRESET="shared_rollout8", MINISW_MAX_STEPS="20")
    assert settings["actor_rollout_ref.rollout.n"] == "4"
    assert settings["actor_rollout_ref.actor.ppo_mini_batch_size"] == "16"
    assert (settings["rllm.ace.g_ref"], settings["rllm.ace.g_cur"], settings["rllm.ace.candidate_eval_rollouts"]) == ("2", "3", "1")
    assert settings["rllm.ace.policy_train_source"] == "seed_and_candidates"
    assert settings["rllm.ace.auxiliary_update_gate"] == "strict_improvement"
    _, oneeval = command("mini_swe_bench", "ace_rl", "instruct", MINISW_MAX_STEPS="20")
    assert oneeval["rllm.ace.policy_train_source"] == "seed"
    assert oneeval["actor_rollout_ref.actor.ppo_mini_batch_size"] == "8"
    assert "_rollout844_seed_oneeval_" in oneeval["trainer.experiment_name"]
    assert oneeval["rllm.ace.flush_playbook_every_n_steps"] == "3"
    _, deeper = command("mini_swe_bench", "ace_rl", "instruct", PRESET="shared_rollout8", FLUSH_PLAYBOOK_EVERY_N_STEPS="8")
    assert deeper["rllm.ace.flush_playbook_every_n_steps"] == "8"


def test_appworld_shared_rollout8_composes_against_pinned_verl():
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("Set VERL_RESUME_SOURCE to validate the actual pinned Hydra schema")
    argv, _ = command("appworld", "ace_rl", "instruct", PRESET="shared_rollout8")
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        cfg = compose(config_name="agent_ppo_trainer", overrides=argv[argv.index("-m") + 2 :] + [f"hydra.searchpath=[file://{source}/verl/trainer/config]"])
    assert cfg.actor_rollout_ref.rollout.n == 4
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 16
    assert cfg.rllm.ace.policy_train_source == "seed_and_candidates"
    assert (cfg.rllm.ace.g_ref, cfg.rllm.ace.g_cur, cfg.rllm.ace.candidate_eval_rollouts) == (2, 3, 1)


@pytest.mark.parametrize("preset", ("standard", "singlemode", "one_curator_group", "one_curator_group_no_rejection", "seed_oneeval", "seed_oneeval_no_rejection", "iterative_singlemode", "delayed_joint"))
def test_appworld_auxiliary_ablations_default_to_eight_policy_rollouts(preset):
    _, settings = command("appworld", "ace_rl", "thinking", PRESET=preset)
    assert settings["actor_rollout_ref.rollout.n"] == "8"


def test_appworld_shared_rollout8_trains_candidate_rollouts_within_grpo_budget():
    _, settings = command("appworld", "ace_rl", "instruct", PRESET="shared_rollout8")
    assert settings["actor_rollout_ref.rollout.n"] == "4"
    assert settings["rllm.ace.g_ref"] == "2"
    assert settings["rllm.ace.g_cur"] == "3"
    assert settings["rllm.ace.singlemode_curator_one_group"] == "True"
    assert settings["rllm.ace.candidate_eval_rollouts"] == "1"
    assert settings["rllm.ace.policy_train_source"] == "seed_and_candidates"
    assert settings["rllm.ace.training_mode"] == "standard"
    # Same 64-row actor mini-batch as GRPO (8 prompts x 8 rollouts).
    assert int(settings["actor_rollout_ref.actor.ppo_mini_batch_size"]) * int(settings["actor_rollout_ref.rollout.n"]) == 64
    with pytest.raises(subprocess.CalledProcessError) as error:
        command("appworld", "ace_rl", "instruct", PRESET="shared_rollout8", TRAINING_MODE="delayed_joint")
    assert "requires TRAINING_MODE=standard" in error.value.stderr


@pytest.mark.parametrize("benchmark", ("mastermind", "mbpp"))
def test_small_model_shared_rollout8_matches_grpo_budget_and_keeps_oneeval_identity(benchmark):
    turns = {TURN_OVERRIDES[benchmark]: STANDARD_ACE_MAX_TURNS[benchmark]}
    handoff = ("+rllm.ace.handoff_protocol=feedback_v1", "+rllm.agent.preserve_terminal_feedback=True")
    argv, settings = command(benchmark, "ace_rl", "nonthinking", *handoff, PRESET="shared_rollout8", **turns)
    assert argv[-2:] == list(handoff)
    assert settings["actor_rollout_ref.rollout.n"] == "4"
    # Same 64-row actor mini-batch as GRPO (8 prompts x 8 rollouts).
    assert settings["actor_rollout_ref.actor.ppo_mini_batch_size"] == "16"
    expected = {
        "g_ref": "2",
        "g_cur": "3",
        "singlemode": "True",
        "singlemode_curator_one_group": "True",
        "guidebook_aggregation_level": "single",
        "policy_train_source": "seed_and_candidates",
        "candidate_eval_rollouts": "1",
        "selection_mode": "best_rejection_update",
        "reflection_source_selection": "random",
        "auxiliary_update_gate": "strict_improvement",
        "flush_playbook_every_n_steps": "3",
        "training_mode": "standard",
        "max_turns": STANDARD_ACE_MAX_TURNS[benchmark],
        "handoff_protocol": "feedback_v1",
    }
    assert {key: settings[f"rllm.ace.{key}"] for key in expected} == expected
    assert settings["rllm.agent.max_steps"] == STANDARD_ACE_MAX_TURNS[benchmark]
    assert settings["rllm.agent.preserve_terminal_feedback"] == "True"
    assert "/runs/shared_rollout8/" in settings["trainer.default_local_dir"]
    assert "_rollout423_seed_and_candidates_" in settings["trainer.experiment_name"]
    _, ungated = command(benchmark, "ace_rl", "nonthinking", PRESET="shared_rollout8", AUXILIARY_UPDATE_GATE="none", **turns)
    assert ungated["rllm.ace.auxiliary_update_gate"] == "none"
    assert ungated["trainer.experiment_name"] != settings["trainer.experiment_name"]
    assert {k: v for k, v in ungated.items() if k not in {"rllm.ace.auxiliary_update_gate", "trainer.experiment_name", "rllm.ace.handoff_protocol", "rllm.agent.preserve_terminal_feedback"}} == {
        k: v for k, v in settings.items() if k not in {"rllm.ace.auxiliary_update_gate", "trainer.experiment_name", "rllm.ace.handoff_protocol", "rllm.agent.preserve_terminal_feedback"}
    }
    _, oneeval = command(benchmark, "ace_rl", "nonthinking", **turns)
    assert oneeval["rllm.ace.policy_train_source"] == "seed"
    assert oneeval["actor_rollout_ref.actor.ppo_mini_batch_size"] == "8"
    assert oneeval["trainer.experiment_name"] != settings["trainer.experiment_name"]


@pytest.mark.parametrize("benchmark", ("mastermind", "mbpp"))
def test_small_model_shared_rollout8_composes_against_pinned_verl(benchmark):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("Set VERL_RESUME_SOURCE to validate the actual pinned Hydra schema")
    argv, _ = command(benchmark, "ace_rl", "nonthinking", "+rllm.ace.handoff_protocol=feedback_v1", "+rllm.agent.preserve_terminal_feedback=True", PRESET="shared_rollout8")
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        cfg = compose(config_name="agent_ppo_trainer", overrides=argv[argv.index("-m") + 2 :] + [f"hydra.searchpath=[file://{source}/verl/trainer/config]"])
    assert cfg.actor_rollout_ref.rollout.n == 4
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 16
    assert cfg.rllm.ace.policy_train_source == "seed_and_candidates"
    assert (cfg.rllm.ace.g_ref, cfg.rllm.ace.g_cur, cfg.rllm.ace.candidate_eval_rollouts) == (2, 3, 1)
    assert cfg.rllm.ace.handoff_protocol == "feedback_v1" and cfg.rllm.agent.preserve_terminal_feedback is True


@pytest.mark.parametrize("benchmark", ("mastermind", "mbpp"))
def test_small_model_shared_rollout8_rejects_delayed_and_other_group_sizes(benchmark):
    for env, message in (
        ({"ACE_TRAINING_MODE": "delayed_joint"}, "conflicts with PRESET"),
        ({"POLICY_ROLLOUTS": "8"}, "conflicts with PRESET"),
    ):
        with pytest.raises(subprocess.CalledProcessError) as error:
            command(benchmark, "ace_rl", "nonthinking", PRESET="shared_rollout8", **env)
        assert message in error.value.stderr
    with pytest.raises(subprocess.CalledProcessError) as error:
        command(benchmark, "ace_rl", "nonthinking", PRESET="delayed_oneeval_rollout8", POLICY_TRAIN_SOURCE="seed_and_candidates")
    assert "requires standard ACE-RL training" in error.value.stderr
    # Candidate rows join the seed group, so one evaluation per candidate is complete.
    _, settings = command(benchmark, "ace_rl", "nonthinking", POLICY_TRAIN_SOURCE="seed_and_candidates")
    assert settings["rllm.ace.candidate_eval_rollouts"] == "1"


def test_appworld_retired_single_rollout_preset_is_rejected():
    with pytest.raises(subprocess.CalledProcessError) as error:
        command("appworld", "ace_rl", "thinking", PRESET="delayed_joint_rollout1")
    assert "Unknown ACE-RL PRESET" in error.value.stderr


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_reflection_source_selection_is_explicit_and_validated(benchmark):
    _, default = command(benchmark, "ace_rl", "thinking")
    _, ablation = command(benchmark, "ace_rl", "thinking", REFLECTION_SOURCE_SELECTION="min_reward")
    assert default["rllm.ace.reflection_source_selection"] == "random"
    assert ablation["rllm.ace.reflection_source_selection"] == "min_reward"
    assert ablation["trainer.experiment_name"] != default["trainer.experiment_name"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        command(benchmark, "ace_rl", "thinking", REFLECTION_SOURCE_SELECTION="max_reward")
    assert "REFLECTION_SOURCE_SELECTION" in error.value.stderr


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_auxiliary_gate_override_is_explicit_and_delayed_stays_legacy(benchmark):
    _, standard = command(benchmark, "ace_rl", "nonthinking")
    _, legacy = command(benchmark, "ace_rl", "nonthinking", AUXILIARY_UPDATE_GATE="none")
    assert legacy["rllm.ace.auxiliary_update_gate"] == "none"
    assert legacy["trainer.experiment_name"] != standard["trainer.experiment_name"]
    delayed_preset = ACE_PRESETS[benchmark][2]
    _, delayed = command(benchmark, "ace_rl", "nonthinking", PRESET=delayed_preset)
    assert delayed["rllm.ace.max_turns"] == MAX_TURNS[benchmark]
    assert delayed["rllm.ace.auxiliary_update_gate"] == "none"
    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero"):
        command(benchmark, "ace_rl", "nonthinking", AUXILIARY_UPDATE_GATE="unknown")
    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero"):
        command(benchmark, "ace_rl", "nonthinking", PRESET=delayed_preset, AUXILIARY_UPDATE_GATE="strict_improvement")


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_hydra_overrides_reach_the_command(benchmark):
    argv, settings = command(benchmark, "ace_rl", "thinking", "trainer.total_training_steps=3", "++rllm.ace.g_ref=2")
    assert argv[-2:] == ["trainer.total_training_steps=3", "++rllm.ace.g_ref=2"]
    assert settings["rllm.ace.g_ref"] == "2"


def test_maintained_training_shells_parse_and_have_no_site_identity():
    for benchmark in BENCHMARKS:
        for path in (ROOT / "examples" / benchmark / "train").rglob("*"):
            if path.suffix not in {".slurm", ".sh"}:
                continue
            subprocess.run(["bash", "-n", str(path)], check=True)
            assert not any(tag in path.name for tag in ("_kailin", "_doyoung", "_empire", "_torch"))
            if path.suffix == ".slurm":
                assert "#SBATCH --account=" not in path.read_text()


def test_small_models_reject_unavailable_posttrained_variants():
    with pytest.raises(subprocess.CalledProcessError):
        command("mastermind", "grpo", "instruct")


@pytest.mark.parametrize("benchmark", ("mbpp", "mastermind"))
@pytest.mark.parametrize("source", ("selected_candidate", "both"))
def test_selected_policy_defaults_to_all_candidate_rollouts(benchmark, source):
    _, settings = command(benchmark, "ace_rl", "nonthinking", POLICY_TRAIN_SOURCE=source)
    assert settings["rllm.ace.policy_train_source"] == source
    assert settings["rllm.ace.candidate_eval_rollouts"] == "0"


@pytest.mark.parametrize("source", ("selected_candidate", "both"))
@pytest.mark.parametrize("cap", ("1", "3", "4", "7"))
def test_selected_policy_rejects_explicit_partial_candidate_rollouts(source, cap):
    with pytest.raises(subprocess.CalledProcessError) as error:
        command("mbpp", "ace_rl", "nonthinking", POLICY_TRAIN_SOURCE=source, CANDIDATE_EVAL_ROLLOUTS=cap)
    assert "requires CANDIDATE_EVAL_ROLLOUTS=0 (all) or at least 8" in error.value.stderr


@pytest.mark.parametrize("cap", ("0", "8", "16"))
def test_selected_policy_accepts_explicit_complete_candidate_rollouts(cap):
    _, settings = command("mastermind", "ace_rl", "thinking", POLICY_TRAIN_SOURCE="both", CANDIDATE_EVAL_ROLLOUTS=cap)
    assert settings["rllm.ace.candidate_eval_rollouts"] == cap


@pytest.mark.parametrize("prefix", ("", "+", "++"))
def test_positional_policy_source_selects_all_rollouts(prefix):
    _, settings = command("mbpp", "ace_rl", "thinking", f"{prefix}rllm.ace.policy_train_source=both")
    assert settings["rllm.ace.candidate_eval_rollouts"] == "0"


def test_final_hydra_overrides_control_rollout_compatibility():
    with pytest.raises(subprocess.CalledProcessError):
        command("mbpp", "ace_rl", "thinking", "++rllm.ace.policy_train_source=both", "++rllm.ace.candidate_eval_rollouts=1")
    _, settings = command(
        "mbpp",
        "ace_rl",
        "thinking",
        "++rllm.ace.policy_train_source=both",
        "actor_rollout_ref.rollout.n=2",
        "++rllm.ace.candidate_eval_rollouts=2",
    )
    assert settings["rllm.ace.candidate_eval_rollouts"] == "2"
    _, settings = command(
        "mbpp",
        "ace_rl",
        "thinking",
        "++rllm.ace.candidate_eval_rollouts=0",
        POLICY_TRAIN_SOURCE="both",
        CANDIDATE_EVAL_ROLLOUTS="1",
    )
    assert settings["rllm.ace.candidate_eval_rollouts"] == "0"


def test_quoted_hydra_source_and_explicit_cap_are_checked():
    _, settings = command("mbpp", "ace_rl", "thinking", '++rllm.ace.policy_train_source="both"')
    assert settings["rllm.ace.candidate_eval_rollouts"] == "0"
    with pytest.raises(subprocess.CalledProcessError):
        command("mbpp", "ace_rl", "thinking", "++rllm.ace.candidate_eval_rollouts='1'", POLICY_TRAIN_SOURCE="selected_candidate")


@pytest.mark.parametrize("setting,value", (("POLICY_ROLLOUTS", "0"), ("CANDIDATE_EVAL_ROLLOUTS", "-1"), ("CANDIDATE_EVAL_ROLLOUTS", "invalid")))
def test_invalid_rollout_counts_fail_before_runtime(setting, value):
    with pytest.raises(subprocess.CalledProcessError):
        command("mastermind", "ace_rl", "nonthinking", **{setting: value})


@pytest.mark.parametrize("benchmark,method,variant", MATRIX)
def test_exact_ordered_argv_matches_reviewed_baseline(benchmark, method, variant):
    # Reviewed command snapshots include the maintained eight-rollout defaults.
    # Only interpreter and repository locations are normalized.
    baseline = json.loads((Path(__file__).with_name("training_argv_sha256.json")).read_text())
    argv, _ = command(benchmark, method, variant)
    canonical = ["<python>", *[arg.replace(str(ROOT), "<repo>") for arg in argv[1:]]]
    digest = hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()
    assert digest == baseline["|".join((benchmark, method, variant))]


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_method_owned_launcher_layout(benchmark):
    base = ROOT / "examples" / benchmark / "train"
    assert not (base / "harness").exists()
    assert not (base / "launch.sh").exists()
    assert not (ROOT / "scripts/train_benchmark.sh").exists()
    assert (base / "common.sh").is_file()
    for method in ("grpo", "ace_rl"):
        for name in ("presets.sh", "train.py", "train.slurm"):
            assert (base / method / name).is_file()
        assert not (base / method / "run.sh").exists()
        assert [path.name for path in (base / method).glob("*.slurm")] == ["train.slurm"]
        for launcher in (base / method).glob("*.slurm"):
            source = launcher.read_text()
            assert f"examples/{benchmark}/train/{method}/presets.sh" in source
            assert "train_benchmark.sh" not in source


@pytest.mark.parametrize("benchmark", ("appworld", "mini_swe_bench"))
def test_explicit_launch_values_preserve_literal_shell_metacharacters(tmp_path, benchmark):
    sentinel = tmp_path / "unexpected-shell-command"
    literal = f"quoted ' $(touch {sentinel}) `touch {sentinel}`\nnew line"
    run = tmp_path / literal
    _, settings = command(benchmark, "ace_rl", "thinking", RUN_DIR=str(run), EXPERIMENT_NAME=literal)

    assert settings["trainer.default_local_dir"] == str(run / "checkpoints")
    assert settings["trainer.experiment_name"] == literal
    assert not sentinel.exists()
    assert not run.exists()


@pytest.mark.parametrize("benchmark", ("appworld", "mini_swe_bench"))
def test_explicit_python_symlink_keeps_virtualenv_identity(tmp_path, benchmark):
    environment = tmp_path / "venv 'with spaces"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    python = environment / "bin" / "python"
    assert python.is_symlink()
    argv, _ = command(benchmark, "ace_rl", "thinking", PYTHON_BIN=str(python))

    assert argv[0] == str(python)
    prefix = subprocess.check_output([argv[0], "-c", "import sys; print(sys.prefix)"], text=True).strip()
    assert prefix == str(environment)
