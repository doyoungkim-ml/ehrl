"""Check additive PPO recipes against unchanged benchmark launch contracts."""

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("appworld", "mbpp", "mastermind", "mini_swe_bench")
METHODS = ("ppo", "delayed_joint_ppo")
# Tasks per step for delayed joint PPO (one ACE chain each), and the delayed
# ACE-RL parent preset whose benchmark settings it inherits.
JOINT_BATCH = {"appworld": 8, "mbpp": 6, "mastermind": 8, "mini_swe_bench": 8}
DELAYED_PARENT = {"appworld": "delayed_joint", "mbpp": "delayed_oneeval_rollout8",
                  "mastermind": "delayed_oneeval_rollout8", "mini_swe_bench": "delayed_oneeval_rollout8"}


def preset(benchmark, method):
    return f"batch{JOINT_BATCH[benchmark]}_rollout8" if method == "delayed_joint_ppo" else method
MATRIX = [
    (benchmark, method, variant)
    for benchmark in BENCHMARKS
    for method in METHODS
    for variant in (("nonthinking", "thinking", "instruct", "thinking_2507") if benchmark in {"appworld", "mini_swe_bench"} else ("nonthinking", "thinking"))
]


def command(benchmark, method, variant="nonthinking", *overrides, capture_runtime=False, **env):
    clean = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    clean.update(RLLM_DIR=str(ROOT), RLLM_DRY_RUN="1", MODEL_VARIANT=variant, PYTHON_BIN=sys.executable, **env)
    filename = "train.slurm"
    source = ROOT / "examples" / benchmark / "train" / method / filename
    shell = ["bash", str(source), *overrides]
    if capture_runtime:
        # Source the real launcher so settings exported by presets/runtime are
        # observable too; argv alone cannot see AppWorld ACE's backend choice.
        script = (
            'source "$1" "${@:2}"\n"$PYTHON_BIN" -c \'import json, os; '
            'print(json.dumps({key: value for key, value in os.environ.items() '
            'if key.startswith(("VLLM_", "PYTORCH_", "TORCH_", "CUDA_", "NCCL_", '
            '"RLLM_GPU_KEEPALIVE", "RLLM_DISABLE_GPU_KEEPALIVE"))}))\''
        )
        shell = ["bash", "-c", script, "launcher-runtime-test", str(source), *overrides]
    result = subprocess.run(shell, cwd=ROOT, env=clean, check=True, capture_output=True, text=True)
    argv, end = json.JSONDecoder().raw_decode(result.stdout)
    settings = {arg.lstrip("+").split("=", 1)[0]: arg.split("=", 1)[1] for arg in argv if "=" in arg}
    if capture_runtime:
        return argv, settings, json.loads(result.stdout[end:])
    return argv, settings


@pytest.mark.parametrize("benchmark,method,variant", MATRIX)
def test_ppo_recipes_preserve_benchmark_settings_and_isolate_outputs(benchmark, method, variant):
    argv, settings = command(benchmark, method, variant)
    parent = "grpo" if method == "ppo" else "ace_rl"
    parent_env = {"PRESET": "grpo" if method == "ppo" else DELAYED_PARENT[benchmark]}
    _, original = command(benchmark, parent, variant, **parent_env)
    if method == "delayed_joint_ppo":
        # Omitted historical gate fields and explicit none have the same
        # effective objective; joint PPO must never inherit standard ACE gating.
        settings.setdefault("rllm.ace.auxiliary_update_gate", "none")
        original.setdefault("rllm.ace.auxiliary_update_gate", "none")
        assert settings["rllm.ace.auxiliary_update_gate"] == "none"
    # These are the declared algorithm changes and artifact identities. All other
    # generated benchmark settings, including model lanes, masks, sampling, limits,
    # prompts, LR, evaluation cadence, and sidecars, must remain identical.
    changed = {
        "algorithm.adv_estimator", "trainer.critic_warmup", "trainer.experiment_name", "trainer.default_local_dir",
        "trainer.val_before_train", "trainer.test_freq", "trainer.val_after_train",
        "trainer.episode_log_dir", "trainer.trajectory_log_dir", "rllm.env.env_args.experiment_name",
        "rllm.rejection_sample.enable", "rllm.rejection_sample.multiplier", "critic.checkpoint.save_contents",
        "critic.ppo_infer_max_token_len_per_gpu", "rllm.ace.local_task_artifact_root",
    }
    if method == "delayed_joint_ppo":
        changed |= {"actor_rollout_ref.rollout.n", "data.train_batch_size", "actor_rollout_ref.actor.ppo_mini_batch_size", "rllm.ace.candidate_eval_batch_size"}
        changed |= {f"rllm.ace.{key}" for key in (
            "g_ref", "g_cur", "singlemode", "singlemode_curator_one_group", "guidebook_aggregation_level",
            "policy_train_source", "selection_mode", "training_mode", "candidate_eval_rollouts",
        )}
    assert {key: value for key, value in original.items() if key not in changed} == {
        key: settings[key] for key in original if key not in changed
    }
    assert settings["algorithm.adv_estimator"] == "gae"
    assert settings["algorithm.gamma"] == "1.0"
    assert settings["algorithm.lam"] == "0.95"
    assert settings["critic.enable"] == "True"
    assert settings["critic.optim.lr"] == "1e-5"
    assert settings["critic.model.path"] == settings["actor_rollout_ref.model.path"]
    assert settings["critic.ppo_infer_max_token_len_per_gpu"] == settings["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"]
    assert settings["trainer.critic_warmup"] == "10"
    assert settings["trainer.val_before_train"] == "False"
    assert settings["trainer.test_freq"] == "-1"
    assert settings["trainer.val_after_train"] == "False"
    assert settings["trainer.save_at_end"] == "True"
    assert settings["rllm.rejection_sample.enable"] == "False"
    assert settings["rllm.rejection_sample.multiplier"] == "1"
    for owner in ("actor_rollout_ref.actor", "critic"):
        for operation in ("save", "load"):
            assert settings[f"{owner}.checkpoint.{operation}_contents"] == "['model','optimizer','extra']"
    assert "trainer.max_actor_ckpt_to_keep" not in settings and "trainer.max_critic_ckpt_to_keep" not in settings
    assert settings["trainer.total_training_steps"] == "301"
    assert settings["trainer.save_freq"] == "10"
    if method == "delayed_joint_ppo":
        batch = str(JOINT_BATCH[benchmark])
        assert settings["data.train_batch_size"] == settings["actor_rollout_ref.actor.ppo_mini_batch_size"] == batch
        assert settings["actor_rollout_ref.rollout.n"] == "8"
        assert settings["actor_rollout_ref.rollout.val_kwargs.n"] == "8"
        expected = {
            "training_mode": "joint_ppo", "g_ref": "1", "g_cur": "1", "singlemode": "False",
            "singlemode_curator_one_group": "False", "guidebook_aggregation_level": "task",
            "policy_train_source": "both", "selection_mode": "best", "candidate_eval_rollouts": "1",
        }
        assert {key: settings[f"rllm.ace.{key}"] for key in expected} == expected
    assert f"/{benchmark}/train/{method}/runs/{preset(benchmark, method)}/" in settings["trainer.default_local_dir"]
    assert settings["trainer.default_local_dir"].endswith("/checkpoints")
    assert settings["trainer.experiment_name"] != original["trainer.experiment_name"]
    module = argv[argv.index("-m") + 1]
    assert module == f"examples.{benchmark}.train.{method}.train"
    assert (ROOT / (module.replace(".", "/") + ".py")).is_file()
    if benchmark == "mini_swe_bench":
        assert settings["rllm.workflow.use_workflow"] == ("True" if method == "ppo" else "False")


@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("method", ("erl", *METHODS))
@pytest.mark.parametrize("runtime_case", ("h200", "h100", "explicit_overrides"))
def test_baselines_inherit_parent_gpu_runtime_and_explicit_overrides(benchmark, method, runtime_case):
    """Method additions must not replace a benchmark's working engine recipe.

    Compare both directions so new engine kwargs (such as optimization_level)
    cannot slip through by checking only the parent's already-existing keys.
    ERL's larger prompt/token budgets remain covered by its objective tests.
    """
    variant = "instruct" if benchmark in {"appworld", "mini_swe_bench"} else "nonthinking"
    parent = "ace_rl" if method == "delayed_joint_ppo" else "grpo"
    parent_preset = DELAYED_PARENT[benchmark] if method == "delayed_joint_ppo" else "grpo"
    env = {"RLLM_GPU_FAMILY": "h200" if runtime_case == "explicit_overrides" else runtime_case}
    overrides = ()
    if runtime_case == "explicit_overrides":
        env.update(
            ROLLOUT_ENFORCE_EAGER="True",
            VLLM_ATTENTION_BACKEND="FLASH_ATTN",
            VLLM_ATTENTION_CONFIG="{backend:FLASH_ATTN,flash_attn_version:2}",
            VLLM_COMPILATION_CONFIG="{cudagraph_mode:NONE}",
            ROLLOUT_GPU_MEMORY_UTILIZATION="0.3",
            RLLM_GPU_KEEPALIVE="0",
            RLLM_DISABLE_GPU_KEEPALIVE="1",
        )
        overrides = (
            "actor_rollout_ref.rollout.enforce_eager=False",
            "++actor_rollout_ref.rollout.engine_kwargs.vllm.optimization_level=0",
        )
    _, actual, actual_env = command(benchmark, method, variant, *overrides, capture_runtime=True, **env)
    _, expected, expected_env = command(benchmark, parent, variant, *overrides, capture_runtime=True, PRESET=parent_preset, **env)
    runtime_keys = {
        "actor_rollout_ref.rollout.name", "actor_rollout_ref.rollout.mode",
        "actor_rollout_ref.rollout.tensor_model_parallel_size", "actor_rollout_ref.rollout.data_parallel_size",
        "actor_rollout_ref.rollout.enforce_eager", "actor_rollout_ref.rollout.free_cache_engine",
        "actor_rollout_ref.rollout.gpu_memory_utilization", "trainer.n_gpus_per_node", "trainer.nnodes",
    }

    def runtime(settings):
        return {key: value for key, value in settings.items() if key in runtime_keys or key.startswith("actor_rollout_ref.rollout.engine_kwargs.")}

    assert runtime(actual) == runtime(expected)
    assert actual_env == expected_env
    if runtime_case != "explicit_overrides":
        assert "actor_rollout_ref.rollout.engine_kwargs.vllm.optimization_level" not in actual


@pytest.mark.parametrize("benchmark,method,variant", MATRIX)
def test_ppo_commands_compose_with_pinned_critic_schema(benchmark, method, variant):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from rllm.algorithms.ppo.configuration import validate_ppo_config

    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("Set VERL_RESUME_SOURCE for pinned critic Hydra-schema coverage")
    argv, _ = command(benchmark, method, variant)
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    search = f"hydra.searchpath=[file://{source}/verl/trainer/config]"
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        config = compose(config_name="agent_ppo_trainer", overrides=argv[argv.index("-m") + 2 :] + [search])
    validate_ppo_config(config, delayed_joint=method == "delayed_joint_ppo")
    assert config.data.gen_batch_size == (JOINT_BATCH[benchmark] if method == "delayed_joint_ppo" else 8)
    assert config.critic.use_dynamic_bsz
    assert config.critic.ppo_mini_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size
    assert list(config.critic.checkpoint.load_contents) == ["model", "optimizer", "extra"]
    if benchmark == "mini_swe_bench" and method == "ppo":
        config.trainer.use_legacy_worker_impl = "enable"
        with pytest.raises(ValueError, match="masked-padding critic loss"):
            validate_ppo_config(config)


@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("method", METHODS)
def test_explicit_names_and_final_overrides_survive_parent_presets(benchmark, method):
    argv, settings = command(
        benchmark, method, "nonthinking", "trainer.total_training_steps=3", "critic.optim.lr=2e-5",
        EXPERIMENT_NAME="custom-experiment", WANDB_NAME="custom-dashboard", RUN_NAME="custom-run",
    )
    assert settings["trainer.experiment_name"] == "custom-experiment"
    assert f"/train/{method}/runs/{preset(benchmark, method)}/" in settings["trainer.default_local_dir"]
    assert argv[-2:] == ["trainer.total_training_steps=3", "critic.optim.lr=2e-5"]


def test_ppo_binding_rejects_silent_grpo_and_incomplete_checkpoint_state():
    from omegaconf import OmegaConf

    from rllm.algorithms.ppo.configuration import validate_ppo_config

    config = OmegaConf.create({"algorithm": {"adv_estimator": "grpo"}})
    with pytest.raises(ValueError, match="requires algorithm.adv_estimator=gae"):
        validate_ppo_config(config)
    config.algorithm.adv_estimator = "gae"
    with pytest.raises(ValueError, match="must save model, optimizer, and extra"):
        validate_ppo_config(config)
    with pytest.raises(ValueError, match="training_mode=joint_ppo"):
        validate_ppo_config(config, delayed_joint=True)


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_joint_ppo_semantic_overrides_are_validated_before_parent_entrypoint(benchmark):
    from types import SimpleNamespace

    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from rllm.algorithms.ppo.configuration import validate_ppo_config

    argv, _ = command(benchmark, "delayed_joint_ppo", TRAINING_MODE="joint_ppo", ACE_TRAINING_MODE="joint_ppo")
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    source = os.environ["VERL_RESUME_SOURCE"]
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        config = compose(config_name="agent_ppo_trainer", overrides=argv[argv.index("-m") + 2 :] + [f"hydra.searchpath=[file://{source}/verl/trainer/config]"])
    # Execute the dedicated entrypoint's real body with only the GPU-owning
    # parent replaced, proving validation happens before it can allocate Ray.
    path = ROOT / f"examples/{benchmark}/train/delayed_joint_ppo/train.py"
    main = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main.decorator_list = []
    calls = []
    namespace = {"validate_ppo_config": validate_ppo_config, "parent_main": SimpleNamespace(__wrapped__=calls.append)}
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace)
    namespace["main"](config)
    assert calls == [config]
    grouped = copy.deepcopy(config)
    OmegaConf.update(grouped, "actor_rollout_ref.rollout.n", 8)
    calls.clear()
    namespace["main"](grouped)
    assert calls == [grouped]
    invalid = {
        "rllm.ace.training_mode": "delayed_joint",
        "actor_rollout_ref.rollout.n": 0,
        "rllm.ace.g_ref": 4,
        "rllm.ace.g_cur": 4,
        "rllm.ace.candidate_eval_rollouts": 0,
        "rllm.ace.policy_train_source": "selected_candidate",
        "rllm.ace.selection_mode": "best_rejection",
        "rllm.ace.guidebook_aggregation_level": "single",
        "rllm.ace.singlemode": True,
        "rllm.ace.singlemode_curator_one_group": True,
        "rllm.rejection_sample.enable": True,
        "rllm.rejection_sample.multiplier": 2,
        "rllm.stepwise_advantage.enable": True,
        "rllm.mask_truncated_samples": True,
        "rllm.agent.overlong_filter": True,
    }
    for key, value in invalid.items():
        bad = copy.deepcopy(config)
        OmegaConf.update(bad, key, value)
        calls.clear()
        with pytest.raises(ValueError, match=key):
            namespace["main"](bad)
        assert not calls


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_joint_ppo_environment_overrides_are_not_hidden_by_parent_defaults(benchmark):
    _, settings = command(
        benchmark, "delayed_joint_ppo", REFLECTOR_ROLLOUTS="2", CURATOR_ROLLOUTS="3",
        REJECTION_SAMPLE_MULTIPLIER="4", TRAINING_MODE="standard", VAL_ROLLOUTS="3",
    )
    assert settings["rllm.ace.g_ref"] == "2"
    assert settings["rllm.ace.g_cur"] == "3"
    assert settings["rllm.rejection_sample.multiplier"] == "4"
    assert settings["rllm.ace.training_mode"] == "standard"
    assert settings["actor_rollout_ref.rollout.val_kwargs.n"] == "3"


@pytest.mark.parametrize("method", METHODS)
def test_mini_swe_critic_active_inference_budget_accepts_long_trajectories(method):
    """Execute the pin's active config mapping and sequence-budget guard on CPU.

    This targets the new EngineWorker field, which is independent of the legacy
    forward_max_token_len_per_gpu. No distributed worker imports are needed.
    """
    import torch
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    source = Path(os.environ["VERL_RESUME_SOURCE"])
    argv, _ = command("mini_swe_bench", method)
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        config = compose(config_name="agent_ppo_trainer", overrides=argv[argv.index("-m") + 2 :] + [f"hydra.searchpath=[file://{source}/verl/trainer/config]"])
    tree = ast.parse((source / "verl/trainer/ppo/ray_trainer.py").read_text())
    mapping = next(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(ast.unparse(target) == "engine_config.infer_max_token_len_per_gpu" for target in node.targets)
        and "critic_cfg" in ast.unparse(node.value)
    )
    budget = eval(compile(ast.Expression(mapping), "<pinned-critic-engine-mapping>", "eval"), {"critic_cfg": config.critic})
    assert budget >= config.data.max_prompt_length + config.data.max_response_length
    source_file = source / "verl/utils/seqlen_balancing.py"
    function = copy.deepcopy(next(node for node in ast.parse(source_file.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "rearrange_micro_batches"))
    guard = next(index for index, node in enumerate(function.body) if isinstance(node, ast.Assert))
    function.body = function.body[:guard + 1]
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source_file), "exec"), namespace)
    batch = {"input_ids": torch.zeros((1, 32769), dtype=torch.long), "attention_mask": torch.ones((1, 32769), dtype=torch.long)}
    namespace["rearrange_micro_batches"](batch, max_token_len=budget)
    with pytest.raises(AssertionError, match="max_token_len must be greater"):
        namespace["rearrange_micro_batches"](batch, max_token_len=32768)


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_joint_ppo_preset_name_and_batch_must_match_the_benchmark_setting(benchmark):
    for env in ({"PRESET": "delayed_joint_ppo"}, {"PRESET": "batch99"}, {"PRESET": f"batch{JOINT_BATCH[benchmark]}"}, {"TRAIN_BATCH_SIZE": "99"}, {"POLICY_ROLLOUTS": "1"}):
        with pytest.raises(subprocess.CalledProcessError) as error:
            command(benchmark, "delayed_joint_ppo", **env)
        assert "PRESET" in error.value.stderr
    _, settings = command(benchmark, "delayed_joint_ppo", ACTOR_PPO_MINI_BATCH_SIZE="2", PPO_MINI_BATCH_SIZE="2")
    assert settings["actor_rollout_ref.actor.ppo_mini_batch_size"] == "2"
    assert f"_batch{JOINT_BATCH[benchmark]}_" in settings["trainer.experiment_name"]


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_joint_group_eight_default_preserves_role_groups_and_task_batch(benchmark):
    _, settings = command(benchmark, "delayed_joint_ppo", "nonthinking")
    assert settings["actor_rollout_ref.rollout.n"] == "8"
    assert settings["rllm.ace.g_ref"] == settings["rllm.ace.g_cur"] == "1"
    assert settings["data.train_batch_size"] == str(JOINT_BATCH[benchmark])
    assert settings["actor_rollout_ref.actor.ppo_mini_batch_size"] == str(JOINT_BATCH[benchmark])
