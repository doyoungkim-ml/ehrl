"""Exercise original ERL task bindings and the pinned backend's Hydra schema."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("mastermind", "mbpp", "appworld", "mini_swe_bench")
ROLLOUTS = dict.fromkeys(BENCHMARKS, 8)
MAX_TURNS = {"mastermind": 5, "mbpp": 6, "appworld": 10, "mini_swe_bench": 10}


def command(benchmark, *overrides, **updates):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(ROOT), RLLM_DRY_RUN="1", PYTHON_BIN=sys.executable)
    env["MODEL_VARIANT"] = "instruct" if benchmark in {"appworld", "mini_swe_bench"} else "nonthinking"
    env.update(updates)
    result = subprocess.run(
        ["bash", str(ROOT / f"examples/{benchmark}/train/erl/train.slurm"), *overrides],
        env=env,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    argv = json.loads(result.stdout)
    return argv, {item.lstrip("+").split("=", 1)[0]: item.split("=", 1)[1] for item in argv if "=" in item}


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_original_erl_recipe_composes_and_keeps_distinct_role_budgets(benchmark):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    argv, settings = command(benchmark)
    overrides = argv[argv.index("-m") + 2 :]
    keys = [item.lstrip("+").split("=", 1)[0] for item in overrides]
    assert len(keys) == len(set(keys)), "Duplicate Hydra keys can invalidate +add overrides"
    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("Set VERL_RESUME_SOURCE for exact-pin Hydra schema coverage")
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        config = compose(config_name="agent_ppo_trainer", overrides=overrides + [f"hydra.searchpath=[file://{source}/verl/trainer/config]"])
    assert config.algorithm.adv_estimator == "grpo"
    assert config.critic.enable is False
    assert config.rllm.rejection_sample.enable is False
    assert config.data.gen_batch_size == config.data.train_batch_size == 8
    assert config.actor_rollout_ref.rollout.n == ROLLOUTS[benchmark]
    assert config.rllm.agent.max_steps == MAX_TURNS[benchmark]
    assert config.trainer.total_training_steps == 301 and config.trainer.save_freq == 10 and config.trainer.save_at_end
    assert config.trainer.max_actor_ckpt_to_keep is None
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 32
    assert config.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-sum"
    assert config.actor_rollout_ref.actor.use_kl_loss
    assert config.actor_rollout_ref.actor.kl_loss_coef == 0.001
    assert config.actor_rollout_ref.actor.clip_ratio_high == 0.28
    assert config.trainer.critic_warmup == 0
    assert not config.trainer.val_before_train and not config.trainer.val_after_train
    assert config.trainer.test_freq == -1
    assert config.rllm.erl.upstream_commit == "20cd6460753973b9a5834769e2feed02736f44c6"
    assert config.rllm.erl.updater_max_response_length == 8192
    assert config.rllm.erl.retry_max_prompt_length == config.data.max_prompt_length + 8192
    assert config.rllm.erl.updater_max_prompt_length >= config.data.max_prompt_length + config.data.max_response_length + 8192
    window = config.actor_rollout_ref.rollout.max_model_len
    assert window >= config.rllm.erl.updater_max_prompt_length + config.rllm.erl.updater_max_response_length
    assert window >= config.rllm.erl.retry_max_prompt_length + config.data.max_response_length
    for budget in (
        config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
        config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu,
        config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu,
    ):
        assert budget >= window
    assert f"/train/erl/runs/rollout{ROLLOUTS[benchmark]}/" in config.trainer.default_local_dir
    assert f"_erl_rollout{ROLLOUTS[benchmark]}_" in config.trainer.experiment_name
    assert settings["rllm.disable_thinking"] == "True"
    expected = "Qwen/Qwen3-4B-Instruct-2507" if benchmark in {"appworld", "mini_swe_bench"} else "Qwen/Qwen3-0.6B"
    assert config.actor_rollout_ref.model.path == expected
    assert not any(key.startswith("rllm.ace.") for key in keys)


def test_explicit_erl_identity_and_attention_survive_parent_preset():
    argv, settings = command(
        "appworld",
        "trainer.total_training_steps=3",
        EXPERIMENT_NAME="custom-erl",
        VLLM_ATTENTION_CONFIG="{backend:FLASH_ATTN,flash_attn_version:2}",
    )
    assert argv[-1] == "trainer.total_training_steps=3"
    assert settings["trainer.experiment_name"] == "custom-erl"
    assert settings["actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config"] == "{backend:FLASH_ATTN,flash_attn_version:2}"


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_erl_preset_must_match_the_benchmark_rollouts(benchmark):
    for env in ({"PRESET": "original"}, {"PRESET": "rollout2"}, {"PRESET": "rollout3"}, {"PRESET": "rollout4"}, {"PRESET": "rollout5"}, {"PRESET": "rollout99"}, {"POLICY_ROLLOUTS": "99"}):
        with pytest.raises(subprocess.CalledProcessError) as error:
            command(benchmark, **env)
        assert "PRESET" in error.value.stderr
