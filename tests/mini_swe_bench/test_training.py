"""CPU contracts for MiniSWEBench adapters, launchers, and controller wiring.

Execute selected production controller methods from AST, without importing Ray,
VeRL, TensorDict, or GPU workers. Agent/environment/reward classes remain real;
container execution and parquet/registry I/O are test doubles. Hydra reads the
prepared source's YAML files via VERL_RESUME_SOURCE (or VERL_SOURCE_DIR), not an
import of the VeRL package. A prepared source on PYTHONPATH also works.

Run with unittest, or pytest --confcutdir=tests/mini_swe_bench to exclude the
repository-wide optional-dependency stubs. No integration coverage is skipped.
"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from examples.mini_swe_bench.env import MiniSWEnv
from examples.mini_swe_bench.train.ace_rl.adapter import MiniSweReflectorAdapter
from examples.mini_swe_bench.train.ace_rl.agent import MiniSweACEAgent
from examples.mini_swe_bench.train.ace_rl.env import MiniSweACEEnv
from examples.mini_swe_bench.train.ace_rl.train import (
    ACE_DEFAULTS,
    AGENT_CLASS,
    ENV_CLASS,
    _resolve_trainer_class_path,
)
from examples.mini_swe_bench.train.ace_rl.train import (
    main as train_local_ace_rl,
)
from examples.mini_swe_bench.train.common import load_datasets, preflight_checks, select_rows
from examples.mini_swe_bench.workflow import MiniSweWorkflow
from rllm.agents.agent import Episode, Step, Trajectory
from rllm.algorithms.ace.context.playbook import Playbook
from rllm.data import Dataset, DatasetRegistry

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "examples/mini_swe_bench/train"
LAUNCHERS = {
    "grpo": TRAIN / "grpo/train.slurm",
    "ace_rl": TRAIN / "ace_rl/train.slurm",
}


def execute_source(nodes, path, namespace):
    """Execute production AST nodes, preserving decorators and method bodies."""
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *copy.deepcopy(nodes)],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def candidate_controller():
    """Load only the candidate-cloning, prompt, and task-ID production methods."""
    namespace = {"copy": copy, "os": os, "hashlib": hashlib, "logger": logging.getLogger(__name__)}
    selections = (
        (
            "trainer_base.py",
            "ACERLAgentPPOTrainer",
            {"_render_template"},
            {
                "_parse_candidate_eval_rollouts",
                "_get_full_agent_args",
                "_get_full_env_args",
                "_get_candidate_eval_source_envs",
                "_render_base_system_prompt",
                "_apply_playbook_prompt_to_agent",
                "_inject_playbook_into_agents",
            },
        ),
        (
            "trainer.py",
            "LocalACERLAgentPPOTrainer",
            {"_safe_task_id_fragment", "_fallback_task_id_from_payload"},
            {
                "_extract_batch_task_ids",
                "_build_local_candidate_eval_envs_and_agents",
            },
        ),
    )
    for filename, class_name, helpers, methods in selections:
        path = ROOT / "rllm/algorithms/ace" / filename
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helpers]
        assert {node.name for node in functions} == helpers
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        assert {node.name for node in cls.body} == methods
        if filename == "trainer_base.py":
            cls.bases = []
        execute_source([*functions, cls], path, namespace)
    return namespace["LocalACERLAgentPPOTrainer"]()


def verl_config_override():
    """Find YAML without importing VeRL's package or its optional dependencies."""
    source = os.environ.get("VERL_RESUME_SOURCE") or os.environ.get("VERL_SOURCE_DIR")
    roots = [Path(source)] if source else [Path(entry) for entry in sys.path if entry]
    for root in roots:
        config_dir = root / "verl/trainer/config"
        if (config_dir / "ppo_trainer.yaml").is_file():
            return f"hydra.searchpath=[{config_dir.resolve().as_uri()}]"
    raise FileNotFoundError("Set VERL_RESUME_SOURCE to the prepared VeRL source for CPU Hydra configuration tests")


def task(instance_id="sympy__sympy-123"):
    return {
        "instance": {"instance_id": instance_id, "problem_statement": "Repair simplify()", "image_name": "test/sympy:latest"},
        "data_source": "SumanthRH/SWE-Gym-Subset",
        "prompt": [{"role": "user", "content": "Repair simplify()"}],
        "env_class": "null",
    }


def dry_run(method, variant="nonthinking", overrides=(), *, environ=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_", "MINISW_", "TRAIN_", "TRAINING_", "EXPERIMENT_", "MAX_", "PPO_", "POLICY_"))}
    env.update(RLLM_DIR=str(ROOT), RLLM_DRY_RUN="1", MODEL_VARIANT=variant)
    env.update(environ or {})
    result = subprocess.run(["bash", str(LAUNCHERS[method]), *overrides], env=env, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


class LauncherTests(unittest.TestCase):
    def test_full_model_matrix_composes_with_real_hydra_config(self):
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y), replace=True)
        models = {
            "nonthinking": ("Qwen/Qwen3-4B", True),
            "thinking": ("Qwen/Qwen3-4B", False),
            "instruct": ("Qwen/Qwen3-4B-Instruct-2507", True),
            "thinking_2507": ("Qwen/Qwen3-4B-Thinking-2507", False),
        }
        for method in LAUNCHERS:
            for variant, (model, disable) in models.items():
                with self.subTest(method=method, variant=variant):
                    argv = dry_run(method, variant, ["trainer.total_training_steps=2"])
                    self.assertEqual(argv[2:4], ["-m", f"examples.mini_swe_bench.train.{method}." + "train"])
                    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
                        cfg = compose(config_name="agent_ppo_trainer", overrides=[*argv[4:], verl_config_override()])
                    with contextlib.redirect_stdout(io.StringIO()):
                        preflight_checks(cfg)
                    self.assertEqual(cfg.actor_rollout_ref.model.path, model)
                    self.assertEqual(cfg.rllm.disable_thinking, disable)
                    self.assertEqual(cfg.rllm.accumulate_reasoning, not disable)
                    self.assertEqual(cfg.trainer.total_training_steps, 2)
                    self.assertEqual(cfg.trainer.n_gpus_per_node, 2)
                    if method == "ace_rl":
                        self.assertFalse(cfg.rllm.workflow.use_workflow)
                        self.assertEqual((cfg.actor_rollout_ref.rollout.n, cfg.rllm.ace.g_ref, cfg.rllm.ace.g_cur), (8, 4, 4))
                        self.assertEqual(cfg.actor_rollout_ref.rollout.val_kwargs.n, 8)
                        self.assertEqual(cfg.rllm.ace.reflection_source_selection, "random")
                        self.assertEqual(cfg.rllm.ace.auxiliary_update_gate, "strict_improvement")
                        self.assertEqual((cfg.rllm.agent.max_steps, cfg.rllm.env.env_args.max_steps, cfg.rllm.ace.max_turns), (20, 20, 20))
                        self.assertTrue(cfg.rllm.ace.singlemode)
                        self.assertTrue(cfg.rllm.ace.singlemode_curator_one_group)
                        self.assertEqual(cfg.rllm.ace.policy_train_source, "seed")
                        self.assertEqual(cfg.rllm.ace.selection_mode, "best_rejection_update")
                        self.assertEqual(cfg.rllm.ace.candidate_eval_rollouts, 1)
                        self.assertEqual(cfg.rllm.ace.training_mode, "standard")
                        expected_name = f"mini_swe_bench_harness_{model.split('/')[-1]}_{variant}_shaped_tasksall_offset0_2gpu_singlemode1curgroup_rollout844_seed_oneeval_best_rejection_update_random_reflection_source_aux_strict_improvement"
                        self.assertEqual(cfg.trainer.experiment_name, expected_name)
                    else:
                        self.assertTrue(cfg.rllm.workflow.use_workflow)
                        self.assertEqual(cfg.actor_rollout_ref.rollout.n, 8)
                        self.assertEqual(cfg.rllm.workflow.workflow_args.max_steps, 10)
                        self.assertEqual(cfg.rllm.rejection_sample.mode, "variance")

    def test_delayed_joint_preset_keeps_comparison_controls_and_separate_identity(self):
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y), replace=True)
        for variant in ("nonthinking", "thinking", "instruct", "thinking_2507"):
            with self.subTest(variant=variant):
                standard_args = dry_run("ace_rl", variant)
                delayed_args = dry_run("ace_rl", variant, environ={"TRAINING_MODE": "delayed_joint"})
                with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
                    standard = compose(config_name="agent_ppo_trainer", overrides=[*standard_args[4:], verl_config_override()])
                    delayed = compose(config_name="agent_ppo_trainer", overrides=[*delayed_args[4:], verl_config_override()])
                with contextlib.redirect_stdout(io.StringIO()):
                    preflight_checks(delayed)
                self.assertEqual(delayed.rllm.ace.training_mode, "delayed_joint")
                self.assertEqual((delayed.actor_rollout_ref.rollout.n, delayed.rllm.ace.g_ref, delayed.rllm.ace.g_cur), (8, 1, 1))
                self.assertNotEqual(delayed.trainer.experiment_name, standard.trainer.experiment_name)
                self.assertNotEqual(delayed.trainer.default_local_dir, standard.trainer.default_local_dir)
                self.assertIn("mini_swe_bench_delayed_joint_", delayed.trainer.experiment_name)
                self.assertIn("_rollout8_aligned11_seed_oneeval_best_rejection_update", delayed.trainer.experiment_name)
                self.assertEqual(standard.rllm.ace.reflection_source_selection, "random")
                self.assertEqual(delayed.rllm.ace.reflection_source_selection, "min_reward")
                self.assertEqual(standard.rllm.ace.auxiliary_update_gate, "strict_improvement")
                self.assertEqual(delayed.rllm.ace.auxiliary_update_gate, "none")
                self.assertEqual((standard.rllm.agent.max_steps, standard.rllm.env.env_args.max_steps, standard.rllm.ace.max_turns), (20, 20, 20))
                self.assertEqual((delayed.rllm.agent.max_steps, delayed.rllm.env.env_args.max_steps, delayed.rllm.ace.max_turns), (10, 10, 10))
                # Delayed reflection uses every chain, so its source setting is
                # inactive. Its historical auxiliary objective and turn default
                # also stay unchanged; all other comparison controls match.
                for key in standard.rllm.ace:
                    if key not in {"training_mode", "g_ref", "g_cur", "local_task_artifact_root", "reflection_source_selection", "auxiliary_update_gate", "max_turns"}:
                        self.assertEqual(delayed.rllm.ace[key], standard.rllm.ace[key], key)
                for key in ("data", "actor_rollout_ref", "algorithm", "rllm.rejection_sample", "rllm.disable_thinking", "rllm.accumulate_reasoning"):
                    standard_value = OmegaConf.select(standard, key)
                    delayed_value = OmegaConf.select(delayed, key)
                    if OmegaConf.is_config(standard_value):
                        # Trace paths interpolate the intentionally different run
                        # identity; compare the declared scientific settings.
                        standard_value = OmegaConf.to_container(standard_value, resolve=False)
                        delayed_value = OmegaConf.to_container(delayed_value, resolve=False)
                    if key == "actor_rollout_ref":
                        # Each preset's policy group size is part of its name.
                        self.assertEqual((standard_value["rollout"].pop("n"), delayed_value["rollout"].pop("n")), (8, 8))
                    self.assertEqual(delayed_value, standard_value, key)

    def test_explicit_experiment_name_is_already_complete(self):
        argv = dry_run("ace_rl", environ={"EXPERIMENT_NAME": "existing_run"})
        self.assertIn("trainer.experiment_name=existing_run", argv)

    def test_unsupported_training_modes_fail_before_runtime(self):
        for mode in ("iterative", "unknown"):
            with self.subTest(mode=mode):
                with self.assertRaises(subprocess.CalledProcessError) as failure:
                    dry_run("ace_rl", environ={"TRAINING_MODE": mode, "VENV_DIR": "/does/not/exist"})
                self.assertIn("TRAINING_MODE must be standard or delayed_joint", failure.exception.stderr)
                config = OmegaConf.create({"rllm": {"ace": {"training_mode": mode}}})
                with patch("examples.mini_swe_bench.train.ace_rl.train.load_datasets") as load:
                    with self.assertRaisesRegex(ValueError, "Unsupported MiniSWEBench ACE training mode"):
                        train_local_ace_rl.__wrapped__(config)
                    load.assert_not_called()

    def test_runtime_is_not_loaded_in_dry_run_and_invalid_variant_fails(self):
        # Even an unusable runtime environment must not be touched in dry-run.
        with patch.dict(os.environ, {"VENV_DIR": "/does/not/exist", "GPU_FAMILY": "invalid"}):
            self.assertIn("algorithm.adv_estimator=grpo", dry_run("ace_rl"))
        with self.assertRaises(subprocess.CalledProcessError):
            dry_run("ace_rl", "unsupported")

    def test_short_token_budget_fails_preflight(self):
        cfg = OmegaConf.create(
            {
                "data": {"max_prompt_length": 8192, "max_response_length": 32768},
                "actor_rollout_ref": {"actor": {"ppo_max_token_len_per_gpu": 8192}},
            }
        )
        with self.assertRaisesRegex(ValueError, "ppo_max_token_len_per_gpu"):
            preflight_checks(cfg)


class AdapterTests(unittest.TestCase):
    def test_prompt_survives_reset_and_preserves_swe_tool_protocol(self):
        agent = MiniSweACEAgent()
        original = agent.system_prompt
        agent.set_ace_playbook_prompt(template="{{playbook}}", playbook="first tip", rendered_prompt="first tip")
        agent.reset()
        self.assertIn(original, agent.chat_completions[0]["content"])
        self.assertIn("first tip", agent.chat_completions[0]["content"])
        agent.update_from_env("Repair simplify()", 0, False, {"max_steps": 30})
        action = agent.update_from_model("<function=execute_bash><parameter=command>git diff</parameter></function>")
        self.assertIn("execute_bash", action.action)
        agent.set_ace_playbook_prompt(template="{{playbook}}", playbook="second tip", rendered_prompt="second tip")
        agent.reset()
        self.assertIn("second tip", agent.chat_completions[0]["content"])
        self.assertNotIn("first tip", agent.chat_completions[0]["content"])
        self.assertEqual(len(agent.trajectory.steps), 0)

    def test_terminal_rewards_preserve_workflow_semantics(self):
        counts = {"ftp_passed": 2, "ftp_total": 4, "ptp_passed": 9, "ptp_total": 10}
        for mode, expected in [("shaped", 0.2), ("binary", 0), ("frac", 0.56), ("frac2", 0.45), ("frac3", 0.4)]:
            with self.subTest(mode=mode):
                env = MiniSweACEEnv(task=task(), reward_mode=mode)
                env._last_test_counts = counts
                with patch.object(env, "_get_git_diff", return_value="diff --git a/foo b/foo"), patch.object(env, "_compute_reward", return_value=(0, "tests_failed")):
                    _, reward, done, info = env.step("<function=submit></function>")
                self.assertTrue(done)
                self.assertAlmostEqual(reward, expected)
                self.assertFalse(info["all_passed"])
                # The workflow receives precisely the same raw final-step info.
                workflow = object.__new__(MiniSweWorkflow)
                workflow.configure_reward(mode)
                trajectory = Trajectory(steps=[Step(metadata={"git_diff": "diff --git a/foo b/foo", "reward_reason": "tests_failed", "test_counts": counts})])
                workflow.compute_trajectory_reward(trajectory)
                self.assertAlmostEqual(trajectory.reward, expected)
                episode = Episode(trajectories=[trajectory])
                with patch.dict(os.environ, {"MINISW_CORRECT_FLAG": "resolved"}):
                    workflow.assign_episode_correctness(episode)
                self.assertFalse(episode.is_correct)

    def test_no_patch_resolved_and_max_steps(self):
        for diff, reason, expected in [("", "no_patch", 0), ("patch", "tests_passed", 1), ("patch", "grading_harness_unavailable", 0.05)]:
            with self.subTest(reason=reason):
                env = MiniSweACEEnv(task=task(), max_steps=1)
                with patch.object(env, "_get_git_diff", return_value=diff), patch.object(env, "_compute_reward", return_value=(expected, reason)):
                    _, reward, done, info = env.step("<function=unknown></function>")
                self.assertTrue(done)
                self.assertEqual(reward, expected)
                self.assertEqual(info["all_passed"], expected == 1)

    def test_container_failures_are_retryable_and_cleanup_is_scoped(self):
        with tempfile.TemporaryDirectory() as root:
            env = MiniSweACEEnv(task=task())
            testbed = Path(root) / "testbed"
            overlay = Path(root) / "overlay"
            unrelated = Path(root) / "other-job"
            for path in (testbed, overlay, unrelated):
                path.mkdir()
            env._host_testbed, env._overlay_dir = str(testbed), str(overlay)
            env.close()
            self.assertFalse(testbed.exists())
            self.assertFalse(overlay.exists())
            self.assertTrue(unrelated.exists())
        with patch("examples.mini_swe_bench.env.run_apptainer_exec", side_effect=RuntimeError("container creation failed")):
            with self.assertRaisesRegex(RuntimeError, "container creation failed"):
                env.reset()
        self.assertIsNone(env._host_testbed)
        self.assertIsNone(env._overlay_dir)
        self.assertTrue(env.is_retryable_error(RuntimeError("container creation failed")))
        with patch.object(MiniSWEnv, "step", return_value=("pull failure", 0, True, {"infra_failed": True})):
            with self.assertRaisesRegex(RuntimeError, "container backend failed"):
                env.step("submit")

    def test_reflector_does_not_call_partial_credit_success(self):
        trajectory = SimpleNamespace(task=task(), info={"all_passed": False}, steps=[Step(observation="repository", action="inspect()", model_response="inspect()", reward=0.2)])
        values = MiniSweReflectorAdapter(max_turns=30).get_template_variables(trajectory, 0.2, "initial")
        self.assertEqual(values["outcome"], "FAILED")
        self.assertEqual(values["problem_statement"], "Repair simplify()")
        self.assertIn("inspect()", values["conversation_history"])
        self.assertEqual(values["playbook"], "initial")

    def test_packaged_candidate_cloning_keeps_distinct_task_playbooks(self):
        trainer = candidate_controller()
        trainer.config = OmegaConf.create({"rllm": {"agent": {"agent_args": {}}, "env": {"env_args": {}}}})
        trainer.agent_args = {}
        trainer.env_args = {"reward_mode": "frac3", "max_steps": 30}
        trainer.agent_class = MiniSweACEAgent
        trainer.env_class = MiniSweACEEnv
        trainer.candidate_eval_rollouts = 1
        trainer.base_system_prompt_template = (TRAIN / "ace_rl/prompts/base_system_prompt.txt").read_text()
        first = MiniSweACEEnv.from_dict(task("sympy__sympy-1"))
        second = MiniSweACEEnv.from_dict(task("sympy__sympy-2"))
        trainer.agent_execution_engine = SimpleNamespace(envs=[first, second, first])
        self.assertEqual(trainer._extract_batch_task_ids(), [first.task_id, second.task_id])
        candidates = [
            (0, {"task_id": first.task_id, "guidebook_uid": "first", "playbook": Playbook("[1] Inspect alpha.")}),
            (1, {"task_id": second.task_id, "guidebook_uid": "second", "playbook": Playbook("[1] Inspect beta.")}),
        ]
        envs, agents, specs = trainer._build_local_candidate_eval_envs_and_agents(candidates, {first.task_id: [first] * 8, second.task_id: [second] * 8})
        self.assertEqual(len(envs), 2)  # one evaluation per candidate
        self.assertEqual([env.task_id for env in envs], [first.task_id, second.task_id])
        self.assertTrue(all(env.reward_mode == "frac3" for env in envs))
        envs[0].task["instance"]["problem_statement"] = "changed clone"
        self.assertEqual(first.task["instance"]["problem_statement"], "Repair simplify()")
        for agent, tip, spec, source in zip(agents, ("alpha", "beta"), specs, (first, second), strict=True):
            agent.reset()
            self.assertIn(tip, agent.chat_completions[0]["content"])
            self.assertEqual(spec["task_id"], source.task_id)

    def test_task_runner_uses_packaged_controller_and_benchmark_assets(self):
        path = TRAIN / "ace_rl/train.py"
        tree = ast.parse(path.read_text())
        factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_task_runner")
        cls = next(node for node in factory.body if isinstance(node, ast.ClassDef))
        self.assertEqual([ast.unparse(base) for base in cls.bases], ["TaskRunner"])
        self.assertEqual([ast.unparse(decorator) for decorator in cls.decorator_list], ["ray.remote(num_cpus=1)"])
        # Exercise the production run method, replacing only actor registration
        # and the backend dispatcher. No backend modules enter sys.modules.
        cls.bases = []
        cls.decorator_list = []
        bind = Mock(return_value="bound")
        namespace = {
            "run_local_ace_rl_task": bind,
            "ACE_DEFAULTS": ACE_DEFAULTS,
            "_resolve_trainer_class_path": _resolve_trainer_class_path,
            "AGENT_CLASS": AGENT_CLASS,
            "ENV_CLASS": ENV_CLASS,
        }
        execute_source([cls], path, namespace)
        runner = namespace[cls.name]()
        configs = (
            ({}, "rllm.algorithms.ace.trainer.LocalACERLAgentPPOTrainer"),
            ({"rllm": {"ace": {"training_mode": "standard"}}}, "rllm.algorithms.ace.trainer.LocalACERLAgentPPOTrainer"),
            ({"rllm": {"ace": {"training_mode": "delayed_joint"}}}, "rllm.algorithms.ace.trainer_delayed_joint.LocalACERLDelayedJointAgentPPOTrainer"),
        )
        for config, trainer_path in configs:
            with self.subTest(config=config):
                bind.reset_mock()
                config = OmegaConf.create(config)
                self.assertEqual(runner.run(config), "bound")
                bind.assert_called_once()
                self.assertEqual(bind.call_args.kwargs["trainer_class_path"], trainer_path)
                self.assertIs(bind.call_args.kwargs["config"], config)
                self.assertEqual(bind.call_args.kwargs["default_agent_class_path"], AGENT_CLASS)
                self.assertEqual(bind.call_args.kwargs["default_env_class_path"], ENV_CLASS)
                self.assertIs(bind.call_args.kwargs["ace_defaults"], ACE_DEFAULTS)
        bind.reset_mock()
        with self.assertRaisesRegex(ValueError, "Unsupported MiniSWEBench ACE training mode"):
            runner.run(OmegaConf.create({"rllm": {"ace": {"training_mode": "iterative"}}}))
        bind.assert_not_called()
        self.assertTrue(Path(ACE_DEFAULTS["initial_playbook"]).is_file())
        self.assertTrue(Path(ACE_DEFAULTS["prompts_dir"]).is_dir())


class DatasetTests(unittest.TestCase):
    def test_offset_without_cap_and_empty_selections(self):
        self.assertEqual(select_rows([0, 1, 2], split="train", environ={"TRAIN_TASK_OFFSET": "1"}), [1, 2])
        for env in ({"TRAIN_TASK_OFFSET": "3"}, {"MAX_TRAIN_TASKS": "0"}, {"TRAIN_TASK_OFFSET": "-1"}):
            with self.assertRaises(ValueError):
                select_rows([0, 1, 2], split="train", environ=env)

    def test_same_size_changed_data_gets_fresh_registry_snapshot(self):
        # Test the real source fingerprint + selection + registry-name logic.
        # File decoding and persistent registry writes are isolated I/O adapters;
        # this does not claim to validate pandas/PyArrow's parquet implementation.
        snapshots = {}

        def read_rows(path):
            return Dataset(json.loads(Path(path).read_text()))

        def register(name, rows, split):
            snapshots[name, split] = Dataset(copy.deepcopy(rows), name=name, split=split)

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            rows = [task("sympy__sympy-1"), task("sympy__sympy-2")]
            for split in ("train", "validation"):
                (root / f"{split}.parquet").write_text(json.dumps(rows))
            with (
                patch.object(Dataset, "load_data", side_effect=read_rows),
                patch.object(DatasetRegistry, "dataset_exists", side_effect=lambda name, split: (name, split) in snapshots),
                patch.object(DatasetRegistry, "register_dataset", side_effect=register),
                patch.object(DatasetRegistry, "load_dataset", side_effect=lambda name, split: snapshots.get((name, split))),
                patch.dict(os.environ, {"MINISW_DATA_DIR": str(root), "MAX_TRAIN_TASKS": "", "TRAIN_TASK_OFFSET": "1", "MAX_VAL_TASKS": "1"}),
            ):
                first, val = load_datasets()
                self.assertEqual(len(first), 1)
                self.assertEqual(len(val), 1)
                rows[1]["instance"]["problem_statement"] = "new issue content"
                (root / "train.parquet").write_text(json.dumps(rows))
                second, same_val = load_datasets()
                self.assertNotEqual(first.name, second.name)
                self.assertEqual(first.get_data()[0]["instance"]["problem_statement"], "Repair simplify()")
                self.assertEqual(second.get_data()[0]["instance"]["problem_statement"], "new issue content")
                self.assertIs(same_val, val)
                self.assertEqual(len(snapshots), 3)


if __name__ == "__main__":
    unittest.main()
