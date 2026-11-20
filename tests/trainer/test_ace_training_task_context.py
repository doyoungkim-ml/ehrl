"""Execute production training prompt builders and producers without GPU imports."""

import ast
import copy
import json
import logging
import random
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from test_ace_package_boundaries import _benchmark_defaults, _initialize
from test_standard_rejection_update import execute, generation

from rllm.algorithms.ace.agents import SingleTurnCuratorAgent, SingleTurnReflectorAgent
from rllm.algorithms.ace.context.playbook import Playbook
from rllm.algorithms.ace.environments import SingleTurnCuratorEnv, SingleTurnReflectorEnv

ROOT = Path(__file__).resolve().parents[2]
BASE_METHODS = {
    "_build_curator_prompt", "_get_policy_task_for_trajectory", "_get_policy_task_instruction",
    "_trajectory_steps_from_chat", "_build_reflector_prompt_from_dict", "_make_guidebook_uid",
    "_uses_single_guidebook_aggregation", "_should_use_singlemode_curator_one_group",
    "_expand_curator_prompt_specs_for_sampling",
}
HELPERS = {"_render_template", "_concatenate_insights", "_parse_json_response", "_coerce_reflection_payload", "_truncate_response_text"}


def bind_production(instance, relative_path, names, namespace):
    tree = ast.parse((ROOT / relative_path).read_text())
    definitions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    execute(definitions, namespace)
    for definition in definitions:
        if definition.args.args and definition.args.args[0].arg == "self":
            setattr(instance, definition.name, MethodType(namespace[definition.name], instance))


def training_fixture():
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split(), decode=lambda tokens, **kwargs: " ".join(tokens))
    defaults = _benchmark_defaults("appworld")
    defaults["handoff_protocol"] = "feedback_v1"
    defaults["max_turns"] = 20
    instance = _initialize(defaults, tokenizer, preserve_terminal_feedback=True)
    namespace = {"SimpleNamespace": SimpleNamespace, "logger": logging.getLogger(__name__), "json": json, "random": random}
    bind_production(instance, "rllm/algorithms/ace/trainer_base.py", BASE_METHODS | HELPERS, namespace)
    # Original source indices differ from logical task indices and group order.
    instance._original_envs = [SimpleNamespace(task={"task_id": f"task-{index % 2}", "metadata": {"index": index}}) for index in range(4)]
    instance._original_agents = [SimpleNamespace(_trajectory=SimpleNamespace(task={"instruction": f"Original instruction for source {index}."})) for index in range(4)]
    instance.agent_execution_engine = SimpleNamespace(envs=[], agents=[], rollout_engine=SimpleNamespace())
    instance.component_token_limits["reflector"]["max_prompt_length"] = 100000
    instance.singlemode_enabled = False
    instance.guidebook_aggregation_level = "single"
    return instance, namespace


def policy_row(index):
    return {"idx": index, "trajectory_reward": 0.0, "step_rewards": [0.0], "chat_completions": [{"role": "user", "content": "attempt observation"}, {"role": "assistant", "content": "attempt action"}]}


def assert_task_prompt(prompt, index):
    assert f"Original instruction for source {index}." in prompt
    for other in range(4):
        if other != index:
            assert f"Original instruction for source {other}." not in prompt
    assert "{{task_instr}}" not in prompt and "{{ task_instr }}" not in prompt


@pytest.mark.parametrize("index", [0, 2, 3])
def test_actual_training_role_inputs_include_original_task_and_playbook(index):
    instance, _ = training_fixture()
    before = copy.deepcopy([env.task for env in instance._original_envs])
    row = policy_row(index)
    reflector = instance._build_reflector_prompt_from_dict(row, "UNIQUE_CURRENT_PLAYBOOK", task_idx=0)
    curator = instance._build_curator_prompt("UNIQUE_INSIGHT", "UNIQUE_CURRENT_PLAYBOOK", task_instr=instance._get_policy_task_instruction(row, 0))
    for prompt in (reflector, curator):
        assert_task_prompt(prompt, index)
        assert "UNIQUE_CURRENT_PLAYBOOK" in prompt
    assert "UNIQUE_INSIGHT" in curator
    assert [env.task for env in instance._original_envs] == before


def test_instruction_attribute_fallback_preserves_environment_payload_and_identity():
    instance, _ = training_fixture()
    env = instance._original_envs[3]
    original = env.task
    instance._original_agents[3] = SimpleNamespace(task_instruction="Original instruction for source 3.")
    task = instance._get_policy_task_for_trajectory(policy_row(3), 0)
    assert task == {**original, "instruction": "Original instruction for source 3."}
    assert task is not original
    assert env.task is original and "instruction" not in original
    assert task["metadata"] is original["metadata"]


def test_environment_and_task_index_fallback_remain_supported():
    instance, _ = training_fixture()
    instance._original_agents = []
    instance._original_envs[1].task["instruction"] = "env original"
    assert instance._get_policy_task_for_trajectory({"idx": 99}, 1) is instance._original_envs[1].task
    assert instance._get_policy_task_instruction({}, 1) == "env original"
    assert instance._get_policy_task_for_trajectory({}, 99) == {}


def test_legacy_curator_template_and_positional_history_keep_their_contract():
    instance, _ = training_fixture()
    instance.curator_prompt_template = "{{guidebook}}|{{current_playbook}}|{{conversation_history}}"
    assert instance._build_curator_prompt("insight", "playbook", "history") == "insight|playbook|history"
    assert instance._build_curator_prompt("insight", "playbook", "history", task_instr="task") == "insight|playbook|history"


@pytest.mark.parametrize("kind", ["standard", "base"])
@pytest.mark.parametrize("aggregation", ["single", "task"])
def test_standard_training_curator_producers_deliver_source_task(kind, aggregation):
    instance, namespace = training_fixture()
    instance.guidebook_aggregation_level = aggregation
    groups = [[policy_row(2), policy_row(0)], [policy_row(3), policy_row(1)]]
    reflections = [[[{"key_insight": f"insight-{task}-{source}"}] for source in range(2)] for task in range(2)]
    loop = next(
        node for node in ast.walk(generation(kind))
        if isinstance(node, ast.For) and "task_curator_specs = []" in ast.unparse(node)
        and ast.unparse(node.iter) in {"enumerate(task_groups)", "enumerate(task_ids)"}
    )
    namespace.update(self=instance, task_groups=groups, task_ids=["task-0", "task-1"],
                     grouped_policy_trajectories=dict(zip(["task-0", "task-1"], groups, strict=True)),
                     playbook_before_text_by_task={"task-0": "PLAYBOOK_A", "task-1": "PLAYBOOK_B"},
                     playbook_text="PLAYBOOK_SHARED", effective_g_ref=1, effective_g_cur=1, curator_prompts=[])
    if kind == "base":
        namespace.update(reflections_by_task=reflections, reflection_source_indices_by_task=[[0, 1], [0, 1]])
    else:
        namespace.update(reflections_by_task=dict(zip(["task-0", "task-1"], reflections, strict=True)), reflection_source_indices_by_task={"task-0": [0, 1], "task-1": [0, 1]})
    execute([loop], namespace)
    prompts = namespace["curator_prompts"]
    assert len(prompts) == (4 if aggregation == "single" else 2)
    for spec in prompts:
        source = groups[spec["task_idx"]][spec.get("traj_idx", 0)]["idx"]
        assert_task_prompt(spec["prompt_text"], source)


def test_local_delayed_training_curator_producer_delivers_source_task():
    instance, namespace = training_fixture()
    instance._build_delayed_joint_insight_text = lambda record: "Recorded insight"
    groups = {"task-0": [policy_row(2), policy_row(0)], "task-1": [policy_row(3), policy_row(1)]}
    records = [{"task_id": task, "task_idx": task_idx, "task_uid": task, "traj_idx": traj_idx} for task_idx, task in enumerate(groups) for traj_idx in range(2)]
    loop = next(
        node for node in ast.walk(generation("delayed_joint"))
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "enumerate(reflector_records)" and "curator_prompts.append" in ast.unparse(node)
    )
    namespace.update(self=instance, reflector_records=records, grouped_seed_policy=groups,
                     playbook_before_text_by_task={task: "CURRENT_PLAYBOOK" for task in groups}, curator_prompts=[])
    execute([loop], namespace)
    assert len(namespace["curator_prompts"]) == 4
    for spec in namespace["curator_prompts"]:
        assert_task_prompt(spec["prompt_text"], groups[spec["task_id"]][spec["traj_idx"]]["idx"])


def test_base_delayed_training_collectors_carry_original_task_through_role_swaps():
    instance, namespace = training_fixture()
    namespace.update(SingleTurnCuratorAgent=SingleTurnCuratorAgent, SingleTurnReflectorAgent=SingleTurnReflectorAgent,
                     SingleTurnCuratorEnv=SingleTurnCuratorEnv, SingleTurnReflectorEnv=SingleTurnReflectorEnv, Playbook=Playbook)
    bind_production(instance, "rllm/algorithms/ace/trainer_delayed_joint_base.py", {"_collect_rollout_aligned_reflector_rollouts", "_collect_rollout_aligned_curator_rollouts"}, namespace)
    engine = instance.agent_execution_engine
    engine.update_envs_and_agents = lambda envs, agents: (setattr(engine, "envs", envs), setattr(engine, "agents", agents))
    instance._set_component_token_limits = lambda component: setattr(instance, "component", component)
    instance._decode_trajectory_response = lambda row: row["text"]
    sampled = {}

    def sample(**kwargs):
        sampled[instance.component] = [env.prompt_text for env in engine.envs]
        if instance.component == "curator":
            return []
        return [{"idx": index, "text": '{"key_insight": "Recorded insight"}'} for index in range(len(engine.envs))]

    instance._collect_token_rollouts = sample
    groups = [[policy_row(2), policy_row(0)], [policy_row(3), policy_row(1)]]
    _, records = instance._collect_rollout_aligned_reflector_rollouts(groups, ["task-0", "task-1"], "CURRENT_PLAYBOOK", timing_raw={})
    assert len(records) == 4
    instance._collect_rollout_aligned_curator_rollouts(records, "CURRENT_PLAYBOOK", timing_raw={})
    for component in ("reflector", "curator"):
        assert len(sampled[component]) == 4
        for prompt, index in zip(sampled[component], [2, 0, 3, 1], strict=True):
            assert_task_prompt(prompt, index)
    assert engine.envs is instance._original_envs and engine.agents is instance._original_agents
