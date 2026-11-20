"""CPU chain orchestration tests using production methods and real DataProto tensors.

Only model sampling, benchmark environments, and GPU-only trainer imports are
replaced. The full chain controller, parsers, playbook mutations, transform,
metadata expansion, and token accounting execute their production code.
"""

import ast
import copy
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_mixed_advantages import compute_advantage_with_overrides, extract, verl_math  # noqa: F401
from verl import DataProto

from rllm.algorithms.ace.context.playbook import Playbook

ROOT = Path(__file__).resolve().parents[2]


def production_definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    definitions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == set(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), path, "exec"), namespace)


class LocalBase:
    def _init_ace_components(self):
        pass

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        return "ace_validation", meta_info


NAMESPACE = {"json": json, "np": np, "torch": torch, "os": os, "copy": copy, "logger": logging.getLogger(__name__), "DataProto": DataProto, "Playbook": Playbook}
production_definitions(
    "rllm/algorithms/ace/trainer_base.py",
    {
        "_parse_json_response",
        "_coerce_reflection_payload",
        "_concatenate_insights",
        "_get_playbook_token_length",
        "_truncate_response_text",
        "_get_candidate_eval_source_envs",
        "_parse_candidate_eval_rollouts",
        "_compute_rollout_token_totals",
        "_render_template",
        "_build_curator_prompt",
        "_get_policy_task_for_trajectory",
        "_get_policy_task_instruction",
    },
    NAMESPACE,
)
production_definitions(
    "rllm/algorithms/ace/trainer.py",
    {
        "_task_id_from_env",
        "_build_task_index_groups",
        "_build_local_candidate_eval_envs_and_agents",
        "_prepare_task_playbooks_for_step",
        "_get_task_playbook",
        "_task_flush_payload",
        "_add_task_flush_metrics",
    },
    NAMESPACE,
)
production_definitions("rllm/trainer/verl/agent_ppo_trainer.py", {"_transform_agent_trajectories", "_compute_training_advantages"}, NAMESPACE)
for name, method in list(NAMESPACE.items()):
    if callable(method) and name.startswith("_"):
        setattr(LocalBase, name, method)
# Production static helpers retain their method descriptors.
for name in ("_compute_rollout_token_totals", "_parse_candidate_eval_rollouts"):
    setattr(LocalBase, name, staticmethod(NAMESPACE[name]))

NAMESPACE["LocalACERLAgentPPOTrainer"] = LocalBase
module = ast.parse((ROOT / "rllm/algorithms/ppo/trainer_joint.py").read_text())
module.body = [node for node in module.body if not (isinstance(node, ast.ImportFrom) and node.module in {"rllm.algorithms.ace.trainer", "rllm.algorithms.ace.trainer_base"})]
exec(compile(ast.fix_missing_locations(module), "rllm/algorithms/ppo/trainer_joint.py", "exec"), NAMESPACE)
Trainer = NAMESPACE["LocalACEJointPPOTrainer"]


class Env:
    def __init__(self, task):
        self.task = task
        self.task_id = task["task_id"]

    @classmethod
    def from_dict(cls, task):
        return cls(task)


class Agent:
    pass


class Engine:
    def __init__(self, envs, agents):
        self.envs, self.agents = envs, agents

    def update_envs_and_agents(self, envs, agents):
        self.envs, self.agents = envs, agents


def make_trainer(tmp_path, *, task_count=2, group_size=1, curator_response=None, reflector_response=None, max_playbook_length=None):
    trainer = object.__new__(Trainer)
    trainer.config = OmegaConf.create({
        "actor_rollout_ref": {"rollout": {"n": group_size}},
        "rllm": {"rejection_sample": {"enable": False}, "stepwise_advantage": {"enable": False}},
        "data": {"max_prompt_length": 6, "max_response_length": 7},
        "algorithm": {"adv_estimator": "gae", "gamma": 1.0, "lam": 1.0, "norm_adv_by_std_in_grpo": True},
        "critic": {"enable": True},
        "trainer": {"default_local_dir": str(tmp_path)},
    })
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda text, **kwargs: text.split())
    trainer.g_ref = trainer.g_cur = trainer.candidate_eval_rollouts = 1
    trainer._init_ace_components()
    trainer._artifact_task_id = None
    trainer._artifact_task_ids = []
    trainer._suppress_inner_flush = False
    trainer.local_train_playbooks = {}
    trainer.local_eval_playbooks = {}
    trainer.local_train_task_seen_counts = {}
    trainer._current_task_seen_counts = {}
    trainer._current_task_flush_info = {}
    trainer.initial_playbook_text = "[b00001] Original guidance."
    trainer.flush_playbook_every_n_steps = 3
    trainer.max_playbook_length = max_playbook_length
    trainer.global_steps = 4
    trainer._current_epoch = 1
    trainer._epoch_task_data = []
    trainer._estimate_epoch = lambda: 1
    trainer._policy_train_source_metric_id = lambda source: 2
    trainer._keep_rollout_engine_awake = lambda **kwargs: nullcontext()
    trainer._set_component_token_limits = lambda component: setattr(trainer, "active_component", component)
    trainer.reflector_system_prompt = "Reflect"
    trainer.curator_system_prompt = "Curate"
    trainer._build_reflector_prompt_from_dict = lambda row, playbook, index: f"Reflect on {row['source']} with {playbook}"
    trainer._build_curator_prompt = lambda insight, playbook, *, task_instr="": f"Curate {insight} with {playbook} for {task_instr}"
    trainer._decode_trajectory_response = lambda row: row["text"]
    trainer._get_full_env_args = lambda: {}
    trainer._get_full_agent_args = lambda: {}
    trainer.env_class, trainer.agent_class = Env, Agent
    trainer._inject_playbook_into_agents = lambda agents, text: [setattr(agent, "playbook", text) for agent in agents]

    def inject(task_indices, **kwargs):
        for task_id, indices in task_indices.items():
            text = trainer._get_task_playbook(task_id, is_eval=False).to_string()
            trainer._inject_playbook_into_agents([trainer.agent_execution_engine.agents[index] for index in indices], text)

    trainer._inject_task_playbooks_into_agents = inject
    trainer.agent_execution_engine = Engine([Env({"task_id": f"task-{i}", "value": i}) for i in range(task_count)], [Agent() for _ in range(task_count)])
    trainer._current_batch = DataProto.from_dict(non_tensors={
        "uid": np.array([f"uid-{i}" for i in range(task_count)], dtype=object),
        "data_source": np.array([f"source-{i}" for i in range(task_count)], dtype=object),
        "extra_info": np.array([{"task_id": f"task-{i}"} for i in range(task_count)], dtype=object),
        "raw_prompt_ids": np.array([[10 + i, 20 + i] for i in range(task_count)], dtype=object),
    })
    if group_size > 1:
        trainer.agent_execution_engine = Engine([Env({"task_id": f"task-{i}", "value": i}) for i in range(task_count) for _ in range(group_size)], [Agent() for _ in range(task_count * group_size)])
        trainer._current_batch = trainer._current_batch.repeat(group_size, interleave=True)
    trainer._save_task_policy_chat_logs = lambda grouped, **kwargs: setattr(trainer, "logged_policy", copy.deepcopy(grouped)) or {}
    trainer._save_local_snapshot = lambda *args, **kwargs: "snapshot"
    trainer._artifact_mode = lambda meta: "train"
    trainer._save_task_ace_logs = lambda payloads, **kwargs: setattr(trainer, "payloads", payloads)
    trainer.visualize_trajectory = lambda batch: None
    trainer.samples = []
    trainer.generated = []
    original_transform = trainer._transform_agent_trajectories

    def transform(rows):
        trainer.training_rows = rows
        return original_transform(rows)

    trainer._transform_agent_trajectories = transform
    curator_response = curator_response if curator_response is not None else json.dumps({"operations": [{"type": "ADD", "content": "New downstream guidance."}]})
    trainer.curator_response = curator_response
    reflector_response = reflector_response if reflector_response is not None else '{"key_insight": "Try another approach"}'

    def sample(**kwargs):
        engine = trainer.agent_execution_engine
        phase = ["initial", "reflector", "curator", "downstream"][len(trainer.samples)]
        trainer.samples.append((phase, list(engine.envs), list(engine.agents), kwargs))
        phase_number = len(trainer.samples)
        for index in reversed(range(len(engine.envs))):
            response_length = phase_number + 1
            reward = (1.0 - index) if phase == "initial" else float(index)
            text = "attempt"
            if phase == "reflector":
                text = reflector_response
            elif phase == "curator":
                text = trainer.curator_response(index) if callable(trainer.curator_response) else trainer.curator_response
            row = {
                "idx": index,
                "source": f"{phase}-{index}",
                "trajectory_reward": reward,
                "step_rewards": [reward],
                "prompt_tokens": torch.tensor([10 + index, 11 + index]),
                "response_tokens": torch.arange(1, response_length + 1),
                "response_masks": torch.tensor([1] + [0] * (response_length - 2) + [1]),
                "chat_completions": [{"role": "assistant", "content": f"{phase}-{index}"}],
                "text": text,
                "metrics": {"total_prompt_tokens": 2, "total_completion_tokens": response_length, "total_gradient_update_tokens": response_length},
            }
            trainer.generated.append(row)
            yield row

    trainer.generate_agent_trajectories_async = sample
    return trainer


def test_complete_chains_preserve_all_segments_ownership_rewards_and_masks(tmp_path):
    trainer = make_trainer(tmp_path)
    original_envs = list(trainer.agent_execution_engine.envs)
    output, metrics = trainer.generate_agent_trajectory()
    batch = trainer._current_batch.union(output)

    assert [call[0] for call in trainer.samples] == ["initial", "reflector", "curator", "downstream"]
    assert all(len(call[1]) == 2 for call in trainer.samples)
    assert len(batch) == 8
    assert batch.non_tensor_batch["data_source"].tolist() == ["source-0", "source-1"] * 4
    assert batch.non_tensor_batch["extra_info"].tolist() == [{"task_id": "task-0"}, {"task_id": "task-1"}] * 4
    assert batch.non_tensor_batch["chain_id"].tolist() == ["uid-0", "uid-1"] * 4
    assert batch.non_tensor_batch["chain_segment"].tolist() == [segment for segment in ("initial_actor", "reflector", "curator", "downstream_actor") for _ in range(2)]
    assert len(set(batch.non_tensor_batch["uid"])) == 8
    assert torch.equal(batch.batch["token_level_scores"].sum(-1), torch.tensor([0.0, 1.0] * 4))
    assert batch.batch["response_mask"].sum(-1).tolist() == [2] * 8
    assert [row["source"] for row in trainer.training_rows] == [f"{phase}-{index}" for phase in ("initial", "reflector", "curator", "downstream") for index in range(2)]
    assert all(row["trajectory_reward"] == float(index % 2) for index, row in enumerate(trainer.training_rows))
    assert trainer.generated[0]["trajectory_reward"] == 0.0  # initial task 1, unchanged
    assert trainer.generated[1]["trajectory_reward"] == 1.0  # initial task 0, unchanged
    assert trainer.logged_policy["task-0"][0]["trajectory_reward"] == 1.0
    assert trainer.logged_policy["task-0"][1]["trajectory_reward"] == 0.0
    assert "New downstream guidance." in trainer.local_train_playbooks["task-0"].to_string()
    downstream_envs, downstream_agents = trainer.samples[-1][1:3]
    assert all(env is not original for env, original in zip(downstream_envs, original_envs, strict=True))
    assert all("New downstream guidance." in agent.playbook for agent in downstream_agents)
    assert trainer.agent_execution_engine.envs == original_envs
    assert trainer.active_component == "policy"
    assert trainer.local_train_task_seen_counts == {"task-0": 1, "task-1": 1}
    assert trainer._current_step_rewards_list == [[0.0], [1.0]]
    assert trainer._current_step_reward_tasks == ["task-0", "task-1"]
    assert trainer.payloads["task-0"]["chain"]["terminal_reward"] == 0.0
    assert trainer.payloads["task-0"]["chain"]["initial_reward"] == 1.0
    assert metrics["rollout/total_trajectories"] == 8
    assert metrics["rollout/total_prompt_tokens"] == 16
    assert metrics["rollout/total_completion_tokens"] == 28
    assert sum(row["metrics"]["total_gradient_update_tokens"] for row in trainer.generated) == 28
    assert metrics["rollout/total_gradient_update_tokens"] == 16
    assert metrics["joint_ppo/playbook_updates"] == 2


def test_joint_chain_renders_original_task_in_actual_appworld_curator_prompt(tmp_path):
    trainer = make_trainer(tmp_path, task_count=2, group_size=2)
    del trainer._build_curator_prompt
    trainer.curator_prompt_template = (ROOT / "examples/appworld/train/ace_rl/prompts/curator_prompt_handoff_v1.txt").read_text()
    for index, agent in enumerate(trainer.agent_execution_engine.agents):
        agent._trajectory = SimpleNamespace(task={"instruction": f"Original task for chain {index}."})
    trainer.generate_agent_trajectory()
    curator_envs = trainer.samples[2][1]
    assert len(curator_envs) == 4
    for index, env in enumerate(curator_envs):
        assert f"Original task for chain {index}." in env.prompt_text
        assert "{{task_instr}}" not in env.prompt_text
        assert "Original guidance." in env.prompt_text


def test_actor_reward_metrics_keep_fractional_seed_rewards_before_terminal_rewriting(tmp_path):
    trainer = make_trainer(tmp_path)
    sample = trainer.generate_agent_trajectories_async

    def distinct_rewards(**kwargs):
        for row in sample(**kwargs):
            reward = [0.0, 0.25][row["idx"]] if row["source"].startswith("initial") else 1.0
            row["trajectory_reward"] = reward
            row["step_rewards"] = [reward]
            yield row

    trainer.generate_agent_trajectories_async = distinct_rewards
    output, metrics = trainer.generate_agent_trajectory()
    assert output.batch["token_level_scores"].sum().item() == 8.0
    assert metrics["train/actor_reward_sum"] == 0.25
    assert metrics["train/actor_reward_count"] == 2
    assert metrics["train/actor_reward_invalid_count"] == 0


@pytest.mark.parametrize(("curation", "limit", "status"), [
    ("not valid JSON", None, "invalid_operations"),
    ('{"operations": []}', None, "no_op"),
    ('{"operations": [{"type": "ADD", "content": "too many new words"}]}', 1, "max_playbook_length"),
    ('{"operations": [{"type": "ADD", "content": "partial mutation"}, {"type": 99}]}', None, "invalid_operations"),
])
def test_invalid_or_noop_curation_still_trains_every_generated_segment(tmp_path, curation, limit, status):
    trainer = make_trainer(tmp_path, task_count=1, curator_response=curation, reflector_response="raw malformed reflection", max_playbook_length=limit)
    output, metrics = trainer.generate_agent_trajectory()
    assert len(output) == 4
    assert len(trainer.samples) == 4
    assert "raw malformed reflection" in trainer.payloads["task-0"]["curator"][0]["prompt_text"]
    assert trainer.payloads["task-0"]["curator"][0]["response_text"] == curation
    assert trainer.payloads["task-0"]["curator"][0]["update_status"] == status
    assert trainer.local_train_playbooks["task-0"].to_string() == trainer.initial_playbook_text
    assert trainer.samples[-1][2][0].playbook == trainer.initial_playbook_text
    assert metrics["joint_ppo/playbook_updates"] == 0
    assert len(trainer.training_rows) == 4
    assert all(row["trajectory_reward"] == 0 for row in trainer.training_rows)


@pytest.mark.parametrize("bad_indices", [[], [0, 0], [0, 2]])
def test_missing_duplicate_or_unknown_rollout_indices_fail_without_training(tmp_path, bad_indices):
    trainer = make_trainer(tmp_path)
    envs = list(trainer.agent_execution_engine.envs)
    agents = list(trainer.agent_execution_engine.agents)
    trainer.generate_agent_trajectories_async = lambda **kwargs: iter({"idx": index} for index in bad_indices)
    with pytest.raises(RuntimeError, match="expected one trajectory"):
        trainer.generate_agent_trajectory()
    assert not hasattr(trainer, "training_rows")
    assert trainer.agent_execution_engine.envs == envs
    assert trainer.agent_execution_engine.agents == agents
    assert trainer._artifact_task_id is None
    assert trainer._artifact_task_ids == []
    assert trainer._suppress_inner_flush is False
    assert trainer.active_component == "policy"


def test_empty_reflection_keeps_curator_and_downstream_segments(tmp_path):
    trainer = make_trainer(tmp_path, task_count=1, reflector_response="")
    output, _ = trainer.generate_agent_trajectory()
    assert len(output) == 4
    assert "no usable insight" in trainer.payloads["task-0"]["curator"][0]["prompt_text"]
    assert trainer.payloads["task-0"]["reflector"][0]["response_text"] == ""
    assert [call[0] for call in trainer.samples] == ["initial", "reflector", "curator", "downstream"]


def test_single_task_takes_same_complete_chain_path(tmp_path):
    trainer = make_trainer(tmp_path, task_count=1)
    output, _ = trainer.generate_agent_trajectory()
    assert len(output) == 4
    assert trainer._current_batch.non_tensor_batch["chain_id"].tolist() == ["uid-0"] * 4


def test_validation_preserves_existing_ace_protocol(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.config.actor_rollout_ref.rollout.n = 8
    assert trainer.generate_agent_trajectory(meta_info={"validate": True}) == ("ace_validation", {"validate": True})
    assert not trainer.samples


@pytest.mark.parametrize("invalid_setting", ["n", "g_ref", "g_cur", "candidate_eval_rollouts", "rejection", "stepwise", "gae", "critic", "use_critic", "truncation", "overlong"])
def test_chain_contract_rejects_grouping_filtering_or_incompatible_advantages(tmp_path, invalid_setting):
    trainer = make_trainer(tmp_path)
    if invalid_setting == "n":
        trainer.config.actor_rollout_ref.rollout.n = 0
    elif invalid_setting == "rejection":
        trainer.config.rllm.rejection_sample.enable = True
    elif invalid_setting == "stepwise":
        trainer.config.rllm.stepwise_advantage.enable = True
    elif invalid_setting == "gae":
        trainer.config.algorithm.adv_estimator = "grpo"
    elif invalid_setting == "critic":
        trainer.config.critic.enable = False
    elif invalid_setting == "use_critic":
        trainer.use_critic = False
    elif invalid_setting == "truncation":
        trainer.config.rllm.mask_truncated_samples = True
    elif invalid_setting == "overlong":
        trainer.config.rllm.agent = {"overlong_filter": True}
    else:
        setattr(trainer, invalid_setting, 8)
    with pytest.raises(ValueError, match="Joint PPO requires"):
        trainer._init_ace_components()


@pytest.mark.parametrize("curation", [
    "malformed curation",
    '{"operations": []}',
    '{"operations": [{"type": "ADD", "content": "partial mutation"}, {"type": 99}]}',
])
def test_chain_preserves_restored_playbook_mode_and_next_id_after_no_update(tmp_path, curation):
    trainer = make_trainer(tmp_path, task_count=1, curator_response=curation)
    trainer.flush_playbook_every_n_steps = 0
    restored = Playbook("", flat_mode=False)
    restored.next_global_id = 42
    trainer.local_train_playbooks["task-0"] = restored
    original_batch = copy.deepcopy(trainer._current_batch)
    trainer.generate_agent_trajectory()
    stored = trainer.local_train_playbooks["task-0"]
    assert stored.to_string() == ""
    assert stored.flat_mode is False
    assert stored.next_global_id == 42
    assert trainer.playbook.flat_mode is False
    assert trainer.playbook.next_global_id == 42
    assert restored.next_global_id == 42
    assert restored.to_string() == ""

    trainer.samples.clear()
    trainer._current_batch = copy.deepcopy(original_batch)
    trainer.curator_response = '{"operations": [{"type": "ADD", "section": "others", "content": "new guidance"}]}'
    trainer.generate_agent_trajectory()
    assert "[misc-00042]" in trainer.local_train_playbooks["task-0"].to_string()
    assert trainer.local_train_playbooks["task-0"].next_global_id == 43


def test_removing_highest_bullet_preserves_next_id_for_next_chain(tmp_path):
    trainer = make_trainer(tmp_path, task_count=1, curator_response='{"operations": [{"type": "REMOVE", "bullet_id": "misc-00042"}]}')
    trainer.flush_playbook_every_n_steps = 0
    trainer.local_train_playbooks["task-0"] = Playbook("## OTHERS\n[misc-00042] Old guidance", flat_mode=False)
    original_batch = copy.deepcopy(trainer._current_batch)
    trainer.generate_agent_trajectory()
    assert trainer.local_train_playbooks["task-0"].next_global_id == 43
    assert "[misc-00042]" not in trainer.local_train_playbooks["task-0"].to_string()

    trainer.samples.clear()
    trainer._current_batch = copy.deepcopy(original_batch)
    trainer.curator_response = '{"operations": [{"type": "ADD", "section": "others", "content": "replacement guidance"}]}'
    trainer.generate_agent_trajectory()
    assert "[misc-00043]" in trainer.local_train_playbooks["task-0"].to_string()
    assert trainer.local_train_playbooks["task-0"].next_global_id == 44


@pytest.mark.parametrize("group_size", [1, 8])
def test_collected_chain_batch_uses_critic_gae_without_uid_group_normalization(tmp_path, verl_math, group_size):
    trainer = make_trainer(tmp_path, group_size=group_size)
    output, _ = trainer.generate_agent_trajectory()
    batch = trainer._current_batch.union(output)
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    mask = batch.batch["response_mask"].bool()
    # Observation values deliberately differ wildly; action values are .25.
    batch.batch["values"] = torch.where(mask, 0.25, 999.0)
    source = Path(os.environ["VERL_RESUME_SOURCE"])
    math_namespace = {
        "core_algos": SimpleNamespace(**verl_math),
        "AdvantageEstimator": SimpleNamespace(GAE="gae", GRPO="grpo"),
    }
    extract(source / "verl/trainer/ppo/ray_trainer.py", {"compute_advantage"}, math_namespace)
    NAMESPACE["compute_advantage"] = math_namespace["compute_advantage"]
    NAMESPACE["compute_advantage_with_overrides"] = compute_advantage_with_overrides
    result = trainer._compute_training_advantages(batch)
    torch.testing.assert_close(result.batch["returns"][mask], torch.tensor([float(i) for _ in range(4) for i in range(2 * group_size) for _ in range(2)]))
    row_advantages = (result.batch["advantages"] * mask).sum(-1)
    assert (row_advantages.reshape(4, 2 * group_size)[:, 0] < 0).all()
    assert (row_advantages.reshape(4, 2 * group_size)[:, -1] > 0).all()

    # UID changes cannot change PPO advantages; a GRPO path would normalize
    # each singleton UID to zero or center the newly combined group differently.
    expected = result.batch["advantages"].clone()
    result.non_tensor_batch["uid"] = np.array(["same-task"] * len(result), dtype=object)
    shared_uid_result = trainer._compute_training_advantages(result)
    torch.testing.assert_close(shared_uid_result.batch["advantages"], expected)


def test_eight_chains_preserve_branch_rewards_metadata_and_private_playbooks(tmp_path):
    def curation(index):
        return json.dumps({"operations": [{"type": "ADD", "content": f"Branch {index} guidance."}]})

    trainer = make_trainer(tmp_path, group_size=8, curator_response=curation)
    original_envs = list(trainer.agent_execution_engine.envs)
    output, metrics = trainer.generate_agent_trajectory()
    batch = trainer._current_batch.union(output)
    assert len(batch) == 64  # two tasks x eight chains x four segments
    assert all(len(call[1]) == 16 for call in trainer.samples)
    assert len(set(batch.non_tensor_batch["chain_id"])) == 16
    assert len(set(batch.non_tensor_batch["uid"])) == 64
    assert batch.non_tensor_batch["data_source"].tolist() == (["source-0"] * 8 + ["source-1"] * 8) * 4
    assert batch.batch["token_level_scores"].sum(-1).tolist() == list(map(float, range(16))) * 4
    for index, agent in enumerate(trainer.samples[-1][2]):
        assert f"Branch {index} guidance." in agent.playbook
        assert sum(f"Branch {other} guidance." in agent.playbook for other in range(16)) == 1
    for task, first in (("task-0", 0), ("task-1", 8)):
        payload = trainer.payloads[task]
        assert len(payload["chains"]) == len(payload["reflector"]) == len(payload["curator"]) == 8
        assert [chain["terminal_reward"] for chain in payload["chains"]] == list(map(float, range(first, first + 8)))
        assert all(len(chain["segments"]) == 4 for chain in payload["chains"])
        assert payload["persisted_chain_id"] == payload["chains"][0]["id"]
        assert f"Branch {first} guidance." in trainer.local_train_playbooks[task].to_string()
        assert len(trainer.logged_policy[task]) == 16
    assert trainer.local_train_task_seen_counts == {"task-0": 1, "task-1": 1}
    assert trainer._current_step_reward_tasks == ["task-0"] * 8 + ["task-1"] * 8
    assert metrics["joint_ppo/chains"] == 16
    assert metrics["joint_ppo/chains_per_task"] == 8
    assert metrics["local/task_batch_size"] == 2
    assert metrics["rollout/total_gradient_update_tokens"] == 128
    assert trainer.agent_execution_engine.envs == original_envs


def test_group_size_must_match_actual_parent_rollouts(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.config.actor_rollout_ref.rollout.n = 8
    with pytest.raises(ValueError, match="exactly rollout.n"):
        trainer.generate_agent_trajectory()
    assert not trainer.samples


def test_grouped_phase_failure_restores_original_engine(tmp_path):
    trainer = make_trainer(tmp_path, group_size=8)
    original_envs = trainer.agent_execution_engine.envs
    original_agents = trainer.agent_execution_engine.agents
    sample = trainer.generate_agent_trajectories_async

    def incomplete(**kwargs):
        rows = list(sample(**kwargs))
        return iter(rows[:-1] if len(trainer.samples) == 2 else rows)

    trainer.generate_agent_trajectories_async = incomplete
    with pytest.raises(RuntimeError, match="expected one trajectory"):
        trainer.generate_agent_trajectory()
    assert trainer.agent_execution_engine.envs == original_envs
    assert trainer.agent_execution_engine.agents == original_agents
    assert trainer._artifact_task_ids == []
    assert trainer._suppress_inner_flush is False
    assert not hasattr(trainer, "training_rows")
