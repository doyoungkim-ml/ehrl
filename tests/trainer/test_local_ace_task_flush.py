import numpy as np
import pytest

from rllm.algorithms.ace import trainer_base as ace_rl_trainer_base
from rllm.algorithms.ace.context.playbook import Playbook
from rllm.algorithms.ace.trainer import (
    LocalACERLAgentPPOTrainer,
)
from rllm.algorithms.ace.trainer_base import ACERLAgentPPOTrainer


def _trainer_with_flush_interval(interval: int = 3):
    trainer = object.__new__(LocalACERLAgentPPOTrainer)
    trainer.local_train_playbooks = {}
    trainer.local_eval_playbooks = {}
    trainer.local_train_task_seen_counts = {}
    trainer._current_task_seen_counts = {}
    trainer._current_task_flush_info = {}
    trainer.flush_playbook_every_n_steps = interval
    trainer.initial_playbook_text = "initial playbook"
    trainer.global_steps = 1
    return trainer


def test_task_playbook_flush_counts_training_task_sightings():
    trainer = _trainer_with_flush_interval(interval=3)
    trainer.local_train_playbooks["task_a"] = Playbook("learned task playbook")

    assert trainer._prepare_task_playbooks_for_step(["task_a"], is_eval=False) == {}
    assert trainer.local_train_task_seen_counts["task_a"] == 1
    assert trainer.local_train_playbooks["task_a"].to_string() == "learned task playbook"

    assert trainer._prepare_task_playbooks_for_step(["task_a"], is_eval=False) == {}
    assert trainer.local_train_task_seen_counts["task_a"] == 2
    assert trainer.local_train_playbooks["task_a"].to_string() == "learned task playbook"

    flushed = trainer._prepare_task_playbooks_for_step(["task_a"], is_eval=False)

    assert flushed["task_a"]["task_seen_count"] == 3
    assert flushed["task_a"]["interval"] == 3
    assert trainer.local_train_task_seen_counts["task_a"] == 3
    assert trainer.local_train_playbooks["task_a"].to_string() == "initial playbook"


def test_task_playbook_flush_ignores_validation_sightings():
    trainer = _trainer_with_flush_interval(interval=1)
    trainer.local_train_playbooks["task_a"] = Playbook("learned task playbook")

    assert trainer._prepare_task_playbooks_for_step(["task_a"], is_eval=True) == {}

    assert trainer.local_train_task_seen_counts == {}
    assert trainer.local_train_playbooks["task_a"].to_string() == "learned task playbook"


def test_task_flush_metrics_report_interval_counts_and_flushes():
    trainer = _trainer_with_flush_interval(interval=2)
    trainer._prepare_task_playbooks_for_step(["task_a", "task_b"], is_eval=False)
    trainer._prepare_task_playbooks_for_step(["task_a"], is_eval=False)

    metrics = {}
    trainer._add_task_flush_metrics(metrics, ["task_a", "task_b"], is_eval=False)

    assert metrics["local/task_playbook_flush_interval"] == 2
    assert metrics["local/task_playbook_flushes"] == 1
    assert metrics["local/task_seen_count_mean"] == 1.5
    assert metrics["local/task_seen_count_max"] == 2.0


def _trainer_with_policy_train_source(source: str):
    trainer = object.__new__(ACERLAgentPPOTrainer)
    trainer.policy_train_source = source
    return trainer


def test_policy_train_source_seed_uses_seed_rollouts_only():
    trainer = _trainer_with_policy_train_source("seed")
    seed_rollouts = [{"idx": 0, "source": "seed"}]
    selected_rollouts = [{"idx": 0, "source": "selected"}]

    rows, source = trainer._select_policy_trajectories_for_training(
        seed_rollouts,
        selected_rollouts,
        expected_policy_count=1,
        log_prefix="[TEST]",
    )

    assert source == "seed"
    assert [row["source"] for row in rows] == ["seed"]
    assert rows[0] is not seed_rollouts[0]


def test_policy_train_source_selected_candidate_uses_updated_playbook_rollouts():
    trainer = _trainer_with_policy_train_source("selected_candidate")
    seed_rollouts = [{"idx": 0, "source": "seed"}]
    selected_rollouts = [{"idx": 0, "source": "selected"}]

    rows, source = trainer._select_policy_trajectories_for_training(
        seed_rollouts,
        selected_rollouts,
        expected_policy_count=1,
        log_prefix="[TEST]",
    )

    assert source == "selected_candidate"
    assert [row["source"] for row in rows] == ["selected"]
    assert rows[0] is not selected_rollouts[0]


def test_policy_train_source_both_keeps_seed_and_updated_playbook_rollouts():
    trainer = _trainer_with_policy_train_source("both")
    seed_rollouts = [{"idx": 0, "source": "seed"}]
    selected_rollouts = [{"idx": 0, "source": "selected"}]

    rows, source = trainer._select_policy_trajectories_for_training(
        seed_rollouts,
        selected_rollouts,
        expected_policy_count=1,
        log_prefix="[TEST]",
    )

    assert source == "both"
    assert [row["source"] for row in rows] == ["seed", "selected"]


def test_policy_train_source_controls_candidate_eval_capture():
    assert not _trainer_with_policy_train_source("seed")._should_capture_selected_candidate_policy_trajectories()
    assert _trainer_with_policy_train_source("selected_candidate")._should_capture_selected_candidate_policy_trajectories()
    assert _trainer_with_policy_train_source("both")._should_capture_selected_candidate_policy_trajectories()


def test_policy_train_source_seed_and_candidates_captures_rollouts_and_rejects_shared_selector():
    trainer = _trainer_with_policy_train_source("seed_and_candidates")

    assert trainer._should_capture_selected_candidate_policy_trajectories()
    assert trainer._policy_train_source_metric_id() == 4
    with pytest.raises(ValueError, match="local standard ACE-RL trainer"):
        trainer._select_policy_trajectories_for_training([{"idx": 0}], [{"idx": 0}], expected_policy_count=1)


def test_rollout_completion_totals_include_unretained_candidate_evaluations_but_gradient_totals_do_not():
    totals = LocalACERLAgentPPOTrainer._compute_rollout_token_totals(
        policy_trajectories=[
            {
                "metrics": {
                    "total_prompt_tokens": 10,
                    "total_completion_tokens": 4,
                    "total_gradient_update_tokens": 3,
                }
            }
        ],
        reflector_trajectories=[],
        curator_trajectories=[],
        # This is all that seed-policy training retains for candidate scoring.
        candidate_eval_token_metrics=[
            {
                "total_prompt_tokens": 12,
                "total_completion_tokens": 5,
                "total_gradient_update_tokens": 4,
            },
            {
                "total_prompt_tokens": 14,
                "total_completion_tokens": 6,
                "total_gradient_update_tokens": 5,
            },
        ],
    )

    assert totals == {
        "rollout/total_prompt_tokens": 36,
        "rollout/total_completion_tokens": 15,
        "rollout/total_gradient_update_tokens": 3,
        "rollout/total_trajectories": 3,
    }


def test_rollout_gradient_tokens_follow_explicit_ppo_rows():
    totals = LocalACERLAgentPPOTrainer._compute_rollout_token_totals(
        policy_trajectories=[{"metrics": {"total_completion_tokens": 4, "total_gradient_update_tokens": 3}}],
        candidate_eval_token_metrics=[{"total_completion_tokens": 5, "total_gradient_update_tokens": 4}],
        ppo_trajectories=[{"metrics": {"total_gradient_update_tokens": 7}}],
    )

    assert totals["rollout/total_completion_tokens"] == 9
    assert totals["rollout/total_gradient_update_tokens"] == 7


def test_keep_rollout_engine_awake_with_free_cache_style_lifecycle():
    class RolloutEngine:
        def __init__(self):
            self.wake_calls = 0
            self.sleep_calls = 0

        async def wake_up(self):
            self.wake_calls += 1

        async def sleep(self):
            self.sleep_calls += 1

    rollout_engine = RolloutEngine()
    trainer = object.__new__(LocalACERLAgentPPOTrainer)
    trainer.agent_execution_engine = type(
        "ExecutionEngine",
        (),
        {"engine_name": "verl", "rollout_engine": rollout_engine},
    )()
    trainer._manage_rollout_engine_state_default = None
    trainer._rollout_engine_external_management_depth = 0

    with trainer._keep_rollout_engine_awake("test"):
        assert rollout_engine.wake_calls == 1
        assert rollout_engine.sleep_calls == 0
        assert trainer._rollout_engine_external_management_depth == 1
        assert trainer._manage_rollout_engine_state_default is False

    assert rollout_engine.sleep_calls == 1
    assert trainer._rollout_engine_external_management_depth == 0
    assert trainer._manage_rollout_engine_state_default is None


def test_curator_sampling_keeps_all_groups_by_default():
    trainer = object.__new__(ACERLAgentPPOTrainer)
    trainer.singlemode_enabled = True
    trainer.singlemode_curator_one_group = False

    specs = [{"guidebook_uid": f"gb_{idx}"} for idx in range(3)]
    expanded = trainer._expand_curator_prompt_specs_for_sampling(specs, effective_g_cur=4)

    assert len(expanded) == 12
    assert [row["guidebook_uid"] for row in expanded].count("gb_0") == 4
    assert [row["guidebook_uid"] for row in expanded].count("gb_1") == 4
    assert [row["guidebook_uid"] for row in expanded].count("gb_2") == 4
    assert all(row["curator_grouped"] for row in expanded)
    assert {row["curator_group_sampling_mode"] for row in expanded} == {"all_groups"}


def test_singlemode_curator_sampling_uses_one_group_and_one_shot_others(monkeypatch):
    trainer = object.__new__(ACERLAgentPPOTrainer)
    trainer.singlemode_enabled = True
    trainer.singlemode_curator_one_group = True
    monkeypatch.setattr(ace_rl_trainer_base.random, "randrange", lambda upper: 1)

    specs = [{"guidebook_uid": f"gb_{idx}"} for idx in range(4)]
    expanded = trainer._expand_curator_prompt_specs_for_sampling(specs, effective_g_cur=4)

    guidebook_counts = {uid: 0 for uid in ["gb_0", "gb_1", "gb_2", "gb_3"]}
    grouped_by_uid = {uid: [] for uid in guidebook_counts}
    for row in expanded:
        guidebook_counts[row["guidebook_uid"]] += 1
        grouped_by_uid[row["guidebook_uid"]].append(row["curator_grouped"])

    assert guidebook_counts == {"gb_0": 1, "gb_1": 4, "gb_2": 1, "gb_3": 1}
    assert grouped_by_uid["gb_1"] == [True, True, True, True]
    assert grouped_by_uid["gb_0"] == [False]
    assert grouped_by_uid["gb_2"] == [False]
    assert grouped_by_uid["gb_3"] == [False]
    assert {row["curator_group_sampling_mode"] for row in expanded} == {"single_group_per_task"}


def test_non_seed_policy_train_source_disables_iterative_policy_only_fast_path():
    seed_trainer = _trainer_with_policy_train_source("seed")
    seed_trainer.policy_only_phase_enabled = True
    selected_trainer = _trainer_with_policy_train_source("selected_candidate")
    selected_trainer.policy_only_phase_enabled = True
    both_trainer = _trainer_with_policy_train_source("both")
    both_trainer.policy_only_phase_enabled = True

    assert seed_trainer._should_run_policy_only_phase("policy")
    assert not selected_trainer._should_run_policy_only_phase("policy")
    assert not both_trainer._should_run_policy_only_phase("policy")


def test_flatten_grouped_trajectories_preserves_both_policy_source_rows():
    trainer = object.__new__(LocalACERLAgentPPOTrainer)
    grouped = {
        "task_a": [
            {"source": "task_a_seed"},
            {"source": "task_a_selected"},
        ],
        "task_b": [
            {"source": "task_b_seed"},
            {"source": "task_b_selected"},
        ],
    }
    task_to_indices = {"task_a": [0], "task_b": [1]}

    flattened = trainer._flatten_grouped_trajectories_in_env_order(grouped, task_to_indices)

    assert [row["source"] for row in flattened] == [
        "task_a_seed",
        "task_b_seed",
        "task_a_selected",
        "task_b_selected",
    ]
    assert [row["idx"] for row in flattened] == [0, 1, 2, 3]


def test_batch_metadata_expansion_preserves_object_vector_shape_for_both_source():
    trainer = _trainer_with_policy_train_source("both")
    object_column = np.empty((2,), dtype=object)
    object_column[0] = np.array([1, 2], dtype=np.int64)
    object_column[1] = np.array([3, 4], dtype=np.int64)

    class FakeBatch:
        pass

    batch = FakeBatch()
    batch.non_tensor_batch = {
        "uid": np.array(["u0", "u1"], dtype=object),
        "token_ids": object_column,
        "matrix": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "score": np.array([0.5, 0.75], dtype=np.float32),
    }
    batch.batch = None

    trainer._append_policy_and_component_rows_to_batch(
        batch,
        extra_policy_count=2,
        component_extra_uids=["ref::0", "cur::0"],
    )

    assert batch.non_tensor_batch["uid"].shape == (6,)
    assert batch.non_tensor_batch["token_ids"].shape == (6,)
    assert batch.non_tensor_batch["matrix"].shape == (6, 2)
    assert batch.non_tensor_batch["score"].shape == (6,)
    assert batch.non_tensor_batch["uid"].tolist() == [
        "u0",
        "u1",
        "u0::policy_selected_0",
        "u1::policy_selected_1",
        "ref::0",
        "cur::0",
    ]
    assert np.array_equal(batch.non_tensor_batch["token_ids"][2], object_column[0])
    assert np.array_equal(batch.non_tensor_batch["token_ids"][3], object_column[1])
    assert batch.batch.batch_size == (6,)


def test_flatten_base_indices_name_the_seed_row_of_uneven_candidate_rows():
    trainer = object.__new__(LocalACERLAgentPPOTrainer)
    grouped = {
        "task_a": [{"source": "a_seed0"}, {"source": "a_seed1"}, {"source": "a_cand0"}, {"source": "a_cand1"}, {"source": "a_cand2"}],
        "task_b": [{"source": "b_seed0"}, {"source": "b_seed1"}],
    }
    task_to_indices = {"task_a": [0, 1], "task_b": [2, 3]}

    flattened = trainer._flatten_grouped_trajectories_in_env_order(grouped, task_to_indices)
    base_indices = trainer._flatten_grouped_base_indices_in_env_order(grouped, task_to_indices)

    assert [row["source"] for row in flattened] == ["a_seed0", "a_seed1", "b_seed0", "b_seed1", "a_cand0", "a_cand1", "a_cand2"]
    assert base_indices == [0, 1, 2, 3, 0, 1, 0]


def test_batch_metadata_expansion_shares_seed_group_uid_for_candidate_rows():
    trainer = _trainer_with_policy_train_source("seed_and_candidates")

    class FakeBatch:
        pass

    batch = FakeBatch()
    batch.non_tensor_batch = {
        "uid": np.array(["task_a", "task_a", "task_b", "task_b"], dtype=object),
        "extra_info": np.array(["a", "a", "b", "b"], dtype=object),
    }
    batch.batch = None

    trainer._append_policy_and_component_rows_to_batch(
        batch,
        extra_policy_count=3,
        component_extra_uids=["ref::0"],
        extra_policy_base_indices=[0, 1, 2],
        share_extra_policy_uids=True,
    )

    assert batch.non_tensor_batch["uid"].tolist() == ["task_a", "task_a", "task_b", "task_b", "task_a", "task_a", "task_b", "ref::0"]
    assert batch.non_tensor_batch["extra_info"].tolist() == ["a", "a", "b", "b", "a", "a", "b", ""]

    with pytest.raises(ValueError, match="extra policy base indices"):
        trainer._append_policy_and_component_rows_to_batch(batch, extra_policy_count=2, extra_policy_base_indices=[0])


class _CandidateEvalSourceEnv:
    def __init__(self, idx: int):
        self.task = {"idx": idx}


class _CandidateEvalEnv:
    @classmethod
    def from_dict(cls, payload):
        return dict(payload)


class _CandidateEvalAgent:
    pass


def _candidate_eval_trainer(trainer_cls, candidate_eval_rollouts: int):
    trainer = object.__new__(trainer_cls)
    trainer.candidate_eval_rollouts = candidate_eval_rollouts
    trainer.env_class = _CandidateEvalEnv
    trainer.agent_class = _CandidateEvalAgent
    trainer._get_full_agent_args = lambda: {}
    trainer._get_full_env_args = lambda: {}
    trainer._inject_playbook_into_agents = lambda agents, playbook_text: None
    return trainer


def test_candidate_eval_rollouts_caps_global_candidate_eval_envs():
    trainer = _candidate_eval_trainer(ACERLAgentPPOTrainer, candidate_eval_rollouts=2)
    trainer._original_envs = [_CandidateEvalSourceEnv(idx) for idx in range(5)]
    candidate = {"guidebook_uid": "gb_0", "playbook": Playbook("updated playbook")}

    eval_envs, eval_agents, eval_specs = trainer._build_candidate_eval_envs_and_agents([(0, candidate)])

    assert [env["idx"] for env in eval_envs] == [0, 1]
    assert len(eval_agents) == 2
    assert [spec["rollout_idx"] for spec in eval_specs] == [0, 1]


def test_candidate_eval_rollouts_caps_local_candidate_eval_envs_per_task():
    trainer = _candidate_eval_trainer(LocalACERLAgentPPOTrainer, candidate_eval_rollouts=1)
    original_envs_by_task = {
        "task_a": [_CandidateEvalSourceEnv(idx) for idx in range(3)],
        "task_b": [_CandidateEvalSourceEnv(idx) for idx in range(10, 13)],
    }
    candidates = [
        (
            0,
            {
                "task_id": "task_a",
                "guidebook_uid": "gb_a",
                "playbook": Playbook("task a playbook"),
            },
        ),
        (
            1,
            {
                "task_id": "task_b",
                "guidebook_uid": "gb_b",
                "playbook": Playbook("task b playbook"),
            },
        ),
    ]

    eval_envs, eval_agents, eval_specs = trainer._build_local_candidate_eval_envs_and_agents(
        candidates,
        original_envs_by_task=original_envs_by_task,
    )

    assert [env["idx"] for env in eval_envs] == [0, 10]
    assert len(eval_agents) == 2
    assert [(spec["candidate_idx"], spec["task_id"], spec["rollout_idx"]) for spec in eval_specs] == [
        (0, "task_a", 0),
        (1, "task_b", 0),
    ]
