"""Original ERL control flow and real sampled-token transport on CPU."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rllm.algorithms.erl.agent import PromptOverrideAgent
from rllm.algorithms.erl.controller import ERLController, UpdaterPromptOverflow
from rllm.algorithms.erl.replay import make_internalization_copy, pack_replay_tensors, verified_replay_reward
from rllm.algorithms.erl.upstream import extract_prompt_from_response

ROOT = Path(__file__).resolve().parents[2]


def trainer_class(parent=object):
    """Execute the actual ERL module replacing only its distributed parent."""
    source = ROOT / "rllm/algorithms/erl/trainer.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == "rllm.trainer.verl.agent_ppo_trainer")]
    namespace = {"AgentPPOTrainer": parent}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["OriginalERLAgentPPOTrainer"]


def trajectory(reward, prompt=(1, 2), response=(151667, 198, 4, 5, 6), mask=(1, 1, 0, 1, 1), correct=None):
    return {
        "idx": 0,
        "trajectory_reward": reward,
        "prompt_tokens": torch.tensor(prompt),
        "response_tokens": torch.tensor(response),
        "response_masks": torch.tensor(mask),
        "chat_completions": [{"role": "system", "content": "plain policy"}, {"role": "assistant", "content": "action"}],
        "metrics": {"total_prompt_tokens": len(prompt), "total_completion_tokens": 4, "total_gradient_update_tokens": 4},
        "step_rewards": [reward],
        "all_passed": reward >= 1 if correct is None else correct,
        "old_log_probs": torch.ones(len(response)),
    }


def collect(controller, first_rewards, second_rewards, prompts=None):
    calls = []
    states = []

    async def attempt(index, prompt):
        calls.append((index, prompt))
        reward = first_rewards[index] if prompt is None else second_rewards[index]
        return trajectory(reward, prompt=(1, 2) if prompt is None else (7, 8, 9))

    async def updater(index, state, current):
        states.append((index, state, current))
        return (prompts[index] if prompts else f"improved-{index}"), trajectory(0)

    result = asyncio.run(controller.collect(len(first_rewards), attempt=attempt, update=updater, succeeded=lambda row: row["all_passed"]))
    return result, calls, states


def test_successful_plain_attempts_never_reflect_or_retry():
    controller = ERLController(2)
    controller.memories = ["remembered-a", "remembered-b"]
    result, calls, states = collect(controller, [1, 1], [])
    assert calls == [(0, None), (1, None)]
    assert states == [] and result.additional == []
    assert controller.memories == ["remembered-a", "remembered-b"]


def test_failure_triggered_retry_credit_and_positive_partial_raft():
    controller = ERLController(3)
    result, calls, states = collect(controller, [1, 0, 0.4], [None, 1, 0.6])
    assert calls == [(0, None), (1, None), (1, "improved-1"), (2, None), (2, "improved-2")]
    assert [item[0] for item in states] == [1, 2]
    assert len(result.generated) == 7  # copies are not model generation
    assert [kind for _, kind, _ in result.additional] == ["updater", "second_raw", "second_distilled"] * 2
    assert [row["trajectory_reward"] for _, _, row in result.additional] == [1, 1, 1, 0.6, 0.6, 0.6]
    assert verified_replay_reward(result.additional[-1][2]) == 0.6
    assert controller.memories == [None, "improved-1", None]
    for _, kind, row in result.additional:
        if kind == "second_distilled":
            assert row["prompt_tokens"].tolist() == [1, 2]
            assert row["response_tokens"].tolist() == [151667, 198, 4, 5, 6]
            assert row["response_masks"].tolist() == [1, 1, 0, 1, 1]
            assert "old_log_probs" not in row


def test_memory_conditions_next_updater_not_first_and_failed_retry_does_not_replace():
    controller = ERLController(1)
    collect(controller, [0], [1], ["retained prompt"])
    result, calls, states = collect(controller, [0.3], [-0.1], ["unsuccessful prompt"])
    assert calls[0] == (0, None)
    assert states[0][2] == "retained prompt"
    assert "retained prompt" in states[0][1]
    assert controller.memories == ["retained prompt"]
    assert result.additional[-1][2]["trajectory_reward"] == -0.1
    assert verified_replay_reward(result.additional[-1][2]) == 0


def test_fifo_pool_order_is_preserved_across_collections():
    controller = ERLController(3)
    controller.memories = ["a", "b", "c"]
    controller.slot_order = [2, 0, 1]
    _, _, states = collect(controller, [0], [0.2])
    assert states[0][2] == "c"
    assert controller.slot_order == [0, 1, 2]
    document = controller.state_dict()
    restored = ERLController(3)
    restored.load_state_dict(document)
    assert restored.state_dict() == document


@pytest.mark.parametrize("corruption", [{"memories": [None], "slot_order": [0, 0]}, {"memories": [3], "slot_order": [0]}, {"memories": [None], "slot_order": [True]}])
def test_bad_memory_state_does_not_mutate_controller(corruption):
    controller = ERLController(1)
    original = controller.state_dict()
    with pytest.raises(ValueError):
        controller.load_state_dict(corruption)
    assert controller.state_dict() == original


def test_pack_replay_preserves_masks_negative_rewards_and_rejects_truncation():
    row = trajectory(-0.2)
    tensors = pack_replay_tensors([row], max_prompt_length=8, max_response_length=8, pad_token_id=0)
    assert tensors["responses"][0, :5].tolist() == row["response_tokens"].tolist()
    assert tensors["response_mask"][0].tolist() == [1, 1, 0, 1, 1, 0, 0, 0]
    assert tensors["token_level_scores"][0, 4].item() == pytest.approx(-0.2)
    for prompt, response in ((1, 8), (8, 4)):
        with pytest.raises(ValueError, match="exceeds padded token widths"):
            pack_replay_tensors([row], max_prompt_length=prompt, max_response_length=response, pad_token_id=0)


def test_mixed_augmentation_keeps_task_roles_and_real_dataproto_rows():
    cls = trainer_class()
    subject = cls.__new__(cls)
    collection, _, _ = collect(ERLController(2), [0, 1], [0.6, None])
    subject._erl_pending = collection
    subject.record_prompt_width = subject.record_response_width = 10
    subject.tokenizer = SimpleNamespace(pad_token_id=0)
    first = subject._pack(collection.first)
    first.non_tensor_batch = {"uid": np.array(["same-task", "same-task"], dtype=object), "extra_info": np.array([{"id": "t"}, {"id": "t"}], dtype=object)}
    first.batch["old_log_probs"] = torch.ones(2, 10)
    first.batch["dummy_tensor"] = torch.tensor([[7], [9]], dtype=torch.uint8)
    mixed, metrics = subject._augment_accepted_training_batch(first)
    assert len(mixed) == 5
    assert mixed.non_tensor_batch["uid"].tolist() == ["first::same-task", "first::same-task", "updater::same-task", "second_raw::same-task", "second_distilled::same-task"]
    assert mixed.non_tensor_batch["adv_estimator"].tolist() == ["grpo"] * 4 + ["raft"]
    assert mixed.meta_info["mask_training_padding"] is True
    assert "old_log_probs" not in mixed.batch
    assert mixed.batch["dummy_tensor"].flatten().tolist() == [7, 9, 7, 7, 7]
    assert metrics["erl/positive_internalization_rows"] == 1
    assert torch.equal(mixed.batch["responses"][-1], mixed.batch["responses"][-2])
    assert torch.equal(mixed.batch["response_mask"][-1], mixed.batch["response_mask"][-2])
    assert not torch.equal(mixed.batch["prompts"][-1], mixed.batch["prompts"][-2])


def test_actor_reward_metrics_cover_first_attempts_before_retry_and_internalization():
    cls = trainer_class()
    subject = cls.__new__(cls)
    collection, _, _ = collect(ERLController(3), [0.0, 0.25, 1.0], [0.8, 0.9, None])

    async def sampled_collection(meta_info):
        return collection

    subject._collect = sampled_collection
    subject.agent_execution_engine = SimpleNamespace(envs=[None] * 3)
    subject.record_prompt_width = subject.record_response_width = 10
    subject.tokenizer = SimpleNamespace(pad_token_id=0)
    subject._success = None
    subject.erl_config = {}
    subject._log_collection = lambda collected: None
    output, metrics = subject.generate_agent_trajectory()
    assert output.batch["token_level_scores"].sum().item() == 1.25
    assert len(collection.additional) == 6
    assert len(collection.generated) == 7
    assert metrics["train/actor_reward_sum"] == 1.25
    assert metrics["train/actor_reward_count"] == 3
    assert metrics["train/actor_reward_invalid_count"] == 0


def test_prompt_override_changes_only_first_instruction_after_reset():
    from rllm.agents.mastermind_agent import MastermindAgent

    plain = MastermindAgent()
    wrapper = PromptOverrideAgent(plain, "revised instruction")
    wrapper.reset()
    wrapper.update_from_env({"code_length": 4}, 0, False, {})
    assert wrapper.chat_completions[0]["content"] == "revised instruction"
    assert wrapper.chat_completions[1]["role"] == "user"
    plain.reset()
    assert plain.chat_completions[0]["content"] == MastermindAgent.SYSTEM_PROMPT


def test_updater_extraction_matches_reference_fallback_contract():
    assert extract_prompt_from_response("reason\n<PROMPT> revised </PROMPT>") == "revised"
    assert extract_prompt_from_response("no wrapped prompt") is None


def test_plain_validation_never_enters_controller():
    class Parent:
        def generate_agent_trajectory(self, **kwargs):
            return "plain-only", kwargs

    cls = trainer_class(Parent)
    subject = cls.__new__(cls)
    subject._attempt_runner = None
    subject.erl_controller = SimpleNamespace(collect=lambda *args, **kwargs: pytest.fail("validation adapted memory"))
    output = subject.generate_agent_trajectory(meta_info={"validate": True})
    assert output[0] == "plain-only"


def test_overlong_updater_raises_overflow_without_generation():
    cls = trainer_class()
    subject = cls.__new__(cls)
    subject.global_steps = 1
    subject.updater_max_prompt = 3
    subject.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3, 4])
    subject.agent_execution_engine = SimpleNamespace(chat_parser=SimpleNamespace(parse=lambda *args, **kwargs: "input"), accumulate_reasoning=False)
    with pytest.raises(UpdaterPromptOverflow, match="updater prompt overflow"):
        asyncio.run(subject._update_prompt(0, "state", "current"))


def overflow_collection(controller):
    calls = []

    async def attempt(index, prompt):
        calls.append((index, prompt))
        return trajectory(0.0 if prompt is None else 1.0, prompt=(1, 2) if prompt is None else (7, 8, 9))

    async def updater(index, state, current):
        if index == 0:
            raise UpdaterPromptOverflow("too long")
        return f"improved-{index}", trajectory(0)

    result = asyncio.run(controller.collect(2, attempt=attempt, update=updater, succeeded=lambda row: row["all_passed"]))
    return result, calls


def test_updater_overflow_drops_only_that_rollout_and_keeps_memory():
    controller = ERLController(2)
    collection, calls = overflow_collection(controller)
    assert collection.dropped == [0]
    assert (0, None) in calls and not any(index == 0 and prompt is not None for index, prompt in calls)
    assert [index for index, _, _ in collection.additional] == [1, 1, 1]
    assert sorted(memory for memory in controller.memories if memory) == ["improved-1"]


def test_dropped_rollout_leaves_the_training_batch_with_its_first_attempt():
    cls = trainer_class()
    subject = cls.__new__(cls)
    collection, _ = overflow_collection(ERLController(2))
    subject._erl_pending = collection
    subject.record_prompt_width = subject.record_response_width = 10
    subject.tokenizer = SimpleNamespace(pad_token_id=0)
    first = subject._pack(collection.first)
    first.non_tensor_batch = {"uid": np.array(["a", "b"], dtype=object)}
    mixed, _ = subject._augment_accepted_training_batch(first)
    assert mixed.non_tensor_batch["uid"].tolist() == ["first::b", "updater::b", "second_raw::b", "second_distilled::b"]


def test_internalization_probabilities_change_with_prompt_while_actions_are_exact():
    sampled = trajectory(0.6, prompt=(7, 8))
    plain = make_internalization_copy(sampled, (1, 2))
    model = torch.nn.Embedding(10, 10)
    # A toy conditional policy makes the denominator distinction observable.
    contextual_logprobs = model(sampled["prompt_tokens"][-1]).log_softmax(-1)
    plain_logprobs = model(plain["prompt_tokens"][-1]).log_softmax(-1)
    assert not torch.equal(contextual_logprobs, plain_logprobs)
    assert torch.equal(sampled["response_tokens"], plain["response_tokens"])
    assert "old_log_probs" not in plain


@pytest.mark.parametrize("explicit", [False, True])
def test_generation_logs_stay_outside_checkpoint_directory(tmp_path, explicit):
    cls = trainer_class()
    subject = cls.__new__(cls)
    subject.global_steps = 1
    checkpoint_root = tmp_path / "run" / "checkpoints"
    episode_root = tmp_path / "custom_episodes" if explicit else tmp_path / "run" / "episodes"
    trainer_config = {"default_local_dir": str(checkpoint_root)}
    if explicit:
        trainer_config["episode_log_dir"] = str(episode_root)
    subject.config = OmegaConf.create({"trainer": trainer_config})
    collection, _, _ = collect(ERLController(1), [0], [0.6])
    subject._log_collection(collection)
    rows = [json.loads(line) for line in (episode_root / "erl" / "step_1.jsonl").read_text().splitlines()]
    assert [row["record_type"] for row in rows] == ["first", "updater", "second_raw", "second_distilled"]
    assert not checkpoint_root.exists()
