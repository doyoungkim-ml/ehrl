"""Matched ACE controls retain role work while isolating context application."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from test_local_evaluation_concurrency import Curator, Engine, Recorder, Reflector, Rule, setup_run

from rllm.algorithms.ace import evaluation
from rllm.utils.provider_seed import apply_provider_seed, provider_seed_scope, record_provider_request, record_provider_response, task_provider_seed
from scripts.evaluation_runtime import validate_sequential_metrics


def dispatch(payload):
    sampling = {}
    apply_provider_seed(sampling)
    record = record_provider_request(endpoint="completion", model="test-model", sampling_params=sampling, payload={"model": "test-model", **payload, **sampling})
    prompt, completion = {"actor": (11, 3), "reflector": (29, 5), "curator": (47, 7)}[payload.get("role", "actor")]
    record_provider_response(record, SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)))


class SeededEngine(Engine):
    async def execute_tasks(self, tasks):
        with provider_seed_scope(task_provider_seed(tasks[0]), call=0):
            dispatch({"prompt": self.agent_class._injected_context})
        return await super().execute_tasks(tasks)


class SeededReflector(Reflector):
    async def analyze(self, payload, before):
        dispatch({"prompt": before, "role": "reflector"})
        return await super().analyze(payload, before)


class SeededCurator(Curator):
    async def generate_operations(self, payload, insights, before):
        dispatch({"prompt": before, "role": "curator"})
        return await super().generate_operations(payload, insights, before)


def setup_control(path, mode, *, concurrency=1, visits=8):
    recorder = Recorder()
    kwargs = setup_run(path, recorder, concurrency=concurrency, tasks=[{"id": "alpha"}, {"id": "beta"}], engine_class=SeededEngine)
    args = kwargs["args"]
    args.adaptation_mode = mode
    args.provider_seed_scheme = "sha256_v1"
    args.provider_seed_namespace = "handoff-test/appworld"
    args.seed = 711
    args.local_eval_epochs = visits

    def create(*_args, **_kwargs):
        rule = Rule(recorder)
        rule.reflector = SeededReflector(recorder, rule)
        rule.curator = SeededCurator(recorder, rule)
        return rule

    kwargs["hooks"].create_context_rule = create
    return kwargs, recorder


def test_apply_discard_match_initial_requests_and_role_budgets(tmp_path):
    entries = {}
    for mode in ("apply", "discard"):
        kwargs, recorder = setup_control(tmp_path / mode, mode)
        entry = asyncio.run(evaluation.run_local_ace(**kwargs)).evaluations[0]
        entries[mode] = entry
        assert len(recorder.generator_calls) == 16
        assert len(recorder.reflector_calls) == len(recorder.curator_calls) == 14
        assert entry["rollout_counts"] == {"policy": 16, "reflector": 14, "curator": 14, "candidate": 0, "total": 44}
        for visit in entry["local_epochs"]:
            for row in visit["tasks"]:
                final = visit["local_eval_epoch"] == 8
                assert row["adaptation_generated"] is (not final)
                assert row["adaptation_applied"] is (mode == "apply" and not final)
                assert row["adaptation_changed"] == row["adaptation_applied"]
                assert len(row["provider_requests"]) == (1 if final else 3)
                if mode == "discard":
                    assert row["policy_playbook"] == ""
                    log = json.loads(open(row["artifacts"]["ace_log"]).read())
                    assert log["playbook"] == {"before": "", "after": ""}
                    if not final:
                        candidate = log["curator"]["candidates"][0]
                        assert candidate["operations"]["operations"]
                        assert candidate["candidate_playbook"]
                        assert candidate["discarded_by_control"] and not candidate["applied"]
        assert entry["actually_applied_context_updates"] == (14 if mode == "apply" else 0)
        assert entry["provider_usage"]["actor"]["completion_tokens"] == 48
        assert entry["provider_usage"]["reflector"]["completion_tokens"] == 70
        assert entry["provider_usage"]["curator"]["completion_tokens"] == 98
        assert entry["provider_usage"]["total"]["completion_tokens"] == 216
        assert entry["provider_usage"]["total"]["prompt_tokens"] == 1240
    for apply, discard in zip(entries["apply"]["local_epochs"], entries["discard"]["local_epochs"], strict=True):
        for left, right in zip(apply["tasks"], discard["tasks"], strict=True):
            assert [request["seed"] for request in left["provider_requests"]] == [request["seed"] for request in right["provider_requests"]]
            if apply["local_eval_epoch"] == 1:
                assert left["provider_requests"] == right["provider_requests"]


def test_seed_receipts_do_not_depend_on_worker_order(tmp_path):
    entries = []
    for concurrency in (1, 2):
        kwargs, _ = setup_control(tmp_path / str(concurrency), "apply", concurrency=concurrency, visits=3)
        entries.append(asyncio.run(evaluation.run_local_ace(**kwargs)).evaluations[0])
    for serial, concurrent in zip(entries[0]["local_epochs"], entries[1]["local_epochs"], strict=True):
        assert [row["provider_requests"] for row in serial["tasks"]] == [row["provider_requests"] for row in concurrent["tasks"]]


@pytest.mark.parametrize("corruption", ["arm", "seed", "book", "missing_requests", "reflector", "curator"])
def test_completion_validator_rejects_changed_control_evidence(tmp_path, corruption):
    kwargs, _ = setup_control(tmp_path, "discard", visits=3)
    entry = asyncio.run(evaluation.run_local_ace(**kwargs)).evaluations[0]
    item = {"expected_epochs": 3, "g_act": 1, "handoff_control": entry["handoff_control"]}
    validate_sequential_metrics(item, entry, entry["local_epochs"], tmp_path / "metrics.json")
    entry = copy.deepcopy(entry)
    row = entry["local_epochs"][0]["tasks"][0]
    if corruption == "arm":
        row["handoff_control"]["adaptation_mode"] = "apply"
    elif corruption == "seed":
        row["provider_requests"][0]["seed"] += 1
    elif corruption == "book":
        row["adaptation_changed"] = True
    elif corruption in ("reflector", "curator"):
        row["provider_requests"] = [request for request in row["provider_requests"] if request["identity"]["role"] != corruption]
    else:
        row["provider_requests"] = []
    with pytest.raises(ValueError):
        validate_sequential_metrics(item, entry, entry["local_epochs"], tmp_path / "metrics.json")


@pytest.mark.parametrize("pre_receipt", [False, True])
@pytest.mark.parametrize("change", [{"adaptation_mode": "discard"}, {"seed": 712}, {"provider_seed_namespace": "other"}, {"provider_seed_scheme": "legacy", "provider_seed_namespace": ""}, {"handoff_protocol": "feedback_v1"}])
def test_resume_rejects_changed_arm_or_provider_schedule(tmp_path, change, pre_receipt):
    kwargs, _ = setup_control(tmp_path, "apply", visits=3)
    kwargs["args"].resume = True
    asyncio.run(evaluation.run_local_ace(**kwargs))
    if pre_receipt:
        for receipt in (tmp_path / "artifacts").glob("task_*/completed_visits/*.json"):
            receipt.unlink()
    retry, recorder = setup_control(tmp_path, "apply", visits=3)
    retry["args"].resume = True
    for key, value in change.items():
        setattr(retry["args"], key, value)
    with pytest.raises(ValueError, match="different handoff control configuration"):
        asyncio.run(evaluation.run_local_ace(**retry))
    assert not recorder.generator_calls


def test_resume_aggregates_saved_usage_without_regenerating(tmp_path):
    kwargs, recorder = setup_control(tmp_path, "apply", visits=3)
    kwargs["args"].resume = True
    original = asyncio.run(evaluation.run_local_ace(**kwargs)).evaluations[0]
    retry, recorder = setup_control(tmp_path, "apply", visits=3)
    retry["args"].resume = True
    resumed = asyncio.run(evaluation.run_local_ace(**retry)).evaluations[0]
    assert not recorder.generator_calls and not recorder.reflector_calls and not recorder.curator_calls
    assert resumed["provider_usage"] == original["provider_usage"]
    assert resumed["provider_usage"]["total"]["request_dispatches"] == 14
    assert resumed["provider_usage"]["total"]["completion_tokens"] == 66


def test_explicit_predispatch_failure_does_not_require_invented_role_receipt(tmp_path):
    class PredispatchFailure(Reflector):
        async def analyze(self, *_args):
            raise ValueError("adapter failed before provider dispatch")

    kwargs, recorder = setup_control(tmp_path, "discard", visits=3)
    create = kwargs["hooks"].create_context_rule

    def broken(*args, **options):
        rule = create(*args, **options)
        rule.reflector = PredispatchFailure(recorder, rule)
        return rule

    kwargs["hooks"].create_context_rule = broken
    entry = asyncio.run(evaluation.run_local_ace(**kwargs)).evaluations[0]
    row = entry["local_epochs"][0]["tasks"][0]
    assert row["provider_role_work"]["reflector"]["calls"] == [{"sample": 0, "request_dispatches": 0, "response_present": False, "predispatch_failure": True}]
    item = {"expected_epochs": 3, "g_act": 1, "handoff_control": entry["handoff_control"]}
    validate_sequential_metrics(item, entry, entry["local_epochs"], tmp_path / "metrics.json")
