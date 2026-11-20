"""Campaign controls are explicit, separately named and immutable on resume."""

import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_sequential_launcher_contract import preview


@pytest.mark.parametrize("benchmark", ["appworld", "mini_swe_bench", "mbpp", "mastermind"])
@pytest.mark.parametrize("mode", ["apply", "discard"])
@pytest.mark.parametrize("protocol", ["legacy", "feedback_v1"])
def test_handoff_environment_reaches_direct_cli_and_separate_output(benchmark, mode, protocol, monkeypatch):
    result = preview(benchmark, "ace", ADAPTATION_MODE=mode, HANDOFF_PROTOCOL=protocol, PROVIDER_SEED_SCHEME="sha256_v1", PROVIDER_SEED_NAMESPACE=f"campaign/{benchmark}", SEED="711")
    if benchmark in {"mini_swe_bench", "mastermind"}:
        command = shlex.split(result.stdout.splitlines()[-1])
        output = command[command.index("--playbook_snapshot_dir") + 1]
    else:
        item = json.loads(result.stdout)["evaluations"][0]
        command, output = item["evaluation"], item["run_dir"]
        assert item["handoff_control"] == {"adaptation_mode": mode, "provider_seed": {"scheme": "sha256_v1", "namespace": f"campaign/{benchmark}", "evaluation_seed": 711}, **({"handoff_protocol": protocol} if protocol != "legacy" else {})}
    assert f"visits8-{mode}-sha256_v1-" in output
    assert "seed711" in output
    module = importlib.import_module(command[command.index("-m") + 1])
    monkeypatch.setattr(sys, "argv", [module.__file__, *command[command.index("-m") + 2:]])
    args = module.parse_args()
    assert args.adaptation_mode == mode
    assert args.provider_seed_scheme == "sha256_v1"
    assert args.provider_seed_namespace == f"campaign/{benchmark}"
    assert args.seed == 711
    assert args.handoff_protocol == protocol
    assert (args.g_act, args.g_ref, args.g_cur, args.local_eval_epochs) == (1, 1, 1, 8)


@pytest.mark.parametrize("benchmark", ["appworld", "mini_swe_bench", "mbpp", "mastermind"])
def test_discard_cannot_run_with_vanilla_role_budget(benchmark):
    with pytest.raises(subprocess.CalledProcessError):
        preview(benchmark, "vanilla", ADAPTATION_MODE="discard")


@pytest.mark.parametrize("benchmark", ["appworld", "mini_swe_bench", "mbpp", "mastermind"])
def test_feedback_protocol_wires_actual_adapter_and_selected_role_prompts(benchmark, monkeypatch, tmp_path):
    module = importlib.import_module(f"examples.{benchmark}.eval.ace.evaluate")
    monkeypatch.setattr(sys, "argv", [module.__file__, "--handoff_protocol", "feedback_v1"])
    args = module.parse_args()
    utils_name = f"{benchmark}_ace_utils"
    utils = importlib.import_module(f"examples.{benchmark}.eval.ace.{utils_name}")
    tokenizer = SimpleNamespace(encode=lambda text, **_kwargs: list(map(ord, text)), decode=lambda values, **_kwargs: "".join(map(chr, values)))
    monkeypatch.setattr(utils.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(utils, "OpenAIEngine", lambda **kwargs: SimpleNamespace(**kwargs))
    if benchmark == "mini_swe_bench":
        monkeypatch.setitem(sys.modules, "examples.mini_swe_bench.train.ace_rl.train", SimpleNamespace(ACE_DEFAULTS={"reflector_system_prompt": "reflect", "curator_system_prompt": "curate"}))
    root = Path(__file__).resolve().parents[2]
    prompts = root / f"examples/{benchmark}/train/ace_rl/prompts"
    initial = tmp_path / "initial.txt"
    initial.write_text("")
    rule = utils.create_ace_context_rule(args, str(prompts), str(initial), None)
    assert rule.reflector.adapter.handoff_protocol == "feedback_v1"
    assert Path(rule.config.curator_prompt_path).name == "curator_prompt_handoff_v1.txt"
    assert Path(rule.config.reflector_prompt_path).name == "reflector_prompt_handoff_v1.txt"
    if benchmark == "appworld":
        from rllm.agents.agent import Step, Trajectory

        # The original instruction must survive independently of capped history.
        instruction = "Rename every report using its creation date, then move older reports."
        book = "[1] Existing advice to check against the original request."
        trajectory = Trajectory(task={"instruction": instruction}, steps=[Step(observation="long initial context " * 200, action="print(1)", next_observation="1")])
        data = {"trajectory": trajectory, "reward": 0, "max_turns": 20}
        reflected = rule.reflector._build_prompt(data, book)
        curated = rule.curator._build_prompt(data, {"key_insight": "Only some reports were renamed."}, book)
        for rendered in (reflected, curated):
            assert f"[ORIGINAL TASK STARTS HERE]\n{instruction}\n[ORIGINAL TASK ENDS HERE]" in rendered
            assert book in rendered
            assert "{{task_instr}}" not in rendered
        assert "Only some reports were renamed." in curated
        assert "ENVIRONMENT REPLY:\n1" in reflected
    if benchmark in {"mbpp", "mastermind"}:
        # Native evaluation steps: the evaluator renders both role prompts with every placeholder filled.
        book = "[1] Existing tip."
        info = {"exact_matches": 1, "value_matches": 2} if benchmark == "mastermind" else {"all_passed": False}
        step = SimpleNamespace(observation="problem", action="0 1 2 3", next_observation="Turn 1:\nterminal reply", info=info, reward=0.5, model_response="0 1 2 3")
        trajectory = SimpleNamespace(task={"code_length": 4, "num_digits": 6}, steps=[step])
        data = {"trajectory": trajectory, "reward": 0.5, "max_turns": 6}
        reflected = rule.reflector._build_prompt(data, book)
        curated = rule.curator._build_prompt(data, {"key_insight": "Keep the confirmed digits."}, book)
        for rendered in (reflected, curated):
            assert "{{" not in rendered and "}}" not in rendered
        assert book in curated and "Keep the confirmed digits." in curated
        assert ("exact=1, value=2" if benchmark == "mastermind" else "ENVIRONMENT REPLY:\nTurn 1:\nterminal reply") in reflected
