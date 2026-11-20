#!/usr/bin/env python3
"""Plan and run AppWorld and MBPP evaluations from method presets.

The planner uses only the standard library. --dry-run never starts services,
loads credentials, creates outputs, downloads models, or contacts a scheduler.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The launcher can use system Python 3.9 while the evaluator uses a newer venv.
JSON_NUMBER_TYPES = (float, int)
COMMON_OPTIONS = (
    "seed",
    "n_parallel_agents",
    "local_eval_epochs",
    "max_turns",
    "g_act",
    "g_ref",
    "g_cur",
    "selection_mode",
    "ace_temperature",
    "max_playbook_length",
    "gen_temperature",
    "gen_top_p",
    "reflector_max_prompt_length",
    "reflector_max_response_length",
    "curator_max_prompt_length",
    "curator_max_response_length",
    "reflector_system_prompt",
    "curator_system_prompt",
    "task_offset",
    "max_tasks",
    "base_system_prompt_filename",
    "prompts_dir",
    "initial_playbook",
    "ace_model",
    "ace_base_url",
    "ace_key_file",
    "ace_project_id",
    "api_key_file",
    "project_id",
)
BENCHMARK_OPTIONS = {
    "appworld": ("test_split", "max_response_length", "max_prompt_length", "reflector_max_history_tokens", "task_id", "trajectory_hard_timeout"),
    "mbpp": ("split", "gen_max_response_length", "gen_max_prompt_length", "data_dir"),
}
API_PREFIXES = ("gemini", "gpt-", "claude-", "o1-", "o3-")
# Each method has one eight-sample protocol: ACE visits8 (eight measured visits)
# and vanilla pass8 (eight independent attempts).
METHOD_PRESETS = {"ace": "visits8", "vanilla": "pass8"}


def boolean(value: str) -> bool:
    if value.lower() in {"true", "1"}:
        return True
    if value.lower() in {"false", "0"}:
        return False
    raise ValueError(f"Expected True/False or 1/0, got {value!r}")


def absolute(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def model_source(env: dict[str, str], root: Path) -> tuple[str, str]:
    model = env.get("MODEL_PATH") or env.get("MODEL_NAME", "")
    if not model:
        raise ValueError("Set MODEL_PATH or MODEL_NAME")
    source = env.get("MODEL_SOURCE", "auto")
    if source == "auto":
        if env.get("CKPT_BASE"):
            source = "checkpoint"
        elif absolute(model, root).is_dir() or env.get("MODEL_PATH"):
            source = "local"
        elif model.lower().startswith(API_PREFIXES):
            source = "api"
        elif "/" in model:
            source = "hub"
        else:
            source = "checkpoint"
    if source not in {"local", "checkpoint", "hub", "api"}:
        raise ValueError("MODEL_SOURCE must be auto, local, checkpoint, hub, or api")
    if source == "api" and not model.lower().startswith(API_PREFIXES):
        raise ValueError(f"The evaluator's external client does not recognize API model {model!r}")
    if source in {"hub", "local", "checkpoint"} and model.lower().startswith(API_PREFIXES):
        raise ValueError("API model names require MODEL_SOURCE=api")
    return source, model


def run_defaults(run_dir: Path) -> dict[str, str]:
    """Model and thinking mode recorded by the training run, for unset evaluation inputs."""
    record_path = run_dir / "run.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    identity = record.get("training_identity") or {}
    variant = record.get("model_variant")
    if not variant:
        # Runs without a record keep the launcher's `<model>-<variant>` directory name.
        name = run_dir.name.lower()
        variant = next((tag for tag in ("nonthinking", "thinking") if name.endswith("-" + tag)), None)
        variant = variant or ("thinking_2507" if "thinking-2507" in name else "instruct" if "instruct-2507" in name else None)
    values = {}
    model = (identity.get("model") or {}).get("path")
    if model and "/" in model and not Path(model).is_absolute():
        values["MODEL_NAME"] = model
    if variant:
        thinking = variant in {"thinking", "thinking_2507"}
        values["MODEL_VARIANT"] = "thinking" if thinking else "nonthinking"
        values["DISABLE_THINKING"] = "False" if thinking else "True"
    disabled = (identity.get("rllm") or {}).get("disable_thinking")
    if isinstance(disabled, bool):
        values["DISABLE_THINKING"] = str(disabled)
        values["MODEL_VARIANT"] = "nonthinking" if disabled else "thinking"
    return values


def method_preset(method: str, env: dict[str, str]) -> str:
    preset = env.get("PRESET") or METHOD_PRESETS[method]
    if preset != METHOD_PRESETS[method]:
        raise ValueError(f"{method} evaluation supports only PRESET={METHOD_PRESETS[method]} (got {preset!r})")
    return preset


def eval_quantization(env: dict[str, str]) -> str:
    quantization = env.get("EVAL_QUANTIZATION") or "none"
    if quantization not in {"none", "fp8"}:
        raise ValueError("EVAL_QUANTIZATION must be none or fp8")
    return quantization


def fp8_kernel(env: dict[str, str]) -> str:
    """verl: VeRL's Triton kernel on the job's GPU (fails without one); replica: explicit CPU fallback."""
    kernel = env.get("EVAL_FP8_KERNEL") or "verl"
    if kernel not in {"verl", "replica"}:
        raise ValueError("EVAL_FP8_KERNEL must be verl or replica")
    return kernel


def serving_hf_overrides(env: dict[str, str], source: str) -> dict:
    """Explicit serving-only config; never infer tying from checkpoint files."""
    if "EVAL_TIE_WORD_EMBEDDINGS" not in env:
        return {}
    try:
        tied = boolean(env["EVAL_TIE_WORD_EMBEDDINGS"])
    except ValueError as exc:
        raise ValueError("EVAL_TIE_WORD_EMBEDDINGS must be True/False or 1/0") from exc
    if source == "api":
        raise ValueError("EVAL_TIE_WORD_EMBEDDINGS requires a local, Hub or checkpoint model")
    return {"tie_word_embeddings": tied}


def fp8_conversion_command(python: str, root: Path, model_source: str, item: dict, env: dict[str, str]) -> list[str]:
    source_flag = "--hub-model" if model_source == "hub" else "--source-dir"
    command = [python, str(root / "scripts/convert_fp8_checkpoint.py"), source_flag, item["quantization_source"], "--target-dir", item["model"], "--kernel", item["quantization_kernel"]]
    if env.get("EVAL_FP8_MIN_CAPABILITY"):
        command.extend(("--min-capability", env["EVAL_FP8_MIN_CAPABILITY"]))
    return command


def fp8_model_path(source: str, model_path: str, env: dict[str, str], root: Path) -> str:
    """Where the block-FP8 copy of a BF16 checkpoint or base model is published."""
    if source == "checkpoint":
        return model_path + "-fp8"
    cache = absolute(env.get("EVAL_FP8_CACHE_DIR") or ".cache/fp8_models", root)
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", model_path.strip("/"))[-80:]
    if source == "local":
        label += "-" + hashlib.sha256(model_path.encode()).hexdigest()[:12]
    return str(cache / f"{label}-fp8")


def mbpp_feedback(benchmark: str, env: dict[str, str]) -> str:
    """MBPP_FEEDBACK=hidden hides failed assertions from the policy; full is the historical view."""
    feedback = env.get("MBPP_FEEDBACK") or "full"
    if feedback not in {"full", "hidden"}:
        raise ValueError(f"MBPP_FEEDBACK must be full or hidden, got {feedback!r}")
    if feedback != "full" and benchmark != "mbpp":
        raise ValueError("MBPP_FEEDBACK applies only to MBPP evaluation")
    return feedback


def trained_mbpp_feedback(run_dir: Path) -> str | None:
    """Feedback mode recorded in a training run's identity, when one exists."""
    record_path = run_dir / "run.json"
    if not record_path.is_file():
        return None
    identity = json.loads(record_path.read_text()).get("training_identity") or {}
    env_args = ((identity.get("rllm") or {}).get("env") or {}).get("env_args") or {}
    return str(env_args.get("feedback", "full"))


def evaluation_paths(benchmark: str, method: str, env: dict[str, str], root: Path) -> dict:
    """Resolve only explicit inputs; never discover experiments or latest steps."""
    preset = method_preset(method, env)
    quantization = eval_quantization(env)
    # FP8 results live beside, never inside, the BF16 results of the same preset.
    preset_dir = preset if quantization == "none" else f"{preset}-{quantization}"
    # Hidden-feedback results never share, or resume from, a full-feedback directory.
    if mbpp_feedback(benchmark, env) == "hidden":
        preset_dir += "-hiddenfb"
    handoff = handoff_control_environment(benchmark, method, env)
    if handoff:
        seeds = handoff["provider_seed"]
        suffix = f"-{handoff['adaptation_mode']}"
        if seeds:
            namespace_hash = hashlib.sha256(seeds["namespace"].encode()).hexdigest()[:12]
            suffix += f"-{seeds['scheme']}-{namespace_hash}-seed{seeds['evaluation_seed']}"
        if handoff.get("handoff_protocol"):
            suffix += f"-{handoff['handoff_protocol']}"
        preset_dir += suffix
    # A named lane keeps a changed protocol out of an existing preset directory for a whole step series.
    lane = env.get("EVAL_LANE_SUFFIX", "")
    if lane:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", lane):
            raise ValueError("EVAL_LANE_SUFFIX must be one name of letters, digits, '.', '_' or '-'")
        preset_dir += f"-{lane}"
    training_run = absolute(env["RUN_DIR"], root) if env.get("RUN_DIR") else None
    if env.get("STEP") and env.get("STEPS") and env["STEPS"].split() != [env["STEP"]]:
        raise ValueError("STEP and STEPS disagree")
    if training_run:
        if env.get("MODEL_SOURCE", "auto") not in {"auto", "checkpoint"} or env.get("CKPT_BASE") or env.get("MODEL_PATH"):
            raise ValueError("RUN_DIR selects training checkpoints; omit MODEL_SOURCE, MODEL_PATH, and CKPT_BASE")
        source, model = "checkpoint", env.get("MODEL_NAME", training_run.name)
    else:
        source, model = model_source(env, root)
    hf_overrides = serving_hf_overrides(env, source)
    if quantization != "none" and source == "api":
        raise ValueError("EVAL_QUANTIZATION=fp8 requires a local, Hub or checkpoint model")
    steps = env.get("STEP", env.get("STEPS", "" if source == "checkpoint" else "0")).split()
    if not steps or any(not step.isdecimal() for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("STEP is required for training checkpoints; STEPS must contain distinct nonnegative integers")
    if source != "checkpoint" and steps != ["0"]:
        raise ValueError("Use STEP=0 for a single local, Hub, or API model")
    if env.get("OUTPUT_DIR") and len(steps) != 1:
        raise ValueError("OUTPUT_DIR requires one checkpoint step")
    model_tag = env.get("RUN_TAG") or re.sub(r"[^A-Za-z0-9._-]+", "_", Path(model).name)
    if not model_tag or Path(model_tag).name != model_tag or model_tag in {".", ".."}:
        raise ValueError("RUN_TAG must be a single directory name")
    results = absolute(env.get("RUNS_BASE", f"examples/{benchmark}/eval/{method}/results"), root)
    items = []
    for step in steps:
        checkpoint_source = None
        if training_run:
            checkpoint_source = training_run / "checkpoints" / f"global_step_{step}"
            model_path = str(training_run / "checkpoints_merged" / f"global_step_{step}")
            output = training_run / "eval" / method / preset_dir / f"step_{step}"
        elif source == "checkpoint":
            if not env.get("CKPT_BASE"):
                raise ValueError("Checkpoint series requires RUN_DIR or explicit CKPT_BASE; use MODEL_SOURCE=local for merged weights")
            model_path = str(absolute(env["CKPT_BASE"], root) / f"global_step_{step}")
            output = results / preset_dir / model_tag / f"step_{step}"
        else:
            model_path = str(absolute(model, root)) if source == "local" else model
            output = results / preset_dir / model_tag
        if env.get("OUTPUT_DIR"):
            output = absolute(env["OUTPUT_DIR"], root)
        item = {"step": int(step), "model": model_path, "run_dir": str(output), "checkpoint_source": str(checkpoint_source) if checkpoint_source else None}
        if hf_overrides:
            item["hf_overrides"] = hf_overrides
        if quantization != "none":
            # The served model is converted from this BF16 source before startup.
            item.update(quantization=quantization, quantization_source=model_path, quantization_kernel=fp8_kernel(env), model=fp8_model_path(source, model_path, env, root))
        items.append(item)
    return {"model_source": source, "model": model, "quantization": quantization, "evaluations": items}


def handoff_control_environment(benchmark: str, method: str, env: dict[str, str]) -> dict:
    mode = env.get("ADAPTATION_MODE") or "apply"
    scheme = env.get("PROVIDER_SEED_SCHEME") or "legacy"
    namespace = env.get("PROVIDER_SEED_NAMESPACE") or ""
    protocol = env.get("HANDOFF_PROTOCOL") or "legacy"
    if protocol not in {"legacy", "feedback_v1"}:
        raise ValueError("HANDOFF_PROTOCOL must be legacy or feedback_v1")
    if mode not in {"apply", "discard"} or scheme not in {"legacy", "sha256_v1"}:
        raise ValueError("Invalid ADAPTATION_MODE or PROVIDER_SEED_SCHEME")
    if scheme == "legacy" and namespace:
        raise ValueError("PROVIDER_SEED_NAMESPACE requires PROVIDER_SEED_SCHEME=sha256_v1")
    if mode == "apply" and scheme == "legacy" and protocol == "legacy":
        return {}
    if method != "ace" or benchmark not in {"appworld", "mini_swe_bench", "mbpp", "mastermind"}:
        raise ValueError("Handoff controls currently require ACE evaluation")
    if env.get("LOCAL_EVAL_PROTOCOL", "sequential_adaptation_v1") != "sequential_adaptation_v1":
        raise ValueError("Handoff controls require LOCAL_EVAL_PROTOCOL=sequential_adaptation_v1")
    if mode == "discard" and (int(env.get("G_REF", "1")), int(env.get("G_CUR", "1"))) != (1, 1):
        raise ValueError("Discard control requires G_REF=G_CUR=1")
    seeds = None
    if scheme == "sha256_v1":
        if not namespace.strip() or namespace != namespace.strip():
            raise ValueError("sha256_v1 requires a nonempty, trimmed PROVIDER_SEED_NAMESPACE")
        seed = int(env.get("SEED", "42"))
        if seed < 0:
            raise ValueError("Provider evaluation SEED must be nonnegative")
        seeds = {"scheme": scheme, "namespace": namespace, "evaluation_seed": seed}
    return {"adaptation_mode": mode, "provider_seed": seeds, **({"handoff_protocol": protocol} if protocol != "legacy" else {})}


def plan(benchmark: str, method: str, env: dict[str, str], extra: list[str]) -> dict:
    if extra:
        raise ValueError("Set evaluation options through environment variables, or invoke the benchmark's evaluate.py CLI directly")
    handoff = handoff_control_environment(benchmark, method, env)
    for key in ("LOCAL_EVAL_EPOCHS", "N_PARALLEL_AGENTS", "MAX_TURNS", "MAX_MODEL_LEN", "G_ACT", "VLLM_TP", "VLLM_DP", "MAX_NUM_SEQS"):
        if int(env.get(key, "1")) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("G_REF", "G_CUR"):
        if int(env.get(key, "0")) < 0:
            raise ValueError(f"{key} must be nonnegative")
    local_protocol = env.get("LOCAL_EVAL_PROTOCOL", "sequential_adaptation_v1")
    if local_protocol not in {"sequential_adaptation_v1", "legacy_search_v1"}:
        raise ValueError("Unknown LOCAL_EVAL_PROTOCOL")
    if local_protocol == "sequential_adaptation_v1":
        if boolean(env.get("FREEZE_SOLVED", "False")):
            raise ValueError("Sequential evaluation keeps every policy visit; FREEZE_SOLVED must be False")
        if (int(env.get("G_REF", "0")), int(env.get("G_CUR", "0"))) not in {(0, 0), (1, 1)}:
            raise ValueError("Sequential evaluation supports one reflection and one curator delta; branching requires LOCAL_EVAL_PROTOCOL=legacy_search_v1")
        if env.get("FINAL_POLICY_FROM_RUN_DIR") or env.get("FINAL_POLICY_SOURCE_MANIFEST"):
            raise ValueError("Saved-playbook continuation requires LOCAL_EVAL_PROTOCOL=legacy_search_v1; sequential runs need the complete measured chain")
    task_concurrency = int(env.get("TASK_CONCURRENCY", "1"))
    if task_concurrency < 1:
        raise ValueError("TASK_CONCURRENCY must be positive")
    if task_concurrency * int(env.get("G_ACT", "1")) > int(env.get("N_PARALLEL_AGENTS", "1")):
        raise ValueError("TASK_CONCURRENCY * G_ACT must not exceed N_PARALLEL_AGENTS for parallel attempts")
    if task_concurrency > 1 and benchmark not in ("appworld", "mbpp"):
        raise ValueError("Concurrent local tasks are supported for AppWorld and MBPP here")
    final_policy = boolean(env.get("FINAL_POLICY_EVAL", "False"))
    final_policy_source = env.get("FINAL_POLICY_FROM_RUN_DIR", "")
    final_policy_manifest = env.get("FINAL_POLICY_SOURCE_MANIFEST", "")
    if (final_policy or final_policy_source or final_policy_manifest) and (benchmark != "appworld" or method != "ace"):
        raise ValueError("Final policy evaluation currently requires AppWorld ACE")
    if final_policy_source and not final_policy:
        raise ValueError("FINAL_POLICY_FROM_RUN_DIR requires FINAL_POLICY_EVAL=True")
    if final_policy_manifest and not final_policy_source:
        raise ValueError("FINAL_POLICY_SOURCE_MANIFEST requires FINAL_POLICY_FROM_RUN_DIR")
    parity_flag = boolean(env.get("TRAINING_INPUT_PARITY", "False"))
    protocol = env.get("EVAL_PROTOCOL") or ("training_input_parity" if parity_flag else "legacy")
    if protocol not in {"legacy", "training_input_parity"}:
        raise ValueError("EVAL_PROTOCOL must be legacy or training_input_parity")
    if "TRAINING_INPUT_PARITY" in env and parity_flag != (protocol == "training_input_parity"):
        raise ValueError("EVAL_PROTOCOL conflicts with TRAINING_INPUT_PARITY")
    preset = method_preset(method, env)
    if protocol == "training_input_parity" and benchmark != "appworld":
        raise ValueError("training_input_parity requires AppWorld")
    if env.get("TRAJECTORY_HARD_TIMEOUT"):
        timeout = float(env["TRAJECTORY_HARD_TIMEOUT"])
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("TRAJECTORY_HARD_TIMEOUT must be finite and positive")
    root = absolute(env.get("RLLM_DIR", str(ROOT)), ROOT).resolve()
    feedback = mbpp_feedback(benchmark, env)
    if benchmark == "mbpp" and env.get("RUN_DIR") and not env.get("MBPP_FEEDBACK"):
        trained = trained_mbpp_feedback(absolute(env["RUN_DIR"], root))
        if trained not in (None, "full"):
            raise ValueError(f"RUN_DIR was trained with MBPP feedback={trained}; set MBPP_FEEDBACK explicitly (normally {trained})")
    layout = evaluation_paths(benchmark, method, env, root)
    source, model = layout["model_source"], layout["model"]
    if protocol == "training_input_parity" and source == "api":
        raise ValueError("training_input_parity requires a local or Hub model")
    if benchmark == "appworld" and source in {"local", "checkpoint"} and env.get("EVAL_REASONING_CONFIGURED") != "1":
        raise ValueError("Set MODEL_VARIANT=thinking|nonthinking (or all thinking flags) to match the AppWorld checkpoint")
    if benchmark == "appworld" and source == "api" and boolean(env.get("ACCUMULATE_REASONING", "False")):
        raise ValueError("External AppWorld generators require ACCUMULATE_REASONING=False")
    python = env.get("PYTHON_BIN") or (str(Path(env["VENV_DIR"]) / "bin/python") if env.get("VENV_DIR") else "python3")
    steps = [str(item["step"]) for item in layout["evaluations"]]
    offset = int(env.get("TASK_OFFSET", "0"))
    limit = env.get("MAX_TASKS", "")
    if offset < 0 or (limit and int(limit) <= 0):
        raise ValueError("TASK_OFFSET must be nonnegative and MAX_TASKS must be positive or empty")
    label = f"{offset}-{offset + int(limit) - 1}" if limit and offset else (limit or (f"from{offset}" if offset else "all"))
    if env.get("TASK_ID"):
        if offset or limit:
            raise ValueError("TASK_ID cannot be combined with TASK_OFFSET or MAX_TASKS")
        label = "id_" + hashlib.sha256(env["TASK_ID"].encode()).hexdigest()[:12]
    # Include the method, preset, split and reasoning mode so unrelated configurations
    # do not silently claim one another's completion marker.
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(model).name)
    variant = "nonthinking" if boolean(env.get("DISABLE_THINKING", "False")) else "thinking"
    split = env.get("TEST_SPLIT" if benchmark == "appworld" else "SPLIT", "test")
    protocol_tag = "_training_input_parity" if protocol == "training_input_parity" else ""
    if layout["quantization"] != "none":
        protocol_tag += "_" + layout["quantization"]
    if final_policy:
        protocol_tag += "_final_policy"
    if feedback == "hidden":
        protocol_tag += "_hiddenfb"
    tag = env.get("RUN_TAG") or f"{model_tag}_{method}_{preset}_{split}_{variant}{protocol_tag}"
    if not tag or Path(tag).name != tag or tag in {".", ".."}:
        raise ValueError("RUN_TAG must be a single directory name")
    if final_policy_source and len(steps) != 1:
        raise ValueError("FINAL_POLICY_FROM_RUN_DIR requires exactly one checkpoint step")
    resume = boolean(env.get("RESUME", "1"))
    vllm_port = int(env.get("VLLM_PORT", "30000"))
    app_port = int(env.get("APPWORLD_PORT", "9000"))
    for port in (vllm_port, app_port):
        if not 1 <= port <= 65535:
            raise ValueError("Server ports must be between 1 and 65535")
    if benchmark == "appworld" and source != "api" and vllm_port == app_port:
        raise ValueError("VLLM_PORT and APPWORLD_PORT must differ")
    base_url = env.get("BASE_URL", "") if source == "api" else f"http://127.0.0.1:{vllm_port}/v1"
    app_url = env.get("APPWORLD_URL") or f"http://127.0.0.1:{app_port}"
    app_servers = int(env.get("N_SERVERS", env["N_PARALLEL_AGENTS"])) if benchmark == "appworld" and not env.get("APPWORLD_URL") else 1
    if benchmark == "appworld" and int(env["N_PARALLEL_AGENTS"]) > app_servers:
        raise ValueError("AppWorld needs one stateful server per parallel agent; existing APPWORLD_URL requires N_PARALLEL_AGENTS=1")
    if benchmark == "appworld" and task_concurrency * int(env.get("G_ACT", "1")) > min(app_servers, int(env["N_PARALLEL_AGENTS"])):
        raise ValueError("TASK_CONCURRENCY * G_ACT needs one AppWorld server and parallel agent per trajectory")
    items = []
    for paths in layout["evaluations"]:
        step = str(paths["step"])
        model_path = paths["model"]
        run_tag = f"{tag}_step{step}_tasks{label}"
        run_dir = Path(paths["run_dir"])
        metrics = run_dir / f"metrics_step{step}.json"
        cmd = [
            python,
            "-m",
            f"examples.{benchmark}.eval.ace.evaluate",
            "--local_eval_protocol",
            local_protocol,
            "--model_name",
            model_path,
            "--base_url",
            base_url,
            "--epochs",
            "1",
            "--train_batch_size",
            "1",
            "--max_model_length",
            env["MAX_MODEL_LEN"],
            "--playbook_snapshot_dir",
            str(run_dir),
            "--metrics_file",
            str(metrics),
            "--wandb_step",
            step,
        ]
        for key in (*COMMON_OPTIONS, *BENCHMARK_OPTIONS[benchmark]):
            value = env.get(key.upper())
            if value is not None and value != "":
                cmd.extend((f"--{key}", value))
        if handoff:
            cmd.extend(("--adaptation_mode", handoff["adaptation_mode"]))
            if handoff["provider_seed"]:
                cmd.extend(("--provider_seed_scheme", handoff["provider_seed"]["scheme"], "--provider_seed_namespace", handoff["provider_seed"]["namespace"]))
            if handoff.get("handoff_protocol"):
                cmd.extend(("--handoff_protocol", handoff["handoff_protocol"]))
        # Full feedback keeps the historical command, so recorded runs still match.
        if feedback == "hidden":
            cmd.extend(("--feedback", feedback))
        if "--ace_model" not in cmd:
            cmd.extend(("--ace_model", model_path))
        for key in ("disable_thinking", "accumulate_thinking", "accumulate_reasoning"):
            if key.upper() in env and (key == "disable_thinking" or benchmark == "appworld"):
                cmd.append(f"--{'' if boolean(env[key.upper()]) else 'no-'}{key}")
        for key in ("freeze_solved", "include_thoughts"):
            if boolean(env.get(key.upper(), "0")):
                cmd.append(f"--{key}")
        if benchmark == "appworld":
            cmd.extend(("--remote_environment_url", app_url, "--n_servers", str(app_servers), "--base_port", str(app_port)))
        # Serial runs keep the historical command so recorded configurations still match.
        if task_concurrency > 1:
            cmd.extend(("--task_concurrency", str(task_concurrency)))
        if benchmark == "appworld":
            if protocol == "training_input_parity":
                cmd.append("--training_input_parity")
            if final_policy:
                cmd.append("--final_policy_eval")
            if final_policy_source:
                cmd.extend(("--final_policy_from_run_dir", str(absolute(final_policy_source, root))))
            if final_policy_manifest:
                cmd.extend(("--final_policy_source_manifest", str(absolute(final_policy_manifest, root))))
        if env.get("WANDB_PROJECT"):
            cmd.extend(("--wandb_project", env["WANDB_PROJECT"], "--wandb_run_name", run_tag))
        cmd.extend(extra)
        server = []
        if source != "api":
            server = [
                python,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(vllm_port),
                "--trust-remote-code",
                "--dtype",
                "bfloat16",
                "--generation-config",
                "vllm",
                "--max-model-len",
                env["MAX_MODEL_LEN"],
                "--tensor-parallel-size",
                env["VLLM_TP"],
                "--data-parallel-size",
                env["VLLM_DP"],
                "--gpu-memory-utilization",
                env["GPU_MEMORY_UTILIZATION"],
                "--max-num-seqs",
                env["MAX_NUM_SEQS"],
            ]
            if paths.get("quantization"):
                # BF16 activations/unquantized layers with block-FP8 Linear weights,
                # the numerics of VeRL's FP8 training rollout.
                server.extend(("--quantization", paths["quantization"]))
            if paths.get("hf_overrides"):
                server.extend(("--hf-overrides", json.dumps(paths["hf_overrides"], separators=(",", ":"))))
        items.append(
            {
                "benchmark": benchmark,
                "method": method,
                "preset": preset,
                "task_label": label,
                "wandb_entity": (env.get("WANDB_ENTITY") or None) if env.get("WANDB_PROJECT") else None,
                "protocol": protocol,
                "local_eval_protocol": local_protocol,
                **({"handoff_control": handoff} if handoff else {}),
                **({"mbpp_feedback": feedback} if feedback != "full" else {}),
                "expected_epochs": int(env["LOCAL_EVAL_EPOCHS"]),
                "g_act": int(env.get("G_ACT", "1")),
                "final_policy_eval": final_policy,
                "final_policy_from_run_dir": str(absolute(final_policy_source, root)) if final_policy_source else None,
                "final_policy_source_manifest": str(absolute(final_policy_manifest, root)) if final_policy_manifest else None,
                "step": int(step),
                "model": model_path,
                "checkpoint_source": paths["checkpoint_source"],
                **({key: paths[key] for key in ("quantization", "quantization_source", "quantization_kernel")} if paths.get("quantization") else {}),
                **({"hf_overrides": paths["hf_overrides"]} if paths.get("hf_overrides") else {}),
                "run_dir": str(run_dir),
                "metrics_file": str(metrics),
                "resume": resume,
                "evaluation": cmd,
                "server": server,
            }
        )
    return {
        "benchmark": benchmark,
        "method": method,
        "preset": preset,
        "protocol": protocol,
        "local_eval_protocol": local_protocol,
        "repo": str(root),
        "model_source": source,
        "appworld_servers": app_servers if benchmark == "appworld" else 0,
        "task_concurrency": task_concurrency,
        "base_url": base_url,
        "appworld_url": app_url if benchmark == "appworld" else None,
        "evaluations": items,
    }


def free_port(requested: str | None) -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", int(requested or "0")))
        return str(sock.getsockname()[1])


def free_port_range(requested: str | None, count: int, excluded: set[int]) -> str:
    for _ in range(100):
        start = int(requested or free_port(None))
        if start + count > 65536 or excluded.intersection(range(start, start + count)):
            if requested:
                raise ValueError("AppWorld server port range conflicts with the model server or exceeds 65535")
            continue
        try:
            with contextlib.ExitStack() as stack:
                for port in range(start, start + count):
                    sock = stack.enter_context(socket.socket())
                    sock.bind(("127.0.0.1", port))
                return str(start)
        except OSError:
            if requested:
                raise
    raise RuntimeError("Could not find a free contiguous AppWorld server port range")


def runtime_environment(env: dict[str, str], local_model: bool, root: Path) -> dict[str, str]:
    result = dict(env)
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        result.pop(name, None)
    result["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result["PYTHONUNBUFFERED"] = "1"
    if env.get("APPWORLD_DATA_DIR"):
        result["APPWORLD_ROOT"] = env["APPWORLD_DATA_DIR"]
    # Use the same checked interpreter/library discovery as training, creating
    # the checkout's environment for this GPU family when it is missing.
    script = """set -euo pipefail
if [[ -z "${PYTHON_BIN:-}" && -z "${VENV_DIR:-}" ]]; then
    source "$RLLM_DIR/scripts/gpu_venv.sh"
    rllm_resolve_gpu_env "$RLLM_DIR" >&2
    rllm_ensure_gpu_env "$RLLM_DIR"
fi
export PYTHON_BIN=${PYTHON_BIN:-$VENV_DIR/bin/python}
source "$RLLM_DIR/scripts/gpu_runtime.sh"
rllm_configure_gpu_runtime "$PYTHON_BIN" >&2
"$PYTHON_BIN" -c 'import json, os; print(json.dumps(dict(os.environ)))'
"""
    if local_model:
        captured = subprocess.run(["bash", "-c", script], env=result, cwd=root, check=True, text=True, stdout=subprocess.PIPE)
        result = json.loads(captured.stdout)
    else:
        result["PYTHON_BIN"] = env.get("PYTHON_BIN") or (str(Path(env["VENV_DIR"]) / "bin/python") if env.get("VENV_DIR") else sys.executable)
    return result


@contextlib.contextmanager
def owned_process(command: list[str], env: dict[str, str], log: Path):
    """Stop only this process group, including children, on success or failure."""
    with log.open("a") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield process
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_ready(process: subprocess.Popen, url: str, *, model: str | None = None, timeout: float = 1200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with status {process.returncode}; see its log")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if model is not None:
                    names = [entry.get("id") for entry in json.load(response).get("data", [])]
                    if model not in names:
                        raise RuntimeError(f"Server at {url} is serving a different model")
                if process.poll() is not None:
                    raise RuntimeError("Service exited during readiness check")
                return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1)
    raise RuntimeError(f"Service did not become ready within {timeout:g}s")


def recorded_command(command: list[str]) -> list[str]:
    recorded = list(command)
    for index, token in enumerate(recorded):
        if token.startswith(("--api_key=", "--ace_api_key=")):
            recorded[index] = token.split("=", 1)[0] + "=<redacted>"
            continue
        if index + 1 == len(recorded):
            continue
        if token in {"--api_key", "--ace_api_key"}:
            recorded[index + 1] = "<redacted>"
        elif token == "--base_port":
            recorded[index + 1] = "<managed-local-port>"
        elif token in {"--base_url", "--remote_environment_url"} and re.fullmatch(r"http://127\.0\.0\.1:[0-9]+(?:/v1)?", recorded[index + 1]):
            recorded[index + 1] = "<managed-local-service>"
    return recorded


def evaluation_task_ids(entry: dict, epochs: list[dict], path: Path) -> set[str]:
    """Prefer the declared selection when legacy epochs omit frozen solved tasks."""
    observed = set()
    for record in [entry, *epochs]:
        tasks = record.get("tasks", [])
        if not isinstance(tasks, list):
            raise ValueError(f"Evaluation task rows are invalid: {path}")
        for task in tasks:
            if not isinstance(task, dict) or not isinstance(task.get("task_id"), str) or not task["task_id"].strip():
                raise ValueError(f"Evaluation task identity is invalid: {path}")
            observed.add(task["task_id"])
    if "evaluation_task_ids" not in entry:
        return observed
    declared = entry["evaluation_task_ids"]
    if not isinstance(declared, list) or not declared or any(not isinstance(identifier, str) or not identifier.strip() for identifier in declared) or len(declared) != len(set(declared)):
        raise ValueError(f"Declared evaluation_task_ids must be nonempty valid unique IDs: {path}")
    expected = set(declared)
    if not observed <= expected:
        raise ValueError(f"Observed adaptation task IDs are outside evaluation_task_ids: {path}")
    return expected


def validate_sequential_metrics(item: dict, entry: dict, epochs: list[dict], path: Path) -> None:
    """Require the complete fixed policy budget and retained per-sample evidence."""
    count, samples_per_task = item["expected_epochs"], item.get("g_act", 1)
    handoff = item.get("handoff_control", {})
    if entry.get("handoff_control", {}) != handoff:
        raise ValueError(f"Sequential handoff control differs from configuration: {path}")
    expected_ids = evaluation_task_ids(entry, epochs, path)
    if not expected_ids or entry.get("policy_visits") != count or entry.get("g_act") != samples_per_task:
        raise ValueError(f"Sequential evaluation has inconsistent policy budget: {path}")
    attempts = set()
    initial_books = {}
    for epoch in epochs:
        rows = epoch.get("tasks", [])
        ids = [row["task_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f"Sequential evaluation must retain every task in every visit: {path}")
        rewards = []
        for row in rows:
            if row.get("handoff_control", {}) != handoff:
                raise ValueError(f"Task handoff control differs from configuration: {path}")
            if handoff:
                if any(not isinstance(row.get(key), bool) for key in ("adaptation_generated", "adaptation_changed", "adaptation_applied")):
                    raise ValueError(f"Task has no explicit generated/applied/changed handoff evidence: {path}")
                if handoff["adaptation_mode"] == "discard":
                    book = initial_books.setdefault(row["task_id"], row.get("policy_playbook"))
                    if row["adaptation_applied"] or row["adaptation_changed"] or row.get("policy_playbook") != book:
                        raise ValueError(f"Discard control changed the actor playbook: {path}")
                seeds = handoff["provider_seed"]
                if seeds is not None:
                    validate_provider_seed_receipt(row, seeds, epoch["local_eval_epoch"], samples_per_task, path)
            samples = row.get("samples")
            if not isinstance(samples, list) or len(samples) != samples_per_task:
                raise ValueError(f"Sequential evaluation is missing policy samples: {path}")
            if [sample.get("sample_index") for sample in samples] != list(range(samples_per_task)):
                raise ValueError(f"Sequential evaluation sample indices are incomplete: {path}")
            task_rewards = []
            for sample in samples:
                reward, success, attempt = sample.get("reward"), sample.get("success"), sample.get("attempt_id")
                if isinstance(reward, bool) or not isinstance(reward, JSON_NUMBER_TYPES) or not math.isfinite(reward) or not isinstance(success, bool):
                    raise ValueError(f"Sequential evaluation contains invalid sample outcomes: {path}")
                if not isinstance(attempt, str) or not attempt or attempt in attempts:
                    raise ValueError(f"Sequential evaluation contains missing or repeated attempt identities: {path}")
                attempts.add(attempt)
                task_rewards.append(reward)
            row_reward = row.get("reward")
            if isinstance(row_reward, bool) or not isinstance(row_reward, JSON_NUMBER_TYPES) or not math.isclose(row_reward, sum(task_rewards) / samples_per_task, abs_tol=1e-9):
                raise ValueError(f"Sequential task reward is not its raw sample mean: {path}")
            rewards.extend(task_rewards)
        mean = epoch.get("metrics", {}).get("avg_reward")
        if isinstance(mean, bool) or not isinstance(mean, JSON_NUMBER_TYPES) or not math.isclose(mean, sum(rewards) / len(rewards), abs_tol=1e-9):
            raise ValueError(f"Sequential epoch reward is not its raw sample mean: {path}")
        if handoff.get("provider_seed") is not None:
            requests = [request for row in rows for request in row["provider_requests"]]
            if epoch.get("provider_usage") != summarize_provider_usage(requests):
                raise ValueError(f"Epoch provider usage disagrees with retained requests: {path}")
    if handoff.get("provider_seed") is not None:
        requests = [request for epoch in epochs for row in epoch["tasks"] for request in row["provider_requests"]]
        if entry.get("provider_usage") != summarize_provider_usage(requests):
            raise ValueError(f"Evaluation provider usage disagrees with retained requests: {path}")
    final = entry.get("final_policy_evaluation")
    if not isinstance(final, dict) or final.get("kind") != "last_measured_policy_visit" or final.get("adaptation_applied") is not False:
        raise ValueError(f"Sequential evaluation has no last-measured final policy result: {path}")
    if final.get("policy_visit") != count or final.get("after_local_eval_epoch") != count - 1:
        raise ValueError(f"Sequential final policy visit is inconsistent: {path}")
    if final.get("tasks") != epochs[-1]["tasks"] or final.get("metrics") != epochs[-1]["metrics"]:
        raise ValueError(f"Sequential final policy result is not the last measured visit: {path}")
    if final.get("provenance", {}).get("extra_policy_rollouts") != 0:
        raise ValueError(f"Sequential final policy result generated extra rollouts: {path}")
    if final.get("work", {}).get("policy_rollouts") != len(expected_ids) * samples_per_task:
        raise ValueError(f"Sequential final policy sample count differs: {path}")
    if any(final.get("work", {}).get(key) != 0 for key in ("reflector_rollouts", "curator_rollouts", "candidate_rollouts")):
        raise ValueError(f"Sequential final policy visit contains adaptation work: {path}")
    for key, value in final["metrics"].items():
        if entry["metrics"].get(f"final_policy/{key}") != value:
            raise ValueError(f"Sequential final policy metrics disagree: {path}")
    if entry.get("rollout_counts", {}).get("candidate") != 0:
        raise ValueError(f"Sequential evaluation performed candidate scoring: {path}")
    if any(row.get("adaptation_applied") is not False for row in epochs[-1]["tasks"]):
        raise ValueError(f"Sequential final policy rows contain adaptation: {path}")
    if entry.get("rollout_counts", {}).get("policy") != count * len(expected_ids) * samples_per_task:
        raise ValueError(f"Sequential evaluation executed a different policy budget: {path}")


def validate_provider_seed_receipt(row: dict, seeds: dict, visit: int, g_act: int, path: Path) -> None:
    requests = row.get("provider_requests")
    if row.get("provider_seed") != seeds or not isinstance(requests, list) or not requests:
        raise ValueError(f"Task is missing provider seed dispatch receipts: {path}")
    roles = set()
    for request in requests:
        identity = request.get("identity", {})
        role = identity.get("role")
        attempt = identity.get("attempt")
        if role not in {"actor", "reflector", "curator"} or any(identity.get(key) != value for key, value in seeds.items()) or identity.get("task_id") != row["task_id"]:
            raise ValueError(f"Provider request has the wrong task/role/seed identity: {path}")
        expected_attempts = range((visit - 1) * g_act, visit * g_act) if role == "actor" else (visit - 1,)
        if attempt not in expected_attempts or any(isinstance(identity.get(key), bool) or not isinstance(identity.get(key), int) or identity[key] < 0 for key in ("call", "sample")):
            raise ValueError(f"Provider request has the wrong visit/call/sample identity: {path}")
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        expected_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)
        if request.get("seed") != expected_seed or not re.fullmatch(r"[0-9a-f]{64}", str(request.get("request_sha256", ""))):
            raise ValueError(f"Provider dispatch seed/hash disagrees with its logical request: {path}")
        usage = request.get("usage")
        if not isinstance(request.get("response_received"), bool) or (usage is not None and (not request["response_received"] or not isinstance(usage, dict) or set(usage) != {"prompt_tokens", "completion_tokens", "total_tokens"} or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in usage.values()))):
            raise ValueError(f"Provider dispatch has invalid response/usage evidence: {path}")
        roles.add(role)
    if "actor" not in roles:
        raise ValueError(f"Task has no seeded actor request: {path}")
    work = row.get("provider_role_work", {})
    for role in ("reflector", "curator"):
        recorded = work.get(role, {})
        calls = recorded.get("calls")
        if not isinstance(calls, list) or recorded.get("scheduled_calls") != len(calls) or [call.get("sample") for call in calls] != list(range(len(calls))):
            raise ValueError(f"Task has incomplete auxiliary call coverage: {path}")
        if role == "reflector" and len(calls) != (g_act if row.get("adaptation_scheduled") else 0):
            raise ValueError(f"Reflector call count disagrees with the scheduled adaptation slot: {path}")
        if sum(call.get("request_dispatches", 0) for call in calls) != sum(request["identity"]["role"] == role for request in requests):
            raise ValueError(f"Auxiliary request sample has no corresponding role call: {path}")
        for call in calls:
            dispatches = sum(request["identity"]["role"] == role and request["identity"]["sample"] == call["sample"] for request in requests)
            if call.get("request_dispatches") != dispatches or not isinstance(call.get("response_present"), bool) or not isinstance(call.get("predispatch_failure"), bool):
                raise ValueError(f"Auxiliary dispatch receipts disagree with role call evidence: {path}")
            if (dispatches == 0 and (not call["predispatch_failure"] or call["response_present"])) or (dispatches > 0 and call["predispatch_failure"]):
                raise ValueError(f"Auxiliary call has no dispatch or explicit pre-dispatch failure: {path}")
    if row.get("provider_usage") != summarize_provider_usage(requests):
        raise ValueError(f"Task provider usage disagrees with request receipts: {path}")


def summarize_provider_usage(requests: list[dict]) -> dict:
    """Standard-library counterpart of the evaluator's receipt aggregation."""
    def summarize(items):
        known = [item["usage"] for item in items if isinstance(item.get("usage"), dict)]
        observed = {key: sum(usage[key] for usage in known) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return {"request_dispatches": len(items), "responses_received": sum(item.get("response_received") is True for item in items), "usage_known_requests": len(known), "usage_unknown_requests": len(items) - len(known), **{f"observed_{key}": value for key, value in observed.items()}, **{key: value if len(known) == len(items) else None for key, value in observed.items()}}

    return {**{role: summarize([item for item in requests if item["identity"]["role"] == role]) for role in ("actor", "reflector", "curator")}, "total": summarize(requests)}


def validate_independent_metrics(item: dict, payload: dict, path: Path) -> None:
    """Validate Mastermind's flat independent-attempt result without ML imports."""
    if payload.get("model") != item["model"] or payload.get("step") != item["step"]:
        raise ValueError(f"Independent metrics model or step differs: {path}")
    if payload.get("evaluation_protocol") != "independent_samples_v1" or item["expected_epochs"] != 1:
        raise ValueError(f"Independent metrics protocol differs: {path}")
    rows = payload.get("tasks", [])
    ids = evaluation_task_ids(payload, [], path)
    count = item["g_act"]
    if not ids or len(rows) != len(ids) or payload.get("num_tasks") != len(ids):
        raise ValueError(f"Independent metrics task coverage differs: {path}")
    if payload.get("samples_per_task") != count or payload.get("num_samples") != count * len(ids):
        raise ValueError(f"Independent metrics sample budget differs: {path}")
    rewards, successes = [], []
    for row in rows:
        samples = row.get("samples", [])
        if len(samples) != count or [sample.get("sample_index") for sample in samples] != list(range(count)):
            raise ValueError(f"Independent metrics have incomplete samples: {path}")
        for index, sample in enumerate(samples):
            reward = sample.get("reward")
            if isinstance(reward, bool) or not isinstance(reward, JSON_NUMBER_TYPES) or not math.isfinite(reward) or not isinstance(sample.get("success"), bool):
                raise ValueError(f"Independent metrics contain invalid outcomes: {path}")
            if sample.get("task_id") != row["task_id"] or sample.get("attempt_id") != index or sample.get("sample_id") != f"{row['task_id']}/attempt_{index}":
                raise ValueError(f"Independent sample identity differs: {path}")
        mean = sum(sample["reward"] for sample in samples) / count
        solved = any(sample["success"] for sample in samples)
        row_reward = row.get("reward")
        if isinstance(row_reward, bool) or not isinstance(row_reward, JSON_NUMBER_TYPES) or not math.isclose(row_reward, mean, abs_tol=1e-9) or row.get("success") is not solved:
            raise ValueError(f"Independent task summary differs from samples: {path}")
        rewards.append(mean)
        successes.append(sum(sample["success"] for sample in samples))
    expected = {"avg_reward": sum(rewards) / len(rows), "success_rate": sum(n > 0 for n in successes) / len(rows)}
    for k in range(1, count + 1):
        expected[f"pass_at_{k}_success_rate"] = sum(1 - (math.comb(count - n, k) / math.comb(count, k) if count - n >= k else 0) for n in successes) / len(rows)
    for key, value in expected.items():
        actual = payload.get(key)
        if isinstance(actual, bool) or not isinstance(actual, JSON_NUMBER_TYPES) or not math.isclose(actual, value, abs_tol=1e-9):
            raise ValueError(f"Independent summary {key} differs from samples: {path}")


def recorded_evaluation_item(config: Path) -> dict:
    """Recover validation inputs from the immutable shell-launcher command."""
    saved = json.loads(config.read_text())
    command = saved["evaluation"]

    def option(*flags, default=None):
        for flag in flags:
            if flag in command:
                return command[command.index(flag) + 1]
        return default

    step = int(option("--wandb_step", "--step"))
    model = option("--model_name", "--model")
    if step != saved["step"] or model != saved["model_name"] or Path(saved["run_dir"]).resolve() != config.parent.resolve():
        raise ValueError(f"Recorded command identity disagrees with configuration: {config}")
    epochs = int(option("--local_eval_epochs", default="1"))
    samples = int(option("--g_act", "--g-act"))
    if epochs < 1 or samples < 1:
        raise ValueError(f"Invalid recorded evaluation budget: {config}")
    metrics = Path(option("--metrics_file", default=str(config.parent / f"metrics_step{step}.json")))
    if metrics.resolve() != (config.parent / f"metrics_step{step}.json").resolve():
        raise ValueError(f"Recorded metrics path disagrees with run directory: {config}")
    return {**saved, "model": model, "metrics_file": str(metrics), "expected_epochs": epochs, "g_act": samples}


def write_completion(item: dict) -> None:
    """Write only validated evidence; callers hold the run lock after successful exit."""
    metrics_hash = validate_metrics(item)
    marker = Path(item["run_dir"]) / "evaluation_complete.json"
    receipt = {"evaluation": recorded_command(item["evaluation"]), "metrics_sha256": metrics_hash}
    if "hf_overrides" in item:
        receipt["hf_overrides"] = item["hf_overrides"]
    if marker.exists():
        if json.loads(marker.read_text()) != receipt:
            raise ValueError(f"Existing completion receipt disagrees: {marker}")
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=marker.parent, prefix=".completion-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    try:
        temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)


def validate_metrics(item: dict) -> str:
    path = Path(item["metrics_file"])
    raw = path.read_bytes()
    payload = json.loads(raw)
    if item.get("local_eval_protocol") == "independent_samples_v1":
        validate_independent_metrics(item, payload, path)
        return hashlib.sha256(raw).hexdigest()
    if not isinstance(payload, dict) or payload.get("model_name") != item["model"]:
        raise ValueError(f"Metrics model does not match evaluation: {path}")
    evaluations = payload.get("evaluations", [])
    if not isinstance(evaluations, list) or len(evaluations) != 1:
        raise ValueError(f"Expected exactly one completed evaluation: {path}")
    entry = evaluations[0]
    count = item["expected_epochs"]
    epochs = entry.get("local_epochs", [])
    if entry.get("step") != item["step"] or entry.get("local_eval_epochs") != count or len(epochs) != count:
        raise ValueError(f"Metrics step or completed epoch count does not match evaluation: {path}")
    if [epoch.get("local_eval_epoch") for epoch in epochs] != list(range(1, count + 1)):
        raise ValueError(f"Metrics contain missing or repeated epochs: {path}")
    for epoch in epochs:
        metrics = epoch.get("metrics", {})
        if not epoch.get("tasks") and metrics.get("n_solved", 0) <= 0:
            raise ValueError(f"Evaluation contains no tasks: {path}")
    values = entry.get("metrics", {})
    for key in ("success_rate", "avg_reward", "max_reward_mean"):
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, JSON_NUMBER_TYPES) or not math.isfinite(value):
            raise ValueError(f"Evaluation has no finite {key}: {path}")
    local_protocol = item.get("local_eval_protocol", "legacy_search_v1")
    if entry.get("local_eval_protocol", "legacy_search_v1") != local_protocol:
        raise ValueError(f"Metrics local evaluation protocol differs from the launcher: {path}")
    if local_protocol == "sequential_adaptation_v1":
        validate_sequential_metrics(item, entry, epochs, path)
        return hashlib.sha256(raw).hexdigest()
    final = entry.get("final_policy_evaluation")
    if item.get("final_policy_eval", False):
        if not isinstance(final, dict) or final.get("kind") != "final_policy_only" or final.get("adaptation_applied") is not False:
            raise ValueError(f"Evaluation has no completed policy-only final pass: {path}")
        if final.get("after_local_eval_epoch") != count or final.get("policy_visit") != count + 1:
            raise ValueError(f"Final policy pass has an inconsistent visit number: {path}")
        tasks = final.get("tasks")
        if not isinstance(tasks, list) or not tasks or any(not isinstance(task, dict) or not isinstance(task.get("task_id"), str) or not task["task_id"] for task in tasks):
            raise ValueError(f"Final policy pass has no valid task identities: {path}")
        ids = [task["task_id"] for task in tasks]
        expected_ids = evaluation_task_ids(entry, epochs, path)
        if not expected_ids or len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f"Final policy task coverage disagrees with the evaluation task set: {path}")
        final_values = final.get("metrics", {})
        for key in ("success_rate", "avg_reward", "max_reward_mean"):
            value = final_values.get(key)
            if isinstance(value, bool) or not isinstance(value, JSON_NUMBER_TYPES) or not math.isfinite(value) or values.get(f"final_policy/{key}") != value:
                raise ValueError(f"Final policy pass has no consistent finite {key}: {path}")
        work = final.get("work", {})
        if work.get("policy_rollouts") != len(ids) * item.get("g_act", 1) or any(work.get(key) != 0 for key in ("reflector_rollouts", "curator_rollouts", "candidate_rollouts")):
            raise ValueError(f"Final policy pass contains adaptation work or no policy rollouts: {path}")
    elif final is not None:
        raise ValueError(f"Metrics contain an unrequested final policy pass: {path}")
    return hashlib.sha256(raw).hexdigest()


def check_outputs(item: dict, *, ignore_lock: bool = False) -> bool:
    run_dir = Path(item["run_dir"])
    marker = run_dir / "evaluation_complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("evaluation") != recorded_command(item["evaluation"]) or saved.get("hf_overrides") != item.get("hf_overrides") or not Path(item["metrics_file"]).is_file():
            raise ValueError(f"Existing output has a different configuration or missing metrics: {run_dir}")
        if saved.get("metrics_sha256") != validate_metrics(item):
            raise ValueError(f"Completed metrics have changed: {run_dir}")
        return True
    if run_dir.exists() and any(path.name != ".evaluation.lock" for path in run_dir.iterdir()):
        config = run_dir / "evaluation_config.json"
        if item.get("resume") and not config.exists() and all(path.name in {".evaluation.lock", "service_logs"} for path in run_dir.iterdir()):
            # AppWorld sidecars can fail before the evaluator claims the output;
            # their diagnostic logs are the only files created in that phase.
            return False
        if not item.get("resume") or not config.is_file():
            raise ValueError(f"Existing output is incomplete or has no verified completion record: {run_dir}. Choose another OUTPUT_DIR; it will not be deleted.")
        saved = json.loads(config.read_text())
        previous, current = saved.get("evaluation"), recorded_command(item["evaluation"])
        has_completed_tasks = any(run_dir.glob("task_*/completed_visits/*.json")) or any(run_dir.glob("task_*/completed_final_policy/*.json"))
        # A startup-only failure has no samples to mix across runtimes. The
        # interpreter may change during repair (for example, removing a site
        # override) while every evaluator argument remains identical.
        same_command = previous == current or (
            not has_completed_tasks and isinstance(previous, list) and previous[1:] == current[1:]
        )
        if not same_command or saved.get("step") != item["step"] or saved.get("model") != item["model"] or saved.get("hf_overrides") != item.get("hf_overrides"):
            raise ValueError(f"Existing output belongs to a different evaluation: {run_dir}")
    return False


def gpu_keepalive_command(run_plan: dict, env: dict[str, str], root: Path) -> list[str] | None:
    """The utilization keepalive for GPU evaluations inside a scheduler allocation.

    Torch cancels jobs whose average GPU utilization stays low; one-task-at-a-time
    evaluation leaves a served model idle between requests. API models use no GPU,
    and outside Slurm (local checks, tests) nothing is started.
    """
    if run_plan["model_source"] == "api" or not env.get("SLURM_JOB_ID"):
        return None
    if env.get("RLLM_GPU_KEEPALIVE", "1") == "0" or env.get("RLLM_DISABLE_GPU_KEEPALIVE", "0") == "1":
        return None
    return [env.get("PYTHON_BIN", sys.executable), str(root / "scripts/gpu_keepalive.py")]


def execute(run_plan: dict, env: dict[str, str]) -> int:
    root = Path(run_plan["repo"])
    todo = []
    # Check the whole series before launching a model or touching outputs.
    for item in run_plan["evaluations"]:
        if item.get("checkpoint_source"):
            if not (Path(item["checkpoint_source"]) / "actor" / "huggingface" / "config.json").is_file():
                raise ValueError(f"Missing training checkpoint: {item['checkpoint_source']}")
        elif run_plan["model_source"] in {"local", "checkpoint"}:
            checkpoint = Path(item.get("quantization_source") or item["model"])
            if not (checkpoint / "config.json").is_file() or not any(checkpoint.glob("*.safetensors")):
                raise ValueError(f"Expected a merged Hugging Face checkpoint with config.json and .safetensors: {checkpoint}")
        if check_outputs(item):
            print(f"[eval] completed: {item['run_dir']}", flush=True)
        else:
            todo.append(item)
    if not todo:
        return 0
    for item in todo:
        if item.get("checkpoint_source"):
            subprocess.run(
                [
                    env.get("PYTHON_BIN", sys.executable),
                    str(root / "scripts/merge_eval_checkpoint.py"),
                    "--source-dir",
                    item["checkpoint_source"],
                    "--target-dir",
                    item["model"],
                    "--python",
                    env.get("PYTHON_BIN", sys.executable),
                ],
                env=env,
                check=True,
            )
        if item.get("quantization") == "fp8":
            # Runs inside the evaluation allocation, before vLLM starts, with the
            # job's GPUs visible: the default kernel needs one and fails otherwise.
            subprocess.run(fp8_conversion_command(env.get("PYTHON_BIN", sys.executable), root, run_plan["model_source"], item, env), env=env, check=True)
            item["quantization_conversion"] = json.loads((Path(item["model"]) / "fp8_conversion.json").read_text()).get("quantizer")
    with contextlib.ExitStack() as stack:
        keepalive = gpu_keepalive_command(run_plan, env, root)
        if keepalive:
            # Log beside the outputs, not inside one: an output directory must stay
            # empty until its evaluation claims it.
            log_dir = Path(todo[0]["run_dir"]).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            stack.enter_context(owned_process(keepalive, env, log_dir / f"gpu-keepalive-{env['SLURM_JOB_ID']}.log"))
        if run_plan["benchmark"] == "appworld" and not env.get("APPWORLD_URL"):
            data_root = env.get("APPWORLD_DATA_DIR", "")
            server_venv = env.get("APPWORLD_SERVER_VENV", "")
            if not data_root or not (Path(data_root) / "data").is_dir() or not server_venv:
                raise ValueError("Set APPWORLD_DATA_DIR and APPWORLD_SERVER_VENV, or provide an existing APPWORLD_URL")
            runtime_parent = Path(env.get("SLURM_TMPDIR") or env.get("TMPDIR") or tempfile.gettempdir())
            runtime_parent.mkdir(parents=True, exist_ok=True)
            runtime = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=f"rllm-eval-{os.getpid()}-", dir=runtime_parent)))

            def preserve_service_logs():
                destination = Path(todo[0]["run_dir"]) / "service_logs"
                destination.mkdir(parents=True, exist_ok=True)
                for log in runtime.glob("*.log"):
                    shutil.copy2(log, destination / log.name)

            stack.callback(preserve_service_logs)
            shared_data = (Path(data_root) / "data").resolve()
            for index in range(run_plan["appworld_servers"]):
                port = int(env["APPWORLD_PORT"]) + index
                # AppWorld writes and cleans experiments under APPWORLD_ROOT.
                # Even concurrent copies of the same task must own distinct roots.
                server_root = runtime / f"server-{port}"
                server_root.mkdir()
                (server_root / "data").symlink_to(shared_data, target_is_directory=True)
                app_env = dict(env, APPWORLD_ROOT=str(server_root), APPWORLD_PORT=str(port))
                app_command = [
                    str(Path(server_venv) / "bin/python"),
                    "-c",
                    "import os\n"
                    "from appworld import update_root\n"
                    "update_root(os.environ['APPWORLD_ROOT'])\n"
                    "from appworld.serve.environment import app\n"
                    "import uvicorn\n"
                    "uvicorn.run(app, host='127.0.0.1', port=int(os.environ['APPWORLD_PORT']))\n",
                ]
                service = stack.enter_context(owned_process(app_command, app_env, runtime / f"appworld-{port}.log"))
                wait_ready(service, f"http://127.0.0.1:{port}", timeout=120)
        for item in todo:
            output = Path(item["run_dir"])
            # A kernel lock is released after a killed job; an exclusive-create
            # lock file would strand exactly the runs that need resuming.
            output.mkdir(parents=True, exist_ok=True)
            with (output / ".evaluation.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f"Evaluation is already running in {output}") from exc
                if check_outputs(item, ignore_lock=True):
                    continue
                (output / "evaluation_config.json").write_text(json.dumps({**item, "evaluation": recorded_command(item["evaluation"])}, indent=2) + "\n")
                with contextlib.ExitStack() as servers:
                    if item["server"]:
                        # This owned loopback server and its readiness/client calls
                        # use no server credential. Do not inherit unrelated auth.
                        server_env = dict(env)
                        server_env.pop("VLLM_API_KEY", None)
                        service = servers.enter_context(owned_process(item["server"], server_env, output / "vllm.log"))
                        wait_ready(service, run_plan["base_url"] + "/models", model=item["model"], timeout=float(env.get("VLLM_READY_TIMEOUT", "1200")))
                    print(f"[eval] {shlex.join(recorded_command(item['evaluation']))}", flush=True)
                    evaluation_env = dict(env)
                    if env.get("WANDB_PROJECT"):
                        evaluation_env["WANDB_DIR"] = str(output)
                    with owned_process(item["evaluation"], evaluation_env, output / "evaluation.log") as evaluation:
                        returncode = evaluation.wait()
                    if returncode:
                        raise RuntimeError(f"Evaluator failed with status {returncode}; see {output / 'evaluation.log'}")
                if not Path(item["metrics_file"]).is_file():
                    raise RuntimeError(f"Evaluator exited without writing metrics: {item['metrics_file']}")
                write_completion(item)
                print(f"[eval] completed: {output}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=(*BENCHMARK_OPTIONS, "mini_swe_bench", "mastermind"))
    parser.add_argument("method", choices=("ace", "vanilla"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--complete-recorded", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--shell-paths", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-defaults", action="store_true", help=argparse.SUPPRESS)
    args, extra = parser.parse_known_args(argv)
    env = dict(os.environ)
    # Evaluator overrides follow --; argparse options are forwarded as tokens.
    if extra[:1] == ["--"]:
        extra = extra[1:]
    try:
        if args.complete_recorded:
            if extra or args.dry_run:
                raise ValueError("Completion does not accept overrides or --dry-run")
            item = recorded_evaluation_item(args.complete_recorded)
            if item["benchmark"] != args.benchmark or item["method"] != args.method:
                raise ValueError("Completion benchmark/method differs from recorded configuration")
            write_completion(item)
            return 0
        if args.run_defaults:
            if "EVAL_TIE_WORD_EMBEDDINGS" in env:
                # Every maintained shell launcher runs this before benchmark
                # setup. Some benchmarks prepare their runtime before plan().
                source = model_source(env, ROOT)[0] if env.get("MODEL_PATH") or env.get("MODEL_NAME") else env.get("MODEL_SOURCE", "auto")
                serving_hf_overrides(env, source)
            if env.get("RUN_DIR"):
                for name, value in run_defaults(absolute(env["RUN_DIR"], absolute(env.get("RLLM_DIR", str(ROOT)), ROOT))).items():
                    if not env.get(name):
                        print(f"export {name}={shlex.quote(value)}")
            return 0
        if args.shell_paths:
            layout = evaluation_paths(args.benchmark, args.method, env, absolute(env.get("RLLM_DIR", str(ROOT)), ROOT))
            if len(layout["evaluations"]) != 1:
                raise ValueError("Shell path resolution requires one STEP")
            item = layout["evaluations"][0]
            values = {"MODEL_PATH": item["model"], "MODEL_SOURCE": layout["model_source"], "STEP": str(item["step"]), "RUN_DIR": item["run_dir"], "CHECKPOINT_SOURCE": item["checkpoint_source"] or ""}
            values.update(EVAL_QUANTIZATION=layout["quantization"], QUANTIZATION_SOURCE=item.get("quantization_source", ""))
            if "hf_overrides" in item:
                values["EVAL_TIE_WORD_EMBEDDINGS"] = str(item["hf_overrides"]["tie_word_embeddings"]).lower()
            for name, value in values.items():
                print(f"export {name}={shlex.quote(value)}")
            return 0
        if args.benchmark not in BENCHMARK_OPTIONS:
            raise ValueError("Use this benchmark's eval.slurm entrypoint")
        run_plan = plan(args.benchmark, args.method, env, extra)
        if args.dry_run or boolean(env.get("RLLM_DRY_RUN", "0")):
            print(json.dumps(run_plan, indent=2))
            return 0
        env = runtime_environment(env, run_plan["model_source"] != "api", Path(run_plan["repo"]))
        if run_plan["model_source"] != "api":
            env["VLLM_PORT"] = free_port(env.get("VLLM_PORT"))
        if args.benchmark == "appworld" and not env.get("APPWORLD_URL"):
            env["APPWORLD_PORT"] = free_port_range(env.get("APPWORLD_PORT"), run_plan["appworld_servers"], {int(env["VLLM_PORT"])} if env.get("VLLM_PORT") else set())
        run_plan = plan(args.benchmark, args.method, env, extra)
        os.chdir(run_plan["repo"])
        return execute(run_plan, env)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[eval] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
