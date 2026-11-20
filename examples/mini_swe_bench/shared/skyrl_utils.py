"""SWE-Gym / SWE-bench image-name helper.

Previously this module also defined `get_eval_script`, a homemade synthesizer
that built a `python -m pytest ...` command from `FAIL_TO_PASS`. That approach
had multiple correctness bugs (JSON-decode oversight, no `test_patch`
application, wrong runner for Django, etc.). It was replaced by calls into
the official `swebench` package (`make_test_spec` + `get_eval_report`); see
`examples/mini_swe_bench/env.py::MiniSWEnv._run_one_instance`.
"""

from __future__ import annotations


def get_docker_image_name(instance: dict, data_source: str) -> str:
    image_name = instance.get("image_name")
    if image_name is not None:
        return image_name

    instance_id = instance.get("instance_id", "")
    data_source_lower = (data_source or "").lower()
    if "swe-gym" in data_source_lower or "swegym" in data_source_lower:
        image_suffix = instance_id.replace("__", "_s_")
        return f"docker.io/xingyaoww/sweb.eval.x86_64.{image_suffix}:latest".lower()
    if "swe-bench" in data_source_lower or "swebench" in data_source_lower:
        image_suffix = instance_id.replace("__", "_1776_")
        return f"docker.io/swebench/sweb.eval.x86_64.{image_suffix}:latest".lower()
    raise ValueError(f"Unsupported data_source: {data_source}")
