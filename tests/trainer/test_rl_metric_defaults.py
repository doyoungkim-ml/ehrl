from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_NORMALIZATION_PATH = REPO_ROOT / "rllm" / "trainer" / "utils" / "metric_normalization.py"


def _load_metric_normalization_module():
    spec = importlib.util.spec_from_file_location("metric_normalization", METRIC_NORMALIZATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOSS_AGG_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>actor_rollout_ref\.actor\.loss_agg_mode=|\bloss_agg_mode:\s*)"
    r"(?P<value>[^\\\s#]+)"
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            {"loss": 1.5, "grad_norm": 2.5, "lr": 1e-6, "mfu": 0.12},
            {
                "actor/pg_loss": 1.5,
                "actor/grad_norm": 2.5,
                "actor/lr": 1e-6,
                "perf/mfu/actor": 0.12,
            },
        ),
        (
            {
                "loss": 1.5,
                "actor/pg_loss": 9.0,
                "grad_norm": 2.5,
                "actor/grad_norm": 8.0,
                "mfu": 0.12,
                "perf/mfu/actor": 0.34,
            },
            {"actor/pg_loss": 9.0, "actor/grad_norm": 8.0, "perf/mfu/actor": 0.34},
        ),
    ],
)
def test_actor_update_metrics_are_namespaced_like_current_verl(raw, expected):
    normalize_actor_metric_keys = _load_metric_normalization_module().normalize_actor_metric_keys

    normalized = normalize_actor_metric_keys(raw)

    for old_key in ("loss", "grad_norm", "lr", "mfu"):
        assert old_key not in normalized
    for key, value in expected.items():
        assert normalized[key] == value


def test_actor_update_flops_are_derived_from_mfu_and_update_time():
    module = _load_metric_normalization_module()

    enriched = module.add_actor_update_flop_metrics(
        {"mfu": 0.5, "timing_s/update_actor": 2.0},
        n_gpus=4,
        promised_flops_per_gpu=1000.0,
    )

    assert "mfu" not in enriched
    assert enriched["perf/mfu/actor"] == 0.5
    assert enriched["perf/flops/actor_update"] == 4000.0
    assert enriched["perf/mflops/actor_update"] == 0.004


def test_existing_actor_update_flops_metric_wins_over_derived_value():
    module = _load_metric_normalization_module()

    enriched = module.add_actor_update_flop_metrics(
        {
            "perf/mfu/actor": 0.5,
            "timing_s/update_actor": 2.0,
            "perf/flops/actor_update": 123.0,
        },
        n_gpus=4,
        promised_flops_per_gpu=1000.0,
    )

    assert enriched["perf/flops/actor_update"] == 123.0
    assert enriched["perf/mflops/actor_update"] == 0.000123


def test_rl_experiments_use_token_mean_loss_aggregation():
    offenders: list[str] = []
    for path in [
        *REPO_ROOT.glob("examples/**/*"),
        *REPO_ROOT.glob("scripts/agent/**/*"),
        *REPO_ROOT.glob("docs/examples/**/*"),
    ]:
        if not path.is_file() or path.suffix not in {".sh", ".sbatch", ".yaml", ".yml", ".md"}:
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            match = LOSS_AGG_ASSIGNMENT_RE.search(line)
            if match and match.group("value") != "token-mean":
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {match.group('value')}")

    assert offenders == []
