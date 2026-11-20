"""Behavioral checks for artifact selection and the portable plotting entrypoint."""

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/figures"))
spec = importlib.util.spec_from_file_location("plot_eval", ROOT / "scripts/figures/plot_eval.py")
plot_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_eval)


class EvaluationPlotsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def artifact(self, tag="model", step=10, tasks="all", value=0.4, base=None, variant=""):
        path = (base or self.root) / f"{tag}_step{step}{variant}_tasks{tasks}" / f"metrics_step{step}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "evaluations": [
                        {"step": step - 1, "metrics": {"success_rate": 1.0}},
                        {"step": step, "metrics": {"success_rate": value}, "local_eval_epochs": 3},
                    ]
                }
            )
        )
        return path

    def test_exact_task_slice_and_final_matching_step(self):
        self.artifact(tasks="all", value=0.4)
        self.artifact(tasks="0-9", value=0.9)
        self.artifact(tag="other", value=0.8)
        points = plot_eval.load_points(self.root, "success_rate", "all", ["model"])
        self.assertEqual([(p["run_tag"], p["value"], p["local_eval_epochs"]) for p in points], [("model", 0.4, 3)])

    def test_mini_swe_budget_and_thinking_variants_remain_separate(self):
        self.artifact(variant="_32768_50", value=0.4)
        self.artifact(variant="_32768_50_dtFalse_atFalse", value=0.8)
        points = plot_eval.load_points(self.root, "success_rate")
        self.assertEqual([(point["variant"], point["value"]) for point in points], [("32768_50", 0.4), ("32768_50_dtFalse_atFalse", 0.8)])

    def test_duplicate_observations_fail(self):
        self.artifact(base=self.root / "first")
        self.artifact(base=self.root / "second", value=0.9)
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation"):
            plot_eval.load_points(self.root, "success_rate")

    def test_missing_metric_does_not_fall_back(self):
        self.artifact()
        with self.assertRaisesRegex(ValueError, "no metric"):
            plot_eval.load_points(self.root, "max_reward_mean")

    def test_missing_requested_run_fails(self):
        self.artifact()
        with self.assertRaisesRegex(ValueError, "requested run tags"):
            plot_eval.load_points(self.root, "success_rate", run_tags=["model", "missing"])

    def test_nonfinite_metric_fails(self):
        self.artifact(value=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite number"):
            plot_eval.load_points(self.root, "success_rate")

    def canonical_artifact(self, step=215, label="all", preset="visits8"):
        path = self.root / "eval" / preset / f"step_{step}"
        path.mkdir(parents=True, exist_ok=True)
        (path / "evaluation_config.json").write_text(json.dumps({"benchmark": "appworld", "method": "ace", "preset": preset, "step": step, "task_label": label, "run_dir": str(self.root)}))
        (path / f"metrics_step{step}.json").write_text(json.dumps({"evaluations": [{"step": step, "metrics": {"success_rate": 0.6}}]}))

    def test_canonical_run_artifacts_use_recorded_preset_and_task_slice(self):
        self.canonical_artifact()
        points = plot_eval.load_points(self.root / "eval", "success_rate", "auto")
        self.assertEqual([(point["step"], point["variant"], point["task_label"]) for point in points], [(215, "ace/visits8", "all")])

    def test_runtime_config_output_directories_share_the_training_run_curve(self):
        for step in (10, 20):
            self.canonical_artifact(step=step)
            path = self.root / "eval/visits8" / f"step_{step}/evaluation_config.json"
            config = json.loads(path.read_text())
            config["run_dir"] = str(path.parent)
            path.write_text(json.dumps(config))
        points = plot_eval.load_points(self.root / "eval", "success_rate", "auto")
        self.assertEqual({point["run_tag"] for point in points}, {str(self.root)})
        self.assertEqual([point["step"] for point in points], [10, 20])

    def test_canonical_multiple_slices_require_explicit_selection(self):
        self.canonical_artifact()
        self.canonical_artifact(step=220, label="0-9")
        with self.assertRaisesRegex(ValueError, "Multiple task slices"):
            plot_eval.load_points(self.root / "eval", "success_rate", "auto")
        self.assertEqual(len(plot_eval.load_points(self.root / "eval", "success_rate", "all")), 1)

    def launch(self, *args, benchmark="appworld", **env):
        return subprocess.run(
            [sys.executable, str(ROOT / "plot.py"), benchmark, "--preset", "eval", "--metric", "success_rate", "--runs-dir", str(self.root), *args],
            cwd=self.root,
            env={**os.environ, "PYTHON_BIN": sys.executable, "EVAL_RUNS_DIR": str(self.root), **env},
            text=True,
            capture_output=True,
        )

    def test_launcher_works_outside_checkout_and_reports_missing_inputs(self):
        failure = self.launch("--check")
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("No matching evaluation metrics", failure.stderr)
        self.artifact()
        success = self.launch("--check")
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertIn("Validated 1 points", success.stdout)

    def test_plot_defaults_match_default_evaluator_task_labels(self):
        for benchmark, label in (("appworld", "all"), ("mbpp", "100"), ("mini_swe_bench", "100"), ("mastermind", "0-all")):
            with self.subTest(benchmark=benchmark):
                self.artifact(tag=benchmark, tasks=label)
                result = self.launch("--check", "--run-tag", benchmark, benchmark=benchmark)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Validated 1 points", result.stdout)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib not available")
    def test_launcher_renders_and_exports_exact_data(self):
        self.artifact(step=10, value=0.4)
        self.artifact(step=20, value=0.6)
        out = self.root / "plots"
        result = self.launch("--output-dir", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((out / "eval.pdf").read_bytes().startswith(b"%PDF"))
        with (out / "eval.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([float(row["value"]) for row in rows], [0.4, 0.6])
        self.assertEqual([int(row["step"]) for row in rows], [10, 20])


if __name__ == "__main__":
    unittest.main()
