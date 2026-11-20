"""Comparison presets consume each benchmark's maintained evaluation schema."""

import hashlib
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
import plot_comparison as comparison  # noqa: E402


class SharedComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def artifact(self, benchmark, method, *, epochs=8):
        folder = self.root / benchmark / method / "eval" / ("visits8" if method == "ace" else "pass8") / "step_10"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "metrics_step10.json"
        split = {"appworld": "dev", "mini_swe_bench": "validation"}.get(benchmark, "test")
        model = f"/models/{benchmark}/{method}/global_step_10"
        tasks = [{"task_id": "task_a", "reward": 0.1}, {"task_id": "task_b", "reward": 0.3}]
        flat = benchmark == "mastermind" and method == "vanilla"
        if flat:
            payload = {"dataset": benchmark, "model": model, "split": split, "step": 10, "avg_reward": 0.2, "reward_aggregation": "max", "tasks": tasks}
            command = ["python", "evaluate.py", "--model", model, "--step", "10"]
        else:
            count = 1 if method == "vanilla" else epochs
            local_epochs = [{"local_eval_epoch": epoch, "tasks": tasks, "metrics": {"max_reward_mean": 0.2 if method == "vanilla" else epoch / 10}} for epoch in range(1, count + 1)]
            entry = {"step": 10, "metrics": local_epochs[-1]["metrics"], "local_eval_epochs": count, "local_epochs": local_epochs, "evaluation_task_ids": ["task_a", "task_b"]}
            payload = {"model_name": model, "evaluations": [entry]}
            command = ["python", "evaluate.py", "--test_split" if benchmark == "appworld" else "--split", split, "--model_name", model, "--wandb_step", "10", "--local_eval_epochs", str(count)]
        path.write_text(json.dumps(payload))
        recorded = {"benchmark": benchmark, "method": method, "step": 10, "task_label": "all", "evaluation": command}
        (folder / "evaluation_config.json").write_text(json.dumps(recorded))
        # These maintained shell launchers currently record configs, not completion markers.
        if benchmark not in ("mastermind", "mini_swe_bench"):
            (folder / "evaluation_complete.json").write_text(json.dumps({"evaluation": command, "metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        return path, split

    def configuration(self, benchmark, *, epochs=8):
        ace, split = self.artifact(benchmark, "ace", epochs=epochs)
        vanilla, _ = self.artifact(benchmark, "vanilla", epochs=1)
        config = {
            "benchmark": benchmark,
            "title": f"{benchmark} measured comparison",
            "split": split,
            "task_label": "all",
            "series": [
                {"id": "ace_rl", "run": "ace-run", "points": [{"step": 10, "file": str(ace), "provenance": "Synthetic ACE evaluation"}]},
                {"id": "grpo_vanilla", "run": "grpo-run", "points": [{"step": 10, "file": str(vanilla), "provenance": "Synthetic vanilla evaluation"}]},
            ],
        }
        path = self.root / benchmark / "comparison.json"
        path.write_text(json.dumps(config))
        return path, ace, vanilla

    def launch(self, benchmark, config, preset="comparison", *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / "plot.py"), benchmark, "--preset", preset, "--config", str(config), "--out", str(self.root / benchmark / f"{preset}.pdf"), *arguments],
            cwd=self.root,
            env={key: value for key, value in os.environ.items() if key != "PRESET"},
            capture_output=True,
            text=True,
        )

    def test_all_benchmarks_keep_measured_visits_and_single_vanilla_visit(self):
        for benchmark in comparison.BENCHMARKS:
            with self.subTest(benchmark=benchmark):
                path, _, _ = self.configuration(benchmark)
                for visit in (3, 8):
                    config, rows = comparison.load_comparison(path, local_epoch=visit, benchmark=benchmark)
                    self.assertEqual(config["benchmark"], benchmark)
                    self.assertEqual([(row["series"], row["value"], row["local_eval_epochs"]) for row in rows], [("ace_rl", visit / 10, visit), ("grpo_vanilla", 0.2, 1)])
                self.assertEqual({row["task_count"] for row in rows}, {2})
                self.assertNotIn("declared in config", {row["split_evidence"] for row in rows})
                for arguments in ((), ("--local-epoch", "3")):
                    result = self.launch(benchmark, path, "comparison", *arguments, "--check")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Validated 2 evaluation points", result.stdout)

    def test_selected_visit_requires_measurement_on_non_appworld_benchmarks(self):
        path, _, _ = self.configuration("mbpp", epochs=3)
        with self.assertRaisesRegex(ValueError, "epoch 8 unavailable"):
            comparison.load_comparison(path, local_epoch=8, benchmark="mbpp")
        result = self.launch("mbpp", path, "comparison", "--local-epoch", "8", "--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("epoch 8 unavailable", result.stderr)

    def test_split_checks_cover_other_completion_flags_and_config_only_launchers(self):
        for benchmark, filename in (("mbpp", "evaluation_complete.json"), ("mini_swe_bench", "evaluation_config.json")):
            with self.subTest(benchmark=benchmark):
                path, ace, _ = self.configuration(benchmark)
                record = ace.parent / filename
                saved = json.loads(record.read_text())
                saved["evaluation"][saved["evaluation"].index("--split") + 1] = "train"
                record.write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, "split disagrees"):
                    comparison.load_comparison(path, benchmark=benchmark)

    def test_benchmark_metadata_cannot_cross_selected_benchmark(self):
        path, ace, _ = self.configuration("mbpp")
        with self.assertRaisesRegex(ValueError, "benchmark disagrees"):
            comparison.load_comparison(path, benchmark="appworld")
        record = ace.parent / "evaluation_config.json"
        saved = json.loads(record.read_text())
        saved["benchmark"] = "mastermind"
        record.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "benchmark disagrees"):
            comparison.load_comparison(path, benchmark="mbpp")

    def test_flat_vanilla_metric_requires_its_recorded_maximum_aggregation(self):
        path, _, vanilla = self.configuration("mastermind")
        saved = json.loads(vanilla.read_text())
        saved["reward_aggregation"] = "last"
        vanilla.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "maximum-aggregated"):
            comparison.load_comparison(path, benchmark="mastermind")

    def test_canonical_evaluation_metadata_keeps_task_slice_checks(self):
        path, ace, _ = self.configuration("mini_swe_bench")
        record = ace.parent / "evaluation_config.json"
        saved = json.loads(record.read_text())
        saved["task_label"] = "2"
        record.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "task label disagrees"):
            comparison.load_comparison(path, benchmark="mini_swe_bench")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib not available")
    def test_non_appworld_comparison_renders_all_formats(self):
        path, _, _ = self.configuration("mastermind")
        result = self.launch("mastermind", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "mastermind/comparison"
        self.assertTrue(output.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))
        self.assertTrue(output.with_suffix(".png").read_bytes().startswith(b"\x89PNG"))
        self.assertIn("grpo_vanilla", output.with_suffix(".csv").read_text())


if __name__ == "__main__":
    unittest.main()
