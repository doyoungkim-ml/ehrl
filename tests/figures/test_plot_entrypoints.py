"""Preset routing, explicit inputs, rollback history, and preserved plot methods."""

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
PRESETS = json.loads((ROOT / "scripts/figures/presets.json").read_text())
SPEC = importlib.util.spec_from_file_location("plot_training", ROOT / "scripts/figures/plot_training.py")
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


class PlotEntrypointsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {key: value for key, value in os.environ.items() if key not in ("PRESET", "TASK_LABEL", "FIGURE_DATA_DIR", "FIGURE_OUTPUT_DIR")}

    def command(self, benchmark, *args, env=None):
        return subprocess.run([sys.executable, str(ROOT / "plot.py"), benchmark, *args], cwd=self.root, env={**self.env, **(env or {})}, text=True, capture_output=True)

    def test_every_preset_has_help_and_resolves_without_importing_its_dependencies(self):
        for benchmark, presets in PRESETS.items():
            for preset, script in presets.items():
                with self.subTest(benchmark=benchmark, preset=preset):
                    help_result = self.command(benchmark, "--preset", preset, "--help")
                    self.assertEqual(help_result.returncode, 0, help_result.stderr)
                    self.assertIn(f"Selected preset: {preset}", help_result.stdout)
                    resolved = self.command(benchmark, "--preset", preset, "--resolve")
                    self.assertEqual(resolved.returncode, 0, resolved.stderr)
                    data = json.loads(resolved.stdout)
                    self.assertEqual(data["method"], str(ROOT / script))
                    self.assertTrue(Path(data["method"]).is_file())
                    self.assertEqual(data["output_dir"], str(ROOT / "examples" / benchmark / "figures" / "output" / preset))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_argument_preset_overrides_environment_and_run_owns_outputs(self):
        result = self.command("appworld", "--preset", "training", "--run", str(self.root), "--resolve", env={"PRESET": "comparison"})
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["preset"], "training")
        self.assertEqual(data["output_dir"], str(self.root / "figures/training"))

    def test_default_does_not_hunt_for_other_experiments(self):
        result = self.command("appworld", "--preset", "training", "--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --run", result.stderr)
        result = self.command("appworld", "--preset", "eval", "--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --run", result.stderr)

    def history(self):
        records = [
            {"training/global_step": step, "logging/index": index, "training/resume_from_step": 190, "critic/score/mean": value}
            for index, (step, value) in enumerate([(190, 0.2), (191, 0.3), (190, 0.25), (191, 0.35)])
        ]
        path = self.root / "metrics.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        return path

    def test_training_step_does_not_use_logging_index_or_discard_replay(self):
        points = training.load_points(self.history(), ["critic/score/mean"])
        self.assertEqual([point["training_step"] for point in points], [190, 191, 190, 191])
        self.assertEqual([point["logging_index"] for point in points], [0, 1, 2, 3])
        self.assertEqual([point["segment"] for point in points], [0, 0, 1, 1])

    def test_missing_training_step_requires_explicit_legacy_step_key(self):
        path = self.root / "legacy.jsonl"
        path.write_text(json.dumps({"_step": 77, "critic/score/mean": 0.2}) + "\n")
        with self.assertRaisesRegex(ValueError, "no 'training/global_step'"):
            training.load_points(path, ["critic/score/mean"])
        points = training.load_points(path, ["critic/score/mean"], "_step")
        self.assertEqual(points[0]["training_step"], 77)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib not available")
    def test_training_renders_and_exports_all_observations(self):
        self.history()
        result = self.command("mastermind", "--run", str(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.root / "figures/training/training"
        self.assertTrue(out.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))
        self.assertTrue(out.with_suffix(".png").read_bytes().startswith(b"\x89PNG"))
        with out.with_suffix(".csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["training_step"]) for row in rows], [190, 191, 190, 191])

    def test_dated_presets_are_not_active(self):
        result = self.command("appworld", "--preset", "turnfreeze_prior0829", "--resolve")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown preset", result.stderr)


if __name__ == "__main__":
    unittest.main()
