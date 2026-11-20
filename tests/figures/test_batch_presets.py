"""All-presets selection, argument routing, process isolation, and failures."""

import contextlib
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("figure_batch", ROOT / "scripts/figures/batch.py")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)
PRESETS = json.loads((ROOT / "scripts/figures/presets.json").read_text())["appworld"]


class BatchPresetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_batch(self, *, presets=None, run=None, config=None, output_dir=None, remaining=None, jobs=1, resolve=False):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = batch.run_all(
                "appworld",
                PRESETS if presets is None else presets,
                run=run,
                config=config,
                output_dir=output_dir,
                remaining=remaining or [],
                jobs=jobs,
                resolve=resolve,
            )
        return result, stdout.getvalue(), stderr.getvalue()

    def training(self):
        (self.root / "metrics.jsonl").write_text(json.dumps({"training/global_step": 10, "critic/score/mean": 0.5}) + "\n")

    def evaluation(self, preset="visits8", epochs=3):
        directory = self.root / "eval" / preset / "step_10"
        directory.mkdir(parents=True, exist_ok=True)
        config = {"benchmark": "appworld", "method": "ace", "preset": preset, "step": 10, "task_label": "all", "max_turns": 2, "run_dir": str(self.root)}
        (directory / "evaluation_config.json").write_text(json.dumps(config))
        epoch_rows = [{"local_eval_epoch": epoch, "tasks": [{"task_id": "task", "reward": 0.5}], "metrics": {"max_reward_mean": 0.5}} for epoch in range(1, epochs + 1)]
        payload = {"evaluations": [{"step": 10, "local_eval_epochs": epochs, "local_epochs": epoch_rows, "metrics": {"max_reward_mean": 0.5}, "evaluation_task_ids": ["task"]}]}
        (directory / "metrics_step10.json").write_text(json.dumps(payload))
        trajectories = directory / "task_task" / "chat_completions" / "eval"
        trajectories.mkdir(parents=True)
        for epoch in range(epochs):
            (trajectories / f"policy_step000010_iter_{epoch}.json").write_text(json.dumps({"step": 10, "trajectories": [{"steps": [{"reward": 0.0}, {"reward": 0.5}]}]}))
        return directory

    def test_resolve_reports_skips_and_separate_outputs_without_writes(self):
        self.evaluation()
        before = set(self.root.rglob("*"))
        output = self.root / "plots"
        with mock.patch.object(batch.subprocess, "run") as child:
            code, stdout, stderr = self.run_batch(run=self.root, output_dir=output, resolve=True)
        self.assertEqual(code, 0, stderr)
        plans = {plan["preset"]: plan for plan in json.loads(stdout)}
        self.assertNotIn("turnfreeze", plans)
        self.assertEqual(plans["training"]["status"], "skipped")
        self.assertEqual(plans["comparison"]["reason"], "comparison requires --config")
        self.assertEqual(set(plans), {"training", "eval", "turns", "iterations", "comparison", "campaign-report"})
        self.assertEqual(plans["eval"]["status"], "selected")
        self.assertEqual(plans["eval"]["output_dir"], str(output / "eval"))
        child.assert_not_called()
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_options_route_as_complete_groups_including_repeats_inline_and_negative(self):
        self.training()
        self.evaluation()
        options = ["--metrics-file=export.jsonl", "--title", "My plot", "--step", "-2", "--step=10", "--run-tag", "first", "--run-tag=second", "--check"]
        code, stdout, stderr = self.run_batch(run=self.root, remaining=options, resolve=True)
        self.assertEqual(code, 0, stderr)
        plans = {plan["preset"]: plan for plan in json.loads(stdout)}
        self.assertEqual(plans["training"]["arguments"], ["--metrics-file=export.jsonl", "--title", "My plot", "--check"])
        self.assertEqual(plans["eval"]["arguments"], ["--title", "My plot", "--run-tag", "first", "--run-tag=second", "--check"])
        self.assertEqual(plans["turns"]["arguments"], ["--step", "-2", "--step=10", "--run-tag", "first", "--run-tag=second", "--check"])

    def test_unknown_options_missing_values_and_colliding_outputs_fail_before_launch(self):
        for options in (["--made-up", "value"], ["--step"], ["--metric", "--check"], ["--check=yes"], ["--out", "shared.pdf"]):
            with self.subTest(options=options), mock.patch.object(batch.subprocess, "run") as child:
                code, _, stderr = self.run_batch(run=self.root, remaining=options)
                self.assertEqual(code, 2)
                self.assertIn("error:", stderr)
                child.assert_not_called()

    def test_config_routes_only_to_supported_presets_and_preserves_scalar_inputs(self):
        self.training()
        self.evaluation()
        config = self.root / "comparison.json"
        options = [f"--runs-dir={self.root / 'eval'}", "--check"]
        code, stdout, stderr = self.run_batch(run=self.root, config=config, remaining=options, resolve=True)
        self.assertEqual(code, 0, stderr)
        plans = {plan["preset"]: plan for plan in json.loads(stdout)}
        for name in ("training", "eval"):
            self.assertIsNone(plans[name]["config"])
            self.assertNotIn("--config", plans[name]["command"])
        self.assertIn(options[0], plans["eval"]["arguments"])
        for name in ("comparison", "turns", "iterations"):
            self.assertEqual(plans[name]["config"], str(config))
            self.assertNotIn(options[0], plans[name]["arguments"])
        _, stdout, _ = self.run_batch(config=config, resolve=True)
        plans = {plan["preset"]: plan for plan in json.loads(stdout)}
        self.assertEqual(plans["training"]["status"], "skipped")
        self.assertEqual(plans["eval"]["status"], "skipped")
        self.assertEqual(plans["comparison"]["status"], "selected")

    def test_real_check_validates_available_presets_without_rendering(self):
        self.training()
        self.evaluation()
        code, stdout, stderr = self.run_batch(run=self.root, remaining=["--check"], jobs=2)
        self.assertEqual(code, 0, stderr)
        self.assertIn("4 succeeded, 0 failed, 2 skipped", stdout)
        self.assertIn("Validated", stdout)
        self.assertFalse((self.root / "figures").exists())

    def test_entrypoint_runs_all_presets_with_parallel_check(self):
        self.training()
        self.evaluation()
        command = [sys.executable, str(ROOT / "plot.py"), "appworld", "--preset", "all", "--jobs", "2", "--run", str(self.root), "--check"]
        result = subprocess.run(command, cwd=self.root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("4 succeeded, 0 failed, 2 skipped", result.stdout)
        self.assertFalse((self.root / "figures").exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib not available")
    def test_entrypoint_renders_all_measured_presets_to_separate_directories(self):
        self.training()
        self.evaluation()
        output = self.root / "all figures"
        command = [
            sys.executable, str(ROOT / "plot.py"), "appworld",
            "--preset", "all", "--jobs", "2", "--run", str(self.root), "--output-dir", str(output),
        ]
        result = subprocess.run(command, cwd=self.root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("4 succeeded, 0 failed, 2 skipped", result.stdout)
        expected = {"training", "eval", "turns", "iterations"}
        self.assertEqual({path.name for path in output.iterdir()}, expected)
        for preset in expected:
            with self.subTest(preset=preset):
                stem = output / preset / preset
                self.assertTrue(stem.with_suffix(".pdf").read_bytes().startswith(b"%PDF-"))
                self.assertTrue(stem.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                with stem.with_suffix(".csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertGreater(len(rows), 0)
                expected_values = {0.0, 0.5} if preset.startswith("turns") else {0.5}
                self.assertEqual({float(row["value"]) for row in rows}, expected_values)
        self.assertIn("[comparison] SKIPPED: comparison requires --config", result.stdout)
        self.assertNotIn("[turnfreeze]", result.stdout)
        self.assertFalse((self.root / "figures").exists())

    def test_real_failure_continues_other_presets_and_returns_nonzero(self):
        self.training()
        directory = self.evaluation()
        (directory / "task_task/chat_completions/eval/policy_step000010_iter_1.json").unlink()
        code, stdout, stderr = self.run_batch(run=self.root, remaining=["--check"], jobs=2)
        self.assertEqual(code, 1)
        self.assertIn("[turns] FAILED", stdout)
        self.assertIn("[iterations] OK", stdout)
        self.assertIn("3 succeeded, 1 failed, 2 skipped", stdout)
        self.assertIn("policy_step000010_iter_1.json", stderr)

    def test_jobs_runs_separate_interpreters_concurrently_and_waits_for_failures(self):
        self.training()
        self.evaluation()
        barrier = threading.Barrier(2)
        commands = []
        lock = threading.Lock()

        def child(command, **kwargs):
            with lock:
                commands.append(command)
            barrier.wait(timeout=5)
            self.assertEqual(command[0], sys.executable)
            self.assertTrue(kwargs["capture_output"])
            preset = command[command.index("--preset") + 1]
            return subprocess.CompletedProcess(command, int(preset == "eval"), "checked\n", "bad metric\n" if preset == "eval" else "")

        presets = {name: PRESETS[name] for name in ("training", "eval")}
        with mock.patch.object(batch.subprocess, "run", side_effect=child):
            code, stdout, stderr = self.run_batch(presets=presets, run=self.root, jobs=2)
        self.assertEqual(code, 1)
        self.assertEqual(len(commands), 2)
        self.assertIn("1 succeeded, 1 failed, 0 skipped", stdout)
        self.assertIn("bad metric", stderr)

    def test_no_available_presets_and_invalid_jobs_report_failure(self):
        code, stdout, stderr = self.run_batch(run=self.root)
        self.assertEqual(code, 1)
        self.assertIn("No presets could run", stderr)
        self.assertIn("0 succeeded", stdout)
        code, _, stderr = self.run_batch(run=self.root, jobs=0)
        self.assertEqual(code, 2)
        self.assertIn("positive integer", stderr)


if __name__ == "__main__":
    unittest.main()
