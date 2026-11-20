"""Measured turn padding, iteration selection, and visit-limited views."""

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
import plot_eval  # noqa: E402
import plot_turns  # noqa: E402


class MeasuredRewardPlotsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, count=3, preset="visits8", artifact_subdir=""):
        result = self.root / "eval" / preset / "step_10"
        result.mkdir(parents=True, exist_ok=True)
        config = {"preset": preset, "method": "ace", "step": 10, "task_label": "all", "run_dir": str(self.root), "max_turns": 3}
        if artifact_subdir:
            # Mastermind ACE records its snapshot directory below the run directory.
            config["run_dir"] = str(result)
            config["evaluation"] = ["python", "--playbook_snapshot_dir", str(result / artifact_subdir), "--max_turns", "3"]
        (result / "evaluation_config.json").write_text(json.dumps(config))
        epochs = []
        for epoch in range(1, count + 1):
            tasks = [{"task_id": "a", "reward": 1.0}, {"task_id": "b", "reward": 0.3}] if epoch == 1 else [{"task_id": "b", "reward": 0.2}]
            epochs.append({"local_eval_epoch": epoch, "tasks": tasks, "metrics": {"max_reward_mean": 0.65 if epoch == 1 else 0.6}})
            self.policy(result / artifact_subdir, "b", epoch, [[0.6, 0.2], [0.2, 0.4]] if epoch == 1 else [[0.1], [0.3]])
            if epoch == 1:
                self.policy(result / artifact_subdir, "a", epoch, [[0, 1], [0.4, 1]])
        entry = {"step": 10, "local_eval_epochs": count, "local_epochs": epochs, "metrics": epochs[-1]["metrics"], "evaluation_task_ids": ["a", "b"]}
        (result / "metrics_step10.json").write_text(json.dumps({"evaluations": [entry]}))
        return result

    def policy(self, result, task_id, epoch, rewards):
        folder = result / f"task_{task_id}/chat_completions/eval"
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"step": 10, "trajectories": [{"steps": [{"reward": value} for value in trace], "info": {"step_rewards": [99]}} for trace in rewards]}
        (folder / f"policy_step000010_iter_{epoch - 1}.json").write_text(json.dumps(payload))

    def flat_vanilla_fixture(self, budgets=(3, 3), command=None):
        """Mastermind vanilla evaluator output: one flat metrics record plus trajectories.json."""
        result = self.root / "eval" / "vanilla" / "results" / "pass8" / "qwen3-0.6b-nonthinking"
        result.mkdir(parents=True, exist_ok=True)
        config = {"benchmark": "mastermind", "method": "vanilla", "preset": "pass8", "step": 0, "task_label": "all", "run_dir": str(result)}
        if command is not None:
            config["evaluation"] = command
        (result / "evaluation_config.json").write_text(json.dumps(config))
        # Each game records the effective (possibly overridden) turn budget it was played with.
        task = lambda uid, budget: {"uid": uid, **({} if budget is None else {"max_turns": budget})}
        games = [
            {"task_id": "a", "task": task("a", budgets[0]), "reward_max": 1.0, "steps": [{"reward": 0.0}, {"reward": 1.0}]},
            {"task_id": "b", "task": task("b", budgets[1]), "reward_max": 0.4, "steps": [{"reward": 0.4}, {"reward": 0.2}, {"reward": 0.1}]},
        ]
        (result / "trajectories.json").write_text(json.dumps(games))
        plot_turns._FLAT_GAMES.clear()  # the reader caches each trajectories.json once per process
        record = {
            "dataset": "mastermind",
            "num_tasks": 2,
            "success_count": 1,
            "success_rate": 0.5,
            "avg_reward": 0.7,
            "reward_aggregation": "max",
            "tasks": [{"task_id": "a", "success": True, "reward": 1.0}, {"task_id": "b", "success": False, "reward": 0.4}],
            "run_tag": "qwen3-0.6b-nonthinking",
            "step": 0,
        }
        (result / "metrics_step0.json").write_text(json.dumps(record))
        return result

    def test_flat_vanilla_record_is_one_measured_visit(self):
        self.flat_vanilla_fixture()
        points = plot_eval.load_points(self.root / "eval", "max_reward_mean", "all")
        self.assertEqual([(point["variant"], point["step"], point["value"], point["local_eval_epochs"]) for point in points], [("vanilla/pass8", 0, 0.7, 1)])
        rows = plot_turns.load_points(self.root / "eval", task_label="all")
        self.assertEqual([row["turn_global"] for row in rows], [1, 2, 3])
        # Turn 1 averages 0.0 and 0.4; the solved game then pads at 1 while the partial game stays fresh.
        for row, expected in zip(rows, [0.2, 0.6, 0.55], strict=True):
            self.assertAlmostEqual(row["value"], expected)
        self.assertTrue(all(json.loads(row["trajectory_sources"])[0].endswith("trajectories.json") for row in rows))
        iterations = plot_turns.load_points(self.root / "eval", task_label="all", axis="iteration")
        self.assertEqual([(row["local_epoch"], row["value"]) for row in iterations], [(1, 0.7)])

    def test_flat_vanilla_record_requires_maximum_aggregation_and_game_budgets(self):
        result = self.flat_vanilla_fixture()
        metrics = result / "metrics_step0.json"
        record = json.loads(metrics.read_text())
        record["reward_aggregation"] = "last"
        metrics.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "maximum-aggregated"):
            plot_eval.load_points(self.root / "eval", "max_reward_mean", "all")
        self.flat_vanilla_fixture(budgets=(3, 2))
        with self.assertRaisesRegex(ValueError, "cover every"):
            plot_turns.load_points(self.root / "eval", task_label="all", max_turns=2)
        self.assertEqual(len(plot_turns.load_points(self.root / "eval", task_label="all")), 3)

    def test_flat_vanilla_turn_budget_comes_from_recorded_command_or_games(self):
        # Mastermind vanilla spells its option --max-turns; the command wins over game records.
        self.flat_vanilla_fixture(budgets=(5, 5), command=["python", "-m", "examples.mastermind.eval.evaluate", "--max-turns", "4"])
        self.assertEqual([row["turn_global"] for row in plot_turns.load_points(self.root / "eval", task_label="all")], [1, 2, 3, 4])
        self.flat_vanilla_fixture(budgets=(5, 5), command=["python", "-m", "examples.mastermind.eval.evaluate"])
        self.assertEqual(len(plot_turns.load_points(self.root / "eval", task_label="all")), 5)
        self.flat_vanilla_fixture(budgets=(5, None), command=["python"])
        with self.assertRaisesRegex(ValueError, "recorded max_turns budget"):
            plot_turns.load_points(self.root / "eval", task_label="all")

    def test_turn_padding_uses_max_and_all_rollouts_without_smoothing_partials(self):
        self.fixture()
        rows = plot_turns.load_points(self.root / "eval")
        self.assertEqual(len(rows), 9)
        self.assertEqual([row["turn_global"] for row in rows], list(range(1, 10)))
        for row, expected in zip(rows, [0.3, 0.65, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], strict=True):
            self.assertAlmostEqual(row["value"], expected)
        # Task a is proven solved and legitimately has no second-epoch trajectory.
        self.assertEqual(len(json.loads(rows[3]["trajectory_sources"])), 1)

    def test_recorded_playbook_snapshot_directory_locates_nested_artifacts(self):
        result = self.fixture(artifact_subdir="playbooks")
        rows = plot_turns.load_points(self.root / "eval")
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(str(result / "playbooks") in row["trajectory_sources"] for row in rows))
        config = result / "evaluation_config.json"
        data = json.loads(config.read_text())
        data["evaluation"][2] = str(self.root / "elsewhere")
        config.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "outside the recorded run directory"):
            plot_turns.load_points(self.root / "eval")

    def test_missing_unsolved_task_fails_instead_of_padding_as_success(self):
        result = self.fixture()
        path = result / "metrics_step10.json"
        payload = json.loads(path.read_text())
        payload["evaluations"][0]["local_epochs"][1]["tasks"] = []
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "missing unsolved"):
            plot_turns.load_points(self.root / "eval")

    def test_unknown_turn_budget_requires_explicit_selection(self):
        result = self.fixture()
        config = result / "evaluation_config.json"
        data = json.loads(config.read_text())
        del data["max_turns"]
        config.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "pass --max-turns"):
            plot_turns.load_points(self.root / "eval")
        self.assertEqual(len(plot_turns.load_points(self.root / "eval", max_turns=3)), 9)
        with self.assertRaisesRegex(ValueError, "cover every"):
            plot_turns.load_points(self.root / "eval", max_turns=1)

    def test_visit_limited_views_require_actual_data(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, "epoch 8 unavailable"):
            plot_turns.load_points(self.root / "eval", local_epoch=8)
        with self.assertRaisesRegex(ValueError, "epoch 8 is unavailable"):
            plot_eval.load_points(self.root / "eval", "max_reward_mean", local_epoch=8)
        self.fixture(count=8)
        rows = plot_turns.load_points(self.root / "eval", local_epoch=8, eval_preset="visits8")
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(plot_turns.load_points(self.root / "eval", local_epoch=3)), 9)
        points = plot_eval.load_points(self.root / "eval", "max_reward_mean", local_epoch=8, eval_preset="visits8")
        self.assertEqual([(point["local_epoch"], point["value"]) for point in points], [(8, 0.6)])

    def test_iteration_rewards_use_exact_recorded_scalar(self):
        self.fixture()
        points = plot_turns.load_points(self.root / "eval", axis="iteration", local_epoch=2)
        self.assertEqual([point["local_epoch"] for point in points], [1, 2])
        self.assertEqual([point["value"] for point in points], [0.65, 0.6])

    def test_turns_read_trajectories_only_beside_their_metrics(self):
        result = self.fixture()
        path = result / "metrics_step10.json"
        payload = json.loads(path.read_text())
        for epoch in payload["evaluations"][0]["local_epochs"]:
            epoch["source_run_dir"] = str(self.root / "elsewhere")
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "other evaluation directories"):
            plot_turns.load_points(self.root / "eval")

    def test_turns_refuse_missing_earlier_epoch_state(self):
        result = self.fixture(count=8)
        path = result / "metrics_step10.json"
        payload = json.loads(path.read_text())
        payload["evaluations"][0]["local_epochs"] = payload["evaluations"][0]["local_epochs"][3:]
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "starting at 1"):
            plot_turns.load_points(self.root / "eval")
        rows = plot_turns.load_points(self.root / "eval", axis="iteration", local_epoch=6)
        self.assertEqual([row["local_epoch"] for row in rows], [4, 5, 6])

    def test_retired_artifacts_are_not_auto_discovered(self):
        original = self.fixture()
        retired = self.root / "historical"
        retired.mkdir()
        original.rename(retired / "step_10")
        with self.assertRaisesRegex(ValueError, "No matching"):
            plot_eval.load_points(self.root, "max_reward_mean")

    def launch(self, preset, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "plot.py"), "appworld", "--preset", preset, "--run", str(self.root), *args],
            env={key: value for key, value in os.environ.items() if key != "PRESET"},
            cwd=self.root,
            text=True,
            capture_output=True,
        )

    def test_existing_comparison_config_drives_turn_and_iteration_presets(self):
        result = self.fixture(count=8)
        metrics = result / "metrics_step10.json"
        payload = json.loads(metrics.read_text())
        payload["model_name"] = "test-model"
        metrics.write_text(json.dumps(payload))
        config = self.root / "comparison.json"
        config.write_text(
            json.dumps(
                {
                    "title": "Synthetic measured data",
                    "split": "dev",
                    "task_label": "all",
                    "series": [{"id": "ace_rl", "run": "run", "points": [{"file": str(metrics), "step": 10, "provenance": "Synthetic test"}]}],
                }
            )
        )
        rows = plot_turns.load_points(config=config, local_epoch=8)
        self.assertEqual(len(rows), 24)
        for preset in ("turns", "iterations"):
            launched = self.launch(preset, "--config", str(config), "--local-epoch", "3", "--check")
            self.assertEqual(launched.returncode, 0, launched.stderr)

    def test_presets_select_visits_and_removed_six_visit_presets_are_rejected(self):
        self.fixture(count=8)
        for preset, args, count in (("eval", (), 1), ("turns", ("--local-epoch", "3"), 9), ("iterations", (), 8)):
            with self.subTest(preset=preset):
                result = self.launch(preset, *args, "--check")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"Validated {count}", result.stdout)
        for preset in ("eval6", "turns6", "iterations6", "comparison6"):
            with self.subTest(preset=preset):
                result = self.launch(preset, "--check")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unknown preset", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib not available")
    def test_rendered_turn_and_iteration_csv_match_measured_rows(self):
        self.fixture(count=8)
        for preset, count in (("turns", 24), ("iterations", 8), ("eval", 1)):
            result = self.launch(preset)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = self.root / "figures" / preset / preset
            self.assertTrue(output.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))
            self.assertTrue(output.with_suffix(".png").read_bytes().startswith(b"\x89PNG"))
            with output.with_suffix(".csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), count)


if __name__ == "__main__":
    unittest.main()
