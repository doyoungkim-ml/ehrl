"""Explicit comparison selection, provenance, and rendering checks without W&B."""

import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/figures"))
SPEC = importlib.util.spec_from_file_location("plot_comparison", ROOT / "scripts/figures/plot_comparison.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class AppWorldComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_path = self.root / "comparison.json"
        self.base_ace = self.artifact("base_ace", 0, 0.3)
        self.config = {
            "title": "AppWorld — Qwen3-4B (thinking)",
            "split": "test_normal",
            "task_label": "all",
            "require_completion": True,
            "series": [
                {"id": "grpo_vanilla", "run": "exact_grpo_name", "points": [self.point(self.artifact("base_vanilla", 0, 0.2), 0), self.point(self.artifact("grpo_vanilla", 10, 0.4), 10)]},
                {"id": "grpo_ace", "run": "exact_grpo_name", "points": [self.point(self.base_ace, 0), self.point(self.artifact("grpo_ace", 10, 0.5), 10)]},
                {"id": "ace_rl", "run": "exact_ace_rl_name", "points": [self.point(self.base_ace, 0), self.point(self.artifact("ace_rl", 10, 0.6), 10)]},
            ],
        }

    def point(self, path, step):
        return {"file": str(path.relative_to(self.root)), "step": step, "provenance": "Explicit synthetic model evaluation"}

    def artifact(self, tag, step, value, ids=("task_a", "task_b"), split="test_normal"):
        path = self.root / f"{tag}_step{step}_tasksall" / f"metrics_step{step}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        first = [{"task_id": identifier} for identifier in ids]
        last = first[1:]
        model = "Qwen/Qwen3-4B" if step == 0 else f"/checkpoints/{tag}/global_step_{step}"
        payload = {
            "model_name": model,
            "evaluations": [
                {
                    "step": step,
                    "metrics": {"success_rate": 0.1, "max_reward_mean": value},
                    "local_eval_epochs": 2,
                    "tasks": last,
                    "local_epochs": [{"local_eval_epoch": 1, "tasks": first}, {"local_eval_epoch": 2, "tasks": last}],
                }
            ],
        }
        path.write_text(json.dumps(payload))
        command = ["python", "evaluate.py", "--test_split", split, "--model_name", model, "--wandb_step", str(step), "--local_eval_epochs", "2"]
        (path.parent / "evaluation_complete.json").write_text(json.dumps({"metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "evaluation": command}))
        return path

    def load(self, metric="max_reward_mean"):
        self.config_path.write_text(json.dumps(self.config))
        return comparison.load_comparison(self.config_path, metric)

    def mutate_artifact(self, change, update_hash=True):
        payload = json.loads(self.base_ace.read_text())
        change(payload)
        self.base_ace.write_text(json.dumps(payload))
        if update_hash:
            marker = self.base_ace.parent / "evaluation_complete.json"
            data = json.loads(marker.read_text())
            data["metrics_sha256"] = hashlib.sha256(self.base_ace.read_bytes()).hexdigest()
            marker.write_text(json.dumps(data))

    def test_shared_base_ace_is_explicit_and_keeps_provenance(self):
        _, rows = self.load()
        base = [row for row in rows if row["step"] == 0 and row["series"] != "grpo_vanilla"]
        self.assertEqual(len(rows), 6)
        self.assertEqual([row["value"] for row in base], [0.3, 0.3])
        self.assertEqual({row["source"] for row in base}, {str(self.base_ace)})
        self.assertEqual({row["task_count"] for row in rows}, {2})
        self.assertEqual({row["split_evidence"] for row in rows}, {"completion record"})
        self.config["series"][2]["points"].pop(0)
        _, rows = self.load()
        self.assertFalse(any(row["series"] == "ace_rl" and row["step"] == 0 for row in rows))

    def test_missing_expected_input_fails_before_outputs(self):
        self.config["series"][2]["points"][1]["file"] = "missing/metrics_step10.json"
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "plot.pdf").exists())
        self.assertFalse((self.root / "plot.csv").exists())

    def test_metric_is_exact_and_nonfinite_values_fail(self):
        with self.assertRaisesRegex(ValueError, "missing exact metric"):
            self.load("avg_reward")
        _, rows = self.load("success_rate")
        self.assertEqual({row["value"] for row in rows}, {0.1})
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value):
                self.mutate_artifact(lambda payload, value=value: payload["evaluations"][0]["metrics"].update(max_reward_mean=value))
                with self.assertRaisesRegex(ValueError, "finite number"):
                    self.load()

    def test_duplicate_step_and_ambiguous_evaluations_fail(self):
        points = self.config["series"][0]["points"]
        points.append(copy.deepcopy(points[0]))
        with self.assertRaisesRegex(ValueError, "duplicate checkpoint step"):
            self.load()
        points.pop()
        self.mutate_artifact(lambda payload: payload["evaluations"].append(copy.deepcopy(payload["evaluations"][0])))
        with self.assertRaisesRegex(ValueError, "exactly one evaluation"):
            self.load()

    def test_filename_entry_and_directory_steps_must_agree(self):
        self.config["series"][1]["points"][0]["step"] = 1
        with self.assertRaisesRegex(ValueError, "filename disagrees"):
            self.load()
        self.config["series"][1]["points"][0]["step"] = 0
        self.mutate_artifact(lambda payload: payload["evaluations"][0].update(step=1))
        with self.assertRaisesRegex(ValueError, "evaluation step disagrees"):
            self.load()

    def test_task_id_difference_rejected_even_if_task_count_matches(self):
        other = self.artifact("other", 10, 0.5, ids=("task_c", "task_d"))
        self.config["series"][2]["points"][1] = self.point(other, 10)
        with self.assertRaisesRegex(ValueError, "task identities differ"):
            self.load()

    def test_split_and_task_slice_disagreement_rejected(self):
        other = self.artifact("other", 10, 0.5, split="dev")
        self.config["series"][2]["points"][1] = self.point(other, 10)
        with self.assertRaisesRegex(ValueError, "split disagrees"):
            self.load()
        self.config["task_label"] = "0-1"
        with self.assertRaisesRegex(ValueError, "task label disagrees"):
            self.load()

    def test_completion_required_and_metrics_hash_checked(self):
        self.mutate_artifact(lambda payload: payload.update(extra="modified"), update_hash=False)
        with self.assertRaisesRegex(ValueError, "completion hash"):
            self.load()
        (self.base_ace.parent / "evaluation_complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing evaluation_complete"):
            self.load()
        self.config["require_completion"] = False
        _, rows = self.load()
        self.assertIn("declared in config", {row["split_evidence"] for row in rows})

    def test_incomplete_epoch_count_rejected(self):
        self.mutate_artifact(lambda payload: payload["evaluations"][0]["local_epochs"].pop())
        with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
            self.load()

    def test_nonbase_source_sharing_rejected(self):
        self.config["series"][2]["points"][1] = copy.deepcopy(self.config["series"][1]["points"][1])
        with self.assertRaisesRegex(ValueError, "shared only"):
            self.load()
        self.config["series"][0]["points"][0] = self.point(self.base_ace, 0)
        with self.assertRaisesRegex(ValueError, "shared only"):
            self.load()

    def test_training_uses_exact_cache_key_and_rejects_duplicate_steps(self):
        cache = self.root / "train.json"
        cache.write_text(json.dumps({"exact_grpo_name": [[1, 0.2], [2, 0.3]], "unrelated": [[1, 1.0]]}))
        self.config["training"] = [
            {"series": "grpo_vanilla", "run": "exact_grpo_name", "metric": "turn/final_avg_reward", "file": "train.json", "provenance": "Local export of explicitly selected training run"}
        ]
        _, rows = self.load()
        training = [row for row in rows if row["kind"] == "training"]
        self.assertEqual([row["value"] for row in training], [0.2, 0.3])
        self.assertEqual({row["source"] for row in training}, {str(cache)})
        self.config["training"][0]["run"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "must match"):
            self.load()
        self.config["training"][0].update(run="exact_grpo_name", data=[[1, 0.2], [1, 0.3]])
        del self.config["training"][0]["file"]
        with self.assertRaisesRegex(ValueError, "duplicate training step"):
            self.load()

    def test_check_does_not_import_plotting_or_network_clients(self):
        self.config_path.write_text(json.dumps(self.config))
        with mock.patch.dict(sys.modules, {"matplotlib": None, "wandb": None}), mock.patch("socket.socket", side_effect=AssertionError("network access")):
            self.assertEqual(comparison.main(["--config", str(self.config_path), "--out", str(self.root / "none.pdf"), "--check"]), 0)
        self.assertFalse((self.root / "none.pdf").exists())

    def launch(self, *args):
        self.config_path.write_text(json.dumps(self.config))
        return subprocess.run(
            [sys.executable, str(ROOT / "plot.py"), "appworld", "--preset", "comparison", "--config", str(self.config_path), "--out", str(self.root / "plot.pdf"), *args],
            cwd=self.root,
            env={**os.environ, "PYTHON_BIN": sys.executable, "PRESET": "comparison"},
            text=True,
            capture_output=True,
        )

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib not available")
    def test_launcher_renders_pdf_png_and_exact_csv_with_optional_training(self):
        self.config["training"] = [{"series": "ace_rl", "run": "exact_ace_rl_name", "metric": "turn/final_avg_reward", "data": [[1, 0.3], [2, 0.45]], "provenance": "Explicit synthetic training data"}]
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "plot.pdf").read_bytes().startswith(b"%PDF"))
        self.assertTrue((self.root / "plot.png").read_bytes().startswith(b"\x89PNG"))
        with (self.root / "plot.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["metric"] for row in rows if row["kind"] == "evaluation"}, {"max_reward_mean"})
        self.assertEqual({row["metric"] for row in rows if row["kind"] == "training"}, {"turn/final_avg_reward"})
        self.assertEqual(len([row for row in rows if row["source"] == str(self.base_ace)]), 2)

    def test_selected_visit_comparison_requires_measured_ace_and_keeps_vanilla(self):
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "epoch 8 unavailable"):
            comparison.load_comparison(self.config_path, local_epoch=8)
        sources = {self.root / point["file"] for item in self.config["series"] if item["id"] != "grpo_vanilla" for point in item["points"]}
        for path in sources:
            payload = json.loads(path.read_text())
            entry = payload["evaluations"][0]
            first = entry["local_epochs"][0]["tasks"]
            entry["local_epochs"] = [{"local_eval_epoch": epoch, "tasks": first, "metrics": {"max_reward_mean": epoch / 10}} for epoch in range(1, 9)]
            entry["local_eval_epochs"] = 8
            path.write_text(json.dumps(payload))
            marker = path.parent / "evaluation_complete.json"
            saved = json.loads(marker.read_text())
            saved["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            saved["evaluation"][saved["evaluation"].index("--local_eval_epochs") + 1] = "8"
            marker.write_text(json.dumps(saved))
        _, rows = comparison.load_comparison(self.config_path, local_epoch=8)
        self.assertEqual({row["value"] for row in rows if row["series"] != "grpo_vanilla"}, {0.8})
        self.assertEqual({row["value"] for row in rows if row["series"] == "grpo_vanilla"}, {0.2, 0.4})
        result = self.launch("--local-epoch", "8", "--check")
        self.assertEqual(result.returncode, 0, result.stderr)

    def configure_final_policy_comparison(self):
        self.config["evaluation_ylabel"] = "Final policy task success rate"
        for series in self.config["series"]:
            if series["id"] == "grpo_vanilla":
                series["metric"] = "success_rate"
                continue
            series["metric"] = "final_policy/avg_reward"
            for point in series["points"]:
                path = self.root / point["file"]
                payload = json.loads(path.read_text())
                entry = payload["evaluations"][0]
                ids = comparison.task_ids(entry, path)
                final = {
                    "kind": "final_policy_only",
                    "after_local_eval_epoch": 2,
                    "policy_visit": 3,
                    "adaptation_applied": False,
                    "tasks": [{"task_id": identifier, "reward": index} for index, identifier in enumerate(ids)],
                    "metrics": {"avg_reward": 0.5},
                    "work": {"policy_rollouts": len(ids), "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0},
                }
                entry["final_policy_evaluation"] = final
                entry["metrics"]["final_policy/avg_reward"] = 0.5
                path.write_text(json.dumps(payload))
                marker = path.parent / "evaluation_complete.json"
                saved = json.loads(marker.read_text())
                if "--final_policy_eval" not in saved["evaluation"]:
                    saved["evaluation"].append("--final_policy_eval")
                saved["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                marker.write_text(json.dumps(saved))

    def configure_warm_start_final_policy_comparison(self, *, empty=False):
        self.configure_final_policy_comparison()
        paths = {self.root / point["file"] for series in self.config["series"] if series["id"] != "grpo_vanilla" for point in series["points"]}
        for path in paths:
            payload = json.loads(path.read_text())
            entry = payload["evaluations"][0]
            ids = [row["task_id"] for row in entry["final_policy_evaluation"]["tasks"]]
            observed = [] if empty else [{"task_id": ids[-1]}]
            entry["evaluation_task_ids"] = ids
            entry["local_eval_epochs"] = 3
            entry["tasks"] = observed
            entry["local_epochs"] = [{"local_eval_epoch": number, "tasks": observed, "metrics": {"n_solved": len(ids)}} for number in (2, 3)]
            entry["final_policy_evaluation"].update(after_local_eval_epoch=3, policy_visit=4)
            path.write_text(json.dumps(payload))
            marker = path.parent / "evaluation_complete.json"
            saved = json.loads(marker.read_text())
            saved["evaluation"].extend(["--start_epoch", "1"])
            saved["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            marker.write_text(json.dumps(saved))

    def test_final_policy_warm_start_uses_declared_task_set_with_empty_adaptation(self):
        self.configure_warm_start_final_policy_comparison(empty=True)
        _, rows = self.load()
        self.assertEqual({row["task_count"] for row in rows}, {2})
        self.assertEqual({row["value"] for row in rows if row["series"] != "grpo_vanilla"}, {0.5})

    def test_final_policy_warm_start_allows_observed_subset_but_validates_declaration(self):
        self.configure_warm_start_final_policy_comparison()
        _, rows = self.load()
        self.assertEqual({row["task_count"] for row in rows}, {2})
        for declared in (None, "task_a,task_b", [], ["task_a", "task_a"], ["task_a", None], ["task_a", " "]):
            with self.subTest(declared=declared):
                self.mutate_artifact(lambda payload, declared=declared: payload["evaluations"][0].update(evaluation_task_ids=declared))
                with self.assertRaisesRegex(ValueError, "evaluation_task_ids"):
                    self.load()
        self.mutate_artifact(lambda payload: payload["evaluations"][0].update(evaluation_task_ids=["task_a", "task_b"]))
        self.mutate_artifact(lambda payload: payload["evaluations"][0]["local_epochs"][0].update(tasks=[{"task_id": "unknown"}]))
        with self.assertRaisesRegex(ValueError, "outside evaluation_task_ids"):
            self.load()

    def test_final_policy_declared_task_set_requires_exact_final_coverage(self):
        self.configure_warm_start_final_policy_comparison(empty=True)
        for declared in (["task_a"], ["task_a", "task_b", "task_c"], ["task_a", "task_c"]):
            with self.subTest(declared=declared):
                self.mutate_artifact(lambda payload, declared=declared: payload["evaluations"][0].update(evaluation_task_ids=declared))
                with self.assertRaisesRegex(ValueError, "final policy task coverage"):
                    self.load()

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib not available")
    def test_final_policy_warm_start_renders_full_task_coverage(self):
        self.configure_warm_start_final_policy_comparison(empty=True)
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "plot.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["task_count"] for row in rows}, {"2"})
        self.assertEqual({row["metric"] for row in rows}, {"success_rate", "final_policy/avg_reward"})

    def test_final_policy_series_select_exact_metric_without_changing_vanilla(self):
        self.configure_final_policy_comparison()
        _, rows = self.load()
        self.assertEqual({row["metric"] for row in rows if row["series"] == "grpo_vanilla"}, {"success_rate"})
        self.assertEqual({row["value"] for row in rows if row["series"] == "grpo_vanilla"}, {0.1})
        self.assertEqual({row["metric"] for row in rows if row["series"] != "grpo_vanilla"}, {"final_policy/avg_reward"})
        self.assertEqual({row["value"] for row in rows if row["series"] != "grpo_vanilla"}, {0.5})
        del self.config["evaluation_ylabel"]
        with self.assertRaisesRegex(ValueError, "explicit evaluation_ylabel"):
            self.load()

    def test_final_policy_directory_suffix_preserves_step_and_task_validation(self):
        self.configure_final_policy_comparison()
        moved = {}
        for series in self.config["series"]:
            if series["id"] == "grpo_vanilla":
                continue
            for point in series["points"]:
                old = self.root / point["file"]
                if old not in moved:
                    destination = old.parent.with_name(old.parent.name + "_final_policy")
                    old.parent.rename(destination)
                    moved[old] = destination / old.name
                point["file"] = str(moved[old].relative_to(self.root))
        _, rows = self.load()
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["task_label"] for row in rows}, {"all"})
        self.config["task_label"] = "other"
        with self.assertRaisesRegex(ValueError, "directory step or task label"):
            self.load()

    def test_final_policy_metric_requires_complete_task_pass_and_matching_record(self):
        self.configure_final_policy_comparison()
        self.mutate_artifact(lambda payload: payload["evaluations"][0]["final_policy_evaluation"]["tasks"].pop())
        with self.assertRaisesRegex(ValueError, "final policy task coverage"):
            self.load()
        self.configure_final_policy_comparison()
        self.mutate_artifact(lambda payload: payload["evaluations"][0]["final_policy_evaluation"]["metrics"].update(avg_reward=0.7))
        with self.assertRaisesRegex(ValueError, "metric disagrees"):
            self.load()
        self.configure_final_policy_comparison()
        self.mutate_artifact(lambda payload: payload["evaluations"][0]["final_policy_evaluation"]["work"].update(reflector_rollouts=1))
        with self.assertRaisesRegex(ValueError, "contains adaptation"):
            self.load()
        self.configure_final_policy_comparison()
        marker = self.base_ace.parent / "evaluation_complete.json"
        saved = json.loads(marker.read_text())
        saved["evaluation"].remove("--final_policy_eval")
        marker.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "did not request"):
            self.load()

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib not available")
    def test_final_policy_comparison_renders_explicit_series_metrics(self):
        self.configure_final_policy_comparison()
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "plot.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["metric"] for row in rows}, {"success_rate", "final_policy/avg_reward"})


if __name__ == "__main__":
    unittest.main()
