"""The simple plot command selects canonical runs and preserves missing-data evidence."""

import importlib.util
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/figures"))
import gallery_render  # noqa: E402
from gallery_render import COLORS, GRPO_RSI_COLOR, read_tables  # noqa: E402
from quickplot import (  # noqa: E402
    add_turns, eight_visit_evaluations, export_tables, inventory, method_for, padded_reward,
    merge_tables, rsi_success, training_token_history, write_tables,
)
from remote_turns import READER_FILES, REMOTE_CODE, TURN_KEYS, TURN_SPECS  # noqa: E402
from remote_gallery import REMOTE_CODE as GALLERY_CODE, fetch_gallery_rows  # noqa: E402


def visits8_point(step, source, **extra):
    """A loaded ACE point carries its measured RSI@1..8 curve, as extraction attaches it."""
    point = {"step": step, "source": str(source), **extra}
    point["curve"] = [{"k": k, "value": rsi_success(point, k)} for k in range(1, 9)]
    return point


def write_visits8(directory, step, rewards, *, task="a", config=None):
    """Write one sequential visits8 result with one fresh reward per measured visit."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluation_config.json").write_text(json.dumps({
        "step": step, "preset": "visits8", "task_label": "all", "method": "ace", **(config or {}),
    }))
    source = directory / f"metrics_step{step}.json"
    source.write_text(json.dumps({"evaluations": [{
        "step": step, "local_eval_protocol": "sequential_adaptation_v1", "g_act": 1,
        "local_eval_epochs": len(rewards), "evaluation_task_ids": [task],
        "metrics": {"max_reward_mean": rewards[-1]},
        "local_epochs": [{"local_eval_epoch": visit, "tasks": [{"task_id": task, "reward": reward}],
                          "metrics": {"max_reward_mean": reward}}
                         for visit, reward in enumerate(rewards, 1)],
    }]}))
    return source


def csv_roundtrip(data):
    """Pass loaded run records through the extracted CSV files the renderer reads."""
    with tempfile.TemporaryDirectory() as directory:
        write_tables(Path(directory), export_tables(data), {
            "benchmark": "appworld", "model": "model", "loaded_methods": list(data),
            "rollout_metrics": {method: record.get("rollout_keys", []) for method, record in data.items()},
        })
        return read_tables(Path(directory))


def plot_curves(name, benchmark, model, data, out, methods, *, token_axis=None):
    tables, manifest = csv_roundtrip(data)
    gallery_render.plot_curves(name, benchmark, model, tables, manifest, out, methods, token_axis=token_axis)


def plot_turns_method(benchmark, model, method, data, out, **options):
    tables, _ = csv_roundtrip(data)
    gallery_render.plot_turns_method(benchmark, model, method, tables, out, **options)


class AppendTest(unittest.TestCase):
    def test_append_keeps_old_rows_overwrites_measured_rows_and_never_blanks_cells(self):
        row = lambda step, value, tokens: {
            "method": "ace_rl", "measure": "rsi", "k": 3, "step": step,
            "cumulative_total_gradient_update_tokens": tokens,
            "cumulative_total_completion_tokens": None, "value": value,
        }
        old = {"training": [], "turns": [], "evaluation": [row(10, 0.1, 100.0), row(20, 0.2, 200.0)]}
        new = {"training": [], "turns": [], "evaluation": [row(20, 0.25, None), row(30, 0.3, 300.0)]}
        merged = merge_tables(old, new)["evaluation"]
        self.assertEqual([(r["step"], r["value"], r["cumulative_total_gradient_update_tokens"]) for r in merged],
                         [(10, 0.1, 100.0), (20, 0.25, 200.0), (30, 0.3, 300.0)])


class SelectionTest(unittest.TestCase):
    def test_token_history_counts_replayed_steps_and_uses_latest_exact_step(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "metrics.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in (
                {"training/global_step": 1, "rollout/total_gradient_update_tokens": 10,
                 "rollout/total_completion_tokens": 15},
                {"training/global_step": 2, "rollout/total_gradient_update_tokens": 20,
                 "rollout/total_completion_tokens": 25},
                {"training/global_step": 1, "training/resume_from_step": 0,
                 "rollout/total_gradient_update_tokens": 7,
                 "rollout/total_completion_tokens": 9},
                {"training/global_step": 2, "rollout/total_gradient_update_tokens": 8,
                 "rollout/total_completion_tokens": 11},
            )) + "\n")
            by_record, by_step, keys = training_token_history(source)
            self.assertEqual(by_record["total_gradient_update_tokens"],
                             {0: 10, 1: 30, 2: 37, 3: 45})
            self.assertEqual(by_step["total_gradient_update_tokens"], {1: 37, 2: 45})
            self.assertEqual(by_step["total_completion_tokens"], {1: 49, 2: 60})
            self.assertIn("rollout/total_completion_tokens", keys)

    def test_token_history_resets_at_a_fresh_step_one(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "metrics.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in (
                {"training/global_step": 1, "rollout/total_gradient_update_tokens": 10,
                 "rollout/total_completion_tokens": 15},
                {"training/global_step": 2, "rollout/total_gradient_update_tokens": 20,
                 "rollout/total_completion_tokens": 25},
                {"training/global_step": 1, "rollout/total_gradient_update_tokens": 7,
                 "rollout/total_completion_tokens": 9},
                {"training/global_step": 2, "rollout/total_gradient_update_tokens": 8,
                 "rollout/total_completion_tokens": 11},
            )) + "\n")
            by_record, by_step, _ = training_token_history(source)
            self.assertEqual(by_record["total_completion_tokens"], {2: 9, 3: 20})
            self.assertEqual(by_step["total_completion_tokens"], {1: 9, 2: 20})

    def test_token_history_does_not_resume_after_an_unaccounted_step(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "metrics.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in (
                {"training/global_step": 1, "critic/score/mean": 0.1,
                 "rollout/total_gradient_update_tokens": 10,
                 "rollout/total_completion_tokens": 15},
                {"training/global_step": 2, "critic/score/mean": 0.2},
                {"training/global_step": 3, "critic/score/mean": 0.3,
                 "rollout/total_gradient_update_tokens": 7,
                 "rollout/total_completion_tokens": 9},
            )) + "\n")
            _, by_step, _ = training_token_history(source)
            self.assertEqual(by_step["total_gradient_update_tokens"], {1: 10})
            self.assertEqual(by_step["total_completion_tokens"], {1: 15})

    def test_paper_methods_use_exact_canonical_presets(self):
        self.assertEqual(method_for("grpo", "grpo"), "grpo")
        self.assertIsNone(method_for("grpo", "no_rejection"))
        self.assertEqual(method_for("ace_rl", "oneeval_rollout8"), "ace_rl")
        self.assertEqual(method_for("ace_rl", "oneeval_rollout4"), "ace_rl")
        self.assertEqual(method_for("ace_rl", "delayed_oneeval_rollout5"), "delayed_joint")
        self.assertIsNone(method_for("ace_rl", "delayed_oneeval"))
        self.assertEqual(method_for("erl", "rollout5"), "erl")
        self.assertEqual(method_for("delayed_joint_ppo", "batch6"), "joint_ppo")
        self.assertEqual(method_for("delayed_joint_ppo", "batch6_rollout8"), "joint_ppo")
        self.assertEqual(method_for("delayed_joint_ppo", "batch8_rollout8"), "joint_ppo")
        self.assertEqual(method_for("delayed_joint_ppo", "batch10_rollout8"), "joint_ppo")
        self.assertIsNone(method_for("delayed_joint_ppo", "batch8_rollout"))
        self.assertIsNone(method_for("erl", "original"))
        self.assertIsNone(method_for("ace_rl", "historical"))

    def test_inventory_finds_exact_model_base_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "examples/mini_swe_bench/train/grpo/runs/grpo/qwen3-4b-instruct-2507").mkdir(parents=True)
            base = root / "examples/mini_swe_bench/eval/ace/results/visits8/qwen3-4b-instruct-2507"
            base.mkdir(parents=True)
            rows = inventory("local", str(root), "mini_swe_bench")
            self.assertEqual([row["path"] for row in rows if row["training_method"] == "base"],
                             [str(base.resolve())])

    def test_gallery_evaluation_scan_stays_at_step_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            step = run / "eval/ace/visits8/step_10"
            nested = step / "task_a/chat_completions/eval"
            incoming = run / "eval/ace/visits8/step_30.vast-1-incoming"
            for directory, number in ((step, 10), (nested, 20), (incoming, 30)):
                write_visits8(directory, number, [1.0] * 8)
            points, errors = eight_visit_evaluations(run, "ace", "visits")
            self.assertEqual(errors, [])
            self.assertEqual([point["step"] for point in points], [10])

    def test_list_needs_no_source(self):
        result = subprocess.run([sys.executable, str(ROOT / "plot.py"), "--list"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("learning", result.stdout)

    def test_turns_include_every_checkpoint_and_use_three_of_eight_visits(self):
        result = {"visits8": [{"step": step, "local_eval_epochs": 8} for step in (0, 10, 20)], "turns": {},
                  "turns_nopad": {}, "turns_extended": {}, "turns_extended_nopad": {},
                  "_turn_missing": [], "_errors": []}

        def curve(point, **kwargs):
            return [{"step": point["step"], "visits": kwargs["local_epoch"], "pad_solved": kwargs["pad_solved"]}]

        with mock.patch("quickplot.curve_rows", side_effect=curve):
            add_turns(result)
        for key in ("turns", "turns_nopad", "turns_extended", "turns_extended_nopad"):
            self.assertEqual(sorted(result[key]), [0, 10, 20])
        self.assertEqual(result["turns"][10], [{"step": 10, "visits": 3, "pad_solved": True}])
        self.assertEqual(result["turns_nopad"][10], [{"step": 10, "visits": 3, "pad_solved": False}])
        self.assertEqual(result["turns_extended"][10], [{"step": 10, "visits": 8, "pad_solved": True}])
        self.assertEqual(result["turns_extended_nopad"][10], [{"step": 10, "visits": 8, "pad_solved": False}])

    def test_remote_turn_aggregation_returns_compact_measured_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            evaluation = run / "eval/ace/visits8/step_10"
            evaluation.mkdir(parents=True)
            (evaluation / "evaluation_config.json").write_text(json.dumps({"max_turns": 2}))
            (evaluation / "metrics_step10.json").write_text(json.dumps({"evaluations": [{
                "step": 10, "local_eval_epochs": 3, "evaluation_task_ids": ["task"],
                "local_epochs": [{"local_eval_epoch": visit,
                                  "tasks": [{"task_id": "task", "reward": 0.5}]}
                                 for visit in (1, 2, 3)],
            }]}))
            trajectories = evaluation / "task_task/chat_completions/eval"
            trajectories.mkdir(parents=True)
            for visit in (1, 2, 3):
                (trajectories / f"policy_step000010_iter_{visit - 1}.json").write_text(json.dumps({
                    "step": 10, "trajectories": [{"steps": [{"reward": 0.25}, {"reward": 0.5}]}],
                }))
            hashes = {name: hashlib.sha256((ROOT / "scripts/figures" / name).read_bytes()).hexdigest()
                      for name in READER_FILES}
            request = {"reader_hashes": hashes, "turn_keys": TURN_KEYS, "turn_specs": TURN_SPECS,
                       "points": [{"source_key": "visits8", "relative_source": "eval/ace/visits8/step_10/metrics_step10.json",
                                   "point": {"run_tag": "test", "variant": "ace", "step": 10,
                                             "task_label": "all", "local_eval_epochs": 3}}]}
            result = subprocess.run([sys.executable, "-c", REMOTE_CODE, str(ROOT), str(run)],
                                    input=json.dumps(request), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(len(payload["curves"]["turns"]["10"]), 6)
            self.assertEqual(payload["curves"]["turns"]["10"][0]["value"], 0.25)
            self.assertEqual(payload["curves"]["turns_extended"], {})
            self.assertEqual(payload["errors"], [])

    def test_turn_views_are_aggregated_together_per_checkpoint(self):
        result = {"visits8": [{"step": step, "local_eval_epochs": 8}
                                         for step in (10, 20)],
                  "turns": {}, "turns_nopad": {}, "turns_extended": {}, "turns_extended_nopad": {},
                  "_turn_missing": [], "_errors": []}
        seen = []

        def curve(point, **kwargs):
            seen.append(point["step"])
            return [{"visits": kwargs["local_epoch"]}]

        with mock.patch("quickplot.curve_rows", side_effect=curve):
            add_turns(result)
        self.assertEqual(seen, [10] * 4 + [20] * 4)

    def test_padded_reward_carries_only_full_success(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "metrics_step20.json"
            source.write_text(json.dumps({"evaluations": [{"step": 20, "evaluation_task_ids": ["a", "b"],
                "local_epochs": [
                    {"local_eval_epoch": 1, "tasks": [{"task_id": "a", "reward": 1.0},
                                                        {"task_id": "b", "reward": 0.6}]},
                    {"local_eval_epoch": 2, "tasks": [{"task_id": "a", "reward": 0.0},
                                                        {"task_id": "b", "reward": 0.2}]},
                ]}]}))
            point = {"source": str(source), "step": 20}
            self.assertAlmostEqual(padded_reward(point, 2), 0.6)
            self.assertAlmostEqual(rsi_success(point, 2), 0.5)

    def test_eight_visit_rsi_is_padded_from_tasks_even_when_final_reward_falls(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            result = run / "eval/ace/visits8/step_20"
            result.mkdir(parents=True)
            (result / "evaluation_config.json").write_text(json.dumps({
                "step": 20, "preset": "visits8", "task_label": "all",
            }))
            entry = {"step": 20, "local_eval_protocol": "sequential_adaptation_v1",
                     "g_act": 1, "local_eval_epochs": 8, "evaluation_task_ids": ["task"],
                     "metrics": {"max_reward_mean": 0.0}, "local_epochs": [
                         {"local_eval_epoch": visit,
                          "tasks": [{"task_id": "task", "reward": float(visit == 1)}],
                          "metrics": {"max_reward_mean": float(visit == 1)}}
                         for visit in range(1, 9)
                     ]}
            (result / "metrics_step20.json").write_text(json.dumps({"evaluations": [entry]}))
            points, errors = eight_visit_evaluations(run, "ace", "visits")
            self.assertEqual(errors, [])
            self.assertEqual([row["value"] for row in points[0]["curve"]], [1.0] * 8)
            self.assertEqual(points[0]["value"], 0.0)

    def test_rsi3_counts_success_across_visits(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "metrics_step10.json"
            source.write_text(json.dumps({"evaluations": [{"step": 10, "evaluation_task_ids": ["a", "b"],
                "local_epochs": [
                    {"local_eval_epoch": 1, "tasks": [{"task_id": "a", "reward": 1.0},
                                                      {"task_id": "b", "reward": 0.0}]},
                    {"local_eval_epoch": 2, "tasks": [{"task_id": "a", "reward": 0.0},
                                                      {"task_id": "b", "reward": 0.0}]},
                    {"local_eval_epoch": 3, "tasks": [{"task_id": "a", "reward": 0.0},
                                                      {"task_id": "b", "reward": 1.0}]},
                ]}]}))
            self.assertEqual(rsi_success({"source": str(source), "step": 10}, 3), 1.0)

    def test_remote_turns_reuse_by_completion_record_unless_full_check(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            evaluation = run / "eval/ace/visits8/step_10"
            trajectories = evaluation / "task_a/chat_completions/eval"
            trajectories.mkdir(parents=True)
            write_visits8(evaluation, 10, [0.5] * 8, config={"max_turns": 2})
            complete = evaluation / "evaluation_complete.json"
            complete.write_text(json.dumps({"attempt": 1}))

            def write_trajectory(first_reward):
                for visit in range(1, 9):
                    (trajectories / f"policy_step000010_iter_{visit - 1}.json").write_text(json.dumps({
                        "step": 10, "trajectories": [{"steps": [{"reward": first_reward}, {"reward": 0.5}]}],
                    }))

            write_trajectory(0.25)
            cache = Path(folder) / "cache"
            actual_run = subprocess.run

            def fetch(**options):
                def local_ssh(_command, *, input, text, stdout, check):
                    return actual_run([sys.executable, "-c", GALLERY_CODE, str(run)], input=input,
                                      text=text, stdout=stdout, check=check)
                with mock.patch("remote_gallery.subprocess.run", side_effect=local_ssh):
                    return fetch_gallery_rows("testhost", str(run), cache, {"turns"}, **options)

            self.assertEqual(fetch()["turns"][10][0]["value"], 0.25)
            # An in-place trajectory rewrite with an unchanged completion record is
            # the documented blind spot of the default check.
            write_trajectory(0.4)
            stale = fetch()
            self.assertEqual(stale["_turn_reused"], [10])
            self.assertEqual(stale["turns"][10][0]["value"], 0.25)
            checked = fetch(full_turn_check=True)
            self.assertEqual(checked["_turn_reused"], [])
            self.assertEqual(checked["turns"][10][0]["value"], 0.4)
            write_trajectory(0.3)
            complete.write_text(json.dumps({"attempt": 2}))
            rerun = fetch()
            self.assertEqual(rerun["_turn_reused"], [])
            self.assertEqual(rerun["turns"][10][0]["value"], 0.3)

    def test_remote_gallery_returns_computed_rows_without_copying_evaluations(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            evaluation = run / "eval/ace/visits8/step_10"
            run.mkdir()
            (run / "metrics.jsonl").write_text(json.dumps({
                "training/global_step": 10, "critic/score/mean": 0.5,
                "rollout/total_completion_tokens": 12,
            }) + "\n")
            source = write_visits8(evaluation, 10, [float(visit == 1) for visit in range(1, 9)],
                                   config={"max_turns": 2})
            trajectories = evaluation / "task_a/chat_completions/eval"
            trajectories.mkdir(parents=True)
            for visit in range(1, 9):
                (trajectories / f"policy_step000010_iter_{visit - 1}.json").write_text(json.dumps({
                    "step": 10, "trajectories": [{"steps": [{"reward": 0.25}, {"reward": 0.5}]}],
                }))
            cache = Path(folder) / "cache"
            actual_run = subprocess.run

            def local_ssh(_command, *, input, text, stdout, check):
                return actual_run([sys.executable, "-c", GALLERY_CODE, str(run)], input=input,
                                  text=text, stdout=stdout, check=check)

            with mock.patch("remote_gallery.subprocess.run", side_effect=local_ssh):
                record = fetch_gallery_rows("testhost", str(run), cache,
                                            {"learning", "comparison", "turns"})
            self.assertEqual(record["visits8"][0]["curve"][2], {"k": 3, "value": 1.0})
            self.assertEqual(padded_reward(record["visits8"][0], 3), 1.0)
            self.assertEqual(record["token_by_step"]["total_completion_tokens"], {10: 12.0})
            self.assertEqual(len(record["turns"][10]), 6)
            self.assertEqual(len(list(cache.rglob("*metrics_step*.json"))), 0)
            with mock.patch("remote_gallery.subprocess.run", side_effect=local_ssh):
                repeated = fetch_gallery_rows("testhost", str(run), cache,
                                              {"learning", "comparison", "turns"})
            self.assertEqual(repeated["_turn_reused"], [10])
            first_trajectory = trajectories / "policy_step000010_iter_0.json"
            first_trajectory.write_text(json.dumps({
                "step": 10, "trajectories": [{"steps": [{"reward": 0.4}, {"reward": 0.5}]}],
            }))
            with mock.patch("remote_gallery.subprocess.run", side_effect=local_ssh):
                changed = fetch_gallery_rows("testhost", str(run), cache,
                                             {"learning", "comparison", "turns"})
            self.assertEqual(changed["_turn_reused"], [])
            self.assertEqual(changed["turns"][10][0]["value"], 0.4)
            with mock.patch("remote_gallery.subprocess.run", side_effect=local_ssh):
                fetch_gallery_rows("testhost", str(run), cache, {"comparison"})
            source.unlink()
            offline = fetch_gallery_rows("testhost", str(run), cache,
                                         {"comparison", "learning", "turns"}, offline=True)
            self.assertEqual(offline["visits8"][0]["curve"][2], {"k": 3, "value": 1.0})
            self.assertEqual(len(offline["turns"][10]), 6)

            # A base-model result root holds its pass8 evaluation directly.
            base = Path(folder) / "results/pass8/model"
            base.mkdir(parents=True)
            (base / "evaluation_config.json").write_text(json.dumps({
                "step": 0, "preset": "pass8", "task_label": "all", "method": "vanilla",
            }))
            (base / "metrics_step0.json").write_text(json.dumps({"evaluations": [{
                "step": 0, "local_eval_protocol": "sequential_adaptation_v1", "g_act": 8,
                "local_eval_epochs": 1, "evaluation_task_ids": ["a"],
                "metrics": {"max_reward_mean": 1.0, **{f"pass_at_{k}_success_rate": 1.0 for k in range(1, 9)}},
                "local_epochs": [{"local_eval_epoch": 1, "tasks": [{"task_id": "a", "reward": 1.0}]}],
            }]}))
            with mock.patch("remote_gallery.subprocess.run", side_effect=lambda command, **kwargs:
                            actual_run([sys.executable, "-c", GALLERY_CODE, str(base)], **kwargs)):
                baseline = fetch_gallery_rows("testhost", str(base), cache,
                                              {"comparison"}, baseline=True)
            self.assertEqual(baseline["visits8"], [])
            self.assertEqual(baseline["pass8"][0]["curve"][0], {"k": 1, "value": 1.0})


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is needed for rendering")
class RenderingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "examples/appworld/train/ace_rl/runs/oneeval_rollout8/model"
        self.run.mkdir(parents=True)
        (self.run / "metrics.jsonl").write_text(
            json.dumps({"training/global_step": 1, "critic/score/mean": 0.25}) + "\n"
        )
        write_visits8(self.run / "eval/ace/visits8/step_1", 1, [0.1 * i for i in range(1, 9)], task="task")

    def command(self, *options):
        extracted = subprocess.run([
            sys.executable, str(ROOT / "scripts/figures/quickplot.py"), "--source", "local:appworld",
            "--source-root", str(self.root), "--model", "model", "--output-dir", str(self.root / "figures"),
            "--cache-dir", str(self.root / "cache"), *options,
        ], capture_output=True, text=True)
        if extracted.returncode:
            return extracted
        return subprocess.run([
            sys.executable, str(ROOT / "scripts/figures/gallery_render.py"), "--data", str(self.root / "figures/model/data"), "--no-open",
        ], capture_output=True, text=True)

    def test_renders_clean_gallery_and_files(self):
        result = self.command("--plots", "learning,comparison,comparison_extended")
        self.assertEqual(result.returncode, 0, result.stderr)
        folder = self.root / "figures/model"
        for name in ("learning", "comparison", "comparison_extended"):
            self.assertTrue((folder / f"{name}.pdf").read_bytes().startswith(b"%PDF-"))
            self.assertTrue((folder / f"{name}.png").read_bytes().startswith(b"\x89PNG"))
            for axis in ("total_gradient_update_tokens", "total_completion_tokens"):
                self.assertTrue((folder / f"{name}_{axis}.png").is_file())
        self.assertTrue((folder / "index.html").is_file())
        self.assertFalse(list(folder.glob("*.csv")))
        with (folder / "data/evaluation.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertIn({"method": "ace_rl", "measure": "rsi", "k": "3", "step": "1",
                       "cumulative_total_gradient_update_tokens": "",
                       "cumulative_total_completion_tokens": "", "value": "0.0"}, rows)
        with (folder / "data/training.csv").open() as handle:
            self.assertEqual(next(csv.DictReader(handle))["value"], "0.25")

    def test_append_adds_plots_without_dropping_earlier_extraction(self):
        self.assertEqual(self.command("--plots", "learning").returncode, 0)
        self.assertEqual(self.command("--plots", "comparison", "--append").returncode, 0)
        tables, manifest = read_tables(self.root / "figures/model/data")
        self.assertEqual(manifest["plots"], ["learning", "comparison"])
        self.assertTrue(tables["training"])
        self.assertEqual(self.command("--plots", "comparison").returncode, 0)
        tables, manifest = read_tables(self.root / "figures/model/data")
        self.assertEqual(manifest["plots"], ["comparison"])
        self.assertFalse(tables["training"])

    def test_render_rejects_plots_that_were_not_extracted(self):
        self.assertEqual(self.command("--plots", "learning").returncode, 0)
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/figures/gallery_render.py"), "--data", str(self.root / "figures/model/data"),
            "--plots", "comparison", "--no-open",
        ], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("extract those plots first", result.stderr)

    def test_missing_metrics_are_left_empty_without_stopping_gallery(self):
        incomplete = self.run / "eval/ace/visits8/step_2"
        incomplete.mkdir(parents=True)
        (incomplete / "evaluation_config.json").write_text(json.dumps({
            "step": 2, "preset": "visits8", "task_label": "all",
        }))
        result = self.command("--plots", "learning")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "figures/model/learning.pdf").is_file())

    def test_requested_figures_exist_when_data_is_missing(self):
        result = self.command("--plots", "comparison,comparison_extended,variants")
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("comparison", "comparison_extended", "variants"):
            self.assertTrue((self.root / f"figures/model/{name}.pdf").is_file(), name)
            self.assertTrue((self.root / f"figures/model/{name}.png").is_file(), name)

    def test_all_renders_with_no_evaluation_data(self):
        shutil.rmtree(self.run / "eval")
        obsolete = self.root / "figures/model/iterations.png"
        obsolete.parent.mkdir(parents=True)
        obsolete.write_bytes(b"old figure")
        result = self.command("--all")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list((self.root / "figures/model").glob("*.pdf"))), 19)
        self.assertTrue((self.root / "figures/model/turns_ace_rl_extended.png").is_file())
        self.assertTrue((self.root / "figures/model/turns_ace_rl_nopad.png").is_file())
        self.assertTrue((self.root / "figures/model/turns_ace_rl_extended_nopad.png").is_file())
        self.assertFalse(obsolete.exists())

    def test_all_renders_with_no_selected_run(self):
        result = self.command("--all", "--methods", "ppo")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list((self.root / "figures/model").glob("*.pdf"))), 19)
        self.assertTrue((self.root / "figures/model/turns_ppo.png").is_file())

    def test_turn_figure_uses_base_and_all_checkpoint_colors(self):
        import matplotlib.pyplot as plt

        rows = lambda step: [{"turn_global": 1, "turn": 1, "local_epoch": 1, "value": step / 20},
                             {"turn_global": 2, "turn": 1, "local_epoch": 2, "value": step / 20}]
        data = {"base": {"turns": {0: rows(0)}},
                "ace_rl": {"turns": {10: rows(10), 20: rows(20)}}}
        with mock.patch("gallery_render._save_figure") as save:
            plot_turns_method("appworld", "model", "ace_rl", data, self.root / "turns_ace_rl", extended=False)
            figure, axes = save.call_args.args[:2]
            data_lines = [line for line in axes[0].lines if line.get_label().startswith("step ")]
            self.assertEqual([line.get_label() for line in data_lines], ["step 0", "step 10", "step 20"])
            self.assertEqual(len({line.get_color() for line in data_lines}), 3)
            self.assertTrue(any(line.get_linestyle() == ":" for line in axes[0].lines))
            plt.close(figure)

    def test_learning_and_variants_use_padded_evaluation_reward(self):
        import matplotlib.pyplot as plt

        source = self.root / "sequential.json"
        source.write_text(json.dumps({"evaluations": [{"step": 20, "evaluation_task_ids": ["task"],
            "local_epochs": [
                {"local_eval_epoch": visit, "tasks": [{"task_id": "task", "reward": float(visit == 1)}]}
                for visit in (1, 2, 3)
            ]}]}))
        data = {"ace_rl": {"visits8": [visits8_point(20, source, local_eval_epochs=3)],
                           "training": [{"training_step": 20, "record": 0, "segment": 0, "value": 0.5}],
                           "token_by_record": {"total_completion_tokens": {0: 1000}},
                           "token_by_step": {"total_completion_tokens": {20: 1000}}}}
        with mock.patch("gallery_render._save_figure") as save:
            plot_curves("learning", "appworld", "model", data, self.root / "learning", {"ace_rl"})
            figure, axes = save.call_args.args[:2]
            self.assertEqual(list(axes[1].lines[0].get_ydata()), [1.0])
            plt.close(figure)
            plot_curves("learning", "appworld", "model", data, self.root / "learning_tokens",
                        {"ace_rl"}, token_axis="total_completion_tokens")
            figure, axes = save.call_args.args[:2]
            self.assertEqual(list(axes[0].lines[0].get_xdata()), [1000])
            self.assertEqual(list(axes[1].lines[0].get_xdata()), [1000])
            plt.close(figure)
            plot_curves("variants", "appworld", "model", data, self.root / "variants", {"ace_rl"})
            figure, axes = save.call_args.args[:2]
            self.assertEqual(list(axes[0].lines[0].get_ydata()), [1.0])
            plt.close(figure)

    def test_comparison_merges_base_into_each_measured_method(self):
        import matplotlib.pyplot as plt

        base = self.root / "base.json"
        base.write_text(json.dumps({"evaluations": [{"step": 0, "evaluation_task_ids": ["task"],
            "local_epochs": [
                {"local_eval_epoch": visit, "tasks": [{"task_id": "task", "reward": float(visit == 1)}]}
                for visit in (1, 2, 3)
            ]}]}))
        ace = self.root / "grpo_ace.json"
        ace.write_text(json.dumps({"evaluations": [{"step": 10, "evaluation_task_ids": ["task"],
            "local_epochs": [
                {"local_eval_epoch": visit, "tasks": [{"task_id": "task", "reward": float(visit == 2)}]}
                for visit in range(1, 9)
            ]}]}))
        curve = [{"k": k, "value": k / 10} for k in range(1, 9)]
        data = {"base": {"visits8": [visits8_point(0, base)]},
                "grpo": {"pass8": [{"step": 10, "curve": curve}],
                         "visits8": [visits8_point(10, ace)],
                         "token_by_step": {"total_completion_tokens": {10: 1000}}}}
        with mock.patch("gallery_render._save_figure") as save:
            plot_curves("comparison_extended", "appworld", "model", data,
                        self.root / "comparison_extended", {"grpo"})
            figure, axes = save.call_args.args[:2]
            lines = {line.get_label(): line for line in axes[0].lines}
            self.assertNotIn("Base RSI@1", lines)
            self.assertEqual(list(lines["GRPO pass@1"].get_xdata()), [0, 10])
            self.assertEqual(list(lines["GRPO RSI@1"].get_xdata()), [0, 10])
            self.assertEqual(list(lines["GRPO RSI@2"].get_ydata()), [1.0, 1.0])
            self.assertEqual(list(lines["GRPO RSI@4"].get_xdata()), [10])
            self.assertEqual(lines["GRPO pass@1"].get_color(), COLORS["grpo"])
            self.assertTrue(all(lines[f"GRPO RSI@{k}"].get_color() == GRPO_RSI_COLOR for k in (1, 2, 4, 8)))
            plt.close(figure)
            plot_curves("comparison", "appworld", "model", data,
                        self.root / "comparison", {"grpo"})
            figure, axes = save.call_args.args[:2]
            lines = {line.get_label(): line for line in axes[0].lines}
            self.assertEqual(list(lines["GRPO pass@1"].get_xdata()), [0, 10])
            self.assertEqual(list(lines["GRPO RSI@3"].get_xdata()), [0, 10])
            self.assertEqual(lines["GRPO pass@1"].get_color(), COLORS["grpo"])
            self.assertEqual(lines["GRPO RSI@3"].get_color(), GRPO_RSI_COLOR)
            plt.close(figure)
            plot_curves("comparison", "appworld", "model", data,
                        self.root / "comparison_tokens", {"grpo"},
                        token_axis="total_completion_tokens")
            figure, axes = save.call_args.args[:2]
            lines = {line.get_label(): line for line in axes[0].lines}
            self.assertEqual(list(lines["GRPO RSI@3"].get_xdata()), [0, 1000])
            plt.close(figure)

    def test_extended_comparison_token_axis_uses_matching_checkpoint_totals(self):
        import matplotlib.pyplot as plt

        points = [{"step": step, "curve": [{"k": k, "value": k / 10} for k in range(1, 9)]}
                  for step in (10, 20, 30)]
        tokens = {"total_completion_tokens": {10: 1000, 20: 2500}}
        data = {"grpo": {"pass8": points, "token_by_step": tokens}}
        with mock.patch("gallery_render._save_figure") as save:
            plot_curves("comparison_extended", "appworld", "model", data,
                        self.root / "comparison_extended_tokens", {"grpo"},
                        token_axis="total_completion_tokens")
            figure, axes = save.call_args.args[:2]
            self.assertEqual(len(axes[0].lines), 4)
            self.assertTrue(all(list(line.get_xdata()) == [1000, 2500] for line in axes[0].lines))
            plt.close(figure)

    def test_rollout_panels_use_step_local_values_and_cumulative_token_axis(self):
        import matplotlib.pyplot as plt

        metrics = ("rollout/total_completion_tokens", "rollout/total_prompt_tokens")
        rollout = {metric: [
            {"training_step": step, "record": step - 1, "segment": 0, "value": step * 10}
            for step in (1, 2)
        ] for metric in metrics}
        data = {"ace_rl": {"rollout_keys": list(metrics), "rollout": rollout,
                           "token_by_record": {"total_completion_tokens": {0: 10, 1: 30}}}}
        with mock.patch("gallery_render._save_figure") as save:
            plot_curves("rollout", "appworld", "model", data, self.root / "rollout",
                        {"ace_rl"}, token_axis="total_completion_tokens")
            figure, axes = save.call_args.args[:2]
            self.assertEqual(len(axes), 2)
            self.assertTrue(all(list(ax.lines[0].get_xdata()) == [10, 30] for ax in axes))
            self.assertTrue(all(list(ax.lines[0].get_ydata()) == [10, 20] for ax in axes))
            plt.close(figure)

    def test_comparison_rsi3_uses_first_three_of_eight_visits(self):
        import matplotlib.pyplot as plt

        source = self.root / "eight_visits.json"
        source.write_text(json.dumps({"evaluations": [{
            "step": 20, "evaluation_task_ids": ["task"],
            "local_epochs": [
                {"local_eval_epoch": visit, "tasks": [{"task_id": "task", "reward": float(visit == 2)}]}
                for visit in range(1, 9)
            ],
        }]}))
        data = {"delayed_joint": {"visits8": [visits8_point(20, source)]}}
        with mock.patch("gallery_render._save_figure") as save:
            plot_curves("comparison", "appworld", "model", data, self.root / "comparison", {"delayed_joint"})
            figure, axes = save.call_args.args[:2]
            rsi = [line for line in axes[0].lines if line.get_label() == "Delayed-joint RSI@3"]
            self.assertEqual(len(rsi), 1)
            self.assertEqual(list(rsi[0].get_xdata()), [20])
            self.assertEqual(list(rsi[0].get_ydata()), [1.0])
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
