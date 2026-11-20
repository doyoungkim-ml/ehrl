"""Exercise filtered SSH retrieval with a real rsync and a local SSH stand-in."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/figures"))
import remote  # noqa: E402


class RemoteSyntaxTest(unittest.TestCase):
    def test_ssh_aliases_and_local_colons(self):
        self.assertEqual(remote.parse_remote("torch:~/harness/run"), ("torch", "~/harness/run"))
        self.assertEqual(remote.parse_remote("user@cluster:relative/run"), ("user@cluster", "relative/run"))
        for value in ("/tmp/a:b", "./torch:run", "relative/run", "~/run"):
            self.assertIsNone(remote.parse_remote(value))
        for value in ("-oProxyCommand=bad:run", "bad host:run", "torch:", "torch:/run\nother", "torch:/archive/run"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                remote.parse_remote(value)

    def test_resolve_is_offline_and_does_not_create_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "cache"
            result = subprocess.run(
                [sys.executable, str(ROOT / "plot.py"), "mbpp", "--preset", "eval", "--run", "unreachable:/some/run", "--cache-dir", str(cache), "--resolve"],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            resolved = json.loads(result.stdout)
            self.assertEqual(resolved["remote_run"], "unreachable:/some/run")
            self.assertEqual(resolved["run"], str(remote.cache_path("unreachable:/some/run", cache)))
            self.assertFalse(cache.exists())


@unittest.skipUnless(shutil.which("rsync"), "rsync is required for the real transport fixture")
class RemoteRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote_home = self.root / "remote"
        self.run = self.remote_home / "run with 'quote"
        self.run.mkdir(parents=True)
        self.cache = self.root / "cache"
        self.source = f"fixture:{self.run}"
        self.ssh_log = self.root / "ssh.log"
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        ssh = bin_dir / "ssh"
        ssh.write_text(
            f"#!{sys.executable}\n"
            "import os, pathlib, subprocess, sys\n"
            "with pathlib.Path(os.environ['FIGURE_TEST_SSH_LOG']).open('a') as f: f.write('call\\n')\n"
            "raise SystemExit(subprocess.run(' '.join(sys.argv[2:]), shell=True, cwd=os.environ['FIGURE_TEST_REMOTE_HOME']).returncode)\n"
        )
        ssh.chmod(0o755)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FIGURE_TEST_SSH_LOG": str(self.ssh_log), "FIGURE_TEST_REMOTE_HOME": str(self.remote_home)}).start()
        (self.run / "metrics.jsonl").write_text(json.dumps({"training/global_step": 10, "critic/score/mean": 0.6}) + "\n")
        (self.run / "run.json").write_text('{"benchmark": "appworld"}\n')
        self.result = self.evaluation("visits8", 3)
        for path in (
            "checkpoints/model.pt", "checkpoints/metrics.jsonl",
            "eval/visits8/step_10/playbooks/task_a/database.bin", "eval/visits8/step_10/server.log",
            "eval/historical/step_10/metrics_step10.json", "wandb/metrics.jsonl",
        ):
            target = self.run / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("must not transfer")

    def evaluation(self, preset, count):
        result = self.run / "eval" / preset / "step_10"
        result.mkdir(parents=True, exist_ok=True)
        config = {
            "step": 10, "task_label": "all", "preset": preset, "method": "ace", "training_run_dir": str(self.run), "run_dir": str(result), "max_turns": 2,
            "evaluation": ["python", "--playbook_snapshot_dir", str(result / "playbooks")],
        }
        (result / "evaluation_config.json").write_text(json.dumps(config))
        (result / "evaluation_complete.json").write_text('{}')
        epochs = []
        for number in range(1, count + 1):
            epochs.append({"local_eval_epoch": number, "tasks": [{"task_id": "a", "reward": 0.6}], "metrics": {"max_reward_mean": 0.6}})
            path = result / f"playbooks/task_a/chat_completions/eval/policy_step000010_iter_{number - 1}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"step": 10, "trajectories": [{"steps": [{"reward": 0.2}, {"reward": 0.6}]}]}))
        entry = {"step": 10, "local_eval_epochs": count, "local_epochs": epochs, "metrics": {"max_reward_mean": 0.6}, "evaluation_task_ids": ["a"]}
        (result / "metrics_step10.json").write_text(json.dumps({"evaluations": [entry]}))
        return result

    def command(self, *args):
        env = {key: value for key, value in os.environ.items() if key not in ("PRESET", "TASK_LABEL")}
        return subprocess.run([sys.executable, str(ROOT / "plot.py"), "appworld", "--cache-dir", str(self.cache), *args], env=env, text=True, capture_output=True)

    def test_scalar_transfer_filters_heavy_files_and_refreshes_exact_bytes(self):
        mirror = remote.fetch_run(self.source, self.cache)
        files = {str(path.relative_to(mirror)) for path in mirror.rglob("*") if path.is_file()}
        self.assertEqual(files, {
            remote.MARKER, "run.json", "metrics.jsonl", "eval/visits8/step_10/metrics_step10.json",
            "eval/visits8/step_10/evaluation_config.json", "eval/visits8/step_10/evaluation_complete.json",
        })
        metrics = self.result / "metrics_step10.json"
        self.assertEqual((mirror / metrics.relative_to(self.run)).read_bytes(), metrics.read_bytes())
        # Refresh overwrites same-size content even if mtime did not change.
        timestamp = metrics.stat()
        metrics.write_text(metrics.read_text().replace('0.6', '0.7'))
        os.utime(metrics, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
        (mirror / "figures").mkdir()
        figure = mirror / "figures/saved.pdf"
        figure.write_text("keep local plot")
        marker = self.result / "evaluation_complete.json"
        marker.unlink()
        remote.fetch_run(self.source, self.cache)
        self.assertEqual((mirror / metrics.relative_to(self.run)).read_bytes(), metrics.read_bytes())
        self.assertFalse((mirror / marker.relative_to(self.run)).exists())
        self.assertEqual(figure.read_text(), "keep local plot")

    def test_turns_fetches_nested_trajectories_and_offline_makes_no_ssh_calls(self):
        result = self.command("--preset", "turns", "--run", self.source, "--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Validated 6 measured turn points", result.stdout)
        calls = self.ssh_log.read_text()
        cached = self.command("--preset", "turns", "--run", self.source, "--offline", "--check")
        self.assertEqual(cached.returncode, 0, cached.stderr)
        self.assertEqual(self.ssh_log.read_text(), calls)

    def test_scalar_refresh_reuses_turn_cache_only_when_eval_inputs_match(self):
        mirror = remote.fetch_run(self.source, self.cache, trajectories=True)
        remote.fetch_run(self.source, self.cache)
        remote.fetch_run(self.source, self.cache, offline=True, trajectories=True)

        (self.run / "metrics.jsonl").write_text(json.dumps({"training/global_step": 11, "critic/score/mean": 0.7}) + "\n")
        remote.fetch_run(self.source, self.cache)
        remote.fetch_run(self.source, self.cache, offline=True, trajectories=True)

        metrics = self.result / "metrics_step10.json"
        timestamp = metrics.stat()
        metrics.write_text(metrics.read_text().replace("0.6", "0.7"))
        os.utime(metrics, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
        remote.fetch_run(self.source, self.cache)
        with self.assertRaisesRegex(ValueError, "trajectories"):
            remote.fetch_run(self.source, self.cache, offline=True, trajectories=True)
        self.assertTrue((mirror / "eval/visits8/step_10/metrics_step10.json").is_file())

    def test_offline_requires_complete_cache_and_trajectories(self):
        with self.assertRaisesRegex(ValueError, "No cached"):
            remote.fetch_run(self.source, self.cache, offline=True)
        mirror = remote.fetch_run(self.source, self.cache)
        with self.assertRaisesRegex(ValueError, "trajectories"):
            remote.fetch_run(self.source, self.cache, offline=True, trajectories=True)
        with patch.object(remote, "checked_command", side_effect=[json.dumps(str(self.run)), ValueError("connection lost")]), self.assertRaisesRegex(ValueError, "connection lost"):
            remote.fetch_run(self.source, self.cache)
        self.assertFalse(json.loads((mirror / remote.MARKER).read_text())["complete"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            remote.fetch_run(self.source, self.cache, offline=True)

    def test_remote_evaluation_directory_and_relative_remote_paths(self):
        result = self.command("--preset", "eval", "--runs-dir", f"fixture:{self.result}", "--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.command("--preset", "eval", "--run", f"fixture:{self.run.name}", "--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.command("--preset", "eval", "--run", "fixture:missing-run", "--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Remote retrieval failed", result.stderr)

    def test_direct_legacy_result_keeps_step_and_task_identity(self):
        legacy = self.run / "legacy_step10_tasksall"
        self.result.rename(legacy)
        (legacy / "evaluation_config.json").unlink()
        result = self.command("--preset", "eval", "--runs-dir", f"fixture:{legacy}", "--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Validated 1 points", result.stdout)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is required to verify rendering")
    def test_remote_inputs_render_locally(self):
        output = self.root / "rendered"
        result = self.command("--preset", "eval", "--run", self.source, "--output-dir", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "eval.pdf").read_bytes().startswith(b"%PDF"))
        self.assertTrue((output / "eval.png").read_bytes().startswith(b"\x89PNG"))
        self.assertIn("0.6", (output / "eval.csv").read_text())
        self.assertFalse((self.run / "figures").exists())


if __name__ == "__main__":
    unittest.main()
