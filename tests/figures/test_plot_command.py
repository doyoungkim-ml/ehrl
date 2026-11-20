"""The root plot.py one-line commands: benchmark gallery and single-run routing."""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/figures"))
import plot_eval  # noqa: E402

SPEC = importlib.util.spec_from_file_location("root_plot", ROOT / "plot.py")


class PlotCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_run_path_names_its_benchmark(self):
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        self.assertEqual(module.run_benchmark("/x/examples/mbpp/train/grpo/runs/grpo/m"), "mbpp")
        self.assertEqual(module.run_benchmark("torch:/x/examples/appworld/train/ace_rl/runs/p/m"), "appworld")
        self.assertIsNone(module.run_benchmark("/elsewhere/run"))

    def test_single_run_preset_value_is_not_taken_as_benchmark(self):
        run = self.root / "examples/mbpp/train/grpo/runs/grpo/model"
        result = subprocess.run([sys.executable, str(ROOT / "plot.py"), "--run", str(run), "--preset", "eval", "--resolve"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual((plan["benchmark"], plan["preset"]), ("mbpp", "eval"))
        self.assertEqual(plan["output_dir"], str(run / "figures/eval"))

    def test_method_level_result_names_its_training_run(self):
        run = self.root / "run"
        result = run / "eval/vanilla/pass8/step_4"
        result.mkdir(parents=True)
        (result / "evaluation_config.json").write_text(json.dumps({
            "step": 4, "preset": "pass8", "task_label": "all", "method": "vanilla", "run_dir": str(result),
        }))
        (result / "metrics_step4.json").write_text(json.dumps({"evaluations": [{"step": 4, "metrics": {"success_rate": 0.5}}]}))
        [point] = plot_eval.load_points(run / "eval", "success_rate", "auto")
        self.assertEqual(point["run_tag"], str(run))
        self.assertEqual(point["variant"], "vanilla/pass8")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is needed for rendering")
    def test_benchmark_gallery_is_one_local_command(self):
        checkout = self.root / "checkout"
        checkout.mkdir()
        shutil.copy2(ROOT / "plot.py", checkout / "plot.py")
        shutil.copytree(ROOT / "scripts/figures", checkout / "scripts/figures")
        run = checkout / "examples/mbpp/train/grpo/runs/grpo/qwen3-0.6b-thinking"
        run.mkdir(parents=True)
        (run / "metrics.jsonl").write_text(json.dumps({"training/global_step": 1, "critic/score/mean": 0.5}) + "\n")
        result = subprocess.run([sys.executable, str(checkout / "plot.py"), "mbpp", "--plots", "learning", "--no-open"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = checkout / "examples/mbpp/figures/output/qwen3-0.6b-thinking"
        self.assertTrue((output / "learning.png").is_file())
        self.assertTrue((output / "index.html").is_file())
        self.assertTrue((output / "data/training.csv").is_file())

    def test_benchmark_without_local_runs_fails_clearly(self):
        checkout = self.root / "checkout"
        checkout.mkdir()
        shutil.copy2(ROOT / "plot.py", checkout / "plot.py")
        shutil.copytree(ROOT / "scripts/figures", checkout / "scripts/figures")
        result = subprocess.run([sys.executable, str(checkout / "plot.py"), "mbpp", "--no-open"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No mbpp runs found", result.stderr)


if __name__ == "__main__":
    unittest.main()
