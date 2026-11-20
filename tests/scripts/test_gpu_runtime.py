"""Run with unittest discovery to avoid the repository's optional-module stubs."""

import importlib.util
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gpu_runtime.py"
SPEC = importlib.util.spec_from_file_location("gpu_runtime", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class GpuRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gpu runtime ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.site = self.root / "venv" / "site-packages"
        self.inherited = self.root / "inherited" / "site-packages"
        self.site.mkdir(parents=True)
        self.torch = self.inherited / "torch"
        (self.torch / "lib").mkdir(parents=True)
        (self.torch / "__init__.py").write_text("raise AssertionError('torch must not be imported')\n")
        (self.torch / "version.py").write_text("cuda: str = '12.6'\nraise AssertionError('version must not execute')\n")

    def discover(self, **kwargs):
        return runtime.discover_runtime(search_path=[str(self.site), str(self.inherited)], environ=kwargs.pop("environ", {}), conventional_roots=kwargs.pop("conventional_roots", []), **kwargs)

    def toolkit(self, name, version):
        root = self.root / name
        (root / "bin").mkdir(parents=True)
        compiler = root / "bin" / "nvcc"
        compiler.write_text(f"#!/bin/sh\nprintf '%s\\n' 'Cuda compilation tools, release {version}, V{version}.99'\n")
        compiler.chmod(0o755)
        return root

    def test_inherited_sites_and_canonical_dedup_without_mutation(self):
        runtime_package = self.inherited / "nvidia" / "cuda_runtime"
        (runtime_package / "lib").mkdir(parents=True)
        (runtime_package / "lib" / "libcudart.so.12").touch()
        (runtime_package / "lib64").symlink_to(runtime_package / "lib", target_is_directory=True)
        (self.site / "nvidia" / "nccl" / "lib").mkdir(parents=True)
        (self.inherited / "nvidia" / "nccl" / "lib").mkdir(parents=True)
        (self.site / "vllm").mkdir()
        (self.site / "vllm" / "__init__.py").write_text("raise AssertionError('no import')")
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        report = self.discover(environ={"LD_LIBRARY_PATH": "/existing/runtime"})
        libraries = report["library_directories"]
        self.assertEqual(report["torch_directory"], str(self.torch))
        self.assertEqual(libraries.count(str(runtime_package / "lib")), 1)
        self.assertIn(str(self.site / "nvidia" / "nccl" / "lib"), libraries)
        self.assertNotIn(str(self.inherited / "nvidia" / "nccl" / "lib"), libraries)
        self.assertEqual(report["exports"]["CUDA12_RUNTIME_HOME"], str(runtime_package))
        self.assertTrue(report["exports"]["LD_LIBRARY_PATH"].endswith(":/existing/runtime"))
        self.assertEqual(before, sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")))

    def test_exact_compiler_preferred_over_earlier_minor_match(self):
        minor = self.toolkit("cuda-12.8", "12.8")
        exact = self.toolkit("cuda-12.6", "12.6")
        wrong = self.toolkit("cuda-13.0", "13.0")
        report = self.discover(conventional_roots=[wrong, minor, exact], require_compiler=True)
        self.assertEqual(report["exports"]["CUDA_HOME"], str(exact))
        self.assertEqual(report["warnings"], [])

    def test_same_major_compiler_reports_minor_warning(self):
        minor = self.toolkit("cuda-12.8", "12.8")
        report = self.discover(conventional_roots=[minor], require_compiler=True)
        self.assertEqual(report["compiler"]["cuda_version"], "12.8")
        self.assertIn("different minor", report["warnings"][0])

    def test_wrong_major_never_selected_and_required_fails(self):
        wrong = self.toolkit("cuda-13.0", "13.0")
        report = self.discover(conventional_roots=[wrong])
        self.assertIsNone(report["compiler"])
        self.assertEqual(report["exports"]["CUDA_HOME"], "")
        self.assertIn("incompatible compilers", report["warnings"][0])
        with self.assertRaisesRegex(runtime.RuntimeDiscoveryError, "No usable nvcc"):
            self.discover(conventional_roots=[wrong], require_compiler=True)

    def test_no_compiler_optional_and_required(self):
        report = self.discover()
        self.assertEqual(report["exports"]["CUDA_HOME"], "")
        self.assertEqual(report["exports"]["CUDA12_RUNTIME_HOME"], "")
        self.assertNotIn("PATH", report["exports"])
        self.assertIn("prebuilt runtime mode", report["warnings"][0])
        with self.assertRaises(runtime.RuntimeDiscoveryError):
            self.discover(require_compiler=True)

    def test_explicit_invalid_or_wrong_major_is_an_error(self):
        with self.assertRaisesRegex(runtime.RuntimeDiscoveryError, "Correct CUDA_HOME/CUDA_PATH"):
            self.discover(environ={"CUDA_HOME": str(self.root / "missing")})
        wrong = self.toolkit("cuda-13.0", "13.0")
        with self.assertRaisesRegex(runtime.RuntimeDiscoveryError, "same CUDA major"):
            self.discover(environ={"CUDA_PATH": str(wrong)})

    def test_explicit_valid_toolkit_is_honored(self):
        selected = self.toolkit("user selected cuda", "12.8")
        exact = self.toolkit("cuda-12.6", "12.6")
        report = self.discover(environ={"CUDA_HOME": str(selected)}, conventional_roots=[exact])
        self.assertEqual(report["exports"]["CUDA_HOME"], str(selected))

    def test_explicit_runtime_only_root_is_optional_but_not_a_compiler(self):
        root = self.inherited / "nvidia" / "cuda_runtime"
        (root / "lib").mkdir(parents=True)
        (root / "lib" / "libcudart.so.12").touch()
        report = self.discover(environ={"CUDA_HOME": str(root)})
        self.assertEqual(report["exports"]["CUDA_HOME"], "")
        self.assertIn(str(root / "lib"), report["library_directories"])
        self.assertIn("supplies runtime libraries", report["warnings"][0])
        with self.assertRaisesRegex(runtime.RuntimeDiscoveryError, "no working bin/nvcc"):
            self.discover(environ={"CUDA_HOME": str(root)}, require_compiler=True)

    def test_explicit_runtime_library_override_is_first_and_deduplicated(self):
        torch_lib = str(self.torch / "lib")
        report = self.discover(environ={"RUNTIME_LD_LIBRARY_PATH": f"/explicit:{torch_lib}:", "LD_LIBRARY_PATH": f"{torch_lib}:/existing"})
        self.assertEqual(report["exports"]["RUNTIME_LD_LIBRARY_PATH"], f"/explicit:{torch_lib}")
        self.assertEqual(report["exports"]["LD_LIBRARY_PATH"], f"/explicit:{torch_lib}:/existing")
        self.assertEqual(report["exports"]["RLLM_RUNTIME_LD_LIBRARY_PATH"], report["exports"]["RUNTIME_LD_LIBRARY_PATH"])

    def test_path_and_packaged_compiler_discovery(self):
        compiler = self.toolkit("path compiler", "12.6")
        report = self.discover(environ={"PATH": str(compiler / "bin")})
        self.assertEqual(report["exports"]["CUDA_HOME"], str(compiler))
        packaged = self.toolkit("inherited/site-packages/nvidia/cu13", "13.0")
        (self.torch / "version.py").write_text("cuda = '13.0'\n")
        report = self.discover(environ={"PATH": str(compiler / "bin")})
        self.assertEqual(report["exports"]["CUDA_HOME"], str(packaged))

    def test_shell_quoting_preserves_paths_and_prevents_execution(self):
        compiler = self.toolkit("cuda ' spaces $(touch INJECTED)", "12.6")
        existing = "/existing ' $(touch INJECTED)"
        report = self.discover(environ={"CUDA_HOME": str(compiler), "LD_LIBRARY_PATH": existing, "PATH": "/usr/bin"})
        command = runtime.shell_exports(report) + "\n/usr/bin/env -0"
        result = subprocess.run(["/bin/bash", "-c", command], capture_output=True, check=True, cwd=self.root, env={})
        exports = dict(value.split("=", 1) for value in result.stdout.decode().split("\0") if "=" in value)
        for name, value in report["exports"].items():
            self.assertEqual(exports[name], value)
        self.assertFalse((self.root / "INJECTED").exists())

    def test_shell_helper_with_selected_python_and_compiler_requirement(self):
        compiler = self.toolkit("cuda ' spaces $(touch INJECTED)", "12.6")
        command = f"set -euo pipefail; source {shlex.quote(str(SCRIPT.with_suffix('.sh')))}; rllm_configure_gpu_runtime {shlex.quote(sys.executable)}; /usr/bin/env -0"
        env = {"CUDA_HOME": str(compiler), "PYTHONPATH": str(self.inherited), "PATH": "/usr/bin", "RLLM_REQUIRE_CUDA_COMPILER": "1"}
        result = subprocess.run(["/bin/bash", "-c", command], capture_output=True, check=True, cwd=self.root, env=env)
        exports = dict(value.split("=", 1) for value in result.stdout.decode().split("\0") if "=" in value)
        self.assertEqual(exports["CUDA_HOME"], str(compiler))
        self.assertIn(str(self.torch / "lib"), exports["LD_LIBRARY_PATH"])
        self.assertFalse((self.root / "INJECTED").exists())
        (compiler / "bin" / "nvcc").unlink()
        (compiler / "lib").mkdir()
        (compiler / "lib" / "libcudart.so.12").touch()
        result = subprocess.run(["/bin/bash", "-c", command], capture_output=True, cwd=self.root, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no working bin/nvcc", result.stderr.decode())

    def test_cpu_or_missing_torch_is_an_actionable_error(self):
        (self.torch / "version.py").write_text("cuda = None\n")
        with self.assertRaisesRegex(runtime.RuntimeDiscoveryError, "CUDA-enabled"):
            self.discover()
        with self.assertRaisesRegex(runtime.RuntimeDiscoveryError, "No torch"):
            runtime.discover_runtime(search_path=[], environ={}, conventional_roots=[])


if __name__ == "__main__":
    unittest.main()
