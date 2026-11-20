"""Verify temporary installed-package fixtures without importing or modifying VeRL."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[2] / "scripts/verify_verl_compat.py"
spec = importlib.util.spec_from_file_location("installed_verl_compat", TOOL)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class InstalledPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.site = self.root / "site-packages"
        self.package = self.site / "verl"
        self.package.mkdir(parents=True)
        (self.package / "__init__.py").write_text('raise AssertionError("VeRL must not be imported")\n')
        self.production = ("verl/trainer/ppo/ray_trainer.py", "verl/experimental/separation/ray_trainer.py")
        self.test_path = "tests/trainer/ppo/test_dataloader_resume_on_cpu.py"
        old, new = b"RESTORED = False\n", b"RESTORED = True\n"
        self.files = {}
        patches = []
        for relative in (*self.production, self.test_path):
            is_test = relative == self.test_path
            self.files[relative] = {
                "base_sha256": None if is_test else compat.sha256(old),
                "patched_sha256": compat.sha256(new),
                "git_mode": "100644",
            }
            if is_test:
                patches.append(f"diff --git a/{relative} b/{relative}\nnew file mode 100644\n--- /dev/null\n+++ b/{relative}\n@@ -0,0 +1 @@\n+RESTORED = True\n")
            else:
                patches.append(f"diff --git a/{relative} b/{relative}\n--- a/{relative}\n+++ b/{relative}\n@@ -1 +1 @@\n-RESTORED = False\n+RESTORED = True\n")
                path = self.site / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(new)
        self.patch_path = self.root / "resume.patch"
        self.patch_path.write_text("".join(patches))
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "base_revision": "a" * 40,
                    "patch": {"id": "fixture-resume", "path": "resume.patch", "sha256": compat.sha256(self.patch_path.read_bytes()), "files": self.files},
                }
            )
        )
        module_patch = patch.dict(sys.modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        sys.modules.pop("verl", None)
        search_path_patch = patch.object(sys, "path", [str(self.site), *sys.path])
        search_path_patch.start()
        self.addCleanup(search_path_patch.stop)

    def verify(self):
        return compat.verify_installed(self.manifest)

    def snapshot(self):
        return {str(path.relative_to(self.site)): (path.read_bytes(), path.stat().st_mtime_ns) for path in self.site.rglob("*") if path.is_file()}

    def test_protected_production_files_verify_without_tests_git_imports_or_writes(self):
        before = self.snapshot()
        with patch.object(compat, "_git", side_effect=AssertionError("Installed check must not invoke Git")):
            result = self.verify()
        self.assertEqual(result["state"], "verified")
        self.assertFalse(result["changed"])
        self.assertEqual(result["package_root"], str(self.package))
        self.assertEqual(result["protected_files"], sorted(self.production))
        self.assertEqual(result["scope"], "protected installed source hashes, not full runtime or GPU validation")
        self.assertNotIn("verl", sys.modules)
        self.assertFalse((self.site / self.test_path).exists())
        self.assertEqual(self.snapshot(), before)

    def test_stale_unpatched_installed_file_is_rejected_without_writes(self):
        (self.site / self.production[0]).write_text("RESTORED = False\n")
        before = self.snapshot()
        with self.assertRaisesRegex(compat.CompatibilityError, "Installed VeRL checksum mismatch.*reinstall"):
            self.verify()
        self.assertEqual(self.snapshot(), before)

    def test_missing_protected_file_is_a_clear_error(self):
        (self.site / self.production[1]).unlink()
        with self.assertRaisesRegex(compat.CompatibilityError, "Missing protected installed VeRL file"):
            self.verify()

    def test_import_precedence_selects_actual_package_instead_of_first_site(self):
        preferred_site = self.root / "preferred"
        preferred_package = preferred_site / "verl"
        preferred_package.mkdir(parents=True)
        (preferred_package / "__init__.py").write_text('raise AssertionError("Do not import")\n')
        for relative in self.production:
            path = preferred_site / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.site / relative).read_bytes())
        (self.site / self.production[0]).write_text("RESTORED = False\n")
        with patch.object(sys, "path", [str(preferred_site), *sys.path]):
            result = self.verify()
        self.assertEqual(result["package_root"], str(preferred_package))
        self.assertNotIn("verl", sys.modules)

    def test_missing_and_namespace_packages_are_rejected(self):
        for resolved in (None, SimpleNamespace(origin=None, submodule_search_locations=[str(self.package)])):
            with self.subTest(resolved=resolved), patch.object(compat.importlib.util, "find_spec", return_value=resolved):
                with self.assertRaisesRegex(compat.CompatibilityError, "ordinary installed VeRL"):
                    self.verify()

    def test_ambiguous_package_search_locations_are_rejected(self):
        resolved = SimpleNamespace(origin=str(self.package / "__init__.py"), submodule_search_locations=[str(self.package), str(self.root / "other")])
        with patch.object(compat.importlib.util, "find_spec", return_value=resolved):
            with self.assertRaisesRegex(compat.CompatibilityError, "resolution is ambiguous"):
                self.verify()

    def test_broken_import_spec_reports_compatibility_error(self):
        with patch.object(compat.importlib.util, "find_spec", side_effect=ValueError("verl.__spec__ is None")):
            with self.assertRaisesRegex(compat.CompatibilityError, "Could not resolve the installed VeRL"):
                self.verify()

    def test_manifest_patch_checksum_is_still_enforced(self):
        self.patch_path.write_bytes(self.patch_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(compat.CompatibilityError, "Patch checksum mismatch"):
            self.verify()

    def test_cli_returns_success_and_reports_selected_package_scope(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = compat.main(["--installed", "--manifest", str(self.manifest)])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["package_root"], str(self.package))
        self.assertEqual(result["state"], "verified")

    def test_cli_mismatch_is_nonzero_with_clear_diagnostic(self):
        (self.site / self.production[0]).write_text("RESTORED = False\n")
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            code = compat.main(["--installed", "--manifest", str(self.manifest)])
        self.assertEqual(code, 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result["state"], "error")
        self.assertIn("Installed VeRL checksum mismatch", result["error"])

    def test_cli_rejects_apply_and_conflicting_targets_before_inspection(self):
        for arguments in (["--installed", "--apply"], ["--installed", "--source", str(self.root)]):
            with self.subTest(arguments=arguments), patch.object(compat, "verify_installed") as verify, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    compat.main(arguments)
                self.assertEqual(error.exception.code, 2)
                verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
