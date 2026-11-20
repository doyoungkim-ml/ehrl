"""Guard the source-only patch workflow using tiny Git repositories and no ML imports."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "scripts/verify_verl_compat.py"
spec = importlib.util.spec_from_file_location("verify_verl_compat", TOOL)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class SourcePatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "verl-source"
        self.source.mkdir()
        self.git("init", "-q")
        (self.source / "verl").mkdir()
        self.target = self.source / "verl/trainer.py"
        self.target.write_text("state = 'old'\n")
        (self.source / "README.md").write_text("VeRL fixture\n")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD").stdout.decode().strip()
        self.target.write_text("state = 'restored'\n")
        self.new_test = self.source / "tests/test_resume.py"
        self.new_test.parent.mkdir()
        self.new_test.write_text("assert True\n")
        self.git("add", "-N", "tests/test_resume.py")
        patch_bytes = self.git("diff", "--").stdout
        self.git("reset", "--", "tests/test_resume.py")
        self.patch = self.root / "resume.patch"
        self.patch.write_bytes(patch_bytes)
        self.manifest = self.root / "manifest.json"
        self.contract = {
            "schema_version": 1,
            "base_revision": self.base,
            "patch": {
                "id": "test-resume",
                "path": "resume.patch",
                "sha256": compat.sha256(patch_bytes),
                "files": {
                    "verl/trainer.py": {
                        "git_mode": "100644",
                        "base_sha256": compat.sha256(b"state = 'old'\n"),
                        "patched_sha256": compat.sha256(self.target.read_bytes()),
                    },
                    "tests/test_resume.py": {
                        "git_mode": "100644",
                        "base_sha256": None,
                        "patched_sha256": compat.sha256(self.new_test.read_bytes()),
                    },
                },
            },
        }
        self.save_manifest()
        self.target.write_text("state = 'old'\n")
        self.new_test.unlink()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.source), *args], capture_output=True, check=True)

    def save_manifest(self):
        self.manifest.write_text(json.dumps(self.contract))

    def verify(self, *, apply=False):
        return compat.verify(self.source, self.manifest, apply=apply)

    def test_default_is_read_only(self):
        result = self.verify()
        self.assertEqual(result["state"], "unpatched")
        self.assertFalse(result["changed"])
        self.assertEqual(self.target.read_text(), "state = 'old'\n")
        self.assertFalse(self.new_test.exists())
        self.assertEqual(self.git("status", "--porcelain").stdout, b"")

    def test_apply_and_repeat_preserve_exact_patch_and_index(self):
        result = self.verify(apply=True)
        self.assertEqual(result["state"], "applied")
        self.assertTrue(result["changed"])
        self.assertEqual(self.target.read_text(), "state = 'restored'\n")
        self.assertTrue(self.new_test.exists())
        self.assertEqual(self.git("diff", "--cached").stdout, b"")
        self.assertEqual(self.verify()["state"], "applied")
        self.assertFalse(self.verify(apply=True)["changed"])

    def test_cli_reports_unpatched_with_nonzero_exit(self):
        result = subprocess.run(
            [sys.executable, str(TOOL), "--source", str(self.source), "--manifest", str(self.manifest)],
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["state"], "unpatched")
        self.assertEqual(self.target.read_text(), "state = 'old'\n")

    def test_wrong_revision_is_rejected_before_writes(self):
        self.contract["base_revision"] = "a" * 40
        self.save_manifest()
        with self.assertRaisesRegex(compat.CompatibilityError, "Wrong VeRL pin"):
            self.verify(apply=True)
        self.assertFalse(self.new_test.exists())

    def test_checksum_mismatch_is_rejected_before_writes(self):
        self.patch.write_bytes(self.patch.read_bytes() + b"\n")
        with self.assertRaisesRegex(compat.CompatibilityError, "Patch checksum mismatch"):
            self.verify(apply=True)
        self.assertEqual(self.target.read_text(), "state = 'old'\n")

    def test_touched_file_drift_is_rejected(self):
        self.target.write_text("state = 'unreviewed'\n")
        with self.assertRaisesRegex(compat.CompatibilityError, "Source checksum drift"):
            self.verify(apply=True)
        self.assertEqual(self.target.read_text(), "state = 'unreviewed'\n")
        self.assertFalse(self.new_test.exists())

    def test_unrelated_tracked_and_untracked_drift_is_rejected(self):
        for name in ("README.md", "notes.txt"):
            with self.subTest(name=name):
                path = self.source / name
                original = path.read_bytes() if path.exists() else None
                path.write_text("do not overwrite\n")
                with self.assertRaisesRegex(compat.CompatibilityError, "Unrelated source drift"):
                    self.verify(apply=True)
                if original is None:
                    path.unlink()
                else:
                    path.write_bytes(original)

    def test_staged_changes_are_rejected(self):
        self.target.write_text("state = 'restored'\n")
        self.git("add", "verl/trainer.py")
        with self.assertRaisesRegex(compat.CompatibilityError, "Staged changes"):
            self.verify(apply=True)

    def test_executable_mode_drift_is_rejected(self):
        self.target.chmod(0o755)
        with self.assertRaisesRegex(compat.CompatibilityError, "executable-mode drift"):
            self.verify(apply=True)
        self.assertFalse(self.new_test.exists())

    def test_patched_new_file_mode_drift_is_rejected(self):
        self.verify(apply=True)
        self.new_test.chmod(0o755)
        with self.assertRaisesRegex(compat.CompatibilityError, "executable-mode drift"):
            self.verify()

    def test_partial_patch_is_rejected(self):
        self.target.write_text("state = 'restored'\n")
        with self.assertRaisesRegex(compat.CompatibilityError, "Partial patch"):
            self.verify(apply=True)
        self.assertFalse(self.new_test.exists())

    def test_symlink_target_is_rejected(self):
        external = self.root / "external.py"
        external.write_bytes(self.target.read_bytes())
        self.target.unlink()
        self.target.symlink_to(external)
        with self.assertRaisesRegex(compat.CompatibilityError, "Symlink"):
            self.verify(apply=True)
        self.assertEqual(external.read_text(), "state = 'old'\n")

    def test_symlink_parent_is_rejected(self):
        tests_outside = self.root / "tests-outside"
        tests_outside.mkdir()
        self.new_test.parent.rmdir()
        self.new_test.parent.symlink_to(tests_outside, target_is_directory=True)
        with self.assertRaises(compat.CompatibilityError):
            self.verify(apply=True)
        self.assertEqual(list(tests_outside.iterdir()), [])

    def test_installed_package_source_is_rejected(self):
        parent = self.root / "site-packages"
        parent.mkdir()
        destination = parent / "verl-source"
        self.source.rename(destination)
        self.source = destination
        with self.assertRaisesRegex(compat.CompatibilityError, "Installed packages"):
            self.verify(apply=True)

    def test_source_must_be_checkout_root(self):
        with self.assertRaisesRegex(compat.CompatibilityError, "root of a VeRL"):
            compat.verify(self.source / "verl", self.manifest, apply=True)

    def test_manifest_cannot_escape_patch_directory(self):
        self.contract["patch"]["path"] = "../resume.patch"
        self.save_manifest()
        with self.assertRaisesRegex(compat.CompatibilityError, "Invalid relative path"):
            self.verify(apply=True)

    def test_manifest_inventory_must_match_actual_patch(self):
        del self.contract["patch"]["files"]["tests/test_resume.py"]
        self.save_manifest()
        with self.assertRaisesRegex(compat.CompatibilityError, "manifest inventory"):
            self.verify(apply=True)

    def test_malformed_manifest_is_clear_error(self):
        self.manifest.write_text('{"schema_version":1}')
        with self.assertRaisesRegex(compat.CompatibilityError, "Invalid compatibility manifest"):
            self.verify(apply=True)

    def test_forged_result_hash_is_rejected_by_reverse_check(self):
        self.contract["patch"]["files"]["verl/trainer.py"]["patched_sha256"] = compat.sha256(b"state = 'forged'\n")
        self.save_manifest()
        self.target.write_text("state = 'forged'\n")
        self.new_test.write_text("assert True\n")
        with self.assertRaisesRegex(compat.CompatibilityError, "Git apply.*failed"):
            self.verify()


class VendoredArtifactTests(unittest.TestCase):
    def test_actual_patch_checksum_and_inventory(self):
        manifest, patch_bytes = compat.load_manifest(compat.DEFAULT_MANIFEST)
        self.assertEqual(manifest["base_revision"], "bcb638649a50e58494a8ddd92085ad1174f674b8")
        self.assertIn("tests/trainer/ppo/test_dataloader_resume_on_cpu.py", manifest["patch"]["files"])
        self.assertIn(b"_dataloader_resume_iterator_finished", patch_bytes)


if __name__ == "__main__":
    unittest.main()
