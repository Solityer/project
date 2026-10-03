#!/usr/bin/env python3
"""Short negative-path checks for ezkl packaging and result tooling."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class PackagingScriptsTest(unittest.TestCase):

    def run_script(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / script), *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_resource_manifest_is_exact(self) -> None:
        completed = self.run_script("validate_resources.py")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("resource manifest valid:", completed.stdout)

    def test_matrix_defaults_to_dry_run(self) -> None:
        completed = self.run_script("run_formal_matrix.py")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("no setup/prove/verify was run", completed.stdout)
        self.assertIn("five-dataset hybrid CPU+GPU chain", completed.stdout)
        self.assertIn("scripts/run_formal_matrix.py --execute-formal", completed.stdout)
        self.assertIn("target/gpu/release/ezkl", completed.stdout)

    def test_single_runner_dry_run_does_not_require_contract(self) -> None:
        completed = self.run_script("run_single_dataset.py", "--dataset", "cora", "--backend", "cpu", "--dry-run")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("no setup, statement wait, final, prove, verify", completed.stdout)

    def test_collector_rejects_incomplete_fixture(self) -> None:
        completed = self.run_script(
            "collect_results.py",
            "--runs-root",
            "tests/fixtures",
            "--output",
            "results/collector_should_not_write.json",
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(
            "missing" in completed.stderr.lower() or "tests/assets" in completed.stderr.lower(),
            completed.stderr,
        )
        self.assertFalse((ROOT / "results/collector_should_not_write.json").exists())

    def test_final_validator_rejects_incomplete_fixture(self) -> None:
        completed = self.run_script(
            "validate_final.py",
            "--matrix",
            "tests/fixtures/invalid_matrix.json",
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("schema mismatch", completed.stderr.lower())

    def test_fixture_is_not_a_formal_success(self) -> None:
        fixture = json.loads((ROOT / "tests/fixtures/invalid_result.json").read_text())
        self.assertNotEqual(fixture.get("status"), "VERIFY_OK")
        self.assertNotIn("proof", fixture)


if __name__ == "__main__":
    unittest.main()
