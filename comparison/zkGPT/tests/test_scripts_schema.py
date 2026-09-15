#!/usr/bin/env python3
"""Bounded negative tests for the production script/verifier contracts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import ChainError, require_result_shape  # noqa: E402
from validate_final_results import validate_cell  # noqa: E402


class ProductionContractTests(unittest.TestCase):
    def test_result_schema_rejects_missing_privacy_field(self) -> None:
        value = {
            "schema": "zkgpt-gat-result-v1",
            "dataset": "cora",
            "backend": "cpu",
            "status": "VERIFY_OK",
            "verification_status": "VERIFY_OK",
            "proof_scope": "full_gat",
            "privacy_policy": "model_and_trace_hidden",
            "output_mode": "public_logits",
        }
        with self.assertRaises(ChainError):
            require_result_shape(value, ROOT / "missing-result.json")

    def test_result_schema_rejects_unsupported_schema(self) -> None:
        value = {
            "schema": "unsupported-result-v0",
            "dataset": "cora",
            "backend": "cpu",
            "status": "VERIFY_OK",
            "verification_status": "VERIFY_OK",
            "proof_scope": "full_gat",
            "privacy_policy": "model_and_trace_hidden",
            "privacy_profile": "masked_full",
            "output_mode": "public_logits",
        }
        with self.assertRaises(ChainError):
            require_result_shape(value, ROOT / "unsupported-result.json")

    def test_final_validator_rejects_unsupported_cell_without_io(self) -> None:
        with self.assertRaises(ChainError):
            validate_cell(ROOT, {"dataset": "not-a-dataset", "backend": "cpu"})

    def test_verifier_rejects_malformed_artifact(self) -> None:
        executable = ROOT / "cmake-build-gat-prod" / "gat_adapter" / "zkgpt_gat_verifier"
        if not executable.is_file():
            self.skipTest("CPU verifier target has not been built")
        with tempfile.TemporaryDirectory(prefix="verifier-negative-", dir=str(ROOT)) as name:
            directory = Path(name)
            context = directory / "verifier_context.bin"
            logits = directory / "public_logits.txt"
            artifact = directory / "proof.artifact"
            context.write_text(json.dumps({
                "schema": "zkgpt-gat-verifier-context-v3",
                "relation_id": "R_GAT-zkGPT",
                "privacy_extension_profile_id": "masked_full",
                "statement_digest": "a" * 64,
                "dataset": "cora",
                "backend": "cpu",
                "public_logits_path": str(logits.relative_to(ROOT)),
            }) + "\n", encoding="utf-8")
            logits.write_text("1 1\n0\n", encoding="utf-8")
            artifact.write_bytes(b"not-a-zkgpt-gat-artifact")
            completed = subprocess.run([
                str(executable),
                "--baseline-root", str(ROOT),
                "--verifier-context", str(context.relative_to(ROOT)),
                "--public-logits", str(logits.relative_to(ROOT)),
                "--artifact", str(artifact.relative_to(ROOT)),
            ], cwd=str(ROOT), text=True, capture_output=True, check=False)
            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
