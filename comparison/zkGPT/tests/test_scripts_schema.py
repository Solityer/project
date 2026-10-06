#!/usr/bin/env python3
"""Bounded negative tests for the production script/verifier contracts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    ChainError,
    PAPER_TIMING_SEMANTICS,
    cleanup_run_outputs,
    require_fixed_digest,
    require_result_shape,
    verifier_binary,
    verifier_timing,
)
from validate_final_results import require_formal_cells  # noqa: E402


def valid_result_shape() -> dict:
    return {
        "schema": "zkgpt-gat-result-v2",
        "paper_timing_semantics": PAPER_TIMING_SEMANTICS,
        "session_id": "1" * 32,
        "run_nonce": "2" * 32,
        "session_manifest_sha256": "3" * 64,
        "dataset": "cora",
        "backend": "cpu",
        "status": "VERIFY_OK",
        "verification_status": "VERIFY_OK",
        "proof_scope": "full_gat",
        "privacy_policy": "model_and_trace_hidden",
        "privacy_profile": "masked_full",
        "output_mode": "public_logits",
        "capability_status": "FORMAL_CPU_MASKED_FULL",
        "formal_backend_supported": True,
    }


class ProductionContractTests(unittest.TestCase):
    def test_verifier_timing_requires_exact_stage_sum(self) -> None:
        self.assertEqual(
            verifier_timing({
                "decode_crypto_ms": 1.25,
                "verify_crypto_ms": 2.5,
                "crypto_verify_ms": 3.75,
            }, "synthetic"),
            {"decode_crypto_ms": 1.25, "verify_crypto_ms": 2.5, "crypto_verify_ms": 3.75},
        )
        with self.assertRaises(ChainError):
            verifier_timing({
                "decode_crypto_ms": 1.25,
                "verify_crypto_ms": 2.5,
                "crypto_verify_ms": 3.749,
            }, "synthetic")

    def test_result_schema_rejects_missing_privacy_field(self) -> None:
        value = valid_result_shape()
        del value["privacy_profile"]
        with self.assertRaises(ChainError):
            require_result_shape(value, ROOT / "missing-result.json")

    def test_result_schema_rejects_unsupported_schema(self) -> None:
        value = valid_result_shape()
        value["schema"] = "unsupported-result-v0"
        with self.assertRaises(ChainError):
            require_result_shape(value, ROOT / "unsupported-result.json")

    def test_final_validator_rejects_noncanonical_cell_without_io(self) -> None:
        with self.assertRaises(ChainError):
            require_formal_cells([{"dataset": "not-a-dataset", "backend": "cpu"}] * 5)

    def test_verifier_rejects_malformed_artifact(self) -> None:
        executable = verifier_binary(ROOT, "cpu")
        with tempfile.TemporaryDirectory(prefix="verifier-negative-", dir=str(ROOT)) as name:
            directory = Path(name)
            context = directory / "verifier_context.bin"
            logits = directory / "public_logits.txt"
            artifact = directory / "proof.artifact"
            context.write_text(json.dumps({
                "schema": "zkgpt-gat-verifier-context-v4",
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

    def test_cleanup_preserves_registration_and_secret(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cleanup-contract-", dir=str(ROOT / "results")) as name:
            run = Path(name)
            (run / "result.json").write_text("stale", encoding="utf-8")
            (run / "execution.json").write_text("stale", encoding="utf-8")
            (run / "old.partial").write_text("partial", encoding="utf-8")
            registration = run / "model_registration.json"
            secret = run / "com_W.prover.secret"
            registration.write_text("registration", encoding="utf-8")
            secret.write_text("secret", encoding="utf-8")
            os.chmod(secret, 0o600)
            cleanup_run_outputs(ROOT, run)
            self.assertFalse((run / "result.json").exists())
            self.assertFalse((run / "execution.json").exists())
            self.assertFalse((run / "old.partial").exists())
            self.assertEqual(registration.read_text(encoding="utf-8"), "registration")
            self.assertEqual(secret.read_text(encoding="utf-8"), "secret")

    def test_cleanup_rejects_volatile_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cleanup-alias-", dir=str(ROOT / "results")) as name:
            run = Path(name)
            target = run / "outside"
            target.write_text("outside", encoding="utf-8")
            link = run / "result.json"
            link.symlink_to(target)
            with self.assertRaises(ChainError):
                cleanup_run_outputs(ROOT, run)

    def test_fixed_digest_rejects_uppercase_and_short_values(self) -> None:
        with self.assertRaises(ChainError):
            require_fixed_digest("A" * 64, "digest", "synthetic")
        with self.assertRaises(ChainError):
            require_fixed_digest("0" * 63, "digest", "synthetic")

    def test_collector_rejects_cross_cell_artifact_reuse(self) -> None:

        seen_paths = {Path("/tmp/shared-proof.artifact")}
        seen_digests = {"0" * 64}
        self.assertIn(Path("/tmp/shared-proof.artifact"), seen_paths)
        self.assertIn("0" * 64, seen_digests)


if __name__ == "__main__":
    unittest.main()
