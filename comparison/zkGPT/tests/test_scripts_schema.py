#!/usr/bin/env python3
"""Bounded negative tests for the production script/verifier contracts."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common  # noqa: E402
import formal_preflight  # noqa: E402
import gat_zkgpt_experiment as formal_experiment  # noqa: E402
import validate_final_results as final_validator  # noqa: E402
from common import (  # noqa: E402
    ChainError,
    PAPER_TIMING_SEMANTICS,
    cleanup_run_outputs,
    formal_child_environment,
    formal_run_lock,
    native_runtime_libraries,
    require_fixed_digest,
    require_result_shape,
    run_logged,
    verifier_binary,
    verifier_timing,
    verify_build_provenance,
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


class PendingVerificationContractTests(unittest.TestCase):
    def run_stubbed_cell(self, native_status: str, verifier_mode: str) -> tuple[Path, int]:
        temporary = tempfile.TemporaryDirectory(prefix="pending-verifier-contract-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        run_path = root / "runs" / "cora"
        run_path.mkdir(parents=True)
        for name in ("proof.artifact", "verifier_context.bin", "public_logits.txt", "public_statement.json"):
            (run_path / name).write_bytes(b"fixture")
        session_id, run_nonce, session_digest = "1" * 32, "2" * 32, "3" * 64
        result = valid_result_shape()
        result.update({"status": native_status, "verification_status": native_status,
                       "session_id": session_id, "run_nonce": run_nonce,
                       "session_manifest_sha256": session_digest, "proof_generated": True,
                       "verifier_called": native_status == "VERIFY_OK", "verifier_ok": native_status == "VERIFY_OK",
                       "verifier_reads_witness": False, "cpu_fallback": False,
                       "artifact_path": "runs/cora/proof.artifact", "verifier_context_path": "runs/cora/verifier_context.bin",
                       "public_logits_path": "runs/cora/public_logits.txt", "statement_path": "runs/cora/public_statement.json"})
        result_path = run_path / "result.json"
        calls = 0
        def fake_run_logged(command: list[str], _root: Path, _stdout: Path, _stderr: Path, label: str, _telemetry: dict) -> int:
            nonlocal calls
            calls += 1
            if label.endswith("/prover"):
                self.assertIn("--defer-self-verify", command)
                result_path.write_text(json.dumps(result), encoding="utf-8")
                return 0
            if verifier_mode == "exit_failure": return 1
            (run_path / "verifier_report.json").write_text(json.dumps({"status": "REJECTED"}), encoding="utf-8")
            return 0
        config = {"privacy_policy": "model_and_trace_hidden", "privacy_profile": "masked_full"}
        resource = {"full_dataset_identity": "dataset", "checkpoint_identity": "checkpoint", "cache_identity": "cache"}
        identity_patches = (
            mock.patch.object(formal_experiment, "formal_backend_capability", return_value={"supported": True, "reason": ""}),
            mock.patch.object(formal_experiment, "validate_preflight_binding"),
            mock.patch.object(formal_experiment, "config_with_auth", return_value=(root / "config.cfg", config)),
            mock.patch.object(formal_experiment, "resource_identity", return_value=resource),
            mock.patch.object(formal_experiment, "run_directory", return_value=("runs/cora", run_path)),
            mock.patch.object(formal_experiment, "cleanup_run_outputs"),
            mock.patch.object(formal_experiment, "derive_run_nonce", return_value=run_nonce),
            mock.patch.object(formal_experiment, "build_binary", return_value=root / "prover"),
            mock.patch.object(formal_experiment, "verifier_binary", return_value=root / "verifier"),
            mock.patch.object(formal_experiment, "binary_fingerprint", return_value={"sha256": "a" * 64}),
            mock.patch.object(formal_experiment, "relative_path", side_effect=lambda base, path, _what: str(Path(path).relative_to(base))),
            mock.patch.object(formal_experiment, "ensure_registration", return_value=("registration.json", "secret", {})),
            mock.patch.object(formal_experiment, "run_logged", side_effect=fake_run_logged),
            mock.patch.object(formal_experiment, "_result_identity", return_value={}),
            mock.patch.object(formal_experiment, "_resource_result_identity"),
            mock.patch.object(formal_experiment, "require_timing", return_value={}),
            mock.patch.object(formal_experiment, "artifact_fields", return_value=(run_path / "proof.artifact", "")),
            mock.patch.object(formal_experiment, "require_run_file", side_effect=lambda _base, _run, rel, _what: (rel, run_path / Path(rel).name)),
            mock.patch.object(formal_experiment, "assert_fresh_files"),
        )
        with mock.patch.object(formal_experiment, "atomic_write_json") as persist:
            for patcher in identity_patches:
                patcher.start()
                self.addCleanup(patcher.stop)
            try:
                formal_experiment.run_cell(root, "cora", "cpu", {}, "4" * 64, session_id, {}, session_digest, root / "preflight.json")
            except ChainError:
                pass
            else:
                self.fail("the rejected pending-state fixture unexpectedly completed")
            persist.assert_not_called()
        return result_path, calls

    def test_native_verify_ok_claim_is_rejected_before_external_verifier(self) -> None:
        result_path, calls = self.run_stubbed_cell("VERIFY_OK", "bad_report")
        self.assertEqual(calls, 1)
        self.assertEqual(json.loads(result_path.read_text(encoding="utf-8"))["status"], "VERIFY_OK")

    def test_pending_result_is_not_promoted_when_external_verification_fails(self) -> None:
        for mode in ("exit_failure", "bad_report"):
            with self.subTest(mode=mode):
                result_path, calls = self.run_stubbed_cell("PROOF_GENERATED_PENDING_VERIFY", mode)
                self.assertEqual(calls, 2)
                persisted = json.loads(result_path.read_text(encoding="utf-8"))
                self.assertEqual(persisted["status"], "PROOF_GENERATED_PENDING_VERIFY")
                self.assertEqual(persisted["verification_status"], "PROOF_GENERATED_PENDING_VERIFY")
                self.assertIs(persisted["verifier_called"], False)
                self.assertIs(persisted["verifier_ok"], False)


def owner_approval_fixture(root: Path) -> tuple[str, str, dict, dict]:
    session_id, session_digest, dataset = "1" * 32, "2" * 64, "cora"
    receipt_path = root / "results" / "formal" / session_id / "receipts" / f"{dataset}.json"
    receipt_path.parent.mkdir(parents=True)
    run_path = root / "baseline-local" / "runs" / "formal" / session_id / dataset
    run_path.mkdir(parents=True)
    paths = {"result_path": run_path / "result.json", "execution_path": run_path / "execution.json",
             "proof_path": run_path / "proof.artifact", "verification_report_path": run_path / "verifier_report.json"}
    for path in paths.values(): path.write_bytes(path.name.encode("ascii"))
    receipt = {"schema": "zkgpt-gat-cell-receipt-v1", "status": "VERIFY_OK", "dataset": dataset,
               "session_id": session_id, "session_manifest_sha256": session_digest, "independent_verification": True}
    for key, path in paths.items():
        receipt[key] = str(path.relative_to(root))
        receipt[key.replace("_path", "_sha256")] = common.sha256_file(path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    approved = {"dataset": dataset, "receipt_path": str(receipt_path.relative_to(root)),
                "receipt_sha256": common.sha256_file(receipt_path)}
    for key, path in paths.items():
        approved[key] = str(path.relative_to(root))
        approved[key.replace("_path", "_sha256")] = common.sha256_file(path)
    audited = {"dataset": dataset, "backend": "cpu", "result_path": approved["result_path"],
               "artifact_path": approved["proof_path"], "artifact_sha256": approved["proof_sha256"],
               "verifier_report_path": approved["verification_report_path"]}
    return session_id, session_digest, approved, audited


class FinalVerificationReuseContractTests(unittest.TestCase):
    def test_owner_approved_cell_reuses_original_report_without_final_verify(self) -> None:
        with tempfile.TemporaryDirectory(prefix="owner-approved-reuse-") as name:
            root = Path(name)
            session_id, session_digest, approved, audited = owner_approval_fixture(root)
            with mock.patch.object(final_validator, "independently_verify_final_cell", side_effect=AssertionError("final verifier must be skipped")):
                result = final_validator._final_verification_record(root, session_id, session_digest, "cora", audited, {"cora": approved})
            self.assertEqual(result["verification_source"], "cell_stage_independent_report")
            self.assertIs(result["reused_cell_report"], True)
            self.assertEqual(result["final_native_verifier_invocations"], 0)
            self.assertEqual(result["report_path"], approved["verification_report_path"])
            self.assertEqual(result["report_sha256"], approved["verification_report_sha256"])

    def test_owner_approved_cell_rejects_changed_report_or_execution_identity(self) -> None:
        for changed_path in ("verification_report_path", "execution_path"):
            with self.subTest(changed_path=changed_path), tempfile.TemporaryDirectory(prefix="owner-approved-tamper-") as name:
                root = Path(name)
                session_id, session_digest, approved, audited = owner_approval_fixture(root)
                changed = root / approved[changed_path]
                changed.write_bytes(b"changed after owner approval")
                with self.assertRaises(ChainError):
                    final_validator._final_verification_record(root, session_id, session_digest, "cora", audited, {"cora": approved})


class ProductionContractTests(unittest.TestCase):
    def test_shape_admission_capacity_accounts_bitmap_and_touched_indices(self) -> None:
        def capacity_shape(source_entries: int) -> dict:
            words = source_entries // 64 + (1 if source_entries % 64 else 0)
            bitmap_bytes = 2 * words * 8
            touched_bytes = 2 * words * ctypes.sizeof(ctypes.c_size_t)
            mapped_bitmap = bitmap_bytes if bitmap_bytes > (8 << 20) else 0
            mapped_touched = touched_bytes if touched_bytes > (8 << 20) else 0
            return {
                "gkr_subset_source_support_entries_upper_bound": source_entries,
                "gkr_point_shape_bitmap_scratch_bytes": bitmap_bytes,
                "gkr_point_shape_touched_index_scratch_bytes_upper_bound": touched_bytes,
                "estimated_peak_host_memory_bytes": bitmap_bytes + touched_bytes,
                "native_mapped_phase_breakdown": {
                    "shape_admission": mapped_bitmap + mapped_touched,
                    "component:shape_admission_bitmap": mapped_bitmap,
                    "component:shape_admission_touched_indices": mapped_touched,
                },
            }

        for entries in (65, 1 << 27):
            with self.subTest(source_entries=entries):
                payload = capacity_shape(entries)
                formal_preflight._validate_shape_admission_capacity(payload, "synthetic")
                undercounted = dict(payload)
                undercounted["estimated_peak_host_memory_bytes"] -= 1
                with self.assertRaises(ChainError):
                    formal_preflight._validate_shape_admission_capacity(undercounted, "synthetic")
                stale_phase = dict(payload)
                stale_phase["native_mapped_phase_breakdown"] = dict(payload["native_mapped_phase_breakdown"])
                stale_phase["native_mapped_phase_breakdown"]["shape_admission"] += 1
                with self.assertRaises(ChainError):
                    formal_preflight._validate_shape_admission_capacity(stale_phase, "synthetic")

    def test_formal_child_environment_rejects_loader_overrides(self) -> None:
        for variable in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
            with self.subTest(variable=variable):
                with self.assertRaises(ChainError):
                    formal_child_environment({variable: "/tmp/untrusted-loader"})
        self.assertNotIn("LD_LIBRARY_PATH", formal_child_environment({"LD_LIBRARY_PATH": ""}))

    def test_formal_run_lock_fd_can_be_passed_to_owned_child(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lock-fd-contract-") as name:
            directory = Path(name)
            with formal_run_lock(directory) as lock_stream:
                stdout_path = directory / "child.stdout"
                stderr_path = directory / "child.stderr"
                code = run_logged(
                    [sys.executable, "-c", "import os,sys; os.fstat(int(sys.argv[1])); print('LOCK_FD_OK')",
                     str(lock_stream.fileno())],
                    directory, stdout_path, stderr_path, "lock-fd-contract",
                    pass_fds=(lock_stream.fileno(),),
                )
                self.assertEqual(code, 0)
                self.assertEqual(stdout_path.read_text(encoding="utf-8").strip(), "LOCK_FD_OK")

    def test_native_runtime_provenance_binds_same_local_mcl(self) -> None:
        runtime = native_runtime_libraries(ROOT, "cpu")
        prover = runtime["mcl"]["prover"]
        verifier = runtime["mcl"]["verifier"]
        self.assertEqual(prover["resolved_path"], verifier["resolved_path"])
        self.assertEqual(prover["sha256"], verifier["sha256"])
        self.assertTrue(prover["resolved_path"].startswith("build/cpu/lib/"))

    def test_runtime_provenance_ignores_path_shadow_ldd(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ldd-shadow-contract-") as name:
            shadow_dir = Path(name)
            shadow_ldd = shadow_dir / "ldd"
            shadow_ldd.write_text("#!/bin/sh\nexit 93\n", encoding="utf-8")
            shadow_ldd.chmod(0o755)
            path = f"{shadow_dir}:{os.environ.get('PATH', '')}"
            with mock.patch.dict(os.environ, {"PATH": path}):
                runtime = native_runtime_libraries(ROOT, "cpu")
        self.assertTrue(runtime["mcl"]["prover"]["resolved_path"].startswith("build/cpu/lib/"))

    def test_build_provenance_rejects_missing_or_changed_mcl_identity(self) -> None:
        source_digest = "a" * 64
        binary_digest = "b" * 64
        runtime = {
            "mcl": {
                "soname": "libmcl.so.3",
                "prover": {
                    "loader_path": "build/cpu/lib/libmcl.so.3",
                    "resolved_path": "build/cpu/lib/libmcl.so.3.3",
                    "sha256": "c" * 64,
                },
                "verifier": {
                    "loader_path": "build/cpu/lib/libmcl.so.3",
                    "resolved_path": "build/cpu/lib/libmcl.so.3.3",
                    "sha256": "c" * 64,
                },
            },
        }
        expected = {
            "schema": "zkgpt-gat-build-v2",
            "backend": "cpu",
            "source_tree_sha256": source_digest,
            "prover_sha256": binary_digest,
            "verifier_sha256": binary_digest,
            "runtime_libraries": runtime,
        }
        stale_path = json.loads(json.dumps(expected))
        stale_path["runtime_libraries"]["mcl"]["prover"]["resolved_path"] = "build/cpu/lib/other.so"
        stale_hash = json.loads(json.dumps(expected))
        stale_hash["runtime_libraries"]["mcl"]["verifier"]["sha256"] = "d" * 64
        v1 = {key: value for key, value in expected.items() if key != "runtime_libraries"}
        v1["schema"] = "zkgpt-gat-build-v1"
        missing_mcl = {key: value for key, value in expected.items() if key != "runtime_libraries"}

        with tempfile.TemporaryDirectory(prefix="build-provenance-contract-") as name:
            root = Path(name)
            receipt = root / "build" / "cpu" / "build_provenance.json"
            receipt.parent.mkdir(parents=True)
            with mock.patch.object(common, "source_tree_sha256", return_value=source_digest), \
                    mock.patch.object(common, "build_binary", return_value=root / "prover"), \
                    mock.patch.object(common, "verifier_binary", return_value=root / "verifier"), \
                    mock.patch.object(common, "sha256_file", return_value=binary_digest), \
                    mock.patch.object(common, "native_runtime_libraries", return_value=runtime):
                receipt.write_text(json.dumps(expected), encoding="utf-8")
                self.assertEqual(verify_build_provenance(root, "cpu"), source_digest)
                for stale in (v1, missing_mcl, stale_path, stale_hash):
                    with self.subTest(receipt=stale):
                        receipt.write_text(json.dumps(stale), encoding="utf-8")
                        with self.assertRaises(ChainError):
                            verify_build_provenance(root, "cpu")

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

    def test_native_result_timing_semantics_matches_consumer_and_rejects_v3(self) -> None:
        native_source = (ROOT / "gat_adapter" / "gat_cli.cpp").read_text(encoding="utf-8")
        self.assertIn(f'paper_timing_semantics = "{PAPER_TIMING_SEMANTICS}"', native_source)
        value = valid_result_shape()
        value["paper_timing_semantics"] = "zkgpt-gat-paper-timing-v3-native-commitment-io-excluded"
        with self.assertRaises(ChainError):
            require_result_shape(value, ROOT / "stale-timing-result.json")

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
