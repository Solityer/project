import hashlib
import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_secure_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_secure_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AGGREGATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATOR)


class SecureAggregationRuntimeTests(unittest.TestCase):
    def digest(self, label):
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    def tamper_map(self):
        names = (
            "codec_tag", "codec_length", "codec_trailing", "public_output",
            "metadata", "schema", "ids", "dynamic_commitment",
            "quotient_commitment", "opening", "reblind_commitment",
            "static_registry", "model_registry", "reference_mismatch",
            "private_opening", "route_padding", "route_query",
            "lookup_inverse", "lookup_multiplicity", "psq", "shift", "logup",
            "aggregate", "average", "division", "transcript_challenge",
            "transcript_value", "srs_metadata", "srs_consistency",
            "srs_resource_identity", "srs_resource_compromised", "proof_scope",
            "observation_ledger", "random_identity_ledger", "integer_bound_ledger",
            "integer_bound_certificate", "security_budget_ledger",
            "security_budget_exact", "public_input_digest",
        )
        return ",".join(f"{name}=reject" for name in names)

    def registration(self):
        return {
            "registry_digest": self.digest("registry"),
            "security_budget_ledger_digest": self.digest("budget-ledger"),
            "security_budget_exact": "1/2",
            "public_input_digest": self.digest("public-input"),
            "topology_digest": self.digest("topology"),
        }

    def report(self):
        return {
            "reference_consistency_passed": True,
            "tamper_tests_passed": True,
            "tamper_test_case_map": self.tamper_map(),
            "security_budget_passed": True,
            "security_budget_ledger_digest": self.digest("budget-ledger"),
            "security_budget_exact": "1/2",
        }

    def acceptance_manifest(self):
        return {
            "status": "VERIFY_OK",
            "verifier_accepted": "true",
            "final_formal_run_accepted": "true",
            "verified": "true",
        }

    def test_single_run_acceptance_rejects_final_false_status_spoof(self):
        manifest = self.acceptance_manifest()
        manifest["final_formal_run_accepted"] = "false"
        with self.assertRaises(ValueError):
            AGGREGATOR.check_single_run_acceptance_state(manifest, "synthetic")

    def test_single_run_acceptance_rejects_missing_final_field(self):
        manifest = self.acceptance_manifest()
        del manifest["final_formal_run_accepted"]
        with self.assertRaises(ValueError):
            AGGREGATOR.check_single_run_acceptance_state(manifest, "synthetic")

    def test_single_run_acceptance_rejects_missing_verifier_field(self):
        manifest = self.acceptance_manifest()
        del manifest["verifier_accepted"]
        with self.assertRaises(ValueError):
            AGGREGATOR.check_single_run_acceptance_state(manifest, "synthetic")

    def test_single_run_acceptance_rejects_verified_verifier_mismatch(self):
        manifest = self.acceptance_manifest()
        manifest["verified"] = "false"
        with self.assertRaises(ValueError):
            AGGREGATOR.check_single_run_acceptance_state(manifest, "synthetic")

    def pair_row(self):
        return {
            field: f"value:{field}"
            for field in AGGREGATOR.PAIR_CONSISTENCY_FIELDS
        }

    def test_cpu_cuda_pair_with_equal_public_and_trace_payloads_passes(self):
        cpu = self.pair_row()
        cuda = dict(cpu)
        AGGREGATOR.compare_cpu_cuda_pair("synthetic", cpu, cuda)

    def test_cpu_cuda_pair_hash_mutation_fails_closed(self):
        cpu = self.pair_row()
        cuda = dict(cpu)
        cuda["reference_quant_logits_hash"] = "mutated"
        with self.assertRaises(ValueError):
            AGGREGATOR.compare_cpu_cuda_pair("synthetic", cpu, cuda)

    def test_cpu_cuda_pair_missing_field_fails_closed(self):
        cpu = self.pair_row()
        cuda = dict(cpu)
        del cuda["secure_trace_logits_hash"]
        with self.assertRaises(ValueError):
            AGGREGATOR.compare_cpu_cuda_pair("synthetic", cpu, cuda)

    def test_single_run_reference_consistency_does_not_create_cpu_gpu_parity(self):
        proof_bytes = b"synthetic proof"
        with tempfile.TemporaryDirectory() as temporary:
            proof_path = pathlib.Path(temporary) / "proof.bin"
            proof_path.write_bytes(proof_bytes)
            manifest = {
                "verifier_reads_witness": "false",
                "private_plain_value_count": "0",
                "proof_payload_audit_status": "passed",
                "reference_consistency_passed": "true",
                "tamper_tests_passed": "true",
                "tamper_tests_ms": "1.0",
                "tamper_test_case_map": self.tamper_map(),
                "security_budget_passed": "true",
                "security_budget_exact": "1/2",
                "proof_hash": hashlib.sha256(proof_bytes).hexdigest(),
                "public_logits_hash": self.digest("public-output"),
                "secure_trace_logits_hash": self.digest("public-output"),
                "reference_quant_logits_hash": self.digest("public-output"),
                "proof_public_output_hash": self.digest("public-output"),
                "public_input_digest": self.digest("public-input"),
                "topology_digest": self.digest("topology"),
                "source_tree_hash": self.digest("source-tree"),
                "source_manifest_hash": self.digest("source-manifest"),
                "binary_hash": self.digest("binary"),
                "build_identity": self.digest("build"),
                "build_receipt_hash": self.digest("build-receipt"),
                "verifier_report_hash": self.digest("verifier-report"),
                "trusted_registry_digest": self.digest("registry"),
                "verifier_registry_digest": self.digest("registry"),
                "security_budget_ledger_digest": self.digest("budget-ledger"),
                "quant_config_hash": self.digest("quant-config"),
                "srs_manifest_hash": self.digest("srs-manifest"),
                "public_metadata_hash": self.digest("proof-specific-metadata"),
                "static_table_hash": self.digest("static-tables"),
            }
            security = AGGREGATOR.check_security(
                manifest,
                proof_path,
                self.report(),
                self.registration(),
                "synthetic/cpu",
            )
            self.assertTrue(security["reference_consistency_passed"])
            self.assertNotIn(
                "cpu_gpu_reference_consistency_passed",
                security,
            )

    def test_manifest_security_mutation_cannot_bypass_independent_report(self):
        proof_bytes = b"synthetic proof"
        with tempfile.TemporaryDirectory() as temporary:
            proof_path = pathlib.Path(temporary) / "proof.bin"
            proof_path.write_bytes(proof_bytes)
            manifest = {
                "verifier_reads_witness": "false",
                "private_plain_value_count": "0",
                "proof_payload_audit_status": "passed",
                "reference_consistency_passed": "true",
                "tamper_tests_passed": "true",
                "tamper_tests_ms": "1.0",
                "tamper_test_case_map": self.tamper_map(),
                "security_budget_passed": "false",
                "security_budget_exact": "1/2",
                "proof_hash": hashlib.sha256(proof_bytes).hexdigest(),
                "secure_trace_logits_hash": self.digest("public-output"),
                "reference_quant_logits_hash": self.digest("public-output"),
                "proof_public_output_hash": self.digest("public-output"),
                "public_input_digest": self.digest("public-input"),
                "topology_digest": self.digest("topology"),
                "source_tree_hash": self.digest("source-tree"),
                "source_manifest_hash": self.digest("source-manifest"),
                "binary_hash": self.digest("binary"),
                "build_identity": self.digest("build"),
                "build_receipt_hash": self.digest("build-receipt"),
                "verifier_report_hash": self.digest("verifier-report"),
                "trusted_registry_digest": self.digest("registry"),
                "verifier_registry_digest": self.digest("registry"),
                "security_budget_ledger_digest": self.digest("budget-ledger"),
                "quant_config_hash": self.digest("quant-config"),
                "srs_manifest_hash": self.digest("srs-manifest"),
                "public_metadata_hash": self.digest("proof-specific-metadata"),
                "static_table_hash": self.digest("static-tables"),
            }
            with self.assertRaises(ValueError):
                AGGREGATOR.check_security(
                    manifest,
                    proof_path,
                    self.report(),
                    self.registration(),
                    "synthetic/cpu",
                )

    def test_wrong_external_registry_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            artifact = root / "registry.bin"
            artifact.write_bytes(b"synthetic registration")
            with self.assertRaises(ValueError):
                AGGREGATOR.inspect_registration_artifact(
                    root,
                    "cora",
                    "registry.bin",
                    self.digest("wrong-registry"),
                    "missing-inspector",
                    "synthetic/cpu",
                )

    def test_security_ledger_mismatch_is_rejected(self):
        proof_bytes = b"synthetic proof"
        with tempfile.TemporaryDirectory() as temporary:
            proof_path = pathlib.Path(temporary) / "proof.bin"
            proof_path.write_bytes(proof_bytes)
            manifest = {
                "verifier_reads_witness": "false",
                "private_plain_value_count": "0",
                "proof_payload_audit_status": "passed",
                "reference_consistency_passed": "true",
                "tamper_tests_passed": "true",
                "tamper_tests_ms": "1.0",
                "tamper_test_case_map": self.tamper_map(),
                "security_budget_passed": "true",
                "security_budget_exact": "1/2",
                "proof_hash": hashlib.sha256(proof_bytes).hexdigest(),
                "secure_trace_logits_hash": self.digest("public-output"),
                "reference_quant_logits_hash": self.digest("public-output"),
                "proof_public_output_hash": self.digest("public-output"),
                "public_input_digest": self.digest("public-input"),
                "topology_digest": self.digest("topology"),
                "source_tree_hash": self.digest("source-tree"),
                "source_manifest_hash": self.digest("source-manifest"),
                "binary_hash": self.digest("binary"),
                "build_identity": self.digest("build"),
                "build_receipt_hash": self.digest("build-receipt"),
                "verifier_report_hash": self.digest("verifier-report"),
                "trusted_registry_digest": self.digest("registry"),
                "verifier_registry_digest": self.digest("registry"),
                "security_budget_ledger_digest": self.digest("budget-ledger"),
                "quant_config_hash": self.digest("quant-config"),
                "srs_manifest_hash": self.digest("srs-manifest"),
                "public_metadata_hash": self.digest("proof-specific-metadata"),
                "static_table_hash": self.digest("static-tables"),
            }
            report = self.report()
            report["security_budget_ledger_digest"] = self.digest("different-ledger")
            with self.assertRaises(ValueError):
                AGGREGATOR.check_security(
                    manifest,
                    proof_path,
                    report,
                    self.registration(),
                    "synthetic/cpu",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
