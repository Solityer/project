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
                "proof_hash": hashlib.sha256(proof_bytes).hexdigest(),
                "checkpoint_model_hash": "checkpoint-model",
                "quant_config_hash": "quant-config",
                "srs_manifest_hash": "srs-manifest",
                "public_metadata_hash": "proof-specific-metadata",
                "static_table_hash": "static-tables",
            }
            security = AGGREGATOR.check_security(
                manifest,
                proof_path,
                "synthetic/cpu",
            )
            self.assertTrue(security["reference_consistency_passed"])
            self.assertNotIn(
                "cpu_gpu_reference_consistency_passed",
                security,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
