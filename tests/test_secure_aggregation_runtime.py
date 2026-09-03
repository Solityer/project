import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_secure_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_secure_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AGGREGATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATOR)
EXPORT_SCRIPT = ROOT / "scripts" / "export_benchmark_table.py"
EXPORT_SPEC = importlib.util.spec_from_file_location("export_benchmark_table", EXPORT_SCRIPT)
assert EXPORT_SPEC is not None and EXPORT_SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(EXPORTER)


class SecureAggregationRuntimeTests(unittest.TestCase):
    def digest(self, label):
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    def test_ppi_srs_domains_match_canonical_planner(self):
        self.assertEqual(
            AGGREGATOR.EXPECTED_SRS_DOMAINS["ppi"],
            {
                "C": 1 << 7,
                "FH": 1 << 22,
                "N": 1 << 16,
                "cat": 1 << 7,
                "d": 1 << 4,
                "edge": 1 << 23,
                "in": 1 << 7,
            },
        )

        envelope = {}
        for line in (ROOT / "configs" / "srs_ppi_max_envelope.cfg").read_text(
            encoding="utf-8"
        ).splitlines():
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            envelope[key] = value
        self.assertEqual(envelope["max_input_dim"], "64")
        self.assertEqual(envelope["max_concat_dim"], "64")

    def test_ogbn_srs_domains_match_canonical_planner(self):
        self.assertEqual(
            AGGREGATOR.EXPECTED_SRS_DOMAINS["ogbn_arxiv"],
            {
                "C": 1 << 6,
                "FH": 1 << 25,
                "N": 1 << 18,
                "cat": 1 << 7,
                "d": 1 << 4,
                "edge": 1 << 24,
                "in": 1 << 8,
            },
        )

        envelope = {}
        for line in (ROOT / "configs" / "srs_ogbn_arxiv_max_envelope.cfg").read_text(
            encoding="utf-8"
        ).splitlines():
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            envelope[key] = value
        self.assertEqual(envelope["max_input_dim"], "128")
        self.assertEqual(envelope["max_concat_dim"], "64")

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

    def export_acceptance_manifest(self):
        manifest = self.acceptance_manifest()
        manifest.update(
            {
                "dataset": "cora",
                "config": "configs/cora_secure.cfg",
                "config_sha256": AGGREGATOR.sha256_file(ROOT / "configs" / "cora_secure.cfg"),
                "input_preprocess_id": "row_normalize",
                "benchmark_mode": "warm",
                "full_dataset": True,
                "is_full_dataset": True,
                "stale_result": False,
                "reference_consistency_passed": True,
                "tamper_tests_passed": True,
                "security_budget_passed": True,
                "verifier_reads_witness": False,
                "private_plain_value_count": 0,
                "proof_payload_audit_status": "passed",
                "tamper_tests_ms": "1.0",
                "tamper_test_case_map": self.tamper_map(),
                "security_budget_ledger_digest": self.digest("budget-ledger"),
                "security_budget_exact": "1/2",
                "timing_schema": EXPORTER.CANONICAL_TIMING_SCHEMA,
                "commitment_time_ms": "1.0",
                "prove_time_ms": "2.0",
                "verify_time_ms": "3.0",
                "proof_size_bytes": "4",
            }
        )
        for field in AGGREGATOR.REQUIRED_HASHES:
            manifest[field] = self.digest(field)
        output_digest = self.digest("public-output")
        manifest.update(
            {
                "secure_trace_logits_hash": output_digest,
                "reference_quant_logits_hash": output_digest,
                "proof_public_output_hash": output_digest,
            }
        )
        return manifest

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

    def test_timing_schema_rejects_missing_or_malformed_value(self):
        manifest = {"timing_schema": AGGREGATOR.CANONICAL_TIMING_SCHEMA}
        self.assertEqual(
            AGGREGATOR.require_canonical_timing_schema(manifest, "synthetic"),
            AGGREGATOR.CANONICAL_TIMING_SCHEMA,
        )
        for malformed_or_missing in ("malformed_timing_schema", "", None):
            mutated = {"timing_schema": malformed_or_missing}
            with self.assertRaises(ValueError):
                AGGREGATOR.require_canonical_timing_schema(mutated, "synthetic")

    def test_benchmark_export_requires_canonical_timing_schema(self):
        manifest = self.export_acceptance_manifest()
        row = EXPORTER.make_success_row("cora", manifest)
        self.assertEqual(row["timing_schema"], EXPORTER.CANONICAL_TIMING_SCHEMA)
        manifest["timing_schema"] = "malformed_timing_schema"
        with self.assertRaises(ValueError):
            EXPORTER.make_success_row("cora", manifest)

    def test_benchmark_export_rejects_missing_or_tampered_input_preprocess_id(self):
        for value in (None, "ogbn_arxiv_raw_float32"):
            manifest = self.export_acceptance_manifest()
            if value is None:
                del manifest["input_preprocess_id"]
            else:
                manifest["input_preprocess_id"] = value
            with self.assertRaises(ValueError):
                EXPORTER.make_success_row("cora", manifest)

    def test_benchmark_export_requires_complete_acceptance_state(self):
        for field in (
            "verifier_accepted",
            "reference_consistency_passed",
            "final_formal_run_accepted",
            "tamper_tests_passed",
            "security_budget_passed",
        ):
            manifest = self.export_acceptance_manifest()
            manifest[field] = "false"
            with self.assertRaises(ValueError, msg=field):
                EXPORTER.make_success_row("cora", manifest)

    def test_benchmark_export_rejects_incomplete_security_audit(self):
        for field in (
            "tamper_test_case_map",
            "proof_payload_audit_status",
            "security_budget_exact",
            "secure_trace_logits_hash",
        ):
            manifest = self.export_acceptance_manifest()
            if field == "proof_payload_audit_status":
                manifest[field] = "failed"
            elif field == "secure_trace_logits_hash":
                manifest[field] = self.digest("different-output")
            else:
                manifest.pop(field)
            with self.assertRaises(ValueError, msg=field):
                EXPORTER.make_success_row("cora", manifest)

    def test_benchmark_export_load_binds_manifest_to_proof_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = pathlib.Path(temporary)
            proof = b"synthetic proof"
            (run_dir / "proof.bin").write_bytes(proof)
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"proof_hash": hashlib.sha256(b"different proof").hexdigest()}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                EXPORTER.load_manifest(run_dir)

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

    def test_cpu_cuda_pair_allows_distinct_backend_provenance(self):
        cpu = self.pair_row()
        cuda = dict(cpu)
        for field in (
            "binary_hash",
            "build_identity",
            "build_receipt_hash",
            "verifier_report_hash",
        ):
            cpu[field] = self.digest(f"cpu:{field}")
            cuda[field] = self.digest(f"cuda:{field}")
            self.assertIn(field, AGGREGATOR.REQUIRED_HASHES)
            self.assertNotIn(field, AGGREGATOR.PAIR_CONSISTENCY_FIELDS)
        AGGREGATOR.compare_cpu_cuda_pair("synthetic", cpu, cuda)

    def test_cpu_cuda_pair_source_tree_mismatch_fails_closed(self):
        cpu = self.pair_row()
        cuda = dict(cpu)
        cuda["source_tree_hash"] = "different-source-tree"
        with self.assertRaises(ValueError):
            AGGREGATOR.compare_cpu_cuda_pair("synthetic", cpu, cuda)

    def test_backend_provenance_hashes_remain_required_and_validated(self):
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
            for field in (
                "binary_hash",
                "build_identity",
                "build_receipt_hash",
                "verifier_report_hash",
            ):
                for invalid in ("", "not-a-sha256"):
                    mutated = dict(manifest)
                    mutated[field] = invalid
                    with self.assertRaises(ValueError):
                        AGGREGATOR.check_security(
                            mutated,
                            proof_path,
                            self.report(),
                            self.registration(),
                            f"synthetic/cpu/{field}",
                        )

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

    def test_cpu_cuda_pair_input_preprocess_mismatch_fails_closed(self):
        cpu = self.pair_row()
        cuda = dict(cpu)
        cuda["input_preprocess_id"] = "tampered_preprocess"
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
