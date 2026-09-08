import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
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
            "protocol_schema_id": self.digest("protocol-schema"),
            "instance_schema_hash": self.digest("instance-schema"),
            "public_input_digest": self.digest("public-input"),
            "topology_digest": self.digest("topology"),
        }

    def report(self):
        return {
            "reference_consistency_passed": True,
            "tamper_tests_passed": True,
            "tamper_test_case_map": self.tamper_map(),
            "tamper_tests_ms": 1.0,
            "security_budget_passed": True,
            "security_budget_ledger_digest": self.digest("budget-ledger"),
            "security_budget_exact": "1/2",
        }

    def exact_attestation(self, manifest=None, **overrides):
        """Build the producer's strict 21-field attestation object."""
        manifest = manifest or self.export_acceptance_manifest()
        value = {
            "binary_sha256": manifest["binary_hash"],
            "build_identity": manifest["build_identity"],
            "build_receipt_sha256": manifest["build_receipt_hash"],
            "config_sha256": manifest["config_sha256"],
            "format": AGGREGATOR.TAMPER_ATTESTATION_FORMAT,
            "instance_schema_hash": manifest["instance_schema_hash"],
            "proof_sha256": manifest["proof_hash"],
            "protocol_schema_id": manifest["protocol_schema_id"],
            "public_input_digest": manifest["public_input_digest"],
            "public_output_digest": manifest["proof_public_output_hash"],
            "source_manifest_sha256": manifest["source_manifest_hash"],
            "source_tree_hash": manifest["source_tree_hash"],
            "srs_manifest_sha256": manifest["srs_manifest_hash"],
            "suite_id": AGGREGATOR.TAMPER_ATTESTATION_SUITE_ID,
            "tamper_test_case_map": self.tamper_map(),
            "tamper_tests_ms": "1.000",
            "tamper_tests_passed": "true",
            "topology_digest": manifest["topology_digest"],
            "trusted_registry_digest": manifest["trusted_registry_digest"],
            "verifier_registry_digest": manifest["verifier_registry_digest"],
        }
        value.update(overrides)
        return value

    def loaded_export_manifest(self, manifest=None, **attestation_overrides):
        manifest = manifest or self.export_acceptance_manifest()
        run_dir = pathlib.Path(
            tempfile.mkdtemp(prefix=".synthetic-export-", dir=ROOT / "tests")
        )
        self.addCleanup(shutil.rmtree, run_dir, ignore_errors=True)
        proof_path = run_dir / "proof.bin"
        proof_path.write_bytes(b"synthetic proof")
        manifest["proof_hash"] = AGGREGATOR.sha256_file(proof_path)
        manifest["proof_size_bytes"] = str(proof_path.stat().st_size)
        attestation = self.exact_attestation(manifest, **attestation_overrides)
        attestation_path, attestation_digest = self.write_attestation(
            run_dir, attestation, "attestation.json"
        )
        manifest_path = run_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return EXPORTER.LoadedManifest(
            manifest,
            manifest_path,
            str(attestation_path.relative_to(ROOT)),
            attestation_digest,
        )

    def write_attestation(self, directory, value=None, filename="attestation.json"):
        path = pathlib.Path(directory) / filename
        path.relative_to(ROOT / "tests")
        path.write_text(
            json.dumps(value or self.exact_attestation(), sort_keys=True),
            encoding="utf-8",
        )
        return path, AGGREGATOR.sha256_file(path)

    def complete_security_manifest(self, manifest, proof_path):
        """Fill the direct security gate's complete binding surface."""
        defaults = self.export_acceptance_manifest()
        for key, value in defaults.items():
            manifest.setdefault(key, value)
        manifest["proof_hash"] = AGGREGATOR.sha256_file(proof_path)
        return manifest

    def security_attestation(self, manifest, proof_path, **overrides):
        manifest = self.complete_security_manifest(manifest, proof_path)
        directory = pathlib.Path(
            tempfile.mkdtemp(prefix=".synthetic-security-", dir=ROOT / "tests")
        )
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return self.write_attestation(
            directory,
            self.exact_attestation(manifest, **overrides),
        )

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
                "protocol_schema_id": self.digest("protocol-schema"),
                "instance_schema_hash": self.digest("instance-schema"),
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
        row = EXPORTER.make_success_row("cora", self.loaded_export_manifest(manifest))
        self.assertEqual(row["timing_schema"], EXPORTER.CANONICAL_TIMING_SCHEMA)
        manifest = self.loaded_export_manifest()
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
                EXPORTER.make_success_row("cora", self.loaded_export_manifest(manifest))

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
                EXPORTER.make_success_row("cora", self.loaded_export_manifest(manifest))

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
                EXPORTER.make_success_row("cora", self.loaded_export_manifest(manifest))

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

    def test_external_tamper_attestation_validates_exact_producer_schema(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            path, digest = self.write_attestation(temporary)
            value = AGGREGATOR.load_tamper_attestation(ROOT, path, digest, "synthetic")
            self.assertTrue(value["tamper_tests_passed"])
            self.assertEqual(value["tamper_test_case_map"], self.tamper_map())
            self.assertEqual(value["tamper_attestation_sha256"], digest)

    def test_external_tamper_attestation_rejects_unknown_missing_and_duplicate_fields(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            value = self.exact_attestation()
            unknown = dict(value)
            unknown["unexpected"] = "test-only"
            path, digest = self.write_attestation(temporary, unknown, "unknown.json")
            with self.assertRaises(ValueError):
                AGGREGATOR.load_tamper_attestation(ROOT, path, digest, "unknown")

            missing = dict(value)
            del missing["proof_sha256"]
            path, digest = self.write_attestation(temporary, missing, "missing.json")
            with self.assertRaises(ValueError):
                AGGREGATOR.load_tamper_attestation(ROOT, path, digest, "missing")

            encoded = json.dumps(value, sort_keys=True)[:-1]
            encoded += ',"format":"zkgat_protocol_tamper_attestation"}'
            path = pathlib.Path(temporary) / "duplicate.json"
            path.write_text(encoded, encoding="utf-8")
            with self.assertRaises(ValueError):
                AGGREGATOR.load_tamper_attestation(
                    ROOT, path, AGGREGATOR.sha256_file(path), "duplicate"
                )

    def test_external_tamper_attestation_rejects_identity_and_state_failures(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            for field in AGGREGATOR.TAMPER_ATTESTATION_HASH_FIELDS:
                value = self.exact_attestation()
                value[field] = "not-a-sha256"
                path, digest = self.write_attestation(temporary, value, f"bad_{field}.json")
                with self.assertRaises(ValueError, msg=field):
                    AGGREGATOR.load_tamper_attestation(ROOT, path, digest, field)

            malformed_values = (
                ("format", "wrong-format"),
                ("suite_id", "wrong-suite"),
                ("tamper_tests_passed", "false"),
                ("tamper_tests_ms", "0"),
                ("tamper_tests_ms", "-1"),
                ("tamper_tests_ms", "nan"),
                ("tamper_test_case_map", ""),
            )
            for field, replacement in malformed_values:
                value = self.exact_attestation()
                value[field] = replacement
                path, digest = self.write_attestation(
                    temporary, value, f"malformed_{field}_{replacement}.json"
                )
                with self.assertRaises(ValueError, msg=f"{field}={replacement}"):
                    AGGREGATOR.load_tamper_attestation(ROOT, path, digest, "malformed")

            value = self.exact_attestation()
            path, _ = self.write_attestation(temporary, value, "wrong_digest.json")
            with self.assertRaises(ValueError):
                AGGREGATOR.load_tamper_attestation(
                    ROOT, path, self.digest("wrong"), "wrong-digest"
                )

    def test_external_tamper_attestation_rejects_unknown_duplicate_missing_and_bad_case_order(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            cases = list(AGGREGATOR.REQUIRED_TAMPER_CASE_ORDER)
            variants = {
                "unknown": "unknown_case=reject," + ",".join(
                    f"{name}=reject" for name in cases[:-1]
                ),
                "duplicate": ",".join(
                    [f"{name}=reject" for name in cases] + ["codec_tag=reject"]
                ),
                "missing": ",".join(f"{name}=reject" for name in cases[:-1]),
                "wrong_order": ",".join(
                    [f"{cases[1]}=reject", f"{cases[0]}=reject"]
                    + [f"{name}=reject" for name in cases[2:]]
                ),
                "failed": ",".join(
                    f"{name}={'fail' if index == 0 else 'reject'}"
                    for index, name in enumerate(cases)
                ),
            }
            for name, case_map in variants.items():
                value = self.exact_attestation(tamper_test_case_map=case_map)
                path, digest = self.write_attestation(temporary, value, f"case_{name}.json")
                with self.assertRaises(ValueError, msg=name):
                    AGGREGATOR.load_tamper_attestation(ROOT, path, digest, name)

    def test_external_tamper_attestation_path_is_tests_canonical_and_symlink_free(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            directory = pathlib.Path(temporary)
            value = self.exact_attestation()
            path, digest = self.write_attestation(directory, value, "regular.json")
            self.assertTrue(
                AGGREGATOR.load_tamper_attestation(ROOT, path, digest, "regular")
            )

            outside = pathlib.Path(tempfile.mkdtemp(prefix=".synthetic-outside-"))
            self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
            outside_path = outside / "outside.json"
            outside_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            outside_digest = AGGREGATOR.sha256_file(outside_path)
            for candidate in (outside_path, ROOT / "results" / "attestation.json"):
                with self.assertRaises(ValueError):
                    AGGREGATOR.load_tamper_attestation(
                        ROOT, candidate, outside_digest, "non-tests"
                    )

            symlink = directory / "symlink.json"
            os.symlink(path, symlink)
            with self.assertRaises(ValueError):
                AGGREGATOR.load_tamper_attestation(
                    ROOT, symlink, digest, "symlink"
                )

    def test_verified_attestation_and_exporter_reject_direct_api_bypass(self):
        value = self.exact_attestation()
        with self.assertRaises(ValueError):
            AGGREGATOR.check_verified_tamper_attestation(
                ROOT, value, self.digest("forged"), "forged-mapping"
            )
        with self.assertRaises(ValueError):
            EXPORTER.make_success_row("cora", self.export_acceptance_manifest())
        with self.assertRaises((TypeError, ValueError)):
            EXPORTER.LoadedManifest(self.export_acceptance_manifest(), value)

    def test_external_tamper_attestation_manifest_binding_covers_all_identity_fields(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            run_dir = pathlib.Path(temporary)
            proof_path = run_dir / "proof.bin"
            proof_path.write_bytes(b"synthetic proof")
            manifest = self.export_acceptance_manifest()
            manifest["proof_hash"] = AGGREGATOR.sha256_file(proof_path)
            value = self.exact_attestation(manifest)
            path, digest = self.write_attestation(run_dir, value)
            result = AGGREGATOR.check_tamper_attestation_manifest_binding(
                ROOT,
                run_dir,
                proof_path,
                manifest,
                str(path.relative_to(ROOT)),
                digest,
                "synthetic/binding",
            )
            self.assertTrue(result["tamper_attestation_verified"])

            for attestation_field in (
                "source_tree_hash",
                "source_manifest_sha256",
                "build_receipt_sha256",
                "binary_sha256",
                "build_identity",
                "srs_manifest_sha256",
                "protocol_schema_id",
                "instance_schema_hash",
                "public_input_digest",
                "public_output_digest",
                "topology_digest",
                "trusted_registry_digest",
                "verifier_registry_digest",
                "config_sha256",
                "proof_sha256",
            ):
                mutated = dict(value)
                mutated[attestation_field] = self.digest("binding-mismatch")
                bad_path, bad_digest = self.write_attestation(
                    run_dir, mutated, f"bad_{attestation_field}.json"
                )
                with self.assertRaises(ValueError, msg=attestation_field):
                    AGGREGATOR.check_tamper_attestation_manifest_binding(
                        ROOT,
                        run_dir,
                        proof_path,
                        manifest,
                        str(bad_path.relative_to(ROOT)),
                        bad_digest,
                        f"synthetic/{attestation_field}",
                    )

    def test_security_gate_requires_external_attestation_and_uses_its_map(self):
        proof_bytes = b"synthetic proof"
        with tempfile.TemporaryDirectory() as temporary:
            proof_path = pathlib.Path(temporary) / "proof.bin"
            proof_path.write_bytes(proof_bytes)
            manifest = {
                "verifier_reads_witness": "false",
                "private_plain_value_count": "0",
                "proof_payload_audit_status": "passed",
                "reference_consistency_passed": "true",
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
            self.complete_security_manifest(manifest, proof_path)
            with self.assertRaises(ValueError):
                AGGREGATOR.check_security(
                    manifest,
                    proof_path,
                    self.report(),
                    self.registration(),
                    "synthetic/missing-attestation",
                )
            attestation_path, attestation_digest = self.security_attestation(
                manifest, proof_path
            )
            security = AGGREGATOR.check_security(
                manifest,
                proof_path,
                self.report(),
                self.registration(),
                "synthetic/external-attestation",
                str(attestation_path.relative_to(ROOT)),
                attestation_digest,
                ROOT,
            )
            self.assertEqual(security["tamper_test_case_map"], self.tamper_map())

    def test_pending_local_tamper_state_does_not_promote_to_export_pass(self):
        manifest = self.export_acceptance_manifest()
        manifest["tamper_tests_passed"] = False
        manifest["tamper_test_case_map"] = ""
        manifest["tamper_tests_ms"] = 0.0
        row = EXPORTER.make_success_row("cora", self.loaded_export_manifest(manifest))
        self.assertTrue(row["tamper_tests_passed"])
        self.assertEqual(row["tamper_test_case_map"], self.tamper_map())

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
                    attestation_path, attestation_digest = self.security_attestation(
                        mutated, proof_path
                    )
                    with self.assertRaises(ValueError):
                        AGGREGATOR.check_security(
                            mutated,
                            proof_path,
                            self.report(),
                            self.registration(),
                            f"synthetic/cpu/{field}",
                            str(attestation_path.relative_to(ROOT)),
                            attestation_digest,
                            ROOT,
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
            self.complete_security_manifest(manifest, proof_path)
            attestation_path, attestation_digest = self.security_attestation(
                manifest, proof_path
            )
            security = AGGREGATOR.check_security(
                manifest,
                proof_path,
                self.report(),
                self.registration(),
                "synthetic/cpu",
                str(attestation_path.relative_to(ROOT)),
                attestation_digest,
                ROOT,
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
            self.complete_security_manifest(manifest, proof_path)
            attestation_path, attestation_digest = self.security_attestation(
                manifest, proof_path
            )
            with self.assertRaises(ValueError):
                AGGREGATOR.check_security(
                    manifest,
                    proof_path,
                    self.report(),
                    self.registration(),
                    "synthetic/cpu",
                    str(attestation_path.relative_to(ROOT)),
                    attestation_digest,
                    ROOT,
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
            self.complete_security_manifest(manifest, proof_path)
            attestation_path, attestation_digest = self.security_attestation(
                manifest, proof_path
            )
            report = self.report()
            report["security_budget_ledger_digest"] = self.digest("different-ledger")
            with self.assertRaises(ValueError):
                AGGREGATOR.check_security(
                    manifest,
                    proof_path,
                    report,
                    self.registration(),
                    "synthetic/cpu",
                    str(attestation_path.relative_to(ROOT)),
                    attestation_digest,
                    ROOT,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
