"""Synthetic zkGPT proof-chain tests.

The default test suite only checks adapter/schema boundaries.  The positive
proof and fresh-process tamper tests are intentionally opt-in because they
execute the native prover once, even though the fixture is tiny:

    ZKGPT_RUN_SYNTHETIC_E2E=1 python -m unittest tests.test_synthetic_e2e -v

No test in this module writes to the production ``results/`` directory or
loads a formal dataset.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from verify_synthetic_proof import (  # type: ignore
    FORBIDDEN_RAW_FIELDS,
    PUBLIC_STATEMENT_SCHEMA,
    PublicInputAccessAudit,
    SyntheticVerificationError,
    _expected_layout,
    _ledger,
    _registry_metadata_seed,
    _public_statement,
    _validate_public_statement,
    verify_serialized_bundle,
)
from synthetic_e2e_support import create_synthetic_bundle, synthetic_graph, synthetic_model, temporary_synthetic_root  # type: ignore
from zkgpt_gat.util import digest_json  # type: ignore
from zkgpt_gat.verifier_context import ContextError  # type: ignore


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADAPTER = pathlib.Path(__file__).with_name("verify_synthetic_proof.py")
RUN_E2E = os.environ.get("ZKGPT_RUN_SYNTHETIC_E2E") == "1"


def _statement_test_metadata() -> dict[str, object]:
    """Return the complete v2 external-statement metadata fixture.

    The statement builder intentionally requires every authenticated public
    binding field.  Keep this schema fixture explicit so its canonical-digest
    tests exercise the same v2 shape as the fresh-process verifier instead of
    relying on production-only metadata defaults.
    """
    return {
        "protocol_id": "zkGPT-gat-v1",
        "dataset": "synthetic-e2e",
        "dataset_display_name": "Synthetic E2E",
        "task_type": "transductive_node_classification",
        "rounding_mode": "nearest_ties_away",
        "integer_bound_ledger_digest": "0" * 64,
        "input_preprocess_id": "feature_int64_identity_v1",
        "public_y_schema": "zkGPT_test_public_output_v1",
        "public_y_path": "public_y.json",
        "public_y_digest": "1" * 64,
        "public_y_descriptor_digest": "2" * 64,
        "X_pub_digest": "3" * 64,
        "full_dataset": True,
        "graph_identity": {"graph_digest": "4" * 64},
        "com_W_identity": {"commitment_digest": "5" * 64},
        "reference_forward_profile_id": "synthetic-test-v1",
        "reference_forward_engine": "test-forward",
        "reference_forward_source_sha256": "6" * 64,
        "public_output_binding_digest": "7" * 64,
        "registration_binding_digest": "8" * 64,
    }


class SyntheticAdapterSchemaTests(unittest.TestCase):
    """Boundary tests which never invoke ``build_proof``."""

    def test_forbidden_private_fields_are_explicitly_rejected(self) -> None:
        self.assertIn("checkpoint_tensors", FORBIDDEN_RAW_FIELDS)
        self.assertIn("raw_witness", FORBIDDEN_RAW_FIELDS)
        self.assertIn("private_trace", FORBIDDEN_RAW_FIELDS)
        with tempfile.TemporaryDirectory(prefix="zkgpt-adapter-schema-") as name:
            root = pathlib.Path(name)
            (root / "proof").mkdir()
            (root / "public_config.json").write_text(
                json.dumps({"checkpoint_tensors": "../checkpoint/tensors.txt"}) + "\n",
                encoding="utf-8",
            )
            result = verify_serialized_bundle(root)
            self.assertEqual(result["status"], "VERIFY_FAILED")
            audit = result["public_input_access_audit"]
            self.assertTrue(audit["forbidden_fields_seen"])
            self.assertFalse(audit["verifier_reads_witness"])

    def test_missing_proof_bundle_fails_closed_without_dataset_access(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-adapter-schema-") as name:
            root = pathlib.Path(name)
            for filename, payload in (
                ("public_config.json", {"schema": "fixture"}),
                ("public_graph.json", {}),
                ("public_model.json", {}),
                ("public_y.json", {}),
            ):
                (root / filename).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            result = verify_serialized_bundle(root)
            self.assertEqual(result["status"], "VERIFY_FAILED")
            self.assertEqual(result["public_input_access_audit"]["witness_paths_read"], [])
            self.assertEqual(result["public_input_access_audit"]["verifier_reads_witness"], False)

    def test_access_audit_rejects_checkpoint_or_private_path_before_open(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-adapter-schema-") as name:
            root = pathlib.Path(name)
            audit = PublicInputAccessAudit(root)
            with self.assertRaises(SyntheticVerificationError):
                audit.record_path(root / "checkpoint" / "tensors.txt", allowed=False)
            self.assertEqual(audit.forbidden_paths_read, ["checkpoint/tensors.txt"])
            self.assertEqual(audit.finish(status="VERIFY_FAILED")["verifier_reads_witness"], False)

    def test_public_statement_digest_is_canonical(self) -> None:
        _dataset, graph = synthetic_graph()
        statement = _public_statement(
            graph,
            np.asarray([[0]], dtype=object),
            _statement_test_metadata(),
        )
        validated = _validate_public_statement(statement)
        self.assertEqual(validated["schema"], PUBLIC_STATEMENT_SCHEMA)
        digest = statement["statement_digest"]
        self.assertEqual(digest, digest_json(validated))

    def test_public_statement_schema_tamper_is_rejected(self) -> None:
        _dataset, graph = synthetic_graph()
        statement = _public_statement(
            graph,
            np.asarray([[0]], dtype=object),
            _statement_test_metadata(),
        )
        statement["schema"] = "zkGPT_attacker_statement_v1"
        with self.assertRaises(SyntheticVerificationError):
            _validate_public_statement(statement)


class SyntheticAdapterRoundingModeTests(unittest.TestCase):
    """The table-registry seed must bind a supported, cross-checked mode."""

    @staticmethod
    def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
        config = {
            "quant_cfg_id": "exact_integer_gat",
            "static_table_id": "tables:lrelu+elu+exp+range",
            "rounding_mode": "nearest_ties_away",
        }
        metadata = {"rounding_mode": "nearest_ties_away"}
        descriptor = {
            "schema": "zkGPT_static_tables_descriptor_v1",
            "content_digest": "content-digest",
            "descriptor_digest": "descriptor-digest",
        }
        ledger = {"rounding_mode": "nearest_ties_away", "records": []}
        return config, metadata, descriptor, ledger

    def test_registry_seed_binds_validated_rounding_mode(self) -> None:
        config, metadata, descriptor, ledger = self._inputs()
        seed = _registry_metadata_seed(config, metadata, descriptor, ledger)
        self.assertEqual(seed["rounding_mode"], "nearest_ties_away")

    def test_registry_seed_rejects_missing_rounding_mode(self) -> None:
        config, metadata, descriptor, ledger = self._inputs()
        metadata.pop("rounding_mode")
        with self.assertRaises(SyntheticVerificationError):
            _registry_metadata_seed(config, metadata, descriptor, ledger)

    def test_registry_seed_rejects_inconsistent_rounding_mode(self) -> None:
        config, metadata, descriptor, ledger = self._inputs()
        metadata["rounding_mode"] = "toward_zero"
        with self.assertRaisesRegex(SyntheticVerificationError, "config/metadata rounding_mode mismatch"):
            _registry_metadata_seed(config, metadata, descriptor, ledger)

    def test_registry_seed_rejects_unknown_rounding_mode(self) -> None:
        config, metadata, descriptor, ledger = self._inputs()
        config["rounding_mode"] = "unknown"
        with self.assertRaises(SyntheticVerificationError):
            _registry_metadata_seed(config, metadata, descriptor, ledger)


class SyntheticAdapterRangeBitsTests(unittest.TestCase):
    """The adapter must bind every range-family descriptor to public config."""

    @staticmethod
    def _inputs() -> tuple[object, dict[str, object], dict[str, object]]:
        _dataset, graph = synthetic_graph()
        model = synthetic_model()
        bounds = {
            "hidden.l1.h0": {
                "B_W": 1,
                "B_a_src": 1,
                "B_a_dst": 1,
                "B_b": 0,
            },
            "output.h0": {
                "B_W": 1,
                "B_a_src": 1,
                "B_a_dst": 1,
                "B_b": 1,
            },
        }
        return graph, model.manifest, bounds

    def test_nondefault_range_bits_are_carried_into_the_rebuilt_layout(self) -> None:
        graph, manifest, bounds = self._inputs()
        ledger = _ledger(
            graph,
            manifest,
            bounds,
            "exact_integer_gat",
            "nearest_ties_away",
            range_bits=4,
        )
        self.assertEqual(ledger["range_bits"], 4)
        layout, lasso, _matrix, _model_arrays = _expected_layout(
            graph, manifest, 4, ledger
        )
        self.assertGreater(layout["N_in"], 0)
        self.assertTrue(lasso)
        self.assertTrue(all(int(item.get("range_bits", 4)) == 4 for item in lasso if item["kind"] == "range"))

    def test_omitted_range_bits_defaults_to_twelve_and_fails_closed_for_four_bit_layout(self) -> None:
        graph, manifest, bounds = self._inputs()
        default_ledger = _ledger(
            graph,
            manifest,
            bounds,
            "exact_integer_gat",
            "nearest_ties_away",
        )
        self.assertEqual(default_ledger["range_bits"], 12)
        with self.assertRaisesRegex(ContextError, "range-family bits differ"):
            _expected_layout(graph, manifest, 4, default_ledger)

    def test_layout_range_bits_mismatch_is_rejected(self) -> None:
        graph, manifest, bounds = self._inputs()
        ledger = _ledger(
            graph,
            manifest,
            bounds,
            "exact_integer_gat",
            "nearest_ties_away",
            range_bits=4,
        )
        with self.assertRaisesRegex(ContextError, "range-family bits differ"):
            _expected_layout(graph, manifest, 12, ledger)


@unittest.skipUnless(RUN_E2E, "set ZKGPT_RUN_SYNTHETIC_E2E=1 to run the native synthetic proof")
class SyntheticFreshProcessE2ETests(unittest.TestCase):
    """One native proof, one fresh positive replay, and four tamper negatives."""

    temp_root: tempfile.TemporaryDirectory[str]
    bundle: dict[str, object]
    bundle_root: pathlib.Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_root = temporary_synthetic_root()
        cls.bundle_root = pathlib.Path(cls.temp_root.name)
        # This is the sole build_proof invocation in the opt-in test class.
        cls.bundle = create_synthetic_bundle(cls.bundle_root)
        internal = cls.bundle.get("internal_verify")
        if internal is None:
            internal = verify_serialized_bundle(cls.bundle_root)
            cls.bundle["internal_verify"] = internal
        if not isinstance(internal, dict) or internal.get("status") != "VERIFY_OK":
            raise AssertionError("synthetic E2E internal verification did not return VERIFY_OK")
        if internal.get("statement_schema") != PUBLIC_STATEMENT_SCHEMA:
            raise AssertionError("synthetic E2E internal statement schema is not source-matched")
        if not isinstance(internal.get("statement_digest"), str) or len(internal["statement_digest"]) != 64:
            raise AssertionError("synthetic E2E internal statement digest is missing")
        cache_source = cls.bundle.get("cache_source")
        if cache_source not in {"temporary", "cache", "rebuilt"}:
            raise AssertionError(f"unexpected synthetic E2E cache source: {cache_source!r}")
        print(
            "synthetic_e2e_internal_verify="
            + json.dumps(
                {
                    "status": internal["status"],
                    "verifier": internal.get("verifier", ""),
                    "statement_schema": internal["statement_schema"],
                    "statement_digest": internal["statement_digest"],
                    "checks": internal.get("checks", {}),
                    "cache_source": cache_source,
                },
                sort_keys=True,
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_root.cleanup()

    def _fresh_verify(self) -> tuple[int, dict[str, object], str]:
        environment = dict(os.environ)
        source_path = str(ROOT / "src")
        environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
        completed = subprocess.run(
            [sys.executable, str(ADAPTER), "--bundle-root", str(self.bundle_root)],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertTrue(completed.stdout.strip(), completed.stderr)
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        return completed.returncode, report, completed.stderr

    @staticmethod
    def _load(path: pathlib.Path) -> object:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _store(path: pathlib.Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_fresh_process_positive_and_public_input_audit(self) -> None:
        code, report, stderr = self._fresh_verify()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(report.get("status"), "VERIFY_OK")
        audit = report["public_input_access_audit"]
        self.assertFalse(audit["verifier_reads_witness"])
        self.assertEqual(audit["witness_paths_read"], [])
        self.assertEqual(audit["witness_fields_read"], [])
        self.assertIn("public_config.json", audit["files_read"])
        self.assertIn("proof/proof_payload.json", audit["files_read"])
        self.assertTrue((self.bundle_root / "public_input_access_audit.json").is_file())

    def test_reused_proof_rejects_public_y_tamper(self) -> None:
        path = self.bundle_root / "public_y.json"
        payload = self._load(path)
        assert isinstance(payload, dict)
        payload["values"][0] = int(payload["values"][0]) + 1
        self._store(path, payload)
        try:
            code, report, _ = self._fresh_verify()
            self.assertNotEqual(code, 0)
            self.assertEqual(report.get("status"), "VERIFY_FAILED")
        finally:
            payload["values"][0] = int(payload["values"][0]) - 1
            self._store(path, payload)

    def test_reused_proof_rejects_privacy_profile_tamper(self) -> None:
        path = self.bundle_root / "proof" / "privacy_profile.json"
        payload = self._load(path)
        assert isinstance(payload, dict)
        original = payload.get("profile_digest")
        payload["profile_digest"] = "0" * 64
        self._store(path, payload)
        try:
            code, report, _ = self._fresh_verify()
            self.assertNotEqual(code, 0)
            self.assertEqual(report.get("status"), "VERIFY_FAILED")
        finally:
            payload["profile_digest"] = original
            self._store(path, payload)

    def test_reused_proof_rejects_gkr_descriptor_tamper(self) -> None:
        path = self.bundle_root / "proof" / "gkr_proof.json"
        payload = self._load(path)
        assert isinstance(payload, dict)
        claims = payload.get("claims")
        assert isinstance(claims, list) and claims
        original = claims[0]["proof"]["spec"]["label"]
        claims[0]["proof"]["spec"]["label"] = "synthetic-attacker-selected"
        self._store(path, payload)
        try:
            code, report, _ = self._fresh_verify()
            self.assertNotEqual(code, 0)
            self.assertEqual(report.get("status"), "VERIFY_FAILED")
        finally:
            claims[0]["proof"]["spec"]["label"] = original
            self._store(path, payload)

    def test_reused_proof_rejects_com_l_tamper(self) -> None:
        path = self.bundle_root / "proof" / "commitments.json"
        payload = self._load(path)
        assert isinstance(payload, dict)
        original = payload["items"]["com_L"]["commitment"]
        payload["items"]["com_L"]["commitment"] = dict(original)
        payload["items"]["com_L"]["commitment"]["x"] = str(int(payload["items"]["com_L"]["commitment"]["x"]) + 1)
        self._store(path, payload)
        try:
            code, report, _ = self._fresh_verify()
            self.assertNotEqual(code, 0)
            self.assertEqual(report.get("status"), "VERIFY_FAILED")
        finally:
            payload["items"]["com_L"]["commitment"] = original
            self._store(path, payload)


if __name__ == "__main__":
    unittest.main()
