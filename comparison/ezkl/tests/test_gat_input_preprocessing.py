from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.append(str(PROJECT_ROOT / "scripts"))

import run_hybrid_formal as formal_runner  # noqa: E402
from generate_gat_resources import feature_bound  # noqa: E402
from produce_public_statement import (  # noqa: E402
    canonical_model_digest,
    digest_i128,
    digest_public_input,
    feature_values,
    i128_bytes,
)
from quantization_scale_chain import canonical_input_preprocess, quantize_array_ties_away  # noqa: E402


class GatInputPreprocessingTest(unittest.TestCase):
    def test_streamed_i128_digests_match_legacy_encoding(self) -> None:
        boundary = [-(1 << 63), -1, 0, 1, (1 << 63) - 1]
        values = np.asarray(boundary * 16385, dtype=np.int64)
        topology = "a5" * 32

        expected_input = hashlib.sha256(
            b"ezkl-gat-public-input-v1"
            + b"".join(i128_bytes(int(value)) for value in values.tolist())
            + bytes.fromhex(topology)
        ).hexdigest()
        expected_output = hashlib.sha256(
            b"ezkl-gat-public-y-v1" + b"".join(i128_bytes(int(value)) for value in values.tolist())
        ).hexdigest()
        expected_model = hashlib.sha256(
            b"model-domain-v1" + b"".join(i128_bytes(int(value)) for value in values.tolist())
        ).hexdigest()

        self.assertEqual(digest_public_input(values.reshape(1, -1), topology), expected_input)
        self.assertEqual(digest_i128(values), expected_output)
        self.assertEqual(canonical_model_digest(values, "model-domain-v1"), expected_model)

    def test_ppi_npy_matches_canonical_normalize_then_quantize(self) -> None:
        raw = np.array([[3.0, 1.0, 0.0], [0.0, 0.0, 0.0], [-1.0, -3.0, 0.0]], dtype="<f4")
        expected = quantize_array_ties_away(canonical_input_preprocess(raw, "ppi", "row_normalize"), 16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "resources/canonical/cache/ppi"
            data_dir.mkdir(parents=True)
            feature_path = data_dir / "features.npy"
            np.save(feature_path, raw, allow_pickle=False)
            meta = {"num_nodes": "3", "num_features": "3", "input_preprocess_id": "row_normalize"}
            actual, actual_path = feature_values(data_dir, meta, 4, "row_normalize", "ppi")
            self.assertEqual(actual_path, feature_path)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(feature_bound(root, "ppi", meta), max(1, int(np.max(np.abs(expected)))))

    def test_ppi_npy_preprocessing_rejects_nonfinite_values(self) -> None:
        raw = np.array([[1.0, np.nan]], dtype="<f4")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "resources/canonical/cache/ppi"
            data_dir.mkdir(parents=True)
            np.save(data_dir / "features.npy", raw, allow_pickle=False)
            meta = {"num_nodes": "1", "num_features": "2", "input_preprocess_id": "row_normalize"}
            with self.assertRaises(SystemExit):
                feature_values(data_dir, meta, 4, "row_normalize", "ppi")
            with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                feature_bound(root, "ppi", meta)

    def test_sparse_rows_normalize_in_feature_order_like_canonical_input(self) -> None:
        dense = np.array([[1.0e16, -1.0e16, 1.0]], dtype=np.float64)
        expected = quantize_array_ties_away(canonical_input_preprocess(dense, "cora", "row_normalize"), 16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "resources/canonical/cache/cora"
            data_dir.mkdir(parents=True)
            (data_dir / "features.txt").write_text("1 3\n1:-1e16 2:1 0:1e16\n", encoding="utf-8")
            meta = {"num_nodes": "1", "num_features": "3", "input_preprocess_id": "row_normalize"}
            actual, actual_path = feature_values(data_dir, meta, 4, "row_normalize", "cora")
            self.assertEqual(actual_path, data_dir / "features.txt")
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(feature_bound(root, "cora", meta), int(np.max(np.abs(expected))))

    def test_sparse_rows_reject_duplicate_columns_and_invalid_values(self) -> None:
        meta = {"num_nodes": "1", "num_features": "2", "input_preprocess_id": "row_normalize"}
        for row, reason in (("0:1 0:2\n", "duplicate"), ("2:1\n", "index"), ("1:nan\n", "non-finite")):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_dir = root / "resources/canonical/cache/cora"
                data_dir.mkdir(parents=True)
                (data_dir / "features.txt").write_text(f"1 2\n{row}", encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, reason):
                    feature_values(data_dir, meta, 4, "row_normalize", "cora")
                with self.assertRaisesRegex(ValueError, reason):
                    feature_bound(root, "cora", meta)

    def test_native_prepare_rejects_domain_size_outside_host_usize(self) -> None:
        budget = {
            "total_limit_bytes": 1,
            "initial_available_bytes": 1,
            "baseline_heap_bytes": 0,
            "baseline_hot_mmap_bytes": 0,
            "tablecache_bytes": 0,
            "shplonk_extensions_bytes": 0,
            "icicle_staging_bytes": 0,
            "device_headroom_bytes": 0,
            "reserve_bytes": 0,
            "incremental_envelope_bytes": 0,
            "future_phase_increment_bytes": 0,
            "minimum_task_bytes": 1,
            "taskcap_bytes": 1,
        }
        with self.assertRaises(formal_runner.ContractError):
            formal_runner._validated_native_prepare_exports(
                budget,
                [1],
                domain_size=sys.maxsize + 1,
                expected_lookup_groups=1,
                expected_h_resident_device_upper_bytes=1,
            )

    def test_admission_rejects_untrusted_or_unbounded_srs_logrows_before_shift(self) -> None:
        valid = {"logrows": 4, "max_public_srs_logrows": 26, "domain_size": 16}
        formal_runner._validate_native_admission_domain("cora", valid, 26)
        with self.assertRaisesRegex(formal_runner.ContractError, "authenticated native SRS contract"):
            formal_runner._validate_native_admission_domain(
                "cora",
                {"logrows": 4, "max_public_srs_logrows": 25, "domain_size": 16},
                26,
            )
        with self.assertRaisesRegex(formal_runner.ContractError, "authenticated native SRS contract"):
            formal_runner._validate_native_admission_domain(
                "cora",
                {"logrows": sys.maxsize + 1, "max_public_srs_logrows": 26, "domain_size": 16},
                26,
            )

    def test_native_prepare_rejects_budget_and_cardinality_mismatch(self) -> None:
        budget = {
            "total_limit_bytes": 4,
            "initial_available_bytes": 4,
            "baseline_heap_bytes": 0,
            "baseline_hot_mmap_bytes": 0,
            "tablecache_bytes": 0,
            "shplonk_extensions_bytes": 0,
            "icicle_staging_bytes": 0,
            "device_headroom_bytes": 0,
            "reserve_bytes": 0,
            "incremental_envelope_bytes": 0,
            "future_phase_increment_bytes": 0,
            "minimum_task_bytes": 1,
            "taskcap_bytes": 1,
        }
        invalid_budgets = []
        oversized_taskcap = dict(budget)
        oversized_taskcap["taskcap_bytes"] = 5
        invalid_budgets.append(oversized_taskcap)
        nonpositive_budget = dict(budget)
        nonpositive_budget["minimum_task_bytes"] = 0
        invalid_budgets.append(nonpositive_budget)
        for candidate in invalid_budgets:
            with self.subTest(budget=candidate), self.assertRaises(formal_runner.ContractError):
                formal_runner._validated_native_prepare_exports(candidate, [1], domain_size=8, expected_lookup_groups=1, expected_h_resident_device_upper_bytes=256)
        with self.assertRaises(formal_runner.ContractError):
            formal_runner._validated_native_prepare_exports(budget, [9], domain_size=8, expected_lookup_groups=1, expected_h_resident_device_upper_bytes=256)

    def test_setup_reuse_receipt_requires_session_anchored_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "cora"
            output.mkdir()
            session_file = root / "session.json"
            session_file.write_text('{"session_id":"session"}\n', encoding="utf-8")
            admission_source = root / "session-admission.json"
            admission_payload = b'{"dataset":"cora","schema_version":10}\n'
            admission_source.write_bytes(admission_payload)
            admission_sha256 = formal_runner.sha256(admission_source)
            for name in formal_runner._SETUP_REUSE_FILES:
                if name != "admission.json":
                    (output / name).write_text(name, encoding="utf-8")
            formal_runner._install_authenticated_admission_snapshot(
                output, admission_source, admission_sha256, "cora"
            )
            admission = {
                "settings_sha256": formal_runner.sha256(output / "settings.json"),
                "compiled_circuit_sha256": formal_runner.sha256(output / "compiled.circuit"),
                "sha256": admission_sha256,
            }
            identity = {"schema": "ezkl-formal-resume-identity-v1", "identity": "stable"}
            native_receipt = {
                "compiled_circuit_sha256": formal_runner.sha256(output / "compiled.circuit"),
            }
            with mock.patch.object(
                formal_runner,
                "_load_native_keygen_receipt",
                return_value=({}, native_receipt, "b" * 64),
            ):
                receipt_digest = formal_runner.write_setup_reuse_receipt(output, "cora", "session", identity, admission)
                formal_runner.anchor_setup_reuse_receipt(session_file, "cora", receipt_digest)
                anchors = json.loads(session_file.read_text(encoding="utf-8"))["setup_reuse_receipts"]
                self.assertTrue(
                    formal_runner.validate_setup_reuse_receipt(
                        output, "cora", "session", identity, admission, anchors["cora"]
                    )
                )

                (output / "admission.json").write_bytes(b'{"dataset":"other"}\n')
                self.assertFalse(
                    formal_runner.validate_setup_reuse_receipt(
                        output, "cora", "session", identity, admission, anchors["cora"]
                    )
                )
                with self.assertRaises(formal_runner.ContractError):
                    formal_runner._install_authenticated_admission_snapshot(
                        output, admission_source, admission_sha256, "cora"
                    )

            with (output / "setup_receipt.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with mock.patch.object(
                formal_runner,
                "_load_native_keygen_receipt",
                return_value=({}, native_receipt, "b" * 64),
            ):
                self.assertFalse(
                    formal_runner.validate_setup_reuse_receipt(
                        output, "cora", "session", identity, admission, anchors["cora"]
                    )
                )


if __name__ == "__main__":
    unittest.main()
