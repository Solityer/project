"""Targeted tests for EZKL final collector/validator capacity arithmetic."""

from __future__ import annotations

import json
import hashlib
import math
import sys
import struct
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from finalization_capacity import (  # noqa: E402
    DATASETS,
    FinalizationCapacityError,
    _atomic_json_bytes,
    _host_requirements,
    _read_meta_cfg,
    _regular_file,
    _require_matrix_size_within_upper,
    _scan_edge_source,
    _session_temp_size,
    _dataset_shape_bounds,
    estimate_finalization_capacity,
    typed_proof_requirements,
    typed_statement_requirements,
)
from collect_results import build_matrix_document  # noqa: E402
from formal_common import CANONICAL_RELATION, resource_manifest_hash  # noqa: E402


class FinalizationCapacityTest(unittest.TestCase):
    def _sizes(self) -> tuple[dict[str, int], dict[str, int]]:
        return ({dataset: (index + 1) * 1000 for index, dataset in enumerate(DATASETS)}, {dataset: (index + 1) * 100 for index, dataset in enumerate(DATASETS)})

    def test_proof_peak_uses_one_largest_proof_and_cpython_factor(self) -> None:
        proofs, results = self._sizes()
        plan = _host_requirements(proofs, results, 4096)
        self.assertEqual(plan["proof_wire_bytes_max"], max(proofs.values()))
        self.assertEqual(plan["proof_peak_bytes_source_derived_single_current_proof"], int(max(proofs.values()) * (plan["proof_dom_bytes_per_wire_byte"] + 10)))
        self.assertGreaterEqual(plan["proof_dom_bytes_per_wire_byte"], 112.5)
        expected_dom_factor = max(112.5, math.ceil((7 * sum(sys.getsizeof(value) for value in ({}, [], "", 0, 0.0)) + 32 * struct.calcsize("P")) / 16))
        self.assertEqual(plan["proof_dom_bytes_per_wire_byte"], expected_dom_factor)
        self.assertEqual(plan["proof_text_bytes_per_wire_byte"], 4.0)
        self.assertEqual(plan["canonical_instances_text_and_bytes_per_wire_byte"], 6.0)
        self.assertGreater(plan["host_required_bytes"], plan["proof_peak_bytes_source_derived_single_current_proof"])

    def test_capacity_admits_when_live_host_and_both_filesystems_fit(self) -> None:
        proofs, results = self._sizes()
        plan = _host_requirements(proofs, results, 8192)
        host = int(plan["host_required_bytes"]) + 1
        result = estimate_finalization_capacity(proof_sizes=proofs, result_sizes=results, matrix_bytes=8192, session_temp_bytes=2048, host_available_bytes=host, filesystem_available_bytes={1: 32768, 2: 32768}, matrix_device_id=1, session_device_id=2)
        self.assertEqual(result["filesystems"]["1"]["required_new_bytes"], 8192)
        self.assertEqual(result["filesystems"]["2"]["required_new_bytes"], 2048)
        self.assertTrue(result["existing_proof_and_result_bytes_already_accounted_by_statvfs"])

    def test_capacity_fails_closed_for_host_or_matrix_filesystem_shortage(self) -> None:
        proofs, results = self._sizes()
        plan = _host_requirements(proofs, results, 8192)
        with self.assertRaisesRegex(FinalizationCapacityError, "host gate"):
            estimate_finalization_capacity(proof_sizes=proofs, result_sizes=results, matrix_bytes=8192, session_temp_bytes=2048, host_available_bytes=int(plan["host_required_bytes"]) - 1, filesystem_available_bytes={1: 32768}, matrix_device_id=1, session_device_id=1)
        with self.assertRaisesRegex(FinalizationCapacityError, "filesystem device"):
            estimate_finalization_capacity(proof_sizes=proofs, result_sizes=results, matrix_bytes=8192, session_temp_bytes=2048, host_available_bytes=int(plan["host_required_bytes"]) + 1, filesystem_available_bytes={1: 10239}, matrix_device_id=1, session_device_id=1)

    def test_matrix_and_atomic_session_size_match_production_serializers(self) -> None:
        cells = [{"dataset": dataset, "_result_path": f"runs/formal/{dataset}/result.json"} for dataset in DATASETS]
        matrix = build_matrix_document(cells, datasets=DATASETS, source_sha256="b" * 64, resource_sha256="c" * 64, collected_at_utc="2099-12-31T23:59:59.999999+00:00", formal_session_id="a" * 32, backends=None)
        expected_document = {"schema_version": 1, "baseline": "ezkl", "collection_status": "COLLECTED_PENDING_FINAL_VALIDATION", "relation_id": CANONICAL_RELATION, "datasets": list(DATASETS), "cell_count": len(cells), "source_tree_sha256": "b" * 64, "resource_manifest_sha256": "c" * 64, "collected_at_utc": "2099-12-31T23:59:59.999999+00:00", "cells": cells, "formal_session_id": "a" * 32}
        expected = len((json.dumps(expected_document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        self.assertEqual(matrix, expected_document)
        self.assertEqual(_atomic_json_bytes(matrix), expected)
        session = {"baseline": "ezkl", "session_id": "a" * 32}
        receipt = {"session_id": "a" * 32, "matrix_output_bytes": expected}
        predicted_temp_bytes = _session_temp_size(session, receipt, receipt_field="finalization_capacity")
        candidate = dict(session)
        candidate["finalization_capacity"] = receipt
        self.assertEqual(predicted_temp_bytes, len((json.dumps(candidate, indent=2, sort_keys=True) + "\n").encode("utf-8")))
        self.assertEqual(receipt["session_atomic_temp_bytes"], predicted_temp_bytes)

    def test_validator_capacity_does_not_charge_existing_matrix_bytes(self) -> None:
        proofs, results = self._sizes()
        plan = _host_requirements(proofs, results, 8192, phase="validator")
        required = int(plan["host_required_bytes"])
        capacity = estimate_finalization_capacity(
            proof_sizes=proofs,
            result_sizes=results,
            matrix_bytes=8192,
            session_temp_bytes=2048,
            host_available_bytes=required + 1,
            filesystem_available_bytes={1: 32768},
            matrix_device_id=1,
            session_device_id=1,
            phase="validator",
        )
        self.assertFalse(capacity["write_matrix"])
        self.assertEqual(capacity["matrix_output_bytes"], 0)
        self.assertEqual(capacity["matrix_existing_bytes"], 8192)
        self.assertEqual(capacity["filesystems"]["1"]["required_new_bytes"], 2048)

    def test_validator_admits_collector_auto_timespec_matrix_size_under_upper(self) -> None:
        cells = [{"dataset": dataset, "_result_path": f"runs/formal/{dataset}/result.json"} for dataset in DATASETS]
        no_fraction = datetime(2026, 10, 7, tzinfo=timezone.utc).isoformat()
        max_fraction = "2099-12-31T23:59:59.999999+00:00"
        actual_document = build_matrix_document(cells, datasets=DATASETS, source_sha256="b" * 64, resource_sha256="c" * 64, collected_at_utc=no_fraction, formal_session_id="a" * 32, backends=None)
        upper_document = build_matrix_document(cells, datasets=DATASETS, source_sha256="b" * 64, resource_sha256="c" * 64, collected_at_utc=max_fraction, formal_session_id="a" * 32, backends=None)
        actual_bytes = _atomic_json_bytes(actual_document)
        upper_bytes = _atomic_json_bytes(upper_document)
        self.assertEqual(upper_bytes - actual_bytes, 7)
        _require_matrix_size_within_upper(actual_bytes, upper_bytes)
        _require_matrix_size_within_upper(upper_bytes, upper_bytes)
        with self.assertRaisesRegex(FinalizationCapacityError, "exceeds the shared-schema serialization upper bound"):
            _require_matrix_size_within_upper(upper_bytes + 1, upper_bytes)

        proofs, results = self._sizes()
        for existing_size in (actual_bytes, upper_bytes):
            plan = _host_requirements(proofs, results, existing_size, phase="validator")
            capacity = estimate_finalization_capacity(
                proof_sizes=proofs,
                result_sizes=results,
                matrix_bytes=existing_size,
                session_temp_bytes=2048,
                host_available_bytes=int(plan["host_required_bytes"]) + 1,
                filesystem_available_bytes={1: 32768},
                matrix_device_id=1,
                session_device_id=1,
                phase="validator",
            )
            self.assertEqual(capacity["matrix_existing_bytes"], existing_size)
            self.assertEqual(capacity["matrix_output_bytes"], 0)

    def test_validator_session_temp_covers_terminal_rewrite(self) -> None:
        session = {"baseline": "ezkl", "session_id": "a" * 32}
        receipt = {"phase": "validator", "matrix_existing_bytes": 8192}
        predicted = _session_temp_size(session, receipt, receipt_field="final_validation_capacity", terminal=True)
        self.assertEqual(receipt["session_atomic_temp_bytes"], predicted)
        recorded = dict(session)
        recorded["final_validation_capacity"] = receipt
        receipt_size = len((json.dumps(recorded, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        terminal = dict(recorded)
        terminal["status"] = "FINAL_VALIDATION_OK"
        terminal["stage"] = "final_validation_complete"
        terminal["updated_at_utc"] = "2099-12-31T23:59:59.999999+00:00"
        terminal["completed_at_utc"] = "2099-12-31T23:59:59.999999+00:00"
        terminal_size = len((json.dumps(terminal, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        self.assertEqual(predicted, max(receipt_size, terminal_size))

    def test_typed_statement_bound_covers_large_feature_table_and_public_y(self) -> None:
        shape = typed_statement_requirements(
            num_nodes=169343,
            num_features=128,
            num_classes=40,
            raw_edge_lines=1_200_000,
            graph_count=1,
        )
        self.assertEqual(shape["feature_values"], 169343 * 128)
        self.assertEqual(shape["public_y_values"], 169343 * 40)
        self.assertEqual(shape["edge_count_upper_after_undirected_expansion_and_self_loops"], 2 * 1_200_000 + 169343)
        self.assertEqual(shape["public_y_digest_bytearray_and_bytes_upper"], 34 * 169343 * 40 + sys.getsizeof(bytearray()) + len(b"ezkl-gat-public-y-v1"))
        self.assertGreater(shape["statement_dom_bytes_upper_typed_int64"], shape["feature_values"] * (sys.getsizeof(1 << 63) + struct.calcsize("P")))
        self.assertLess(shape["statement_wire_bytes_upper"], 1_500_000_000)
        self.assertLess(shape["statement_dom_bytes_upper_typed_int64"] + shape["statement_text_bytes_upper_unicode_safe"], 10_000_000_000)

    def test_typed_snark_bound_avoids_generic_false_rejection_for_official_ogb_shape(self) -> None:
        shapes, _source_stats = _dataset_shape_bounds(resource_manifest_hash())
        ogb = shapes["ogbn_arxiv"]
        # Construct a pure arithmetic wire-size scenario from the source-known
        # minimum public payload plus a 512 MiB native u8 proof. No artifact is
        # created or parsed by this focused capacity regression.
        public_lower = (
            ogb["feature_values"]
            + ogb["num_nodes"]
            + 2 * (ogb["graph_count"] + 1)
            + ogb["public_y_values"]
        )
        known_wire_lower = (
            67 * (public_lower + 1) + 3
            + 68 * public_lower
            + 3 * public_lower
            + 68
        )
        native_proof_bytes = 512 * 1024 * 1024
        proof_sizes = {}
        proof_bounds = {}
        for dataset in DATASETS:
            shape = shapes[dataset]
            lower = (
                shape["feature_values"]
                + shape["num_nodes"]
                + 2 * (shape["graph_count"] + 1)
                + shape["public_y_values"]
            )
            minimum = 67 * (lower + 1) + 3 + 68 * lower + 3 * lower + 68
            payload = native_proof_bytes if dataset == "ogbn_arxiv" else 64 * 1024 * 1024
            proof_sizes[dataset] = minimum + 6 * payload + 2 * 1024 * 1024
            proof_bounds[dataset] = typed_proof_requirements(
                proof_size_bytes=proof_sizes[dataset], statement_shape=shape
            )
        self.assertGreaterEqual(proof_bounds["ogbn_arxiv"]["public_instance_values_upper"], 31_290_144)
        self.assertTrue(proof_bounds["ogbn_arxiv"]["json_u8_scalar_cache_confirmed"])
        self.assertGreaterEqual(
            proof_bounds["ogbn_arxiv"]["proof_vector_bytes_upper_from_remaining_wire_div_4"],
            native_proof_bytes,
        )
        self.assertLess(
            proof_bounds["ogbn_arxiv"]["proof_peak_bytes_source_derived_typed_snark"],
            proof_sizes["ogbn_arxiv"] * 20,
        )
        results = {dataset: 4 * 1024 * 1024 for dataset in DATASETS}
        plan = _host_requirements(
            proof_sizes,
            results,
            32 * 1024 * 1024,
            statement_requirements=shapes,
            proof_requirements=proof_bounds,
        )
        self.assertEqual(plan["proof_peak_model"], "typed-native-Snark-schema")
        self.assertNotIn("proof_dom_bytes_per_wire_byte", plan)
        self.assertEqual(plan["proof_text_bytes_per_wire_byte"], 2.0)
        self.assertLess(plan["host_required_bytes"], 120 * 1024**3)
        generic = _host_requirements(proof_sizes, results, 32 * 1024 * 1024)
        self.assertGreater(generic["host_required_bytes"], 120 * 1024**3)
        admitted = estimate_finalization_capacity(
            proof_sizes=proof_sizes,
            result_sizes=results,
            matrix_bytes=32 * 1024 * 1024,
            session_temp_bytes=1024 * 1024,
            host_available_bytes=plan["host_required_bytes"] + 1,
            filesystem_available_bytes={1: 64 * 1024 * 1024},
            matrix_device_id=1,
            session_device_id=1,
            statement_requirements=shapes,
            proof_requirements=proof_bounds,
        )
        self.assertEqual(admitted["phase"], "collector")
        with self.assertRaisesRegex(FinalizationCapacityError, "host gate"):
            estimate_finalization_capacity(
                proof_sizes=proof_sizes,
                result_sizes=results,
                matrix_bytes=32 * 1024 * 1024,
                session_temp_bytes=1024 * 1024,
                host_available_bytes=0,
                filesystem_available_bytes={1: 64 * 1024 * 1024},
                matrix_device_id=1,
                session_device_id=1,
                statement_requirements=shapes,
                proof_requirements=proof_bounds,
            )

    def test_typed_snark_rejects_impossible_stat_size_before_json_parse(self) -> None:
        shape = typed_statement_requirements(
            num_nodes=10,
            num_features=4,
            num_classes=3,
            raw_edge_lines=12,
            graph_count=1,
        )
        with self.assertRaisesRegex(FinalizationCapacityError, "smaller than its source-required"):
            typed_proof_requirements(proof_size_bytes=1, statement_shape=shape)

    def test_typed_snark_wire_subtraction_matches_single_pretty_debug_copy(self) -> None:
        # These are in-memory Snark-schema serialization cases only. They do
        # not create proof artifacts or claim formal execution evidence.
        for nodes, features, classes, raw_edges, actual_edges, proof_bytes, float_chars in (
            (3, 2, 2, 2, 2, 1, 1),
            (10, 3, 2, 6, 5, 127, 32),
            (20, 4, 3, 12, 15, 255, 328),
        ):
            shape = typed_statement_requirements(
                num_nodes=nodes,
                num_features=features,
                num_classes=classes,
                raw_edge_lines=raw_edges,
                graph_count=1,
            )
            self.assertLessEqual(actual_edges, shape["edge_count_upper_after_undirected_expansion_and_self_loops"])
            feature = ["0" * 64] * (nodes * features)
            node_ids = ["0" * 64] * nodes
            src = ["0" * 64] * actual_edges
            dst = ["0" * 64] * actual_edges
            node_ptr = ["0" * 64] * 2
            edge_ptr = ["0" * 64] * 2
            public_y = ["0" * 64] * (nodes * classes)
            inputs = [feature, node_ids, src, dst, node_ptr, edge_ptr]
            public_values = sum((list(values) for values in inputs), []) + public_y
            fr_debug = "0x" + "0" * 64
            float_text = "0" if float_chars == 1 else "0." + "1" * (float_chars - 2)
            proof = [(index * 53) % 256 for index in range(proof_bytes)]
            document = {
                "protocol": None,
                "instances": [public_values + ["0" * 64]],
                "proof": proof,
                "hex_proof": "0x" + bytes(proof).hex(),
                "split": None,
                "pretty_public_inputs": {
                    "rescaled_inputs": [[float_text] * len(values) for values in inputs],
                    "inputs": [[fr_debug] * len(values) for values in inputs],
                    "processed_inputs": [],
                    "processed_params": [[fr_debug]],
                    "processed_outputs": [],
                    "rescaled_outputs": [[float_text] * len(public_y)],
                    "outputs": [[fr_debug] * len(public_y)],
                },
                "timestamp": 0,
                "version": "0.1.0",
            }
            wire = json.dumps(document, separators=(",", ":")).encode("ascii")
            requirements = typed_proof_requirements(
                proof_size_bytes=len(wire), statement_shape=shape
            )
            lower = requirements["known_native_wire_bytes_lower_for_proof_vector_bound"]
            self.assertGreaterEqual(len(wire), lower)
            self.assertGreaterEqual(
                requirements["proof_vector_bytes_upper_from_remaining_wire_div_4"],
                proof_bytes,
            )
            self.assertEqual(
                len(json.dumps(document["instances"], separators=(",", ":")).encode("ascii")),
                67 * len(document["instances"][0]) + 3,
            )

    def test_canonical_shape_sources_require_manifest_hash_and_immutable_stat(self) -> None:
        meta_bytes = b"name=cora\nnum_nodes=3\nnum_features=2\nnum_classes=1\ngraph_count=1\n"
        edge_bytes = b"0 1\n1 0"
        with tempfile.TemporaryDirectory() as temporary:
            meta_path = Path(temporary) / "meta.cfg"
            edge_path = Path(temporary) / "edges.txt"
            meta_path.write_bytes(meta_bytes)
            edge_path.write_bytes(edge_bytes)
            meta_digest = hashlib.sha256(meta_bytes).hexdigest()
            edge_digest = hashlib.sha256(edge_bytes).hexdigest()

            shape, meta_identity = _read_meta_cfg(meta_path, "cora", meta_digest)
            self.assertEqual(shape, {"num_nodes": 3, "num_features": 2, "num_classes": 1, "graph_count": 1})
            self.assertEqual(meta_identity["sha256"], meta_digest)
            with self.assertRaisesRegex(FinalizationCapacityError, "content digest"):
                _read_meta_cfg(meta_path, "cora", "0" * 64)

            edge_lines, edge_identity = _scan_edge_source(edge_path, "cora", edge_digest)
            self.assertEqual(edge_lines, 2)
            self.assertEqual(edge_identity["sha256"], edge_digest)
            with self.assertRaisesRegex(FinalizationCapacityError, "edge source digest"):
                _scan_edge_source(edge_path, "cora", "0" * 64)

            original_stat = _regular_file
            calls = 0

            def mutate_metadata_between_stat_calls(path: Path, label: str):
                nonlocal calls
                calls += 1
                if calls == 2:
                    path.write_bytes(meta_bytes + b"# changed during read\n")
                return original_stat(path, label)

            with patch("finalization_capacity._regular_file", side_effect=mutate_metadata_between_stat_calls):
                with self.assertRaisesRegex(FinalizationCapacityError, "changed while its authenticated bytes were read"):
                    _read_meta_cfg(meta_path, "cora", meta_digest)

    def test_oversized_small_json_is_rejected_before_admission(self) -> None:
        proofs, results = self._sizes()
        with self.assertRaisesRegex(FinalizationCapacityError, "oversized small JSON"):
            _host_requirements(proofs, results, 4096, small_json_sizes={"artifact_manifest": 1024 * 1024 + 1})

    def test_dataset_axis_is_exactly_five(self) -> None:
        proofs, results = self._sizes()
        del proofs[DATASETS[-1]]
        with self.assertRaisesRegex(FinalizationCapacityError, "exact five"):
            _host_requirements(proofs, results, 1024)


if __name__ == "__main__":
    unittest.main()
