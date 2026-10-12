#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import formal_phase


class NativePayloadCapacityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.admission = {
            # Nodes=5 and classes=3 pad independently to 8 and 4: P=32.
            "native_public_y_padded_domain_elements": 32,
            "native_static_table_raw_i64_capacity_elements": 15,
            "native_static_table_padded_field_elements": 32,
            "native_hidden_matmul_body_max_bytes": 120,
            "native_sparse_route_body_max_bytes": 130,
            "native_fac_line8_body_max_bytes": 140,
            "native_constraint_body_max_bytes": 100,
            "native_padding_record_body_max_bytes": 200,
            "native_sumcheck_body_max_bytes": 10,
            "native_terminal_relation_body_max_bytes": 20,
            "native_sorted_lookup_relation_body_max_bytes": 80,
            "native_sorted_lookup_proof_body_max_bytes": 500,
        }

    def test_padded_public_y_and_phase_copy_bounds(self) -> None:
        bounds = formal_phase._native_public_payload_bounds(
            self.admission, nodes=5, classes=3, proof_bytes=1340,
            operation_descriptor_bytes=100)
        self.assertEqual(bounds["public_y_field_bytes"], 32 * 32)
        self.assertEqual(bounds["public_y_proof_wire_bytes"], 32 * 32 + 16)
        self.assertEqual(bounds["other_proof_object_and_wire_bytes"], 200 * 4 + 100)
        self.assertEqual(bounds["public_y_prove_live_bytes"], 5 * 32 * 32 + 32)
        self.assertEqual(bounds["public_y_verify_live_bytes"], 6 * 32 * 32 + 40 + 5 * 3 * 8)
        self.assertEqual(bounds["public_y_statement_digest_live_bytes"], 2 * 32 * 32 + 8)
        self.assertEqual(bounds["public_y_witness_live_bytes"], 2 * 32 * 32 + 8)
        self.assertEqual(bounds["static_table_raw_i64_bytes"], 15 * 8)
        self.assertEqual(bounds["static_table_padded_field_bytes"], 32 * 32)
        self.assertEqual(bounds["operation_constraint_serialization_scratch_bytes"], 2 * (100 + 4) + 2 * 10 + 2 * 20)
        self.assertEqual(bounds["proof_nested_serialization_scratch_bytes"], 3 * 500 + 2 * 80)
        self.assertEqual(bounds["verify_opaque_typedproof_scratch_bytes"], 8 * 140 + 2 * 130 + 2 * 10 + 8192)
        self.assertEqual(bounds["verify_nested_proof_scratch_bytes"], bounds["verify_opaque_typedproof_scratch_bytes"])

    def test_witness_materialization_and_digest_max_for_small_and_large_domains(self) -> None:
        for nodes, classes in ((5, 3), ((1 << 18) + 1, 1)):
            padded_nodes = 1 << (nodes - 1).bit_length()
            padded_classes = 1 << (classes - 1).bit_length()
            padded_elements = padded_nodes * padded_classes
            admission = dict(self.admission, native_public_y_padded_domain_elements=padded_elements)
            proof_bytes = max(1340, 32 * padded_elements + 16 + 100)
            bounds = formal_phase._native_public_payload_bounds(
                admission, nodes=nodes, classes=classes, proof_bytes=proof_bytes,
                operation_descriptor_bytes=100)
            field_bytes = 32 * padded_elements
            materialization = field_bytes + min(padded_elements, 1 << 18) * 32
            digest = 2 * field_bytes + 8
            self.assertEqual(bounds["public_y_witness_live_bytes"], max(materialization, digest))
            self.assertEqual(bounds["public_y_statement_digest_live_bytes"], digest)

    def test_admission_profile_aggregates_public_and_nested_phase_bounds(self) -> None:
        basis_domain = 1 << 18
        frame_bytes = 40 + basis_domain * 48
        admission = {
            "private_family_storage_upper_bound_bytes": 100,
            "sorted_lookup_scratch_upper_bound_bytes": 200,
            "checkpoint_tensor_count": 1,
            "tensor_family_count": 1,
            "operation_count": 1,
            "sorted_lookup_query_count": 0,
            "native_operation_descriptor_bytes": 100,
            "native_operation_descriptor_max_bytes": 100,
            "native_sumcheck_body_max_bytes": 10,
            "native_terminal_relation_body_max_bytes": 20,
            "native_constraint_body_max_bytes": 100,
            "native_padding_record_body_max_bytes": 200,
            "native_hidden_matmul_body_max_bytes": 120,
            "native_sparse_route_body_max_bytes": 130,
            "native_fac_line8_body_max_bytes": 140,
            "native_sorted_lookup_relation_body_max_bytes": 80,
            "native_sorted_lookup_proof_body_max_bytes": 500,
            "native_relation_proof_sections_wire_bytes_upper_bound": 200,
            "native_proof_wire_bytes_upper_bound": 1340,
            "native_public_y_padded_domain_elements": 32,
            "native_static_table_raw_i64_capacity_elements": 15,
            "native_static_table_padded_field_elements": 32,
            "native_commitment_section_wire_bytes_upper_bound": 0,
            "native_operation_descriptor_resident_bytes_upper_bound": 100,
            "native_trace_i64_payload_bytes": 0,
            "native_trace_field_payload_bytes": 0,
            "native_trace_u8_payload_bytes": 0,
            "native_trace_object_metadata_bytes": 0,
            "native_trace_forward_live_peak_bytes_upper_bound": 0,
            "native_route_sumcheck_live_peak_bytes_upper_bound": 0,
            "native_segmented_sumcheck_live_peak_bytes_upper_bound": 0,
            "native_maximum_route_source_domain": 16,
            "native_maximum_route_edge_domain": 8,
            "native_maximum_route_selector_factor_elements": 1,
            "relation_plan_text_bytes": 0,
            "nodes": 5,
            "edges": 2,
            "graphs": 1,
            "maximum_sorted_lookup_merge_logical_rows": 0,
            "model_commitment_segment_count": 1,
            "model_parameter_elements": 1,
        }
        audit = {
            "srs_plan": {
                "basis_domain": basis_domain,
                "frame_bytes": frame_bytes,
                "families": {"synthetic": {"logical_domain": 1, "segment_size": basis_domain,
                                               "segment_count": 1}},
                "estimated_materialized_bytes": frame_bytes + 176,
            },
            "family_domains": {"synthetic": 1},
            "family_domain_count": 1,
            "input_shape": {"features": 4, "classes": 3},
        }
        preflight = {"datasets": {"cora": {
            "resource_prerequisites": audit,
            "input_identity": {"cache_files": [], "checkpoint_files": []},
        }}}
        with tempfile.TemporaryDirectory() as temporary, patch.object(formal_phase, "DATASETS", ("cora",)):
            profile, diagnostics = formal_phase._admission_profile(
                Path(temporary), "synthetic-session", preflight, {"cora": admission})

        payload = diagnostics["cora"]["native_payload_live_bounds_bytes"]
        phases = diagnostics["cora"]["phase_peak_bytes"]
        self.assertEqual(payload["public_y_verify_live_bytes"], 6 * 32 * 32 + 40 + 5 * 3 * 8)
        self.assertEqual(payload["static_table_raw_i64_bytes"], 15 * 8)
        self.assertEqual(payload["static_table_padded_field_bytes"], 32 * 32)
        self.assertEqual(phases["fresh_independent_verify"], phases["verify"])
        self.assertEqual(profile["datasets"]["cora"]["host_peak_bytes"],
                         max(phases["setup"], phases["witness"], phases["commitment"],
                             phases["prove"], phases["verify"]))
        graph_memory = (5 * 4 * 8 + 2 * 8 + 5 * 48 + 2 * 16 + 6 * 8)
        model_and_graph = graph_memory + 1 * 8 + 15 * 8
        proof_object_and_wire = (1340 - 100 - (32 * 32 + 16)) * 4 + 100
        srs_memory = (frame_bytes + 176) * 4
        expected_verify_peak = (
            model_and_graph + 100 + proof_object_and_wire +
            payload["public_y_verify_live_bytes"] + payload["static_table_padded_field_bytes"] +
            payload["verify_nested_proof_scratch_bytes"] + srs_memory + (16 << 20) + (256 << 20)
        )
        self.assertEqual(phases["verify"], expected_verify_peak)

    def test_missing_or_inconsistent_typed_shape_fails_closed(self) -> None:
        missing = dict(self.admission)
        del missing["native_static_table_padded_field_elements"]
        with self.assertRaises(formal_phase.PhaseError):
            formal_phase._native_public_payload_bounds(
                missing, nodes=5, classes=3, proof_bytes=1340,
                operation_descriptor_bytes=100)

        inconsistent = dict(self.admission, native_public_y_padded_domain_elements=15)
        with self.assertRaises(formal_phase.PhaseError):
            formal_phase._native_public_payload_bounds(
                inconsistent, nodes=5, classes=3, proof_bytes=1340,
                operation_descriptor_bytes=100)

        with self.assertRaises(formal_phase.PhaseError):
            formal_phase._native_public_payload_bounds(
                self.admission, nodes=5, classes=3, proof_bytes=1139,
                operation_descriptor_bytes=100)

    def test_capacity_arithmetic_rejects_u64_overflow(self) -> None:
        overflowing = dict(
            self.admission,
            native_static_table_raw_i64_capacity_elements=(1 << 64) - 1,
        )
        with self.assertRaises(formal_phase.PhaseError):
            formal_phase._native_public_payload_bounds(
                overflowing, nodes=5, classes=3, proof_bytes=1340,
                operation_descriptor_bytes=100)

        overflowing_body = dict(
            self.admission,
            native_sorted_lookup_proof_body_max_bytes=(1 << 64) - 1,
        )
        with self.assertRaises(formal_phase.PhaseError):
            formal_phase._native_public_payload_bounds(
                overflowing_body, nodes=5, classes=3, proof_bytes=1340,
                operation_descriptor_bytes=100)


if __name__ == "__main__":
    unittest.main()
