from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formal_common import DATASETS, ContractError  # noqa: E402
from resource_capacity_precheck import (  # noqa: E402
    _build_receipt_matches_identity,
    _fixed_payload_bounds,
    _identity_matches,
    _setup_files_match,
)


class ResourceCapacityPrecheckFixtureTest(unittest.TestCase):
    def test_native_v4_fixed_payload_lower_bound_is_unknown_for_five_admissions(self) -> None:
        fixture = Path(__file__).parent / "fixtures/resource_capacity_admission_fixture.json"
        rows = []
        for dataset in DATASETS:
            admission = json.loads(fixture.read_text(encoding="utf-8"))
            admission["dataset"] = dataset
            row = _fixed_payload_bounds(admission)
            self.assertEqual(row["fixed_column_count_post_compression"], 8)
            self.assertEqual(row["fixed_payload_lower_bound_bytes"], 0)
            self.assertEqual(row["fixed_payload_lower_bound"]["status"], "unknown_from_admission_missing_per_column_encoding_and_actual_permutation_count")
            self.assertEqual(row["fixed_payload_lower_bound"]["fixed_values_serialized_copies"], 1)
            self.assertFalse(row["fixed_payload_lower_bound"]["fixed_polys_serialized"])
            self.assertIsNone(row["source_model_fixed_payload_upper_bound_bytes"])
            rows.append(row)
        # Native v4 allows both Dense and Compact fixed values; Admission has no
        # per-column encoding/cardinality, so neither layout supports a positive
        # structural lower bound. The source-model upper is gated separately.
        for encoding in ("Dense", "Compact"):
            with self.subTest(v4_fixed_value_encoding=encoding):
                self.assertEqual(sum(row["fixed_payload_lower_bound_bytes"] for row in rows), 0)
                self.assertTrue(all(row["source_model_fixed_payload_upper_bound_bytes"] is None for row in rows))

    def test_invalid_geometry_and_historical_source_mismatch_fail_closed(self) -> None:
        fixture = Path(__file__).parent / "fixtures/resource_capacity_admission_fixture.json"
        admission = json.loads(fixture.read_text(encoding="utf-8"))
        admission["constraint_system"]["degree"] = 0
        with self.assertRaises(ContractError):
            _fixed_payload_bounds(admission)

        current = {
            "schema": "ezkl_capacity_source_identity_v2",
            "geometry_sources_sha256": [["src/gat.rs", "a" * 64]],
        }
        mismatched = {
            "schema": "ezkl_capacity_source_identity_v2",
            "geometry_sources_sha256": [["src/gat.rs", "b" * 64]],
        }
        self.assertFalse(_identity_matches(mismatched, current))
        self.assertFalse(_identity_matches(None, current))
        self.assertFalse(_identity_matches(dict(current, schema="ezkl_capacity_source_identity_v1"), current))

    def test_setup_receipt_exact_files_and_current_build_mismatch_fail_closed(self) -> None:
        fixture = Path(__file__).parent / "fixtures/resource_capacity_provenance_fixture.json"
        document = json.loads(fixture.read_text(encoding="utf-8"))

        self.assertTrue(_setup_files_match(document["setup_files"]))
        missing = dict(document["setup_files"])
        missing.pop("producer_context.json")
        self.assertFalse(_setup_files_match(missing))
        missing_admission = dict(document["setup_files"])
        missing_admission.pop("admission.json")
        self.assertFalse(_setup_files_match(missing_admission))
        extra = {**document["setup_files"], "unanchored.extra": "a" * 64}
        self.assertFalse(_setup_files_match(extra))

        self.assertTrue(_build_receipt_matches_identity(document["build_receipt"], document["resume_identity"]))
        mismatched = dict(document["build_receipt"], binary_sha256="e" * 64)
        self.assertFalse(_build_receipt_matches_identity(mismatched, document["resume_identity"]))


if __name__ == "__main__":
    unittest.main()
