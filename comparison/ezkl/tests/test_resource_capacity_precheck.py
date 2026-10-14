from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formal_common import ContractError  # noqa: E402
from resource_capacity_precheck import (  # noqa: E402
    _build_receipt_matches_identity,
    _fixed_payload_bounds,
    _identity_matches,
    _setup_files_match,
)


class ResourceCapacityPrecheckFixtureTest(unittest.TestCase):
    def test_native_fixed_payload_lower_bound_is_distinct_from_selector_upper_estimate(self) -> None:
        fixture = Path(__file__).parent / "fixtures/resource_capacity_admission_fixture.json"
        admission = json.loads(fixture.read_text(encoding="utf-8"))

        row = _fixed_payload_bounds(admission)

        self.assertEqual(row["fixed_column_count_post_compression"], 8)
        self.assertEqual(row["fixed_payload_lower_bound"]["minimum_fixed_column_count"], 6)
        self.assertEqual(row["fixed_payload_lower_bound_bytes"], 6_144)
        self.assertEqual(row["source_model_fixed_payload_upper_bound_bytes"], 8_192)

    def test_invalid_geometry_and_historical_source_mismatch_fail_closed(self) -> None:
        fixture = Path(__file__).parent / "fixtures/resource_capacity_admission_fixture.json"
        admission = json.loads(fixture.read_text(encoding="utf-8"))
        admission["constraint_system"]["degree"] = 0
        with self.assertRaises(ContractError):
            _fixed_payload_bounds(admission)

        current = {
            "schema": "ezkl_capacity_source_identity_v1",
            "geometry_sources_sha256": [["src/gat.rs", "a" * 64]],
        }
        mismatched = {
            "schema": "ezkl_capacity_source_identity_v1",
            "geometry_sources_sha256": [["src/gat.rs", "b" * 64]],
        }
        self.assertFalse(_identity_matches(mismatched, current))
        self.assertFalse(_identity_matches(None, current))

    def test_setup_receipt_exact_files_and_current_build_mismatch_fail_closed(self) -> None:
        fixture = Path(__file__).parent / "fixtures/resource_capacity_provenance_fixture.json"
        document = json.loads(fixture.read_text(encoding="utf-8"))

        self.assertTrue(_setup_files_match(document["setup_files"]))
        missing = dict(document["setup_files"])
        missing.pop("producer_context.json")
        self.assertFalse(_setup_files_match(missing))
        extra = {**document["setup_files"], "unanchored.extra": "a" * 64}
        self.assertFalse(_setup_files_match(extra))

        self.assertTrue(_build_receipt_matches_identity(document["build_receipt"], document["resume_identity"]))
        mismatched = dict(document["build_receipt"], binary_sha256="e" * 64)
        self.assertFalse(_build_receipt_matches_identity(mismatched, document["resume_identity"]))


if __name__ == "__main__":
    unittest.main()
