#!/usr/bin/env python3
"""Short regression checks for the single internal OGBN dataset identity."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from checkpoint_bundle import canonical_checkpoint_semantics  # noqa: E402
from quantization_scale_chain import (  # noqa: E402
    OGBN_ARXIV_EXTERNAL_NAME,
    OGBN_ARXIV_INTERNAL_NAME,
    canonical_input_preprocess_id,
)

EXPORTER_PATH = ROOT / "scripts" / "export_benchmark_table.py"
EXPORTER_SPEC = importlib.util.spec_from_file_location("export_benchmark_table", EXPORTER_PATH)
assert EXPORTER_SPEC is not None and EXPORTER_SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(EXPORTER_SPEC)
EXPORTER_SPEC.loader.exec_module(EXPORTER)

AGGREGATOR_PATH = ROOT / "scripts" / "aggregate_secure_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_secure_results", AGGREGATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
AGGREGATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATOR)
ARTIFACT_EXPORT_PATH = ROOT / "src" / "protocol" / "artifact_export.cpp"


class OgbnIdentityTests(unittest.TestCase):
    def test_external_spelling_is_only_the_data_boundary(self) -> None:
        self.assertEqual(OGBN_ARXIV_INTERNAL_NAME, "ogbn_arxiv")
        self.assertEqual(OGBN_ARXIV_EXTERNAL_NAME, "ogbn-arxiv")
        self.assertEqual(canonical_input_preprocess_id(OGBN_ARXIV_INTERNAL_NAME), "ogbn_arxiv_raw_float32")
        self.assertEqual(canonical_checkpoint_semantics(OGBN_ARXIV_INTERNAL_NAME)["split_format"], "index_split")

    def test_hyphenated_spelling_is_rejected_as_metadata(self) -> None:
        with self.assertRaises(ValueError):
            canonical_input_preprocess_id(OGBN_ARXIV_EXTERNAL_NAME)
        with self.assertRaises(ValueError):
            canonical_checkpoint_semantics(OGBN_ARXIV_EXTERNAL_NAME)
        with self.assertRaises(ValueError):
            AGGREGATOR.parse_assignment(
                f"{OGBN_ARXIV_EXTERNAL_NAME}=run_manifest.json", "--run"
            )
        with self.assertRaises(ValueError):
            EXPORTER.parse_assignment(
                f"{OGBN_ARXIV_EXTERNAL_NAME}=run_manifest.json", "--run"
            )

    def test_aggregator_accepts_only_internal_identity(self) -> None:
        self.assertEqual(
            AGGREGATOR.parse_assignment(
                f"{OGBN_ARXIV_INTERNAL_NAME}=run_manifest.json", "--run"
            ),
            (OGBN_ARXIV_INTERNAL_NAME, "run_manifest.json"),
        )
        self.assertIn(OGBN_ARXIV_INTERNAL_NAME, AGGREGATOR.CANONICAL_DATASETS)
        self.assertNotIn(OGBN_ARXIV_EXTERNAL_NAME, AGGREGATOR.CANONICAL_DATASETS)
        self.assertEqual(
            EXPORTER.parse_assignment(
                f"{OGBN_ARXIV_INTERNAL_NAME}=run_manifest.json", "--run"
            ),
            (OGBN_ARXIV_INTERNAL_NAME, "run_manifest.json"),
        )

    def test_formal_provenance_and_ogbn_srs_domain_are_canonical(self) -> None:
        artifact_export = ARTIFACT_EXPORT_PATH.read_text(encoding="utf-8")
        self.assertIn('{"config", metrics.config}', artifact_export)
        self.assertIn(
            '{"config_sha256", sha256_file_hex(std::filesystem::path(metrics.config))}',
            artifact_export,
        )
        self.assertEqual(
            AGGREGATOR.EXPECTED_SRS_DOMAINS[OGBN_ARXIV_INTERNAL_NAME]["edge"],
            1 << 24,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
