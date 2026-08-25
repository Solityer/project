"""Lightweight result/resource contract tests; no proof or dataset execution."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from resource_policy import (  # noqa: E402
    CELL_ARTIFACTS,
    atomic_write_json,
    manifest_payload,
    validate_cell_inventory,
)
from collect_results import selected_cells  # noqa: E402
from validate_final_results import validate_matrix_records  # noqa: E402


DATASETS = (
    "cora_full",
    "citeseer_full",
    "pubmed_full",
    "ppi_full_formal",
    "ogbn_arxiv_full",
)


def matrix_fixture() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for dataset in DATASETS:
        for device in ("cpu", "cuda"):
            record: dict[str, object] = {
                "fixture_kind": "tests_only_mock_contract_v1",
                "dataset": dataset,
                "device": device,
                "verified_flag": True,
                "verifier_backend": "cpu_independent",
                "gpu_backend_execution": (
                    "cuda_icicle_prove_only" if device == "cuda" else "not_applicable"
                ),
                "gpu_prove_telemetry": "not_applicable",
                "attestation": {
                    "build_identity_bound": True,
                    "source_bound": True,
                    "source_srs_k": 28,
                    "config_bound": True,
                    "resource_bound": True,
                    "proof_size_bytes": 1,
                    "proof_sha256": "sha256:fixture",
                },
            }
            if device == "cuda":
                record["gpu_prove_telemetry"] = {
                    "schema": "ezkl_cuda_kernel_telemetry_v1",
                    "scope": "formal_prove",
                    "kernel_count": 1,
                    "dropped_records": 0,
                    "active_device_id": 0,
                    "scope_nonce": "fixture",
                    "scope_start_ns": 1,
                    "scope_end_ns": 2,
                    "kernel_names": [{"name": "fixture", "category": "fixture", "count": 1, "device_id": 0}],
                }
            records.append(record)
    return records


class ResultContractTests(unittest.TestCase):
    def test_collector_selection_marks_partial_matrix_without_relaxing_cell_identity(self) -> None:
        selected = selected_cells(("cora_full", "pubmed_full"), ("cpu",))
        self.assertEqual(selected, {("cora_full", "cpu"), ("pubmed_full", "cpu")})
        self.assertNotEqual(selected, selected_cells())

    def test_final_validator_fixture_covers_complete_matrix_contract(self) -> None:
        self.assertEqual(validate_matrix_records(matrix_fixture(), fixture=True), [])

    def test_final_validator_fixture_rejects_missing_unknown_duplicate_and_tamper(self) -> None:
        records = matrix_fixture()
        errors = validate_matrix_records(records[:-1], fixture=True)
        self.assertTrue(any("missing required matrix cell" in item for item in errors))
        unknown = matrix_fixture() + [{"fixture_kind": "tests_only_mock_contract_v1", "dataset": "unknown", "device": "cpu"}]
        self.assertTrue(any("unknown matrix cell" in item for item in validate_matrix_records(unknown, fixture=True)))
        duplicate = matrix_fixture() + [matrix_fixture()[0]]
        self.assertTrue(any("duplicate matrix cell" in item for item in validate_matrix_records(duplicate, fixture=True)))
        tampered = matrix_fixture()
        tampered[0]["attestation"]["source_srs_k"] = 27  # type: ignore[index]
        self.assertTrue(any("source SRS k mismatch" in item for item in validate_matrix_records(tampered, fixture=True)))

    def test_final_validator_fixture_rejects_verified_flag_and_cpu_gpu_tamper(self) -> None:
        records = matrix_fixture()
        records[0]["verified_flag"] = False
        records[0]["gpu_prove_telemetry"] = {"unexpected": True}
        self.assertTrue(any("not verified" in item for item in validate_matrix_records(records, fixture=True)))
        self.assertTrue(any("CPU fixture carries GPU telemetry" in item for item in validate_matrix_records(records, fixture=True)))
        records = matrix_fixture()
        records[-1]["gpu_prove_telemetry"]["kernel_count"] = 0  # type: ignore[index]
        self.assertTrue(any("telemetry/verifier contract mismatch" in item for item in validate_matrix_records(records, fixture=True)))

    def test_final_validator_fixture_rejects_build_source_config_resource_size_digest_tamper(self) -> None:
        fields = (
            ("build_identity_bound", "build identity"),
            ("source_bound", "source binding"),
            ("config_bound", "config binding"),
            ("resource_bound", "resource binding"),
        )
        for field, marker in fields:
            records = matrix_fixture()
            records[0]["attestation"][field] = False  # type: ignore[index]
            self.assertTrue(any(marker in item for item in validate_matrix_records(records, fixture=True)))
        for field, expected_marker in (("proof_size_bytes", "proof size"), ("proof_sha256", "proof digest")):
            records = matrix_fixture()
            records[0]["attestation"][field] = 0 if field == "proof_size_bytes" else "tampered"  # type: ignore[index]
            self.assertTrue(any(expected_marker in item for item in validate_matrix_records(records, fixture=True)))

    def test_exact_inventory_accepts_fixture_without_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            cell.mkdir()
            for name in CELL_ARTIFACTS:
                (cell / name).write_bytes(b"fixture")
            validate_cell_inventory(cell)

    def test_inventory_rejects_extra_directory_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            cell.mkdir()
            for name in CELL_ARTIFACTS:
                (cell / name).write_bytes(b"fixture")
            (cell / "unexpected").mkdir()
            with self.assertRaises(ValueError):
                validate_cell_inventory(cell)
            (cell / "unexpected").rmdir()
            (cell / "unexpected").symlink_to(cell / "run_manifest.json")
            with self.assertRaises(ValueError):
                validate_cell_inventory(cell)

    def test_atomic_json_rejects_symlink_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("original\n", encoding="utf-8")
            link = root / "result.json"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                atomic_write_json(link, {"status": "bad"})
            self.assertEqual(target.read_text(encoding="utf-8"), "original\n")

    def test_manifest_is_baseline_local_and_complete(self) -> None:
        baseline = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=baseline / "tests") as temporary:
            root = Path(temporary) / "resource-root"
            root.mkdir()
            # The hardened lexical path checker intentionally requires every
            # existing parent component before it permits a missing leaf.
            # Build only the baseline-local fixture roots; no production
            # resources or result cells are created by this test.
            for relative in (
                "configs",
                "data/cache",
                "artifacts/checkpoints",
                "resources/public_y",
            ):
                (root / relative).mkdir(parents=True, exist_ok=True)
            payload = manifest_payload(root)
        self.assertEqual(payload["schema"], "ezkl_formal_resource_manifest_v1")
        self.assertEqual(len(payload["datasets"]), 5)
        self.assertEqual(payload["output_policy"]["cell_root"], "results/runs/<backend>/<dataset>")


if __name__ == "__main__":
    unittest.main()
