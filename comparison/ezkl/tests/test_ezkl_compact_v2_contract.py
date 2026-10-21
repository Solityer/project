"""Small synthetic metadata contracts for compact EZKL proofs and GPU budgeting."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import collect_results  # noqa: E402
import formal_common  # noqa: E402
import run_hybrid_formal  # noqa: E402


# Shape-only native admissions captured for the five canonical circuits.
# Tests use these scalar geometry values only; no graph, feature, proof, or
# formal-session artifact is loaded.
FORMAL_GEOMETRY = {
    "cora": (1 << 22, 1 << 25, 9),
    "citeseer": (1 << 24, 1 << 27, 9),
    "pubmed": (1 << 24, 1 << 27, 9),
    "ppi": (1 << 24, 1 << 27, 9),
    "ogbn_arxiv": (1 << 25, 1 << 28, 9),
}


def _ntt_plan(capacity: int, *, base_domain: int | None = None) -> dict[str, int]:
    base_domain = capacity if base_domain is None else base_domain
    return {
        "icicle_ntt_min_log_size": 16,
        "icicle_ntt_max_log_size": 25,
        "icicle_ntt_capacity_elements": capacity,
        "icicle_ntt_device_workspace_upper_bytes": 64 * capacity,
        "icicle_ntt_domain_cache_upper_bytes": 32 * (capacity + 1 + 64 + 6),
        "icicle_ntt_init_scratch_upper_bytes": 4 * 64 * 32,
        "icicle_ntt_device_peak_upper_bytes": 96 * capacity + 10464,
        "icicle_ntt_host_staging_upper_bytes": 32 * capacity,
        "icicle_msm_max_concurrency": 1,
        "icicle_msm_precompute_factor": 4,
        "icicle_msm_precomputed_basis_count": 1,
        "icicle_msm_precomputed_basis_device_upper_bytes": 64 * 4 * base_domain,
        "icicle_msm_point_staging_host_upper_bytes": 64 * min(base_domain, 1 << 20),
    }


class CompactNativeProofContractTest(unittest.TestCase):
    def _compact_fixture(self, root: Path) -> tuple[Path, Path, Path, Path, dict[str, object], dict[str, object]]:
        cell = root / "cora"
        cell.mkdir()
        statement = cell / "statement.json"
        public_output = cell / "public_output.json"
        artifact_manifest = cell / "artifact_manifest.json"
        proof = cell / "proof.json"
        statement.write_bytes(b'{"schema_version":1}\n')
        public_output.write_bytes(b'{"Y":[1,2,3]}\n')
        artifact_manifest.write_bytes(b'{}\n')

        statement_sha = formal_common.sha256(statement)
        output_sha = formal_common.sha256(public_output)
        source_sha = "1" * 64
        binary_sha = "2" * 64
        resource_sha = "3" * 64
        instances = [["1", "2", "3"]]
        instances_sha = formal_common.canonical_json_digest(instances)
        reference = {
            "schema_version": 1,
            "statement_path": "statement.json",
            "statement_sha256": statement_sha,
            "public_output_path": "public_output.json",
            "public_output_sha256": output_sha,
            "artifact_manifest_path": "artifact_manifest.json",
            "source_tree_sha256": source_sha,
            "binary_sha256": binary_sha,
            "resource_manifest_sha256": resource_sha,
            "encoding": "canonical-json",
            "sha256": instances_sha,
            "count": 3,
            "column_counts": [3],
        }
        document: dict[str, object] = {
            "artifact_format": "ezkl-gat-native-proof",
            "schema_version": 2,
            "protocol": None,
            "proof_transcript": {"encoding": "hex-lower", "byte_length": 1, "data": "00"},
            "instances_ref": reference,
            "split": None,
            "timestamp": 1,
            "version": "compact-contract-fixture",
        }
        artifact: dict[str, object] = {
            "statement_sha256": statement_sha,
            "source_tree_sha256": source_sha,
            "binary_sha256": binary_sha,
            "resource_manifest_sha256": resource_sha,
            "files": [
                {"path": "statement.json", "sha256": statement_sha},
                {"path": "public_output.json", "sha256": output_sha},
            ],
        }
        result: dict[str, object] = {
            "proof_instances": {
                "proof_field": "instances_ref",
                "encoding": "canonical-json",
                "sha256": instances_sha,
                "count": 3,
                "column_counts": [3],
                "public_statement_sha256": statement_sha,
                "public_output_sha256": output_sha,
                "native_binding": True,
            },
            "public_statement_sha256": statement_sha,
            "public_output_sha256": output_sha,
        }
        proof.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
        return proof, statement, public_output, artifact_manifest, artifact, result

    def test_five_dataset_ntt_plans_and_integrated_memory_envelopes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-ntt-metadata-") as temporary:
            root = Path(temporary)
            for dataset, (base_domain, proof_domain, degree) in FORMAL_GEOMETRY.items():
                with self.subTest(dataset=dataset):
                    self.assertEqual(proof_domain, 1 << (base_domain * (degree - 1) - 1).bit_length())
                    capacity = min(max(base_domain, proof_domain), 1 << 25)
                    plan = _ntt_plan(capacity, base_domain=base_domain)
                    memory = {
                        "schema": "ezkl-halo2-static-envelope-v20",
                        "proof_domain_size": proof_domain,
                        "size_of_fr_bytes": 32,
                        **plan,
                    }
                    admission = {"dataset": dataset, "domain_size": base_domain, "native_memory_model": memory}
                    admission_path = root / f"{dataset}-admission.json"
                    admission_path.write_text(json.dumps(admission, separators=(",", ":")), encoding="utf-8")
                    budget = {
                        "schema": "ezkl.native-prove-budget.v5",
                        "native_execution_policy": formal_common.FORMAL_GPU_EXECUTION_POLICY,
                        "admission_sha256": formal_common.sha256(admission_path),
                        "icicle_msm_workspace_query_schema": "ezkl.icicle.cub_msm_workspace_query.v3",
                        "icicle_msm_ordinary_factor1_device_workspace_peak_bytes": 2048,
                        "icicle_msm_prepared_factor4_device_workspace_peak_bytes": 3072,
                        "device_workspace_peak_bytes": 3072,
                        **plan,
                    }
                    self.assertEqual(
                        collect_results._validate_native_ntt_profile(
                            admission_path.parent / dataset,
                            admission_path,
                            admission,
                            budget,
                        ),
                        plan,
                    )
                    bad_budget = dict(budget)
                    bad_budget["icicle_ntt_device_peak_upper_bytes"] += 1
                    with self.assertRaises(formal_common.ContractError):
                        collect_results._validate_native_ntt_profile(
                            admission_path.parent / dataset,
                            admission_path,
                            admission,
                            bad_budget,
                        )

            # The task envelope charges host staging in both memory modes and
            # adds one MSM plus one NTT device peak only for unified memory.
            plan = _ntt_plan(1 << 25, base_domain=FORMAL_GEOMETRY["cora"][0])
            base = {
                "proof_nonprepare_heap_host_bytes": 100,
                "proof_nonprepare_hot_mmap_host_bytes": 200,
                "regular_file_mmap_dirty_pagecache_upper_bytes": 300,
                "lookup_table_cache_retained_host_bytes": 400,
                "icicle_msm_host_staging_bytes": 500,
                "icicle_ntt_host_staging_upper_bytes": plan["icicle_ntt_host_staging_upper_bytes"],
                "shplonk_extension_host_upper_bytes": 600,
                "lookup_prepare_task_host_bytes": 70,
                "lookup_commit_task_host_bytes": 80,
                "shplonk_fixed_workspace_host_bytes": 90,
                "shplonk_task_host_bytes": 20,
                "icicle_ntt_device_peak_upper_bytes": plan["icicle_ntt_device_peak_upper_bytes"],
                "icicle_msm_max_concurrency": 1,
                **plan,
            }
            minimum_task = 110
            reserve = 700
            msm_device = 2_000
            for integrated in (False, True):
                device_headroom = (
                    msm_device
                    + plan["icicle_ntt_device_peak_upper_bytes"]
                    + plan["icicle_msm_precomputed_basis_device_upper_bytes"]
                    if integrated else 0
                )
                initial_available = (
                    base["proof_nonprepare_heap_host_bytes"]
                    + base["proof_nonprepare_hot_mmap_host_bytes"]
                    + base["regular_file_mmap_dirty_pagecache_upper_bytes"]
                    + base["lookup_table_cache_retained_host_bytes"]
                    + base["shplonk_extension_host_upper_bytes"]
                    + base["icicle_msm_host_staging_bytes"]
                    + base["icicle_ntt_host_staging_upper_bytes"]
                    + device_headroom
                    + reserve
                    + 3 * minimum_task
                )
                with self.subTest(integrated_device=integrated), \
                     mock.patch("run_hybrid_formal._effective_host_memory_limit_bytes", return_value=(1 << 50, {})), \
                     mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(reserve, {})):
                    budget = run_hybrid_formal._native_prepare_budget(
                        base,
                        initial_available_bytes=initial_available,
                        workspace_query={"integrated_device": integrated, "device_workspace_peak_bytes": msm_device},
                    )
                    self.assertEqual(budget["taskcap_bytes"], 3 * minimum_task)
                    self.assertEqual(budget["ntt_device_workspace_upper_bytes"], plan["icicle_ntt_device_peak_upper_bytes"])
                    self.assertEqual(budget["icicle_ntt_host_staging_upper_bytes"], plan["icicle_ntt_host_staging_upper_bytes"])
                    self.assertEqual(budget["device_headroom_bytes"], device_headroom)
                    exports = run_hybrid_formal._validated_native_prepare_exports(
                        budget,
                        [1],
                        domain_size=FORMAL_GEOMETRY["cora"][0],
                        expected_lookup_groups=1,
                    )
                    self.assertEqual(exports, (str(3 * minimum_task), str(device_headroom), "1"))
                    invalid_budget = dict(budget)
                    invalid_budget["unplanned_extra"] = 0
                    with self.assertRaises(formal_common.ContractError):
                        run_hybrid_formal._validated_native_prepare_exports(
                            invalid_budget,
                            [1],
                            domain_size=FORMAL_GEOMETRY["cora"][0],
                            expected_lookup_groups=1,
                        )

    def test_compact_v2_reference_metadata_and_bad_hashes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-proof-v2-metadata-") as temporary:
            root = Path(temporary)
            proof, statement, public_output, artifact_manifest, artifact, result = self._compact_fixture(root)
            collect_results.validate_proof_instances(
                result,
                proof,
                statement=statement,
                public_output=public_output,
                artifact_manifest=artifact_manifest,
                artifact=artifact,
            )
            original = json.loads(proof.read_text(encoding="utf-8"))
            large_split = copy.deepcopy(original)
            large_split["split"] = {"start": 0, "end": formal_common.NATIVE_USIZE_MAX}
            proof.write_text(json.dumps(large_split, separators=(",", ":")), encoding="utf-8")
            collect_results.validate_proof_instances(
                result,
                proof,
                statement=statement,
                public_output=public_output,
                artifact_manifest=artifact_manifest,
                artifact=artifact,
            )

            mutations = {
                "unknown_top_level_key": lambda doc: doc.update({"legacy_instances": []}),
                "non_null_protocol": lambda doc: doc.update({"protocol": {}}),
                "reversed_split": lambda doc: doc.update({"split": {"start": 2, "end": 1}}),
                "unknown_split_key": lambda doc: doc.update({"split": {"start": 0, "end": 0, "extra": 0}}),
                "empty_version": lambda doc: doc.update({"version": ""}),
                "oversized_version": lambda doc: doc.update({"version": "é" * 129}),
                "oversized_timestamp": lambda doc: doc.update({"timestamp": 1 << 128}),
                "oversized_transcript_length": lambda doc: doc["proof_transcript"].update({"byte_length": formal_common.NATIVE_USIZE_MAX + 1}),
                "wrong_statement_hash": lambda doc: doc["instances_ref"].update({"statement_sha256": "f" * 64}),
                "wrong_instance_digest": lambda doc: doc["instances_ref"].update({"sha256": "0" * 64}),
                "wrong_instance_count": lambda doc: doc["instances_ref"].update({"count": 4, "column_counts": [4]}),
                "oversized_instance_count": lambda doc: doc["instances_ref"].update({"count": formal_common.NATIVE_USIZE_MAX + 1, "column_counts": [formal_common.NATIVE_USIZE_MAX + 1]}),
                "wrong_instance_columns": lambda doc: doc["instances_ref"].update({"column_counts": [2, 1]}),
            }
            for name, mutate in mutations.items():
                with self.subTest(mutation=name):
                    document = copy.deepcopy(original)
                    mutate(document)
                    proof.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
                    with self.assertRaises(formal_common.ContractError):
                        collect_results.validate_proof_instances(
                            result,
                            proof,
                            statement=statement,
                            public_output=public_output,
                            artifact_manifest=artifact_manifest,
                            artifact=artifact,
                        )

    def test_metrics_v4_exact_timing_equations_and_tolerance_reject_drift(self) -> None:
        proof_sha = "a" * 64
        metrics: dict[str, object] = {
            "schema_version": 4,
            "operation": "native_halo2_initial_instance_and_phase0_advice_commitments",
            "included_in": "native_create_proof_wall_ms",
            "measurement_scope": "pre_challenge_public_instance_and_phase0_advice_kzg_commitments_plus_affine_normalization_included_in_native_create_proof_wall_ms",
            "prove_measurement_scope": "native_create_proof_residual_excluding_initial_commitments_plus_authenticated_fixed_value_decode_coefficient_ifft_and_boundary_rebuild",
            "prove_included_in": "native_create_proof_residual_plus_authenticated_pk_materialization",
            "native_pre_create_proof_wall_ms": 1.0,
            "native_post_create_cleanup_wall_ms": 0.0,
            "native_proof_artifact_finalize_wall_ms": 1.0,
            "commitment_time_ms": 3.0,
            "native_create_proof_wall_ms": 8.0,
            "native_create_proof_residual_ms": 5.0,
            "pk_materialize_time_ms": 2.0,
            "prove_time_ms": 7.0,
            "proof_sha256": proof_sha,
            "instance_commitment_count": 1,
            "phase0_advice_commitment_count": 2,
            "affine_normalization_batch_count": 1,
        }
        with tempfile.TemporaryDirectory(prefix="ezkl-metrics-v4-") as temporary:
            path = Path(temporary) / "native_proof_metrics.json"
            path.write_text(json.dumps(metrics), encoding="utf-8")
            formal_common.validate_native_proof_metrics(path, proof_sha256=proof_sha)
            invalid = dict(metrics)
            invalid["native_create_proof_wall_ms"] = 8.00001
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(formal_common.ContractError, "does not equal commitment plus exclusive residual"):
                formal_common.validate_native_proof_metrics(path, proof_sha256=proof_sha)


if __name__ == "__main__":
    unittest.main()
