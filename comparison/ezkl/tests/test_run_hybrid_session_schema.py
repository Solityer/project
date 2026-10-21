"""Regression checks for the EZKL-local formal entry and command plan."""

from __future__ import annotations

import fcntl
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import formal_common  # noqa: E402
import collect_results  # noqa: E402
import run_formal_matrix  # noqa: E402
from run_hybrid_formal import (  # noqa: E402
    DATASETS,
    NATIVE_FORMAL_BACKEND,
    _CUB_QUERY_OPERATION_NAMES,
    _CUB_QUERY_PATHS,
    _admit_local_capacity,
    _authenticated_srs_copy_size_bytes,
    _check_dataset_capacity,
    _disk_budget_components,
    _formal_dataset_command_plan,
    _native_capacity_requirements,
    _native_json_output_bounds,
    _main,
    _native_prepare_budget,
    _rescaled_f64_json_atom_upper,
    _require_formal_dataset_completion,
    _pending_srs_copy_disk_budget,
    _pending_json_output_disk_budget,
    _recheck_local_phase_capacity,
    _remaining_setup_disk_bytes,
    _run_five_admission_preflight,
    _run_five_native_capacity_preflight,
    _run_dataset_cells,
    _set_stage,
    _validate_cub_workspace_query,
    _validate_native_cub_workspace_receipt,
)


class LocalFormalEntryTest(unittest.TestCase):
    def test_native_pk_v4_collector_accepts_producer_bitmap_budget_and_rejects_stale_budget(self) -> None:
        domain_pages = 40
        permutation_cells = domain_pages * 4096 * 8
        admission = {
            "domain_size": permutation_cells,
            "constraint_system": {"num_permutation_columns": 1},
        }
        for successor_edges, expected_backing in ((3, 3 * 4096), (45, domain_pages * 4096)):
            with self.subTest(successor_edges=successor_edges):
                producer_geometry = formal_common.native_pk_v4_read_backing_geometry(
                    permutation_cells, successor_edges
                )
                self.assertEqual(producer_geometry["target_bitmap_backing_bytes"], expected_backing)
                self.assertEqual(producer_geometry["target_bitmap_allocation_bitmap_bytes"], 5)
                producer_budget = {
                    "native_pk_read_backing_required_bytes": expected_backing,
                    "max_total_backing_bytes": expected_backing,
                    "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes": 5,
                }
                self.assertEqual(
                    collect_results._validate_native_pk_v4_read_backing_budget(
                        admission, successor_edges, producer_budget
                    ),
                    producer_geometry,
                )
                legacy_backing = (
                    (16 * successor_edges + 4095) // 4096
                    + (successor_edges + 32767) // 32768
                ) * 4096
                legacy_budget = dict(producer_budget)
                legacy_budget["native_pk_read_backing_required_bytes"] = legacy_backing
                legacy_budget["max_total_backing_bytes"] = legacy_backing
                with self.assertRaisesRegex(formal_common.ContractError, "v4 PK reader's target-bitmap backing"):
                    collect_results._validate_native_pk_v4_read_backing_budget(
                        admission, successor_edges, legacy_budget
                    )
                stale_budget = dict(producer_budget)
                stale_budget["native_pk_read_backing_required_bytes"] = expected_backing + 4096
                stale_budget["max_total_backing_bytes"] = expected_backing + 4096
                with self.assertRaisesRegex(formal_common.ContractError, "v4 PK reader's target-bitmap backing"):
                    collect_results._validate_native_pk_v4_read_backing_budget(
                        admission, successor_edges, stale_budget
                    )

    def test_unknown_v4_payload_report_still_runs_actual_setup_capacity_gate(self) -> None:
        session_id = "a" * 32
        workspace_receipt = {
            "session_id": session_id,
            "query": {
                "native_type_sizes": {"scalar_bytes": 32, "affine_bytes": 64},
                "device_workspace_peak_bytes": 1,
                "integrated_device": False,
            },
        }
        capacity = {
            "setup_key_artifact_upper_bytes": 10,
            "setup_mapped_peak_bytes": 4,
            "proof_spill_known_bytes": 2,
            "host_known_bytes": 8,
        }

        def run_case(root: Path, available_bytes: int) -> tuple[dict[str, object], Path]:
            output_root = root / "runs/formal"
            session_dir = output_root / ".sessions" / session_id
            session_dir.mkdir(parents=True)
            session_file = output_root / ".sessions" / f"{session_id}.json"
            session_file.write_text('{"capacity_checks": []}\n', encoding="utf-8")
            output_root.parent.mkdir(parents=True, exist_ok=True)
            prepared_dir = root.relative_to(ROOT).as_posix()

            class CapacityApi:
                @staticmethod
                def audit_fresh_admission(_session_file: Path, dataset: str, *_args: object, **_kwargs: object):
                    return {"dataset": dataset}

                @staticmethod
                def write_and_require_fresh_report(
                    _session_file: Path,
                    _records: list[dict[str, object]],
                    report_path: Path,
                    **_kwargs: object,
                ):
                    report = {
                        "gate_status": "current_v4_pk_payload_lower_bound_unavailable",
                        "readiness": "NOT_ESTABLISHED",
                    }
                    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
                    return report

            def admit_one(_binary: Path, _srs: Path, _root: Path, _session: str, _device: dict[str, object], *, datasets: tuple[str, ...], **_kwargs: object):
                dataset = datasets[0]
                return {
                    dataset: {
                        "dataset": dataset,
                        "prepared_dir": prepared_dir,
                        "sha256": "d" * 64,
                    }
                }

            with mock.patch("run_hybrid_formal._capacity_precheck_api", return_value=CapacityApi()), \
                 mock.patch("run_hybrid_formal.native_admit_all_datasets", side_effect=admit_one), \
                 mock.patch("run_hybrid_formal._build_native_cub_workspace_query", return_value=workspace_receipt), \
                 mock.patch("run_hybrid_formal._admit_local_capacity", return_value=(workspace_receipt, {name: dict(capacity) for name in DATASETS})), \
                 mock.patch("run_hybrid_formal._sample_native_cuda_memory", return_value={"free_memory_bytes": 100, "total_memory_bytes": 200, "device_uuid": "GPU-test"}), \
                 mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(100, {})), \
                 mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(0, {})), \
                 mock.patch("run_hybrid_formal._filesystem_available_bytes", return_value=(available_bytes, {"available_bytes": available_bytes, "device_id": "test-device"})), \
                 mock.patch("run_hybrid_formal._filesystem_type", return_value="ext4"):
                result = _run_five_native_capacity_preflight(
                    binary=root / "ezkl",
                    srs=root / "srs.kzg",
                    output_root=output_root,
                    session_id=session_id,
                    session_file=session_file,
                    session_dir=session_dir,
                    authenticated_context=object(),
                    selected_device={"uuid": "GPU-test"},
                    saved_admissions={},
                    runtime_identity={},
                    build_environment={},
                    gpu_identity={},
                    resume_identity={},
                    saved_setup_anchors={},
                    archived_outputs=False,
                    is_resume=False,
                    completed_datasets=set(),
                    workspace_receipt=None,
                    srs_copy_size_bytes=5,
                )
            return {"result": result, "session_file": session_file}, session_dir

        with tempfile.TemporaryDirectory(prefix="ezkl-unknown-pk-gate-", dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            response, session_dir = run_case(root, 1000)
            state = json.loads(response["session_file"].read_text(encoding="utf-8"))
            self.assertEqual(
                state["capacity_admission_attempts"]["gate_status"],
                "current_v4_pk_payload_lower_bound_unavailable",
            )
            self.assertEqual(
                [row["phase"] for row in state["capacity_checks"]],
                ["setup"] * len(DATASETS),
            )
            self.assertTrue(all(dataset in response["result"][0] for dataset in DATASETS))
            self.assertFalse(any((root / "runs/formal" / dataset / "pk.key").exists() for dataset in DATASETS))
            self.assertTrue((session_dir / "capacity/resource_capacity_precheck.json").is_file())

        with tempfile.TemporaryDirectory(prefix="ezkl-unknown-pk-gate-fail-", dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            try:
                run_case(root, 78)
            except formal_common.ContractError as exc:
                self.assertIn("mapped/key artifact gate", str(exc))
            else:
                self.fail("unknown PK payload report bypassed the mandatory native Setup capacity gate")

    @staticmethod
    def _capacity_admission() -> dict[str, object]:
        return {
            "logrows": 4,
            "domain_size": 16,
            "constraint_system": {
                "degree": 3,
                "num_fixed_columns": 3,
                "num_advice_columns": 4,
                "num_instance_columns": 1,
                "num_selectors": 2,
                "num_custom_gate_groups": 1,
                "num_blinding_factors": 2,
                "num_lookup_arguments": 2,
                "num_lookup_query_polynomials": 2,
                "max_lookup_query_polynomials_per_argument": 1,
                "num_unique_lookup_table_expression_groups": 1,
                "num_shuffle_arguments": 1,
                "num_permutation_columns": 2,
                "fixed_column_count_post_compression": 5,
                "permutation_copy_call_count": 4,
                "permutation_successor_entry_count_upper": 8,
            },
            "native_memory_model": {
                "native_target_arch": "aarch64",
                "uses_recursive_fft": True,
                "proof_logrows": 5,
                "proof_domain_size": 32,
                "icicle_ntt_min_log_size": 16,
                "icicle_ntt_max_log_size": 25,
                "icicle_ntt_capacity_elements": 32,
                "icicle_ntt_device_workspace_upper_bytes": 64 * 32,
                "icicle_ntt_domain_cache_upper_bytes": 32 * (32 + 1 + 64 + 6),
                "icicle_ntt_init_scratch_upper_bytes": 4 * 64 * 32,
                "icicle_ntt_device_peak_upper_bytes": 96 * 32 + 10464,
                "icicle_ntt_host_staging_upper_bytes": 32 * 32,
                "icicle_msm_max_concurrency": 1,
                "icicle_msm_precompute_factor": 4,
                "icicle_msm_precomputed_basis_count": 1,
                "icicle_msm_precomputed_basis_device_upper_bytes": 16 * 64 * 4,
                "icicle_msm_point_staging_host_upper_bytes": 16 * 64,
                "recursive_fft_cached_domain_lengths": [32],
                "recursive_fft_twiddle_field_elements": 74,
                "fft_cache_supported_length_count": 2,
                "fft_cache_entry_capacity": 3,
                "fft_parallel_thread_count": 2,
                "srs_fft_cache_supported_length_count": 1,
                "srs_fft_cache_entry_capacity": 3,
                "parallel_fft_lookup_table_elements": 1056,
                "size_of_fftdata_bytes": 96,
                "size_of_fftdata_once_lock_bytes": 128,
                "size_of_fftdata_cache_entry_bytes": 24,
                "size_of_fftdata_cache_map_bytes": 48,
                "size_of_arc_header_bytes": 16,
                "custom_gate_dependency_columns": [2],
                "lookup_dependency_columns": [2, 2],
                "lookup_query_polynomial_counts": [1, 1],
                "shuffle_dependency_columns": [2],
                "size_of_fr_bytes": 32,
                "size_of_assigned_fr_bytes": 40,
                "size_of_option_fr_bytes": 48,
                "size_of_batch_invert_entry_bytes": 40,
                "size_of_usize_bytes": 8,
                "size_of_usize_pair_bytes": 16,
                "size_of_permutation_hashmap_header_bytes": 48,
                "size_of_atomic_u64_bytes": 8,
                "size_of_lookup_map_entry_bytes": 32,
                "size_of_lookup_map_header_bytes": 24,
                "size_of_vec_u8_bytes": 24,
                "size_of_lookup_repr_bytes": 32,
                "size_of_vec_header_bytes": 24,
                "size_of_polynomial_header_bytes": 48,
                "size_of_mv_lookup_prepared_bytes": 120,
                "size_of_mv_lookup_prepared_table_bytes": 48,
                "size_of_mv_lookup_committed_bytes": 96,
                "size_of_shplonk_commitment_extension_bytes": 128,
                "size_of_shplonk_rotation_set_extension_bytes": 48,
                "size_of_coeff_polynomial_header_bytes": 48,
                "shplonk_rotation_set_commitment_counts": [8, 5, 3],
                "shplonk_query_polynomial_count": 16,
                "shplonk_rotation_set_count": 3,
                "shplonk_max_rotation_count": 3,
                "shplonk_unique_opening_point_count": 3,
                "shplonk_max_commitments_per_rotation_set": 8,
                "shplonk_permutation_product_set_count": 2,
                "shplonk_advice_query_polynomial_count": 2,
                "shplonk_fixed_query_polynomial_count": 2,
                "shplonk_lookup_argument_count": 2,
                "shplonk_shuffle_argument_count": 1,
                "shplonk_proof_instance_count": 1,
                "shplonk_quotient_piece_count": 2,
                "size_of_g1_affine_bytes": 96,
                "size_of_g1_projective_bytes": 144,
                "size_of_g2_affine_bytes": 192,
                "size_of_fft_stage_bytes": 24,
                "graph_witness_scalar_element_count": 0,
                "size_of_graph_witness_scalar_bytes": 32,
                "selector_activation_bitmap_bytes": 100,
                "fixed_polynomial_read_heap_scratch_bytes_upper": 4096,
                "lookup_table_cardinality_upper_bounds": [4],
            },
        }

    @staticmethod
    def _cub_query(session_id: str = "a" * 32) -> dict[str, object]:
        selected = {"uuid": "GPU-3cde57ba-ae43-4452-bd59-7b156b36db71", "compute_capability": "10.0"}
        type_sizes = {
            "scalar_bytes": 32,
            "affine_bytes": 64,
            "projective_bytes": 96,
            "unsigned_bytes": 4,
        }

        def path(name: str, peak: int) -> dict[str, object]:
            profile = _CUB_QUERY_PATHS[name]
            bucket_count = profile["B"]
            large_count = profile["L_max"]
            counts = (1 << 24, 1 << 24, bucket_count + 1, bucket_count, bucket_count, large_count)
            return {
                **{key: value for key, value in profile.items() if key != "host_resident_points"},
                "point_upload_bytes": (1 << 20) * type_sizes["affine_bytes"] if profile["host_resident_points"] else 0,
                "precompute_device_temporary_bytes": 0,
                "device_workspace_peak_bytes": peak,
                "queries": [
                    {"name": name, "num_items": count, "scratch_bytes": 1}
                    for name, count in zip(_CUB_QUERY_OPERATION_NAMES, counts, strict=True)
                ],
                "scratch_bytes_sum": len(_CUB_QUERY_OPERATION_NAMES),
            }

        return {
            "schema": "ezkl.icicle.cub_msm_workspace_query.v3",
            "session_id": session_id,
            "icicle_source_tree_sha256": "1" * 64,
            "native_backend_manifest_sha256": "2" * 64,
            "device_uuid": selected["uuid"],
            "free_memory_bytes": 1000,
            "total_memory_bytes": 2000,
            "integrated_device": False,
            "scalar_bits": 254,
            "c": 16,
            "chunk_scalars": 1 << 20,
            "total_window_count": 16,
            "M": 1 << 24,
            "supported_paths": ["ordinary_factor1", "prepared_factor4"],
            "cuda_runtime_version": 12080,
            "cuda_driver_version": 12080,
            "cuda_toolkit_version": "12.8.0",
            "cub_version": 200800,
            "device_compute_capability": selected["compute_capability"],
            "native_type_sizes": type_sizes,
            "ordinary_factor1": path("ordinary_factor1", 100),
            "prepared_factor4": path("prepared_factor4", 99),
            "device_workspace_peak_bytes": 100,
        }

    def test_native_capacity_model_counts_setup_and_quotient_mappings(self) -> None:
        requirements = _native_capacity_requirements(
            self._capacity_admission(), {"scalar_bytes": 32, "affine_bytes": 64}
        )
        self.assertEqual(requirements["setup_persisted_mapped_bytes"], (2 * 5 * 16 + 2 * 16 + 3 * 32) * 32)
        self.assertEqual(requirements["setup_transition_mapped_bytes"], (32 + 16) * 32)
        self.assertEqual(
            requirements["setup_mapped_peak_bytes"],
            3 * 16 * 40 + (2 * 5 * 16 + 2 * 16 + 3 * 32) * 32 + (32 + 16) * 32,
        )
        self.assertEqual(requirements["quotient_extended_backing_bytes"], 32 * 32)
        self.assertEqual(requirements["quotient_piece_bytes"], 16 * (3 - 1) * 32)
        self.assertEqual(
            requirements["proof_spill_known_bytes"],
            requirements["setup_persisted_mapped_bytes"]
            + 4 * 16 * 32
            + 2 * 16 * 32
            + 32 * 32
            + 16 * 2 * 32
            + requirements["lookup_spill_known_bytes"]
            + 5 * 32 * 32
            + requirements["shplonk_transient_spill_upper_bytes"],
        )
        self.assertEqual(requirements["proof_evaluation_coset_mapped_window_bytes"], 5 * 32 * 32)
        self.assertEqual(requirements["proof_nonprepare_hot_mmap_host_bytes"], 3 * 32 * 32)
        self.assertEqual(
            requirements["proof_evaluation_hot_mmap_host_bytes"],
            4 * 32 * 32 + 5 * 32 * 32 + 4 * 16 * 32,
        )
        self.assertEqual(requirements["batch_invert_heap_scratch_bytes_upper"], 16 * 40)
        self.assertEqual(
            requirements["lookup_commit_task_host_bytes"],
            16 * (max(2 * 32 + 40, 3 * 32) + 3 * 32) + 5 * 24,
        )
        self.assertEqual(requirements["icicle_msm_host_staging_bytes"], (32 + 64) * (1 << 20))
        self.assertEqual(requirements["shplonk_batch_width_upper"], 2)
        self.assertEqual(requirements["shplonk_task_host_bytes"], 2 * 16 * 32 + 2 * 48)
        self.assertEqual(requirements["shplonk_fixed_workspace_host_bytes"], 4 * 16 * 32 + 4 * 48)
        self.assertEqual(requirements["shplonk_transient_spill_upper_bytes"], 4 * 16 * 32)
        self.assertEqual(
            requirements["proof_host_peak_bytes"],
            max(
                requirements["lookup_prepare_host_peak_bytes"],
                requirements["lookup_commit_host_peak_bytes"],
                requirements["proof_evaluation_host_peak_bytes"],
                requirements["shplonk_host_peak_bytes"],
            ),
        )
        self.assertEqual(
            requirements["permutation_vk_msm_host_known_bytes"],
            100
            + 2 * 16 * 32
            + (32 + 64) * (1 << 20)
            + requirements["permutation_hashmap_heap_peak_bytes"],
        )
        self.assertEqual(
            requirements["host_known_bytes"],
            max(
                requirements["setup_keygen_host_peak_bytes"],
                requirements["srs_setup_host_peak_bytes"],
                requirements["proof_host_peak_bytes"],
            ),
        )
        self.assertGreater(requirements["setup_key_artifact_upper_bytes"], requirements["setup_persisted_mapped_bytes"])

    def test_capacity_gate_accepts_exact_limits_and_rejects_each_shortfall(self) -> None:
        requirements = {"host_known_bytes": 100}
        output_root = Path("/capacity/formal")
        sample = {"free_memory_bytes": 100, "total_memory_bytes": 200, "device_uuid": "GPU-test"}
        query = {"device_workspace_peak_bytes": 100, "integrated_device": False}
        with mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(100, {})), \
             mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(0, {})), \
             mock.patch("run_hybrid_formal._filesystem_available_bytes", return_value=(100, {})):
            check = _check_dataset_capacity(
                "cora", requirements, output_root=output_root, remaining_known_disk_bytes=100,
                phase="setup", device_sample=sample, workspace_query=query,
            )
        self.assertEqual(check["host_required_known_bytes"], 100)
        self.assertEqual(check["disk_required_known_bytes"], 100)
        self.assertEqual(check["cuda_required_workspace_bytes"], 100)

        for host, disk, free, expected in (
            (99, 100, 100, "host allocation"),
            (100, 99, 100, "mapped/key artifact"),
            (100, 100, 99, "CUDA bytes"),
        ):
            with self.subTest(expected=expected), \
                 mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(host, {})), \
                 mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(0, {})), \
                 mock.patch("run_hybrid_formal._filesystem_available_bytes", return_value=(disk, {})):
                with self.assertRaisesRegex(formal_common.ContractError, expected):
                    _check_dataset_capacity(
                        "cora", requirements, output_root=output_root, remaining_known_disk_bytes=100,
                        phase="setup", device_sample={**sample, "free_memory_bytes": free},
                        workspace_query=query,
                )

    def test_native_srs_copy_budget_counts_pending_cells_and_fails_closed_at_disk_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-srs-budget-") as temporary:
            srs_path = Path(temporary) / "trusted.srs"
            srs_path.write_bytes(b"1234567")
            source_bytes = _authenticated_srs_copy_size_bytes(srs_path, expected_size_bytes=7)
            self.assertEqual(source_bytes, 7)
            with self.assertRaisesRegex(formal_common.ContractError, "differs from its validated manifest"):
                _authenticated_srs_copy_size_bytes(srs_path, expected_size_bytes=8)

            all_pending = _pending_srs_copy_disk_budget(source_bytes, set(DATASETS))
            self.assertEqual(all_pending["pending_output_srs_copy_count"], 5)
            self.assertEqual(all_pending["pending_output_srs_copy_bytes"], 35)
            partially_completed = _pending_srs_copy_disk_budget(source_bytes, set(DATASETS[2:]))
            self.assertEqual(partially_completed["pending_output_srs_copy_count"], 3)
            self.assertEqual(partially_completed["pending_output_srs_copy_bytes"], 21)
            self.assertEqual(partially_completed["pending_output_srs_copy_datasets"], list(DATASETS[2:]))

            components = _disk_budget_components(
                pending_srs=partially_completed,
                setup_key_artifact_upper_bytes=10,
                setup_mapped_peak_bytes=4,
            )
            self.assertEqual(components["total_required_bytes"], 35)
            requirements = {"host_known_bytes": 1}
            sample = {"free_memory_bytes": 100, "total_memory_bytes": 200, "device_uuid": "GPU-test"}
            query = {"device_workspace_peak_bytes": 1, "integrated_device": False}
            for available in (35, 34):
                with self.subTest(disk_available=available), \
                     mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(100, {})), \
                     mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(0, {})), \
                     mock.patch("run_hybrid_formal._filesystem_available_bytes", return_value=(available, {})):
                    if available == 35:
                        check = _check_dataset_capacity(
                            "cora", requirements, output_root=Path("/capacity/formal"),
                            remaining_known_disk_bytes=int(components["total_required_bytes"]),
                            phase="setup", device_sample=sample, workspace_query=query,
                            disk_budget_components=components,
                        )
                        self.assertEqual(check["disk_required_known_bytes"], 35)
                        self.assertEqual(check["disk_budget_components"]["pending_output_srs_copy_bytes"], 21)
                    else:
                        with self.assertRaisesRegex(formal_common.ContractError, "mapped/key artifact gate"):
                            _check_dataset_capacity(
                                "cora", requirements, output_root=Path("/capacity/formal"),
                                remaining_known_disk_bytes=int(components["total_required_bytes"]),
                                phase="setup", device_sample=sample, workspace_query=query,
                                disk_budget_components=components,
                            )

    def test_pending_native_witness_and_proof_json_budget_is_cumulative_and_spill_exclusive(self) -> None:
        self.assertEqual(_rescaled_f64_json_atom_upper(0, label="fixture scale"), 42)
        self.assertEqual(_rescaled_f64_json_atom_upper(1000, label="fixture scale"), 324)
        self.assertEqual(_rescaled_f64_json_atom_upper(-1000, label="fixture scale"), 312)
        self.assertEqual(_rescaled_f64_json_atom_upper((1 << 31) - 1, label="fixture scale"), 329)
        self.assertEqual(_rescaled_f64_json_atom_upper(-(1 << 31), label="fixture scale"), 312)
        with tempfile.TemporaryDirectory(prefix="ezkl-json-budget-", dir=ROOT) as temporary:
            prepared = Path(temporary) / "prepared"
            prepared.mkdir()
            settings = {
                "run_args": {
                    "input_visibility": "Public",
                    "output_visibility": "Public",
                    "param_visibility": {"Hashed": {"hash_is_public": True, "outlets": []}},
                },
                "model_instance_shapes": [[2], [1]],
                "model_input_scales": [1000],
                "model_output_scales": [-1000],
                "module_sizes": {"poseidon": [0, [1]], "polycommit": []},
            }
            settings_path = prepared / "settings.json"
            settings_path.write_text(json.dumps(settings, separators=(",", ":")), encoding="utf-8")
            admission = {
                "prepared_dir": prepared.relative_to(ROOT).as_posix(),
                "settings_sha256": formal_common.sha256(settings_path),
                "native_memory_model": {
                    "graph_witness_scalar_element_count": 4,
                    "advice_column_phase_ids": [0],
                    "shplonk_query_polynomial_count": 2,
                    "shplonk_max_rotation_count": 3,
                    "shplonk_lookup_argument_count": 1,
                    "shplonk_shuffle_argument_count": 1,
                    "shplonk_permutation_product_set_count": 2,
                    "shplonk_proof_instance_count": 4,
                    "shplonk_quotient_piece_count": 5,
                },
            }
            bounds = _native_json_output_bounds(admission, "cora")
            self.assertEqual(bounds["public_input_scalar_count"], 2)
            self.assertEqual(bounds["public_output_scalar_count"], 1)
            self.assertEqual(bounds["runtime_parameter_scalar_count"], 1)
            self.assertEqual(bounds["proof_instance_scalar_count"], 4)
            self.assertEqual(bounds["proof_point_write_count_upper"], 21)
            self.assertEqual(bounds["proof_scalar_write_count_upper"], 6)
            self.assertEqual(bounds["proof_transcript_wire_upper_bytes"], 1536)

            pending = _pending_json_output_disk_budget({"cora": bounds}, ("cora",))
            self.assertEqual(pending["pending_json_datasets"], ["cora"])
            witness_bytes = int(pending["pending_witness_json_upper_bytes"])
            proof_bytes = int(pending["pending_proof_json_upper_bytes"])
            self.assertGreater(witness_bytes, 0)
            self.assertGreater(proof_bytes, 0)

            srs_budget = {
                "output_srs_copy_size_bytes": 7,
                "pending_output_srs_copy_count": 1,
                "pending_output_srs_copy_bytes": 7,
                "pending_output_srs_copy_datasets": ["cora"],
            }
            setup_components = _disk_budget_components(
                pending_srs=srs_budget,
                setup_key_artifact_upper_bytes=10,
                setup_mapped_peak_bytes=100,
                pending_witness_json_upper_bytes=witness_bytes,
                pending_proof_json_upper_bytes=proof_bytes,
                pending_json_datasets=["cora"],
            )
            self.assertEqual(setup_components["pending_output_json_upper_bytes"], witness_bytes + proof_bytes)
            self.assertEqual(setup_components["scratch_volume_required_bytes"], 100)

            proof_spill = proof_bytes + 23
            proof_components = _disk_budget_components(
                pending_srs=srs_budget,
                setup_key_artifact_upper_bytes=10,
                proof_spill_known_bytes=proof_spill,
                proof_spill_extra_bytes=23,
                pending_witness_json_upper_bytes=witness_bytes,
                pending_proof_json_upper_bytes=proof_bytes,
                pending_json_datasets=["cora"],
            )
            expected_required = 10 + 7 + witness_bytes + proof_bytes + 23
            self.assertEqual(proof_components["scratch_volume_required_bytes"], 23)
            self.assertEqual(proof_components["total_required_bytes"], expected_required)

            requirements = {"host_known_bytes": 1}
            sample = {"free_memory_bytes": 100, "total_memory_bytes": 200, "device_uuid": "GPU-test"}
            query = {"device_workspace_peak_bytes": 1, "integrated_device": False}

            def disk_available(available: int) -> tuple[int, dict[str, object]]:
                return available, {"device_id": "fixture-fs", "available_bytes": available}

            with mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(100, {})), \
                 mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(0, {})), \
                 mock.patch("run_hybrid_formal._filesystem_available_bytes", side_effect=lambda _path: disk_available(expected_required)), \
                 mock.patch("run_hybrid_formal._filesystem_type", return_value="ext4"):
                check = _check_dataset_capacity(
                    "cora", requirements, output_root=Path("/capacity/formal"),
                    remaining_known_disk_bytes=expected_required, phase="witness_prove_verify",
                    device_sample=sample, workspace_query=query, disk_budget_components=proof_components,
                )
                self.assertEqual(check["disk_required_known_bytes"], expected_required)

            with mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(100, {})), \
                 mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(0, {})), \
                 mock.patch("run_hybrid_formal._filesystem_available_bytes", side_effect=lambda _path: disk_available(expected_required - 1)), \
                 mock.patch("run_hybrid_formal._filesystem_type", return_value="ext4"):
                with self.assertRaisesRegex(formal_common.ContractError, "mapped/key artifact gate"):
                    _check_dataset_capacity(
                        "cora", requirements, output_root=Path("/capacity/formal"),
                        remaining_known_disk_bytes=expected_required, phase="witness_prove_verify",
                        device_sample=sample, workspace_query=query, disk_budget_components=proof_components,
                    )

    def test_formal_initial_gate_consumes_pending_srs_copies_after_completed_cell_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-srs-initial-gate-") as temporary:
            srs_path = Path(temporary) / "trusted.srs"
            srs_path.write_bytes(b"1234567")
            query = {
                "device_uuid": "GPU-test",
                "free_memory_bytes": 100,
                "total_memory_bytes": 200,
                "integrated_device": False,
                "device_workspace_peak_bytes": 1,
                "native_type_sizes": {"scalar_bytes": 32, "affine_bytes": 64},
            }
            receipt = {"query": query, "result_sha256": "a" * 64}
            admissions = {
                name: {"native_memory_model": {"size_of_fr_bytes": 32, "cuda_total_memory_bytes": 200}}
                for name in DATASETS
            }
            per_dataset = {
                "setup_key_artifact_upper_bytes": 10,
                "setup_mapped_peak_bytes": 20,
                "proof_host_peak_bytes": 1,
                "host_known_bytes": 2,
            }
            active = set(DATASETS[2:])
            setup_required = {DATASETS[2], DATASETS[4]}
            with mock.patch("run_hybrid_formal._build_native_cub_workspace_query", return_value=receipt), \
                 mock.patch("run_hybrid_formal._validate_native_cub_workspace_receipt", return_value=receipt), \
                 mock.patch("run_hybrid_formal._native_capacity_requirements", return_value=per_dataset), \
                 mock.patch("run_hybrid_formal._check_dataset_capacity", side_effect=lambda *_args, **kwargs: {
                     "dataset": _args[0], "disk_required_known_bytes": kwargs["remaining_known_disk_bytes"],
                     "disk_budget_components": kwargs["disk_budget_components"],
                 }) as gate:
                _receipt, _requirements, profile = _admit_local_capacity(
                    admissions,
                    output_root=Path(temporary) / "runs/formal",
                    session_dir=Path(temporary) / "session",
                    session_id="a" * 32,
                    runtime_identity={},
                    selected_device={"uuid": "GPU-test"},
                    build_environment={},
                    gpu_identity={},
                    srs_copy_size_bytes=7,
                    active_datasets=active,
                    setup_required_datasets=setup_required,
                    pending_finalize_datasets=active,
                )
            self.assertEqual(gate.call_count, 3)
            # Two setup key artifacts + one setup peak map + three pending output copies.
            self.assertEqual(profile["initial_disk_budget_components"]["total_required_bytes"], 61)
            self.assertEqual(profile["initial_disk_budget_components"]["pending_output_srs_copy_count"], 3)
            self.assertEqual(profile["pending_finalize_datasets"], list(DATASETS[2:]))
            self.assertTrue(all(call.kwargs["remaining_known_disk_bytes"] == 61 for call in gate.call_args_list))
            self.assertTrue(all(call.kwargs["disk_budget_components"]["pending_output_srs_copy_count"] == 3 for call in gate.call_args_list))

    def test_five_dataset_capacity_preflight_precedes_cells_and_preserves_resume_scope(self) -> None:
        session_id = "a" * 32
        native_sizes = {"scalar_bytes": 32, "affine_bytes": 64}
        admissions = {dataset: {"dataset": dataset} for dataset in DATASETS}
        workspace_receipt = {
            "session_id": session_id,
            "query": {"native_type_sizes": native_sizes},
        }
        base_capacity = {
            "setup_key_artifact_upper_bytes": 10,
            "setup_mapped_peak_bytes": 7,
            "proof_spill_known_bytes": 4,
            "proof_host_peak_bytes": 20,
            "gat_transient_witness_bytes_upper": 9,
        }

        def run_preflight(
            root: Path,
            *,
            completed: set[str] | None = None,
            reused: set[str] | None = None,
            final_reused: set[str] | None = None,
            failing_dataset: str | None = None,
        ) -> tuple[list[tuple[object, ...]], dict[str, object], dict[str, dict[str, int]], set[str], set[str]]:
            completed = set() if completed is None else completed
            reused = set() if reused is None else reused
            final_reused = set() if final_reused is None else final_reused
            output_root = root / "runs/formal"
            session_dir = output_root / ".sessions" / session_id
            session_dir.mkdir(parents=True)
            session_file = output_root / ".sessions" / f"{session_id}.json"
            session_file.write_text(json.dumps({"capacity_checks": []}) + "\n", encoding="utf-8")
            for dataset in completed | reused:
                (output_root / dataset).mkdir(parents=True, exist_ok=True)

            events: list[tuple[object, ...]] = []

            def preflight_admissions(**kwargs: object) -> dict[str, dict[str, object]]:
                events.append(("admissions", kwargs.get("capacity_only"), tuple(admissions)))
                return admissions

            def admit_capacity(selected_admissions: dict[str, dict[str, object]], **kwargs: object):
                events.append(("requirements", tuple(selected_admissions)))
                self.assertEqual(set(selected_admissions), set(DATASETS))
                return workspace_receipt, {dataset: dict(base_capacity) for dataset in DATASETS}

            def phase_gate(dataset: str, requirements: dict[str, int], **kwargs: object):
                components = kwargs["disk_budget_components"]
                events.append(("gate", dataset, kwargs["phase"], components))
                if dataset == failing_dataset and kwargs["phase"] == "setup":
                    raise formal_common.ContractError(f"injected {dataset} capacity failure")
                return {"dataset": dataset, "phase": kwargs["phase"], "host_known_bytes": requirements.get("host_known_bytes", 0)}

            def keygen_receipt(output: Path, dataset: str, *_args: object):
                events.append(("reuse_keygen", dataset))
                return (
                    {"gat_transient_witness_bytes_upper": 9},
                    {"native_census": {"authenticated": dataset}},
                    "b" * 64,
                )

            def actual_capacity(admission: dict[str, object], _sizes: dict[str, int], *, native_census: dict[str, object]):
                events.append(("actual_capacity", admission["dataset"], native_census["authenticated"]))
                return dict(base_capacity)

            with mock.patch("run_hybrid_formal._run_five_admission_preflight", side_effect=preflight_admissions), \
                 mock.patch("run_hybrid_formal._build_native_cub_workspace_query", return_value=workspace_receipt), \
                 mock.patch("run_hybrid_formal._admit_local_capacity", side_effect=admit_capacity), \
                 mock.patch("run_hybrid_formal._recheck_local_phase_capacity", side_effect=phase_gate), \
                 mock.patch("run_hybrid_formal.validate_setup_reuse_receipt", side_effect=lambda _output, dataset, *_args: dataset in reused), \
                 mock.patch("run_hybrid_formal._reusable_final_stage_srs", side_effect=lambda **kwargs: kwargs["dataset"] in final_reused), \
                 mock.patch("run_hybrid_formal._load_native_keygen_receipt", side_effect=keygen_receipt), \
                 mock.patch("run_hybrid_formal._native_capacity_requirements", side_effect=actual_capacity):
                result = _run_five_native_capacity_preflight(
                    binary=root / "ezkl",
                    srs=root / "srs.kzg",
                    output_root=output_root,
                    session_id=session_id,
                    session_file=session_file,
                    session_dir=session_dir,
                    authenticated_context=object(),
                    selected_device={"uuid": "GPU-test"},
                    saved_admissions={},
                    runtime_identity={},
                    build_environment={},
                    gpu_identity={},
                    resume_identity={},
                    saved_setup_anchors={dataset: "c" * 64 for dataset in DATASETS},
                    archived_outputs=bool(completed or reused),
                    is_resume=bool(completed or reused),
                    completed_datasets=completed,
                    workspace_receipt=None,
                    srs_copy_size_bytes=5,
                )
            events.append(("cell", DATASETS[0]))
            return events, {"admissions": result[0], "workspace_receipt": result[1]}, result[2], result[3], result[4]

        with tempfile.TemporaryDirectory(prefix="ezkl-all-five-preflight-") as temporary:
            events, state, _requirements, _reused, _pending_finalize = run_preflight(Path(temporary))
            self.assertEqual(events[0], ("admissions", False, tuple(DATASETS)))
            self.assertEqual(events[1], ("requirements", tuple(DATASETS)))
            setup_gates = [event for event in events if event[0] == "gate"]
            self.assertEqual([(event[1], event[2]) for event in setup_gates], [(dataset, "setup") for dataset in DATASETS])
            self.assertEqual(events[-1], ("cell", DATASETS[0]))
            self.assertEqual(set(state["admissions"]), set(DATASETS))
            for event in setup_gates:
                components = event[3]
                self.assertEqual(components["future_setup_key_artifact_upper_bytes"], 50)
                self.assertEqual(components["pending_output_srs_copy_count"], 5)

        with tempfile.TemporaryDirectory(prefix="ezkl-all-five-ppi-failure-") as temporary:
            root = Path(temporary)
            output_root = root / "runs/formal"
            session_dir = output_root / ".sessions" / session_id
            session_dir.mkdir(parents=True)
            session_file = output_root / ".sessions" / f"{session_id}.json"
            session_file.write_text(json.dumps({"capacity_checks": []}) + "\n", encoding="utf-8")
            events: list[tuple[object, ...]] = []

            def preflight_admissions(**kwargs: object) -> dict[str, dict[str, object]]:
                events.append(("admissions", kwargs.get("capacity_only")))
                return admissions

            def fail_on_ppi(dataset: str, _requirements: dict[str, int], **kwargs: object):
                events.append(("gate", dataset, kwargs["phase"]))
                if dataset == "ppi" and kwargs["phase"] == "setup":
                    raise formal_common.ContractError("injected ppi capacity failure")
                return {"dataset": dataset, "phase": kwargs["phase"]}

            with mock.patch("run_hybrid_formal._run_five_admission_preflight", side_effect=preflight_admissions), \
                 mock.patch("run_hybrid_formal._build_native_cub_workspace_query", return_value=workspace_receipt), \
                 mock.patch("run_hybrid_formal._admit_local_capacity", return_value=(workspace_receipt, {dataset: dict(base_capacity) for dataset in DATASETS})), \
                 mock.patch("run_hybrid_formal._recheck_local_phase_capacity", side_effect=fail_on_ppi):
                with self.assertRaisesRegex(formal_common.ContractError, "injected ppi capacity failure"):
                    _run_five_native_capacity_preflight(
                        binary=root / "ezkl", srs=root / "srs.kzg", output_root=output_root,
                        session_id=session_id, session_file=session_file, session_dir=session_dir,
                        authenticated_context=object(), selected_device={"uuid": "GPU-test"}, saved_admissions={},
                        runtime_identity={}, build_environment={}, gpu_identity={}, resume_identity={},
                        saved_setup_anchors={}, archived_outputs=False, is_resume=False, completed_datasets=set(),
                        workspace_receipt=None, srs_copy_size_bytes=5,
                    )
            self.assertEqual(events[0], ("admissions", False))
            self.assertIn(("gate", "ppi", "setup"), events)
            self.assertNotIn(("cell", DATASETS[0]), events)

        with tempfile.TemporaryDirectory(prefix="ezkl-all-five-resume-") as temporary:
            completed = {DATASETS[0]}
            reused = {DATASETS[1]}
            events, _state, _requirements, reused_result, pending_finalize = run_preflight(
                Path(temporary), completed=completed, reused=reused, final_reused=reused
            )
            setup_gates = [event for event in events if event[0] == "gate" and event[2] == "setup"]
            prove_gates = [event for event in events if event[0] == "gate" and event[2] == "witness_prove_verify"]
            self.assertEqual([event[1] for event in setup_gates], list(DATASETS[2:]))
            self.assertEqual([event[1] for event in prove_gates], [DATASETS[1]])
            self.assertEqual(reused_result, reused)
            setup_components = setup_gates[0][3]
            self.assertEqual(setup_components["future_setup_key_artifact_upper_bytes"], 30)
            self.assertEqual(setup_components["pending_output_srs_copy_count"], 3)
            proof_components = prove_gates[0][3]
            self.assertEqual(proof_components["future_setup_key_artifact_upper_bytes"], 0)
            self.assertEqual(proof_components["pending_output_srs_copy_count"], 3)
            self.assertEqual(pending_finalize, set(DATASETS[2:]))
            self.assertNotIn(("gate", DATASETS[0], "setup"), events)
            self.assertNotIn(("gate", DATASETS[0], "witness_prove_verify"), events)

        class FailingCapacityApi:
            @staticmethod
            def audit_fresh_admission(_session_file: Path, dataset: str, *_args: object, **_kwargs: object):
                return {"dataset": dataset}

            @staticmethod
            def write_and_require_fresh_report(_session_file: Path, _records: list[dict[str, object]], report_path: Path, **_kwargs: object):
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text("{}\n", encoding="utf-8")
                return {"gate_status": "current_disk_floor_fail", "readiness": "NOT_ESTABLISHED"}

        for capacity_only in (True, False):
            with self.subTest(capacity_only=capacity_only), tempfile.TemporaryDirectory(prefix="ezkl-capacity-only-arg-") as temporary:
                root = Path(temporary)
                output_root = root / "runs/formal"
                session_id = "d" * 32
                session_dir = output_root / ".sessions" / session_id
                session_dir.mkdir(parents=True)
                session_file = output_root / ".sessions" / f"{session_id}.json"
                session_file.write_text(json.dumps({"status": "RUNNING"}) + "\n", encoding="utf-8")
                with mock.patch("run_hybrid_formal.ROOT", root), \
                     mock.patch("run_hybrid_formal._capacity_precheck_api", return_value=FailingCapacityApi), \
                     mock.patch("run_hybrid_formal.native_admit_all_datasets", side_effect=formal_common.ContractError("injected admission failure")):
                    with self.assertRaises(formal_common.ContractError) as failure:
                        _run_five_admission_preflight(
                            binary=root / "ezkl", srs=root / "srs.kzg", output_root=output_root,
                            session_id=session_id, session_file=session_file, authenticated_context=object(),
                            selected_device={"uuid": "GPU-test"}, saved_admissions={}, capacity_only=capacity_only,
                        )
                self.assertNotIsInstance(failure.exception, NameError)
                self.assertIn("five-dataset admission/capacity preflight", str(failure.exception))

    def test_native_prepare_budget_charges_baseline_and_table_cache_once(self) -> None:
        requirements = _native_capacity_requirements(
            self._capacity_admission(), {"scalar_bytes": 32, "affine_bytes": 64}
        )
        minimum_task = max(
            requirements["lookup_prepare_task_host_bytes"],
            requirements["lookup_commit_task_host_bytes"],
            requirements["shplonk_fixed_workspace_host_bytes"]
            + requirements["shplonk_task_host_bytes"],
        )
        reserve = 100
        msm_workspace = 200
        device = (
            msm_workspace
            + requirements["icicle_ntt_device_peak_upper_bytes"]
            + requirements["icicle_msm_precomputed_basis_device_upper_bytes"]
        )
        expected_taskcap = 7 * minimum_task
        initial_available = (
            requirements["proof_nonprepare_heap_host_bytes"]
            + requirements["proof_nonprepare_hot_mmap_host_bytes"]
            + requirements["lookup_table_cache_retained_host_bytes"]
            + requirements["shplonk_extension_host_upper_bytes"]
            + requirements["icicle_msm_host_staging_bytes"]
            + requirements["icicle_ntt_host_staging_upper_bytes"]
            + reserve
            + device
            + expected_taskcap
        )
        with mock.patch("run_hybrid_formal._effective_host_memory_limit_bytes", return_value=(10**12, {})), \
             mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(reserve, {})):
            budget = _native_prepare_budget(
                requirements,
                initial_available_bytes=initial_available,
                workspace_query={"integrated_device": True, "device_workspace_peak_bytes": msm_workspace},
            )
        self.assertEqual(budget["taskcap_bytes"], expected_taskcap)
        self.assertEqual(
            budget["future_phase_increment_bytes"],
            budget["tablecache_bytes"]
            + budget["shplonk_extensions_bytes"]
            + budget["icicle_staging_bytes"]
            + budget["icicle_ntt_host_staging_upper_bytes"]
            + expected_taskcap,
        )
        self.assertEqual(
            budget["incremental_envelope_bytes"],
            budget["future_phase_increment_bytes"],
        )

    def test_integrated_device_workspace_is_added_to_host_bound(self) -> None:
        requirements = {
            "host_known_bytes": 100,
            "icicle_ntt_device_peak_upper_bytes": 0,
            "icicle_msm_precomputed_basis_device_upper_bytes": 0,
        }
        sample = {"free_memory_bytes": 100, "total_memory_bytes": 200, "device_uuid": "GPU-test"}
        query = {"device_workspace_peak_bytes": 100, "integrated_device": True}
        with mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(199, {})), \
             mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(0, {})), \
             mock.patch("run_hybrid_formal._filesystem_available_bytes", return_value=(1000, {})):
            with self.assertRaisesRegex(formal_common.ContractError, "host allocation"):
                _check_dataset_capacity(
                    "cora", requirements, output_root=Path("/capacity/formal"),
                    remaining_known_disk_bytes=1, phase="prove", device_sample=sample,
                    workspace_query=query,
                )
        with mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(200, {})), \
             mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(0, {})), \
             mock.patch("run_hybrid_formal._filesystem_available_bytes", return_value=(1000, {})):
            _check_dataset_capacity(
                "cora", requirements, output_root=Path("/capacity/formal"),
                remaining_known_disk_bytes=1, phase="prove", device_sample=sample,
                workspace_query=query,
            )

    def test_cub_workspace_receipt_and_phase_recheck_are_session_bound(self) -> None:
        query = self._cub_query()
        selected = {"uuid": query["device_uuid"], "compute_capability": "10.0"}
        checked = _validate_cub_workspace_query(
            query, source_sha256=query["icicle_source_tree_sha256"],
            backend_manifest_sha256=query["native_backend_manifest_sha256"],
            selected_device=selected, session_id="a" * 32,
        )
        self.assertEqual(checked["device_workspace_peak_bytes"], 100)
        with self.assertRaisesRegex(formal_common.ContractError, "constant mismatch: session_id"):
            _validate_cub_workspace_query(
                query, source_sha256=query["icicle_source_tree_sha256"],
                backend_manifest_sha256=query["native_backend_manifest_sha256"],
                selected_device=selected, session_id="b" * 32,
            )

        receipt = {
            "schema": "ezkl.icicle.cub_msm_workspace_query_receipt.v3",
            "session_id": "b" * 32,
            "source_path": "tools/native_msm_workspace_query.cu",
            "source_sha256": "1" * 64,
            "binary_path": "runs/formal/.sessions/session/capacity/query",
            "binary_sha256": "2" * 64,
            "build_log_path": "runs/formal/.sessions/session/capacity/build.log",
            "build_log_sha256": "3" * 64,
            "result_path": "runs/formal/.sessions/session/capacity/result.json",
            "result_sha256": "4" * 64,
            "native_backend_manifest_path": "artifacts/build/icicle-cuda/native_backend_manifest.json",
            "native_backend_manifest_sha256": "5" * 64,
            "icicle_source_tree_sha256": "6" * 64,
            "query": query,
        }
        with self.assertRaisesRegex(formal_common.ContractError, "different formal session"):
            _validate_native_cub_workspace_receipt(
                receipt, Path("/tmp/session"), selected, {}, "a" * 32
            )

        self.assertEqual(
            _remaining_setup_disk_bytes({
                dataset: {"setup_key_artifact_upper_bytes": index + 1, "setup_mapped_peak_bytes": 20}
                for index, dataset in enumerate(DATASETS)
            }, 1, include_current_transient=False),
            sum(range(2, len(DATASETS) + 1)),
        )
        filtered_requirements = {
            dataset: {"setup_key_artifact_upper_bytes": index + 1, "setup_mapped_peak_bytes": 20}
            for index, dataset in enumerate(DATASETS)
        }
        self.assertEqual(
            _remaining_setup_disk_bytes(
                filtered_requirements,
                1,
                include_current_transient=True,
                setup_required_datasets={"citeseer", "ppi"},
            ),
            26,
        )
        self.assertEqual(
            _remaining_setup_disk_bytes(
                filtered_requirements,
                2,
                include_current_transient=False,
                setup_required_datasets={"citeseer", "ppi"},
            ),
            4,
        )
        sample = {
            "session_id": "a" * 32,
            "device_uuid": selected["uuid"],
            "free_memory_bytes": 1000,
            "total_memory_bytes": 2000,
            "integrated_device": False,
        }
        with mock.patch("run_hybrid_formal._sample_native_cuda_memory", return_value=sample) as sample_call, \
             mock.patch("run_hybrid_formal._check_dataset_capacity", return_value={"phase": "setup"}) as gate_call:
            result = _recheck_local_phase_capacity(
                "cora", {"host_known_bytes": 1}, output_root=Path("/formal"),
                remaining_known_disk_bytes=1, phase="setup", receipt={"query": query},
                session_dir=Path("/session"), session_id="a" * 32, selected_device=selected,
                gpu_identity={}, build_environment={},
            )
        self.assertEqual(result["phase"], "setup")
        sample_call.assert_called_once()
        gate_call.assert_called_once()

    def test_failure_marks_local_session_failed_and_releases_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-session-lifecycle-") as temp:
            root = Path(temp)
            output_root = root / "runs" / "formal"
            matrix = root / "results" / "formal_matrix.json"
            with mock.patch("run_hybrid_formal.ROOT", root), mock.patch.object(formal_common, "ROOT", root):
                args = Namespace(binary=None, output_root="runs/formal", matrix="results/formal_matrix.json", preflight_only=False, resume_session=None, _from_ezkl_entry=True)
                with mock.patch("run_hybrid_formal._run_session", side_effect=RuntimeError("bounded injected failure")) as run_session:
                    with self.assertRaisesRegex(RuntimeError, "bounded injected failure"):
                        _main(args)

                run_session.assert_called_once()
                session_id = run_session.call_args.kwargs["session_id"]
                session_file = run_session.call_args.kwargs["session_file"]
                self.assertRegex(session_id, r"^[0-9a-f]{32}$")
                self.assertEqual(session_file, output_root / ".sessions" / f"{session_id}.json")
                document = json.loads(session_file.read_text(encoding="utf-8"))
                self.assertEqual(document["baseline"], "ezkl")
                self.assertEqual(document["session_id"], session_id)
                self.assertEqual(document["status"], "FAILED")
                self.assertNotIn(document["status"], {"FINAL_VALIDATION_OK", "PREFLIGHT_ONLY_READY"})
                self.assertIn("bounded injected failure", document["error"])
                self.assertFalse(matrix.exists())
                self.assertFalse(any((output_root / dataset).exists() for dataset in DATASETS))

                lock_path = root / "runs" / ".ezkl-formal.lock"
                with lock_path.open("r+b") as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def test_released_in_progress_owner_is_marked_interrupted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-session-recovery-") as temp:
            root = Path(temp)
            output_root = root / "runs" / "formal"
            session_dir = output_root / ".sessions"
            session_dir.mkdir(parents=True)
            prior_id = "0123456789abcdef0123456789abcdef"
            prior_file = session_dir / f"{prior_id}.json"
            prior_file.write_text(json.dumps({
                "schema_version": 1,
                "baseline": "ezkl",
                "session_id": prior_id,
                "status": "RUNNING",
                "stage": "citeseer_setup_prove_verify",
            }), encoding="utf-8")
            lock_path = root / "runs" / ".ezkl-formal.lock"
            lock_path.write_text(json.dumps({
                "schema_version": 1,
                "baseline": "ezkl",
                "pid": 99999999,
                "session_id": prior_id,
                "session_file": prior_file.relative_to(root).as_posix(),
            }) + "\n", encoding="utf-8")

            with mock.patch("run_hybrid_formal.ROOT", root), mock.patch.object(formal_common, "ROOT", root):
                args = Namespace(binary=None, output_root="runs/formal", matrix="results/formal_matrix.json", preflight_only=False, resume_session=None, _from_ezkl_entry=True)
                with mock.patch("run_hybrid_formal._run_session", side_effect=RuntimeError("bounded injected failure")):
                    with self.assertRaisesRegex(RuntimeError, "bounded injected failure"):
                        _main(args)

            recovered = json.loads(prior_file.read_text(encoding="utf-8"))
            self.assertEqual(recovered["status"], "INTERRUPTED")
            self.assertEqual(recovered["stage"], "citeseer_setup_prove_verify")
            self.assertIn("released the baseline lock", recovered["error"])

    def test_dataset_failures_continue_persist_and_retry_without_losing_history(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-dataset-retry-") as temporary:
            root = Path(temporary)
            session_file = root / "session.json"
            session_file.write_text(json.dumps({
                "status": "RUNNING",
                "stage": "preflight_complete",
                "dataset_failures": {},
                "dataset_failure_history": {},
            }) + "\n", encoding="utf-8")
            completed: set[str] = set()
            failures: dict[str, dict[str, object]] = {}
            history: dict[str, list[dict[str, object]]] = {}
            attempts: list[str] = []

            def first_attempt(_index: int, dataset: str) -> None:
                attempts.append(dataset)
                stage = f"{dataset}_native_relation_admission" if dataset == "cora" else f"{dataset}_setup"
                _set_stage(session_file, stage, active_dataset=dataset)
                if dataset == "cora":
                    raise formal_common.ContractError("admission failed\nfull native diagnostic")
                if dataset == "pubmed":
                    raise OSError("setup disk failure with detail")

            log_paths = lambda dataset: [f".sessions/test/logs/{dataset}/native_admission.log"]
            _run_dataset_cells(DATASETS, completed, failures, history, session_file, first_attempt, log_paths)
            self.assertEqual(attempts, list(DATASETS))
            self.assertEqual(completed, {"citeseer", "ppi", "ogbn_arxiv"})
            self.assertEqual(set(failures), {"cora", "pubmed"})
            self.assertEqual(failures["cora"]["stage"], "cora_native_relation_admission")
            self.assertEqual(failures["cora"]["exception_type"], "ContractError")
            self.assertEqual(failures["cora"]["error"], "admission failed\nfull native diagnostic")
            self.assertIn("ContractError: admission failed", failures["cora"]["traceback"])
            self.assertEqual(failures["cora"]["log_paths"], log_paths("cora"))
            self.assertEqual(failures["pubmed"]["stage"], "pubmed_setup")
            self.assertEqual(failures["pubmed"]["exception_type"], "OSError")
            self.assertIn("setup disk failure with detail", failures["pubmed"]["error"])
            self.assertEqual(set(history), {"cora", "pubmed"})
            self.assertEqual(len(history["cora"]), 1)
            self.assertEqual(len(history["pubmed"]), 1)
            saved = json.loads(session_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["dataset_failures"], failures)
            self.assertEqual(saved["dataset_failure_history"], history)
            with self.assertRaisesRegex(formal_common.ContractError, "skipping five-result collection"):
                _require_formal_dataset_completion(DATASETS, completed, failures)

            retry_attempts: list[str] = []

            def retry(_index: int, dataset: str) -> None:
                retry_attempts.append(dataset)
                _set_stage(session_file, f"{dataset}_witness_prove_independent_verify", active_dataset=dataset)

            _run_dataset_cells(DATASETS, completed, failures, history, session_file, retry, log_paths)
            self.assertEqual(retry_attempts, ["cora", "pubmed"])
            self.assertEqual(completed, set(DATASETS))
            self.assertEqual(failures, {})
            self.assertEqual(len(history["cora"]), 1)
            self.assertEqual(len(history["pubmed"]), 1)
            self.assertEqual(json.loads(session_file.read_text(encoding="utf-8"))["dataset_failure_history"], history)
            _require_formal_dataset_completion(DATASETS, completed, failures)

            interrupted_file = root / "interrupted.json"
            interrupted_file.write_text(json.dumps({"status": "RUNNING", "stage": "preflight_complete"}) + "\n", encoding="utf-8")
            interrupted_attempts: list[str] = []

            def interrupted_attempt(_index: int, dataset: str) -> None:
                interrupted_attempts.append(dataset)
                _set_stage(interrupted_file, f"{dataset}_setup", active_dataset=dataset)
                raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                _run_dataset_cells(
                    DATASETS,
                    set(),
                    {},
                    {},
                    interrupted_file,
                    interrupted_attempt,
                )
            self.assertEqual(interrupted_attempts, [DATASETS[0]])
            interrupted = json.loads(interrupted_file.read_text(encoding="utf-8"))
            self.assertNotIn("dataset_failures", interrupted)


    def test_formal_command_plan_covers_five_datasets_in_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-command-plan-") as temp:
            root = Path(temp)
            admissions = {dataset: {"prepared_dir": f"runs/admission/{dataset}"} for dataset in DATASETS}
            with mock.patch("run_hybrid_formal.ROOT", root), mock.patch.object(formal_common, "ROOT", root):
                plan = _formal_dataset_command_plan(
                    root / "target/gpu/release/ezkl",
                    root / "resources/native_srs/official.srs",
                    root / "runs/formal",
                    admissions,
                )

            self.assertEqual([item["dataset"] for item in plan], list(DATASETS))
            for item, dataset in zip(plan, DATASETS, strict=True):
                setup = item["setup"]
                producer = item["producer"]
                single = item["single"]
                self.assertEqual(setup[setup.index("--dataset") + 1], dataset)
                self.assertEqual(setup[setup.index("--phase") + 1], "setup")
                self.assertEqual(setup[setup.index("--output-dir") + 1], f"runs/formal/{dataset}")
                self.assertEqual(setup[setup.index("--admission-dir") + 1], f"runs/admission/{dataset}")
                self.assertEqual(setup[setup.index("--srs-path") + 1], "resources/native_srs/official.srs")
                if NATIVE_FORMAL_BACKEND == "gpu":
                    self.assertIn("--gpu", setup)
                self.assertEqual(producer[producer.index("--dataset") + 1], dataset)
                self.assertEqual(producer[producer.index("--config") + 1], f"resources/canonical/configs/{dataset}_secure.cfg")
                self.assertEqual(producer[producer.index("--statement") + 1], f"runs/formal/{dataset}/statement.json")
                self.assertEqual(single[single.index("--dataset") + 1], dataset)
                self.assertIn("--execute-formal", single)
                self.assertIn("--reuse-setup", single)
                self.assertEqual(single[single.index("--config") + 1], f"resources/canonical/configs/{dataset}_secure.cfg")
                self.assertEqual(single[single.index("--output-dir") + 1], f"runs/formal/{dataset}")

    def test_public_formal_entry_executes_only_the_local_runner(self) -> None:
        args = run_formal_matrix.parser().parse_args(["--execute-formal"])
        with mock.patch("run_formal_matrix.os.chdir") as chdir, mock.patch("run_formal_matrix.os.execv") as execv:
            self.assertEqual(run_formal_matrix.main(args), 0)
        chdir.assert_called_once_with(run_formal_matrix.ROOT)
        executable, command = execv.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(command[1], str(run_formal_matrix.ROOT / "scripts" / "run_hybrid_formal.py"))
        self.assertIn("--binary", command)
        self.assertIn("target/gpu/release/ezkl", command)
        self.assertIn("--_from-ezkl-entry", command)
        self.assertNotIn("--internal-phase", command)
        self.assertFalse(any("comparison/run_formal.sh" in arg for arg in command))


if __name__ == "__main__":
    unittest.main()
