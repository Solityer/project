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
import run_formal_matrix  # noqa: E402
from run_hybrid_formal import (  # noqa: E402
    DATASETS,
    NATIVE_FORMAL_BACKEND,
    _CUB_QUERY_COUNTS,
    _check_dataset_capacity,
    _formal_dataset_command_plan,
    _native_capacity_requirements,
    _main,
    _recheck_local_phase_capacity,
    _remaining_setup_disk_bytes,
    _validate_cub_workspace_query,
    _validate_native_cub_workspace_receipt,
)


class LocalFormalEntryTest(unittest.TestCase):
    @staticmethod
    def _capacity_admission() -> dict[str, object]:
        return {
            "domain_size": 16,
            "constraint_system": {
                "degree": 3,
                "num_fixed_columns": 3,
                "num_advice_columns": 4,
                "num_instance_columns": 1,
                "num_permutation_columns": 2,
                "fixed_column_count_post_compression": 5,
                "permutation_successor_entry_count_upper": 8,
            },
            "native_memory_model": {
                "proof_domain_size": 64,
                "size_of_fr_bytes": 32,
                "size_of_assigned_fr_bytes": 40,
                "size_of_option_fr_bytes": 48,
                "size_of_batch_invert_entry_bytes": 40,
                "size_of_g1_affine_bytes": 96,
                "selector_activation_bitmap_bytes": 100,
                "fixed_polynomial_read_heap_scratch_bytes_upper": 4096,
            },
        }

    @staticmethod
    def _cub_query(session_id: str = "a" * 32) -> dict[str, object]:
        selected = {"uuid": "GPU-3cde57ba-ae43-4452-bd59-7b156b36db71", "compute_capability": "10.0"}
        return {
            "schema": "ezkl.icicle.cub_msm_workspace_query.v2",
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
            "window_count": 16,
            "M": 1 << 24,
            "B": 1_048_560,
            "H_max": 1_048_560,
            "L_max": 104_206,
            "cuda_runtime_version": 12080,
            "cuda_driver_version": 12080,
            "cuda_toolkit_version": "12.8.0",
            "cub_version": 200800,
            "device_compute_capability": selected["compute_capability"],
            "native_type_sizes": {
                "scalar_bytes": 32,
                "affine_bytes": 64,
                "projective_bytes": 96,
                "unsigned_bytes": 4,
            },
            "device_workspace_peak_bytes": 100,
            "queries": [
                {"name": name, "num_items": count, "scratch_bytes": 1}
                for name, count in _CUB_QUERY_COUNTS.items()
            ],
            "scratch_bytes_sum": len(_CUB_QUERY_COUNTS),
        }

    def test_native_capacity_model_counts_setup_and_quotient_mappings(self) -> None:
        requirements = _native_capacity_requirements(
            self._capacity_admission(), {"scalar_bytes": 32, "affine_bytes": 64}
        )
        self.assertEqual(requirements["setup_persisted_mapped_bytes"], (2 * 5 * 16 + 2 * 16 + 3 * 64) * 32)
        self.assertEqual(requirements["setup_transition_mapped_bytes"], (64 + 16) * 32)
        self.assertEqual(
            requirements["setup_mapped_peak_bytes"],
            3 * 16 * 40 + (2 * 5 * 16 + 2 * 16 + 3 * 64) * 32 + (64 + 16) * 32,
        )
        self.assertEqual(requirements["quotient_extended_backing_bytes"], 64 * 32)
        self.assertEqual(requirements["quotient_piece_bytes"], 16 * (3 - 1) * 32)
        self.assertEqual(
            requirements["proof_spill_known_bytes"],
            requirements["setup_persisted_mapped_bytes"] + 4 * 16 * 32 + 2 * 16 * 32 + 64 * 32 + 16 * 2 * 32,
        )
        self.assertEqual(requirements["batch_invert_heap_scratch_bytes_upper"], 16 * 40)
        self.assertEqual(requirements["icicle_msm_host_staging_bytes"], (32 + 64) * (1 << 20))
        self.assertEqual(
            requirements["permutation_vk_msm_host_known_bytes"],
            100 + 2 * 16 * 32 + (32 + 64) * (1 << 20),
        )
        self.assertEqual(requirements["host_known_bytes"], requirements["permutation_vk_msm_host_known_bytes"])
        self.assertGreater(requirements["setup_key_artifact_upper_bytes"], requirements["setup_persisted_mapped_bytes"])

    def test_capacity_gate_accepts_exact_limits_and_rejects_each_shortfall(self) -> None:
        requirements = {"host_known_bytes": 100}
        output_root = Path("/capacity/formal")
        sample = {"free_memory_bytes": 100, "total_memory_bytes": 200, "device_uuid": "GPU-test"}
        query = {"device_workspace_peak_bytes": 100, "integrated_device": False}
        with mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(100, {})), \
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
                 mock.patch("run_hybrid_formal._filesystem_available_bytes", return_value=(disk, {})):
                with self.assertRaisesRegex(formal_common.ContractError, expected):
                    _check_dataset_capacity(
                        "cora", requirements, output_root=output_root, remaining_known_disk_bytes=100,
                        phase="setup", device_sample={**sample, "free_memory_bytes": free},
                        workspace_query=query,
                    )

    def test_integrated_device_workspace_is_added_to_host_bound(self) -> None:
        requirements = {"host_known_bytes": 100}
        sample = {"free_memory_bytes": 100, "total_memory_bytes": 200, "device_uuid": "GPU-test"}
        query = {"device_workspace_peak_bytes": 100, "integrated_device": True}
        with mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(199, {})), \
             mock.patch("run_hybrid_formal._filesystem_available_bytes", return_value=(1000, {})):
            with self.assertRaisesRegex(formal_common.ContractError, "host allocation"):
                _check_dataset_capacity(
                    "cora", requirements, output_root=Path("/capacity/formal"),
                    remaining_known_disk_bytes=1, phase="prove", device_sample=sample,
                    workspace_query=query,
                )
        with mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(200, {})), \
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
            "schema": "ezkl.icicle.cub_msm_workspace_query_receipt.v2",
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
                args = Namespace(binary=None, output_root="runs/formal", matrix="results/formal_matrix.json", preflight_only=False, _from_ezkl_entry=True)
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
                args = Namespace(binary=None, output_root="runs/formal", matrix="results/formal_matrix.json", preflight_only=False, _from_ezkl_entry=True)
                with mock.patch("run_hybrid_formal._run_session", side_effect=RuntimeError("bounded injected failure")):
                    with self.assertRaisesRegex(RuntimeError, "bounded injected failure"):
                        _main(args)

            recovered = json.loads(prior_file.read_text(encoding="utf-8"))
            self.assertEqual(recovered["status"], "INTERRUPTED")
            self.assertEqual(recovered["stage"], "citeseer_setup_prove_verify")
            self.assertIn("released the baseline lock", recovered["error"])

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
