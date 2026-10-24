#!/usr/bin/env python3
"""Targeted regressions for baseline-local admission and cell records."""
from __future__ import annotations

import ctypes
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
sys.path.insert(0, str(SCRIPTS))
import formal_runner
import result_artifacts


class FormalLocalRunnerTests(unittest.TestCase):
    def test_hyrax_gpu_offload_metadata_accepts_only_cpu_prove_and_cpu_verify(self) -> None:
        record = {"gpu_offload_stage": "hyrax_ipa_fold", "gpu_forward_kernel_executed": "0",
                  "gpu_proof_kernel_executed": "1", "gpu_kernel_executed": "1",
                  "gpu_proof_kernel_count": "12", "gpu_proof_peak_allocated_bytes": "4096",
                  "gpu_device_name": "NVIDIA GB10", "gpu_device_ordinal": "0", "gpu_device_uuid": "GPU-fixture",
                  "verifier_backend": "cpu", "verify_gpu_kernel_executed": "0"}
        envelope = dict(record)
        verification = {"verifier_backend": "cpu", "verify_gpu_kernel_executed": "0"}
        self.assertEqual(result_artifacts._gpu_offload_evidence(record, envelope, verification, "cpu",
                                                                 "hyrax_ipa_fold")["gpu_device_uuid"], "GPU-fixture")
        for mutated, mutated_verification, backend in (
                ({**envelope, "gpu_offload_stage": "none"}, verification, "cpu"),
                ({**envelope, "gpu_forward_kernel_executed": "1"}, verification, "cpu"),
                ({**envelope, "gpu_proof_kernel_executed": "0"}, verification, "cpu"),
                (envelope, {"verifier_backend": "gpu", "verify_gpu_kernel_executed": "1"}, "cpu"),
                (envelope, verification, "gpu")):
            with self.assertRaises(result_artifacts.ArtifactError):
                result_artifacts._gpu_offload_evidence(record, mutated, mutated_verification, backend,
                                                        "hyrax_ipa_fold")


    def test_finalization_capacity_uses_full_u64_segment_integer_size(self) -> None:
        fields = ("segment_index", "logical_offset", "actual_length", "padded_length", "basis_offset", "basis_count")
        expected_segment_bytes = (
            sys.getsizeof(dict.fromkeys(fields, 0)) +
            sum(sys.getsizeof(field) + sys.getsizeof((1 << 64) - 1) for field in fields) +
            2 * ctypes.sizeof(ctypes.c_void_p)
        )
        profile = formal_runner.formal_phase._srs_manifest_finalization_capacity(1, 1)
        self.assertEqual(profile["python_object_bytes_upper_bound"], 16 * 1024 + expected_segment_bytes)

    def test_cell_stage_markers_report_elapsed_pass_and_failure(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = formal_runner.formal_phase._cell_stage("setup", "cora", lambda: "done")
        self.assertEqual(result, "done")
        markers = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([marker["status"] for marker in markers], ["START", "PASS"])
        self.assertEqual([marker["zkdl_cell_stage"] for marker in markers], ["setup", "setup"])
        self.assertEqual([marker["dataset"] for marker in markers], ["cora", "cora"])
        self.assertGreaterEqual(markers[1]["elapsed_seconds"], 0.0)

        output = io.StringIO()
        def fail() -> None:
            raise formal_runner.formal_phase.PhaseError("fixture fail")

        with redirect_stdout(output), self.assertRaises(formal_runner.formal_phase.PhaseError):
            formal_runner.formal_phase._cell_stage("prove", "cora", fail)
        markers = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([marker["status"] for marker in markers], ["START", "FAIL"])
        self.assertEqual(markers[-1]["error"], "fixture fail")
        self.assertGreaterEqual(markers[-1]["elapsed_seconds"], 0.0)

    def test_cell_failure_stops_later_datasets_but_finalizes_and_success_runs_all_cells(self) -> None:
        def exercise(fail_first_cell: bool, session_suffix: str) -> tuple[int, list[tuple[str, str | None]], dict[str, object]]:
            with tempfile.TemporaryDirectory(prefix="zkdl-formal-loop-") as raw:
                root = Path(raw)
                calls: list[tuple[str, str | None]] = []

                def fake_build(runs: Path, session_id: str) -> Path:
                    session = runs / f"formal_{session_id}"
                    (session / "logs").mkdir(parents=True)
                    (session / "phase").mkdir()
                    return session

                def fake_phase(session: Path, session_id: str, state: dict[str, object],
                               phase: str, dataset: str | None = None) -> dict[str, object]:
                    calls.append((phase, dataset))
                    if phase == "prepare":
                        for item in state["dataset_status"].values():
                            item.update({"status": "READY", "stage": "admitted", "failure": ""})
                        return {"status": "PASS"}
                    if phase == "cell":
                        if fail_first_cell and dataset == formal_runner.DATASETS[0]:
                            raise formal_runner.RunnerError("fixture cell failure")
                        state["dataset_status"][dataset].update({"status": "PASS", "stage": "complete", "failure": ""})
                        return {"status": "PASS"}
                    if phase == "finalize":
                        if fail_first_cell:
                            return {"status": "FAILED", "failure": "fixture partial finalization"}
                        for dataset_name in formal_runner.DATASETS:
                            state["dataset_status"][dataset_name].update(
                                {"status": "PASS", "stage": "complete", "failure": ""})
                        return {"status": "PASS", "dataset_status": state["dataset_status"]}
                    raise AssertionError(f"unexpected fake phase {phase}")

                profile = {
                    "schema": "zkdl_formal_capacity_profile_v4",
                    "backend": "cpu",
                    "gpu_offload_stage": "hyrax_ipa_fold",
                    "device": {"name": "fixture", "ordinal": 0, "uuid": "fixture"},
                    "datasets": {dataset: {} for dataset in formal_runner.DATASETS},
                }
                no_capacity_failure = lambda *_args, **_kwargs: {"checked": True}
                with (
                    patch.object(formal_runner, "ROOT", root),
                    patch.object(formal_runner, "_build_and_publish_session", fake_build),
                    patch.object(formal_runner, "_write_detached_session_link", lambda *_args: None),
                    patch.object(formal_runner, "_prepare_capacity", lambda *_args: profile),
                    patch.object(formal_runner, "_capacity_gate", no_capacity_failure),
                    patch.object(formal_runner, "_run_phase", fake_phase),
                    patch.object(formal_runner, "_validate_final_pass", lambda *_args: "results/final_validation.json"),
                    patch.object(formal_runner.secrets, "token_hex", return_value=session_suffix * 32),
                    patch.object(formal_runner.signal, "signal"),
                    patch.object(formal_runner.formal_phase, "SIGNAL_GRACE_SECONDS", 10.0),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    status = formal_runner.run()
                session = root / "runs" / f"formal_{session_suffix * 32}"
                state = json.loads((session / "state.json").read_text(encoding="utf-8"))
                return status, calls, state

        failed_status, failed_calls, failed_state = exercise(True, "a")
        self.assertEqual(failed_status, 2)
        self.assertEqual(failed_calls, [
            ("prepare", None),
            ("cell", formal_runner.DATASETS[0]),
            ("finalize", None),
        ])
        self.assertEqual(failed_state["dataset_status"][formal_runner.DATASETS[0]]["status"], "FAILED")
        self.assertEqual(failed_state["stage"], "complete")

        pass_status, pass_calls, pass_state = exercise(False, "b")
        self.assertEqual(pass_status, 0)
        self.assertEqual(pass_calls, [("prepare", None), *[("cell", dataset) for dataset in formal_runner.DATASETS],
                                      ("finalize", None)])
        self.assertTrue(all(pass_state["dataset_status"][dataset]["status"] == "PASS"
                            for dataset in formal_runner.DATASETS))

    def _profile(self) -> dict[str, object]:
        rows: dict[str, dict[str, object]] = {}
        finalization_rows: dict[str, dict[str, int]] = {}
        for index, dataset in enumerate(formal_runner.DATASETS, start=1):
            retained = index * 10
            finalization_rows[dataset] = formal_runner.formal_phase._srs_manifest_finalization_capacity(index, index)
            gpu_workspace = 144 * (1 << 18) + 704 + 352
            rows[dataset] = {
                "additional_disk_peak_bytes": retained + index * 3,
                "cell_retained_artifact_bytes_upper_bound": retained,
                "host_peak_bytes": index * 100,
                "device_peak_bytes": gpu_workspace,
                "gpu_authenticated_basis_size": 1 << 18,
                "gpu_max_opening_padded_generator_size": 1,
                "gpu_workspace_required_bytes": gpu_workspace,
                "gpu_cuda_free_bytes_at_prepare": gpu_workspace + 1,
                "gpu_cuda_total_bytes_at_prepare": gpu_workspace + 2,
                "native_route_source_domain": index,
                "native_route_selector_factor_elements": index + 1,
                "phase_coverage": ["setup", "witness", "commitment", "prove", "verify"],
            }
        return {
            "schema": "zkdl_formal_capacity_profile_v4",
            "baseline": "zkdl",
            "backend": "cpu",
            "gpu_offload_stage": "hyrax_ipa_fold",
            "memory_accounting_kind": "shared_host_cuda",
            "device": {"name": "fixture GPU", "ordinal": 0, "uuid": "GPU-fixture"},
            "session_id": "0" * 32,
            "fs_device_id": 1,
            "preexisting_baseline_artifact_bytes": 0,
            "finalization_disk_bytes_upper_bound": 7,
            "finalization_host_peak_bytes_upper_bound": max(row["host_peak_bytes_upper_bound"] for row in finalization_rows.values()),
            "finalization_manifest_capacity": {"schema": "zkdl_srs_manifest_finalization_heap_v1", "datasets": finalization_rows},
            "bound_methods": {
                "additional_disk_peak": "source_derived_upper_bound",
                "host_peak": "source_derived_upper_bound",
                "gpu_workspace": "authenticated_basis_144B_plus_max_opening_704P_plus_352",
                "shared_host_cuda": "shared_pool_min_host_cuda_ge_host_plus_gpu_workspace",
                "finalization_host_peak": "srs_manifest_schema_python_heap_upper_bound",
            },
            "datasets": rows,
        }

    def test_remaining_capacity_accounts_for_future_retained_prefix(self) -> None:
        profile = self._profile()
        all_remaining = formal_runner._remaining_capacity_requirements(profile, ())
        gpu_workspace = 144 * (1 << 18) + 704 + 352
        self.assertEqual(all_remaining["host_bytes"], 500 + gpu_workspace)
        self.assertEqual(all_remaining["device_bytes"], gpu_workspace)
        self.assertEqual(all_remaining["finalization_host_bytes"], profile["finalization_host_peak_bytes_upper_bound"])
        self.assertEqual(all_remaining["disk_bytes"], 172)
        after_two = formal_runner._remaining_capacity_requirements(profile, ("cora", "citeseer"))
        self.assertEqual(after_two["host_bytes"], 500 + gpu_workspace)
        self.assertEqual(after_two["finalization_host_bytes"], profile["finalization_host_peak_bytes_upper_bound"])
        self.assertEqual(after_two["disk_bytes"], 142)
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(profile, ("citeseer",))
        malformed = json.loads(json.dumps(profile))
        malformed["datasets"]["ppi"]["device_peak_bytes"] = 1
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["datasets"]["ppi"]["additional_disk_peak_bytes"] = 1
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["finalization_manifest_capacity"]["datasets"]["ppi"]["segment_count"] += 1
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["schema"] = "zkdl_formal_capacity_profile_v1"
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        del malformed["datasets"]["ppi"]["native_route_selector_factor_elements"]
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["datasets"]["ppi"]["native_route_selector_factor_elements"] = 0
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["datasets"]["ppi"]["native_route_selector_factor_elements"] = 3
        malformed["datasets"]["ppi"]["native_route_source_domain"] = 1
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        del malformed["datasets"]["ppi"]["native_route_source_domain"]
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["datasets"]["ppi"]["native_route_source_domain"] = (1 << 30) + 1
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())

    @staticmethod
    def _pid_is_live(pid: int) -> bool:
        try:
            text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except OSError:
            return False
        close = text.rfind(")")
        if close < 0:
            return False
        fields = text[close + 2:].split()
        return bool(fields) and fields[0] != "Z"



if __name__ == "__main__":
    unittest.main()
