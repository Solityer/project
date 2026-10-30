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

    def test_resume_failed_session_runs_only_unverified_suffix(self) -> None:
        def exercise(fail_dataset: str | None, suffix: str, *, resume_from: str | None = None,
                     session_status: str = "FAILED", initial_pass_count: int = 3,
                     retry_after_failure: bool = False) -> tuple[int, list[tuple[str, str | None]], dict[str, object],
                                                                list[bytes], list[bytes],
                                                                list[tuple[tuple[str, ...], tuple[str, ...]]], Path | None]:
            with tempfile.TemporaryDirectory(prefix="zkdl-formal-resume-") as raw:
                root = Path(raw)
                runs = root / "runs"
                results = root / "results"
                session_id = suffix * 32
                session = runs / f"formal_{session_id}"
                (session / "phase").mkdir(parents=True)
                (session / "logs").mkdir()
                results_dir = results / session.name
                results_dir.mkdir(parents=True)
                calls: list[tuple[str, str | None]] = []
                capacity_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
                build_calls: list[Path | None] = []
                original_receipts: list[bytes] = []
                original_proofs: list[bytes] = []

                def write_json(path: Path, document: dict[str, object]) -> None:
                    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")

                statuses = {
                    dataset: {"dataset": dataset, "status": "PASS" if index < initial_pass_count else "FAILED",
                              "stage": "independently_verified" if index < initial_pass_count else "failed",
                              "failure": "" if index < initial_pass_count else "old failure"}
                    for index, dataset in enumerate(formal_runner.DATASETS)
                }
                state: dict[str, object] = {
                    "schema": "zkdl_formal_runner_state_v1", "baseline": "zkdl", "session_id": session_id,
                    "session_dir": f"runs/{session.name}", "status": session_status, "stage": "complete",
                    "failure": "old finalization failure", "global_failure": "old global failure",
                    "finished_at_utc": "old finish", "completed_phases": [], "dataset_status": statuses,
                    "capacity": {"profile_schema": "zkdl_formal_capacity_profile_v4", "backend": "cpu",
                                 "gpu_offload_stage": "hyrax_ipa_fold", "gpu_device": {"name": "fixture"},
                                 "completed_cells": list(formal_runner.DATASETS[:initial_pass_count])},
                }
                write_json(session / "state.json", state)
                profile = self._profile()
                prepare = {"schema": "zkDL_formal_prepare_receipt_v3", "baseline": "zkdl",
                           "session_id": session_id, "status": "PASS"}
                write_json(session / "phase" / "prepare.json", prepare)
                for index, dataset in enumerate(formal_runner.DATASETS):
                    (session / dataset).mkdir()
                    (session / dataset / "proof.bin").write_bytes(f"proof-{dataset}".encode())
                    if index >= initial_pass_count:
                        continue
                    verification = session / dataset / "verification.result"
                    verification.write_text("status=VERIFIED\n", encoding="utf-8")
                    receipt = {"schema": "zkDL_formal_cell_receipt_v2", "baseline": "zkdl",
                               "session_id": session_id, "dataset": dataset, "status": "PASS",
                               "independent_verification": True,
                               "verification_path": f"runs/{session.name}/{dataset}/verification.result"}
                    receipt_path = session / "phase" / f"{dataset}.json"
                    write_json(receipt_path, receipt)
                    if index < (2 if resume_from else initial_pass_count):
                        original_receipts.append(receipt_path.read_bytes())
                        original_proofs.append((session / dataset / "proof.bin").read_bytes())
                for dataset in formal_runner.DATASETS[initial_pass_count:]:
                    write_json(session / "phase" / f"{dataset}_failure.json", {
                        "schema": "zkdl_formal_dataset_failure_v1", "baseline": "zkdl",
                        "session_id": session_id, "dataset": dataset, "status": "FAILED",
                        "failure": f"old {dataset} failure"})
                write_json(results_dir / "collected.json", {
                    "schema": "zkDL_gat_formal_partial_result_summary_v1", "baseline": "zkdl",
                    "session_id": session_id, "overall_status": "FAILED"})
                write_json(results_dir / "final_validation.json", {
                    "schema": "zkDL_formal_final_validation_receipt_v1", "baseline": "zkdl",
                    "session_id": session_id, "status": "FAILED",
                    "collection_path": f"results/{session.name}/collected.json"})
                if resume_from:
                    (results_dir / "old-results.csv").write_text("historical results\n", encoding="utf-8")

                def fake_build(_runs: Path, received_id: str, *, resume_session: Path | None = None) -> Path:
                    self.assertEqual(received_id, session_id)
                    build_calls.append(resume_session)
                    return session

                def fake_capacity(_profile: dict[str, object], completed: tuple[str, ...], _session: Path,
                                  *, pending: tuple[str, ...] | None = None) -> dict[str, bool]:
                    capacity_calls.append((completed, pending or ()))
                    return {"checked": True}

                def fake_phase(_session: Path, received_id: str, current: dict[str, object],
                               phase: str, dataset: str | None = None) -> dict[str, object]:
                    calls.append((phase, dataset))
                    if phase == "prepare":
                        raise AssertionError("resume must reuse the existing prepare receipt")
                    if phase == "cell":
                        if dataset == fail_dataset:
                            raise formal_runner.RunnerError(f"fixture {dataset} failure")
                        assert dataset is not None
                        (session / dataset).mkdir(exist_ok=True)
                        (session / dataset / "proof.bin").write_bytes(f"new-proof-{dataset}".encode())
                        verification_path = session / dataset / "verification.result"
                        verification_path.write_text("status=VERIFIED\n", encoding="utf-8")
                        receipt = {"schema": "zkDL_formal_cell_receipt_v2", "baseline": "zkdl",
                                   "session_id": received_id, "dataset": dataset, "status": "PASS",
                                   "independent_verification": True,
                                   "verification_path": verification_path.relative_to(root).as_posix()}
                        write_json(session / "phase" / f"{dataset}.json", receipt)
                        current["dataset_status"][dataset].update(
                            {"status": "PASS", "stage": "complete", "failure": ""})
                        return {"status": "PASS", "dataset": dataset}
                    if phase == "finalize":
                        results_dir.mkdir(parents=True, exist_ok=True)
                        if fail_dataset:
                            failure_index = formal_runner.DATASETS.index(fail_dataset)
                            for pending_dataset in formal_runner.DATASETS[failure_index + 1:]:
                                current["dataset_status"][pending_dataset].update(
                                    {"status": "FAILED", "stage": "finalize", "failure": "not run"})
                            write_json(results_dir / "collected.json", {
                                "schema": "zkDL_gat_formal_partial_result_summary_v1", "baseline": "zkdl",
                                "session_id": received_id, "overall_status": "FAILED"})
                            write_json(results_dir / "final_validation.json", {
                                "schema": "zkDL_formal_final_validation_receipt_v1", "baseline": "zkdl",
                                "session_id": received_id, "status": "FAILED",
                                "collection_path": f"results/{session.name}/collected.json"})
                            return {"status": "FAILED", "session_id": received_id, "failure": "fixture final failure"}
                        for completed_dataset in formal_runner.DATASETS:
                            current["dataset_status"][completed_dataset].update(
                                {"status": "PASS", "stage": "complete", "failure": ""})
                        collection_path = results_dir / "collected.json"
                        validation_path = results_dir / "final_validation.json"
                        write_json(collection_path, {
                            "schema": "zkDL_gat_result_collection_v3", "session_id": session.name,
                            "rows": [{"dataset": item} for item in formal_runner.DATASETS]})
                        write_json(validation_path, {
                            "schema": "zkDL_formal_final_validation_receipt_v1", "baseline": "zkdl",
                            "session_id": received_id, "status": "PASS",
                            "validator_output": ["final_validation=PASS records=5 datasets=5 independent_verify=5"],
                            "collection_path": f"results/{session.name}/collected.json",
                            "verified_cells": list(formal_runner.DATASETS)})
                        return {"status": "PASS", "session_id": received_id,
                                "final_validation_path": f"results/{session.name}/final_validation.json",
                                "collection_path": f"results/{session.name}/collected.json",
                                "dataset_status": current["dataset_status"]}
                    raise AssertionError(f"unexpected phase {phase}")

                env = {formal_runner.RESUME_SESSION_ENV: str(session),
                       formal_runner.RESUME_FROM_ENV: resume_from or ""}
                with (
                    patch.object(formal_runner, "ROOT", root),
                    patch.object(formal_runner, "_build_and_publish_session", fake_build),
                    patch.object(formal_runner, "_write_detached_session_link", lambda *_args: None),
                    patch.object(formal_runner, "_prepare_capacity", lambda *_args: profile),
                    patch.object(formal_runner, "_capacity_gate", fake_capacity),
                    patch.object(formal_runner, "_run_phase", fake_phase),
                    patch.dict(formal_runner.os.environ, env),
                    patch.object(formal_runner.signal, "signal"),
                    patch.object(formal_runner.formal_phase, "SIGNAL_GRACE_SECONDS", 10.0),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    status = formal_runner.run()
                    initial_calls = len(calls)
                    if retry_after_failure:
                        formal_runner.os.environ.pop(formal_runner.RESUME_FROM_ENV, None)
                        fail_dataset = None
                        status = formal_runner.run()
                final_state = json.loads((session / "state.json").read_text(encoding="utf-8"))
                self.assertEqual(build_calls, [session] * (2 if retry_after_failure else 1))
                preserved_count = 2 if resume_from else initial_pass_count
                self.assertEqual([(session / "phase" / f"{dataset}.json").read_bytes()
                                  for dataset in formal_runner.DATASETS[:preserved_count]], original_receipts)
                self.assertEqual([(session / dataset / "proof.bin").read_bytes()
                                  for dataset in formal_runner.DATASETS[:preserved_count]], original_proofs)
                archives = list(runs.glob(f".{session.name}.resume-*"))
                archive = archives[0] if archives else None
                if resume_from:
                    self.assertIsNotNone(archive)
                    assert archive is not None
                    self.assertNotEqual(archive.parent, session)
                    self.assertEqual((archive / "pubmed" / "proof.bin").read_bytes(), b"proof-pubmed")
                    self.assertEqual((archive / "phase" / "pubmed.json").exists(), initial_pass_count >= 3)
                    self.assertEqual((archive / "phase" / "ppi_failure.json").is_file(),
                                     initial_pass_count <= formal_runner.DATASETS.index("ppi"))
                    self.assertTrue((archive / "results" / session.name / "old-results.csv").is_file())
                if retry_after_failure:
                    self.assertEqual(calls[:initial_calls], [
                        ("cell", "pubmed"), ("finalize", None)])
                    self.assertTrue((session / "phase" / "pubmed.json").is_file())
                    self.assertFalse((session / "phase" / "pubmed_failure.json").exists())
                    self.assertEqual(calls[initial_calls:], [
                        ("cell", "pubmed"), ("cell", "ppi"), ("cell", "ogbn_arxiv"), ("finalize", None)])
                return status, calls, final_state, original_receipts, original_proofs, capacity_calls, archive

        status, calls, state, _receipts, _proofs, capacity_calls, _archive = exercise(None, "c")
        self.assertEqual(status, 0)
        self.assertEqual(calls, [("cell", "ppi"), ("cell", "ogbn_arxiv"), ("finalize", None)])
        self.assertEqual(capacity_calls[0], (("cora", "citeseer", "pubmed"), ("ppi",)))
        self.assertEqual(capacity_calls[1], (("cora", "citeseer", "pubmed", "ppi"), ("ogbn_arxiv",)))
        self.assertEqual(state["capacity"]["completed_cells"], list(formal_runner.DATASETS))
        self.assertTrue(all(state["dataset_status"][dataset]["status"] == "PASS"
                            for dataset in formal_runner.DATASETS))

        status, calls, state, _receipts, _proofs, _capacity_calls, _archive = exercise("ppi", "d")
        self.assertEqual(status, 2)
        self.assertEqual(calls, [("cell", "ppi"), ("finalize", None)])
        self.assertEqual(state["capacity"]["completed_cells"], ["cora", "citeseer", "pubmed"])
        self.assertEqual(state["dataset_status"]["ppi"]["status"], "FAILED")
        self.assertNotIn("ogbn_arxiv", state["capacity"]["completed_cells"])

        status, calls, state, _receipts, _proofs, _capacity_calls, _archive = exercise(
            None, "e", resume_from="pubmed", initial_pass_count=3)
        self.assertEqual(status, 0)
        self.assertEqual(calls, [("cell", "pubmed"), ("cell", "ppi"), ("cell", "ogbn_arxiv"),
                                 ("finalize", None)])
        self.assertEqual(state["capacity"]["completed_cells"], list(formal_runner.DATASETS))

        status, calls, _state, _receipts, _proofs, _capacity_calls, _archive = exercise(
            None, "f", resume_from="pubmed", session_status="PASS", initial_pass_count=5)
        self.assertEqual(status, 0)
        self.assertEqual(calls, [("cell", "pubmed"), ("cell", "ppi"), ("cell", "ogbn_arxiv"),
                                 ("finalize", None)])

        status, calls, _state, _receipts, _proofs, _capacity_calls, _archive = exercise(
            "pubmed", "1", resume_from="pubmed", initial_pass_count=3, retry_after_failure=True)
        self.assertEqual(status, 0)

    def test_explicit_resume_rejects_running_session_without_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkdl-formal-running-resume-") as raw:
            root = Path(raw)
            session_id = "9" * 32
            session = root / "runs" / f"formal_{session_id}"
            (session / "phase").mkdir(parents=True)
            (session / "logs").mkdir()
            state = {"schema": "zkdl_formal_runner_state_v1", "baseline": "zkdl",
                     "session_id": session_id, "session_dir": f"runs/{session.name}", "status": "RUNNING",
                     "stage": "cell:ppi", "dataset_status": {dataset: {} for dataset in formal_runner.DATASETS}}
            (session / "state.json").write_text(json.dumps(state), encoding="utf-8")
            (session / "phase" / "pubmed.json").write_text("keep receipt\n", encoding="utf-8")
            (session / "pubmed").mkdir()
            (session / "pubmed" / "proof.bin").write_bytes(b"keep proof")
            before = {path.relative_to(root).as_posix(): path.read_bytes()
                      for path in root.rglob("*") if path.is_file()}
            with (
                patch.object(formal_runner, "ROOT", root),
                patch.dict(formal_runner.os.environ, {
                    formal_runner.RESUME_SESSION_ENV: str(session), formal_runner.RESUME_FROM_ENV: "pubmed"}),
                patch.object(formal_runner.signal, "signal"),
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(formal_runner.run(), 2)
            after = {path.relative_to(root).as_posix(): path.read_bytes()
                     for path in root.rglob("*") if path.is_file()}
            self.assertEqual(after, before)
            self.assertFalse(list((root / "runs").glob(".*.resume-*")))

    def test_cell_start_removes_old_failure_marker_before_retry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkdl-formal-cell-retry-") as raw:
            session = Path(raw) / "formal_fixture"
            (session / "phase").mkdir(parents=True)
            (session / "logs").mkdir()
            failure = session / "phase" / "pubmed_failure.json"
            failure.write_text("old failure\n", encoding="utf-8")
            state = {"status": "RUNNING", "dataset_status": {"pubmed": {"status": "READY"}}}
            with (
                patch.object(formal_runner, "ROOT", Path(raw)),
                patch.object(formal_runner.formal_phase, "_run",
                             side_effect=formal_runner.formal_phase.PhaseError("fixture prove failure")),
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
                self.assertRaises(formal_runner.RunnerError),
            ):
                formal_runner._run_phase(session, "a" * 32, state, "cell", "pubmed")
            self.assertFalse(failure.exists())

    @staticmethod
    def _write_kv(path: Path, values: dict[str, object]) -> None:
        path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")

    def _write_proved_fixture(self, run_dir: Path, dataset: str, statement_id: str,
                              proof: bytes) -> tuple[Path, Path, Path, Path]:
        run_dir.mkdir(parents=True, exist_ok=True)
        proof_path = run_dir / "proof.bin"
        proof_path.write_bytes(proof)
        statement_path = run_dir / "statement.txt"
        self._write_kv(statement_path, {
            "schema": "zkDL_gat_statement_v1", "statement_id": statement_id,
            "dataset": dataset, "proof_scope": "full_gat_forward",
        })
        envelope_path = run_dir / "proof.meta"
        self._write_kv(envelope_path, {
            "schema_id": "zkDL_gat_proof_envelope_v1", "status": "PROVED",
            "backend": "cpu", "proof_scope": "full_gat_forward", "statement_id": statement_id,
            "proof_size_bytes": len(proof),
        })
        timing_path = run_dir / "proof.timing"
        self._write_kv(timing_path, {
            "schema_id": "zkDL_gat_proof_timing_v1", "status": "PROVED", "statement_id": statement_id,
        })
        return proof_path, statement_path, envelope_path, timing_path

    def _run_proved_resume_fixture(self, *, verify_failure: bool) -> tuple[int, list[tuple[str, str]], dict[str, object],
                                                                          list[bytes], tuple[bytes, bytes, bytes]]:
        with tempfile.TemporaryDirectory(prefix="zkdl-formal-verify-resume-") as raw:
            root = Path(raw)
            runs = root / "runs"
            results = root / "results"
            session_id = "a" * 32
            session = runs / f"formal_{session_id}"
            phase_dir = session / "phase"
            logs_dir = session / "logs"
            phase_dir.mkdir(parents=True)
            logs_dir.mkdir()
            results_dir = results / session.name
            results_dir.mkdir(parents=True)
            profile = self._profile()
            profile["session_id"] = session_id
            profile["device"] = {"name": "fixture GPU", "ordinal": 0, "uuid": "GPU-fixture"}
            for dataset in formal_runner.DATASETS:
                profile["datasets"][dataset]["gpu_workspace_required_bytes"] = 1

            statuses = {
                dataset: {"dataset": dataset,
                          "status": "PASS" if index < 3 else "FAILED" if index == 3 else "BLOCKED",
                          "stage": "independently_verified" if index < 3 else "failed",
                          "failure": "" if index < 3 else "previous verification failure"}
                for index, dataset in enumerate(formal_runner.DATASETS)
            }
            state: dict[str, object] = {
                "schema": "zkdl_formal_runner_state_v1", "baseline": "zkdl", "session_id": session_id,
                "session_dir": f"runs/{session.name}", "status": "FAILED", "stage": "complete",
                "failure": "previous PPI verification failure", "global_failure": "",
                "finished_at_utc": "old finish", "completed_phases": [], "dataset_status": statuses,
                "capacity": {"profile_schema": profile["schema"], "backend": "cpu",
                             "gpu_offload_stage": "hyrax_ipa_fold", "gpu_device": profile["device"],
                             "completed_cells": list(formal_runner.DATASETS[:3])},
            }
            (session / "state.json").write_text(json.dumps(state), encoding="utf-8")
            prepare = {
                "schema": "zkDL_formal_prepare_receipt_v3", "baseline": "zkdl", "session_id": session_id,
                "status": "PASS",
                "dataset_status": {dataset: {"dataset": dataset, "status": "PASS"}
                                   for dataset in formal_runner.DATASETS},
                "native_admissions": {dataset: {} for dataset in formal_runner.DATASETS},
                "capacity_profile": profile,
            }
            (phase_dir / "prepare.json").write_text(json.dumps(prepare), encoding="utf-8")
            first_receipts: list[bytes] = []
            for dataset in formal_runner.DATASETS[:3]:
                run_dir = session / dataset
                run_dir.mkdir()
                (run_dir / "proof.bin").write_bytes(f"proof-{dataset}".encode())
                verification_path = run_dir / "verification.result"
                verification_path.write_text("status=VERIFIED\n", encoding="utf-8")
                receipt_path = phase_dir / f"{dataset}.json"
                receipt_path.write_text(json.dumps({
                    "schema": "zkDL_formal_cell_receipt_v2", "baseline": "zkdl",
                    "session_id": session_id, "dataset": dataset, "status": "PASS",
                    "independent_verification": True,
                    "verification_path": f"runs/{session.name}/{dataset}/verification.result",
                }), encoding="utf-8")
                first_receipts.append(receipt_path.read_bytes())

            ppi_dir = session / "ppi"
            statement_id = "b" * 64
            ppi_paths = self._write_proved_fixture(ppi_dir, "ppi", statement_id, b"large-proof-fixture")
            (ppi_dir / "run.record").write_text(
                "schema=zkDL_gat_run_record_v2\ndataset=ppi\nbackend=cpu\nstatement_id=" + statement_id +
                "\nverification_status=BLOCKED\nproof_scope=full_gat_forward\nproof_size_bytes=0\n"
                "commitment_time_seconds=0\nprove_time_seconds=0\nverify_time_seconds=0\n", encoding="utf-8")
            (ppi_dir / "verification.result").write_text(
                f"status=BLOCKED\nstatement_id={statement_id}\n", encoding="utf-8")
            ppi_artifacts_before = tuple(path.read_bytes() for path in ppi_paths)

            (phase_dir / "ppi_failure.json").write_text("old PPI failure\n", encoding="utf-8")
            collection_path = results_dir / "collected.json"
            validation_path = results_dir / "final_validation.json"
            collection_path.write_text(json.dumps({
                "schema": "zkDL_gat_formal_partial_result_summary_v1", "baseline": "zkdl",
                "session_id": session_id, "overall_status": "FAILED",
            }), encoding="utf-8")
            validation_path.write_text(json.dumps({
                "schema": "zkDL_formal_final_validation_receipt_v1", "baseline": "zkdl",
                "session_id": session_id, "status": "FAILED",
                "collection_path": f"results/{session.name}/collected.json",
            }), encoding="utf-8")

            native_calls: list[tuple[str, str]] = []
            cell_calls: list[tuple[str, str | None]] = []

            def native_run(command: list[str], *, log_path: Path | None = None,
                           env: dict[str, str] | None = None) -> str:
                action = command[1]
                run_dir = Path(command[-1])
                dataset = run_dir.name
                native_calls.append((dataset, action))
                if action == "prove":
                    generated_id = ("c" if dataset == "ogbn_arxiv" else "d") * 64
                    self._write_proved_fixture(run_dir, dataset, generated_id, f"new-{dataset}-proof".encode())
                elif action == "verify":
                    if dataset == "ppi" and verify_failure:
                        (run_dir / "verification.result").write_text(
                            f"status=BLOCKED\nstatement_id={statement_id}\n", encoding="utf-8")
                        raise formal_runner.formal_phase.PhaseError("fixture native PPI Verify failed")
                    id_value = "c" * 64 if dataset == "ogbn_arxiv" else statement_id
                    (run_dir / "verification.result").write_text(
                        f"status=VERIFIED\nstatement_id={id_value}\n", encoding="utf-8")
                else:
                    raise AssertionError(f"unexpected native action: {action}")
                return ""

            def fake_phase(current_session: Path, received_id: str, current_state: dict[str, object],
                           phase: str, dataset: str | None = None) -> dict[str, object]:
                if phase == "cell":
                    assert dataset is not None
                    cell_calls.append((phase, dataset))
                    formal_phase_failure = current_session / "phase" / f"{dataset}_failure.json"
                    if formal_phase_failure.exists():
                        formal_phase_failure.unlink()
                    formal_runner.formal_phase.cell(current_session, received_id, dataset)
                    current_state["dataset_status"][dataset].update(
                        {"status": "PASS", "stage": "complete", "failure": ""})
                    return {"status": "PASS", "dataset": dataset}
                if phase == "finalize":
                    cell_calls.append((phase, None))
                    if verify_failure:
                        return {"status": "FAILED", "session_id": received_id,
                                "failure": "fixture PPI Verify failed"}
                    for dataset_name in formal_runner.DATASETS:
                        current_state["dataset_status"][dataset_name].update(
                            {"status": "PASS", "stage": "complete", "failure": ""})
                    return {"status": "PASS", "session_id": received_id,
                            "final_validation_path": f"results/{session.name}/final_validation.json",
                            "collection_path": f"results/{session.name}/collected.json",
                            "dataset_status": current_state["dataset_status"]}
                raise AssertionError(f"unexpected phase: {phase}")

            def fake_audit(_root: Path, run_dir: Path, **_kwargs: object) -> dict[str, str]:
                dataset = run_dir.name
                id_value = "c" * 64 if dataset == "ogbn_arxiv" else statement_id
                return {
                    "statement_id": id_value, "commitment_time_seconds": "1.0", "prove_time_seconds": "2.0",
                    "verify_time_seconds": "3.0", "gpu_device_name": "fixture GPU",
                    "gpu_device_ordinal": "0", "gpu_device_uuid": "GPU-fixture",
                    "gpu_proof_peak_allocated_bytes": "0", "gpu_offload_stage": "hyrax_ipa_fold",
                    "gpu_forward_kernel_executed": "0", "gpu_proof_kernel_executed": "1",
                    "gpu_kernel_executed": "1", "gpu_proof_kernel_count": "1",
                    "independent_verification": "1", "verifier_backend": "cpu",
                    "verify_gpu_kernel_executed": "0",
                }

            with (
                patch.object(formal_runner, "ROOT", root),
                patch.object(formal_runner, "_build_and_publish_session", lambda *_args, **_kwargs: session),
                patch.object(formal_runner, "_write_detached_session_link", lambda *_args: None),
                patch.object(formal_runner, "_prepare_capacity", lambda *_args: profile),
                patch.object(formal_runner, "_capacity_gate", lambda *_args, **_kwargs: {"checked": True}),
                patch.object(formal_runner, "_run_phase", fake_phase),
                patch.object(formal_runner, "_validate_final_pass", lambda *_args: "results/final_validation.json"),
                patch.object(formal_runner.formal_phase, "ROOT", root),
                patch.object(formal_runner.formal_phase, "_config", lambda *_args: root / "fixture.cfg"),
                patch.object(formal_runner.formal_phase, "_run", native_run),
                patch.object(result_artifacts, "audit_run", fake_audit),
                patch.dict(formal_runner.os.environ, {
                    formal_runner.RESUME_SESSION_ENV: str(session), formal_runner.RESUME_FROM_ENV: "",
                    formal_runner.RESUME_VERIFY_ENV: "ppi",
                }),
                patch.object(formal_runner.signal, "signal"),
                patch.object(formal_runner.formal_phase, "SIGNAL_GRACE_SECONDS", 10.0),
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
            ):
                status = formal_runner.run()

            final_state = json.loads((session / "state.json").read_text(encoding="utf-8"))
            final_receipts = [(phase_dir / f"{dataset}.json").read_bytes()
                              for dataset in formal_runner.DATASETS[:3]]
            ppi_artifacts_after = tuple(path.read_bytes() for path in ppi_paths)
            self.assertEqual(final_receipts, first_receipts)
            self.assertEqual(ppi_artifacts_after, ppi_artifacts_before)
            if verify_failure:
                self.assertEqual(cell_calls, [("cell", "ppi"), ("finalize", None)])
            else:
                self.assertEqual(cell_calls, [("cell", "ppi"), ("cell", "ogbn_arxiv"), ("finalize", None)])
            return status, native_calls, final_state, final_receipts, ppi_artifacts_after

    def test_resume_verify_reuses_ppi_proof_then_proves_ogbn(self) -> None:
        status, native_calls, state, receipts, ppi_artifacts = self._run_proved_resume_fixture(verify_failure=False)
        self.assertEqual(status, 0)
        self.assertEqual(native_calls, [("ppi", "verify"), ("ogbn_arxiv", "prove"), ("ogbn_arxiv", "verify")])
        self.assertEqual(state["status"], "PASS")
        self.assertTrue(all(state["dataset_status"][dataset]["status"] == "PASS"
                            for dataset in formal_runner.DATASETS))
        self.assertEqual(len(receipts), 3)
        self.assertEqual(len(ppi_artifacts), 4)

    def test_resume_verify_failure_keeps_failed_and_does_not_start_ogbn(self) -> None:
        status, native_calls, state, receipts, ppi_artifacts = self._run_proved_resume_fixture(verify_failure=True)
        self.assertEqual(status, 2)
        self.assertEqual(native_calls, [("ppi", "verify")])
        self.assertEqual(state["status"], "FAILED")
        self.assertEqual(state["dataset_status"]["ppi"]["status"], "FAILED")
        self.assertEqual(len(receipts), 3)
        self.assertEqual(len(ppi_artifacts), 4)

    def test_reuse_proved_preflight_rejects_bad_artifacts_without_proving_or_cleanup(self) -> None:
        cases = ("missing_timing", "mismatched_statement", "wrong_size")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="zkdl-proved-gate-") as raw:
                root = Path(raw)
                session = root / "runs" / f"formal_{'e' * 32}"
                (session / "phase").mkdir(parents=True)
                (session / "logs").mkdir()
                proof_dir = session / "ppi"
                proof_dir.mkdir()
                profile = self._profile()
                (session / "phase" / "prepare.json").write_text(json.dumps({
                    "status": "PASS", "session_id": "e" * 32,
                    "dataset_status": {"ppi": {"status": "PASS"}},
                    "native_admissions": {"ppi": {}},
                    "capacity_profile": {
                        "schema": "zkdl_formal_capacity_profile_v4",
                        "gpu_offload_stage": "hyrax_ipa_fold",
                        "memory_accounting_kind": "shared_host_cuda",
                        "device": profile["device"],
                        "datasets": {"ppi": {"gpu_workspace_required_bytes": 1}},
                    },
                }), encoding="utf-8")
                paths = self._write_proved_fixture(proof_dir, "ppi", "f" * 64, b"proved-fixture")
                proof_path, statement_path, envelope_path, timing_path = paths
                if case == "missing_timing":
                    timing_path.unlink()
                elif case == "mismatched_statement":
                    self._write_kv(envelope_path, {
                        "schema_id": "zkDL_gat_proof_envelope_v1", "status": "PROVED", "backend": "cpu",
                        "proof_scope": "full_gat_forward", "statement_id": "0" * 64,
                        "proof_size_bytes": proof_path.stat().st_size,
                    })
                else:
                    self._write_kv(envelope_path, {
                        "schema_id": "zkDL_gat_proof_envelope_v1", "status": "PROVED", "backend": "cpu",
                        "proof_scope": "full_gat_forward", "statement_id": "f" * 64,
                        "proof_size_bytes": proof_path.stat().st_size + 1,
                    })
                private = proof_dir / ".private_native"
                private.mkdir()
                (private / "keep").write_bytes(b"preserve until gate passes")
                before = {path.name: path.read_bytes() for path in proof_dir.iterdir()
                          if path.is_file() and not path.is_symlink()}
                native_calls: list[list[str]] = []
                with (
                    patch.object(formal_runner.formal_phase, "ROOT", root),
                    patch.object(formal_runner.formal_phase, "_config", lambda *_args: root / "fixture.cfg"),
                    patch.object(formal_runner.formal_phase, "_run",
                                 lambda command, **_kwargs: native_calls.append(command) or ""),
                    patch.dict(formal_runner.os.environ, {formal_runner.RESUME_VERIFY_ENV: "ppi"}),
                    redirect_stdout(io.StringIO()), self.assertRaises(formal_runner.formal_phase.PhaseError),
                ):
                    formal_runner.formal_phase.cell(session, "e" * 32, "ppi")
                after = {path.name: path.read_bytes() for path in proof_dir.iterdir()
                         if path.is_file() and not path.is_symlink()}
                self.assertEqual(after, before)
                self.assertTrue((private / "keep").is_file())
                self.assertEqual(native_calls, [])

    def test_verify_resume_requires_failed_session_first_unverified_cell(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkdl-verify-resume-state-") as raw:
            root = Path(raw)
            session_id = "7" * 32
            session = root / "runs" / f"formal_{session_id}"
            (session / "phase").mkdir(parents=True)
            (session / "logs").mkdir()
            state = {"schema": "zkdl_formal_runner_state_v1", "baseline": "zkdl",
                     "session_id": session_id, "session_dir": f"runs/{session.name}", "status": "FAILED",
                     "dataset_status": {dataset: {"status": "PASS" if index < 3 else "FAILED"}
                                        for index, dataset in enumerate(formal_runner.DATASETS)}}
            (session / "state.json").write_text(json.dumps(state), encoding="utf-8")
            for dataset in formal_runner.DATASETS[:3]:
                (session / "phase" / f"{dataset}.json").write_text(json.dumps({
                    "schema": "zkDL_formal_cell_receipt_v2", "baseline": "zkdl", "session_id": session_id,
                    "dataset": dataset, "status": "PASS", "independent_verification": True,
                    "verification_path": f"runs/{session.name}/{dataset}/verification.result",
                }), encoding="utf-8")
                run_dir = session / dataset
                run_dir.mkdir()
                (run_dir / "verification.result").write_text("status=VERIFIED\n", encoding="utf-8")
            with patch.object(formal_runner, "ROOT", root), patch.dict(formal_runner.os.environ, {
                formal_runner.RESUME_SESSION_ENV: str(session), formal_runner.RESUME_FROM_ENV: "",
                formal_runner.RESUME_VERIFY_ENV: "ogbn_arxiv",
            }):
                with self.assertRaisesRegex(formal_runner.RunnerError, "first unverified"):
                    formal_runner._load_resume_session(root / "runs", resume_verify="ogbn_arxiv")

    def test_verify_resume_cannot_be_combined_with_resume_from(self) -> None:
        with patch.dict(formal_runner.os.environ, {
            formal_runner.RESUME_SESSION_ENV: "/tmp/unused-session",
            formal_runner.RESUME_FROM_ENV: "pubmed",
            formal_runner.RESUME_VERIFY_ENV: "ppi",
        }), self.assertRaisesRegex(formal_runner.RunnerError, "cannot be used together"):
            formal_runner.run()

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
