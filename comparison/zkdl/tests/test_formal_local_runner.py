#!/usr/bin/env python3
"""Targeted regressions for the baseline-local formal lifecycle and admission."""
from __future__ import annotations

import ctypes
import fcntl
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
sys.path.insert(0, str(SCRIPTS))
import formal_runner
import formal_provenance
import session_lifecycle


class FormalLocalRunnerTests(unittest.TestCase):
    def test_session_root_lifecycle_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=TESTS) as temporary:
            project = Path(temporary)
            session_id = uuid.uuid4().hex
            runs = project / "runs"
            session = runs / f"formal_{session_id}"
            for directory in (runs, project / "configs", project / "results"):
                directory.mkdir(parents=True, exist_ok=True)
            session.mkdir()
            identity = session_lifecycle.capture_session_root(session)
            old_identity = formal_runner.ACTIVE_SESSION_IDENTITY
            old_phase_root = formal_runner.formal_phase.ROOT
            old_provenance_root = formal_provenance.ROOT
            environment_names = (
                session_lifecycle.ROOT_ENV,
                session_lifecycle.DEVICE_ENV,
                session_lifecycle.INODE_ENV,
            )
            old_environment = {name: os.environ.get(name) for name in environment_names}
            os.environ.update(identity.child_environment())
            formal_runner.ACTIVE_SESSION_IDENTITY = identity
            formal_runner.formal_phase.ROOT = project
            formal_provenance.ROOT = project
            try:
                session_lifecycle.require_session_root(identity, session)
                for relative in (
                    "phase", "logs", "build", "cora", "ppi",
                    "resources/cache/cora", "resources/cache/ppi",
                    "resources/checkpoints/cora_gat", "resources/checkpoints/ppi_gat",
                    "resources/canonical/tables",
                ):
                    (session / relative).mkdir(parents=True, exist_ok=True)
                (session / "build/gat_runner").write_text("frozen binary\n")
                (session / "logs/cora.log").write_text("formal log\n")
                (session / "resources/canonical/tables/authenticated.json").write_text("authenticated\n")
                (session / "ppi/manifest.json").write_text("authenticated snapshot\n")
                (session / "cora/partial.bin").write_text("failed snapshot\n")
                (project / "configs" / f"{session.name}_cora.cfg").write_text("failed snapshot config\n")
                result = project / "results" / session.name
                result.mkdir()
                (result / "collected.json").write_text("existing result\n")

                formal_runner._write_json(session / "state.json", {"session_id": session_id})
                state = {"session_id": session_id, "dataset_status": {"cora": {}}}
                formal_runner._record_dataset_failure(session, state, "cora", "fixture", "ordinary failure")
                self.assertTrue((session / "phase/cora_failure.json").is_file())

                formal_runner.formal_phase._remove_failed_snapshot(session, "cora")
                self.assertFalse((session / "cora").exists())
                self.assertFalse((session / "resources/cache/cora").exists())
                self.assertFalse((session / "resources/checkpoints/cora_gat").exists())
                self.assertFalse((project / "configs" / f"{session.name}_cora.cfg").exists())
                self.assertTrue((session / "build/gat_runner").is_file())
                self.assertTrue((session / "logs/cora.log").is_file())
                self.assertFalse((session / "results").exists())
                self.assertTrue((result / "collected.json").is_file())
                self.assertTrue((session / "resources/canonical/tables/authenticated.json").is_file())
                self.assertTrue((session / "ppi/manifest.json").is_file())

                tombstone = runs / f"{session.name}.original"
                session.rename(tombstone)
                session.mkdir()
                with self.assertRaisesRegex(session_lifecycle.SessionRootError, "must not mask"):
                    formal_runner._record_dataset_failure(session, state, "cora", "fixture", "must not mask")
                self.assertTrue(session.is_dir())
                self.assertFalse((session / "phase").exists())

                with mock.patch.object(formal_provenance.subprocess, "run") as build:
                    with mock.patch.object(sys, "argv", ["formal_provenance.py", "--output",
                                                           str(session / "build_manifest.json")]):
                        with self.assertRaises(session_lifecycle.SessionRootError):
                            formal_provenance.main()
                    build.assert_not_called()

                shutil.rmtree(session)
                with self.assertRaises(session_lifecycle.SessionRootError):
                    session_lifecycle.require_session_root(identity, session)
                with self.assertRaises(session_lifecycle.SessionRootError):
                    formal_runner._write_json(session / "state.json", {"session_id": session_id})
                self.assertFalse(session.exists())
            finally:
                formal_runner.ACTIVE_SESSION_IDENTITY = old_identity
                formal_runner.formal_phase.ROOT = old_phase_root
                formal_provenance.ROOT = old_provenance_root
                for name, value in old_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                if identity.root_fd is not None:
                    os.close(identity.root_fd)

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

    def _profile(self) -> dict[str, object]:
        rows: dict[str, dict[str, object]] = {}
        finalization_rows: dict[str, dict[str, int]] = {}
        for index, dataset in enumerate(formal_runner.DATASETS, start=1):
            retained = index * 10
            finalization_rows[dataset] = formal_runner.formal_phase._srs_manifest_finalization_capacity(index, index)
            rows[dataset] = {
                "additional_disk_peak_bytes": retained + index * 3,
                "cell_retained_artifact_bytes_upper_bound": retained,
                "host_peak_bytes": index * 100,
                "device_peak_bytes": 0,
                "native_route_source_domain": index,
                "native_route_selector_factor_elements": index + 1,
                "phase_coverage": ["setup", "witness", "commitment", "prove", "verify"],
            }
        return {
            "schema": "zkdl_formal_capacity_profile_v3",
            "baseline": "zkdl",
            "backend": "cpu",
            "memory_accounting_kind": "host-only",
            "device": None,
            "session_id": "0" * 32,
            "fs_device_id": 1,
            "preexisting_baseline_artifact_bytes": 0,
            "finalization_disk_bytes_upper_bound": 7,
            "finalization_host_peak_bytes_upper_bound": max(row["host_peak_bytes_upper_bound"] for row in finalization_rows.values()),
            "finalization_manifest_capacity": {"schema": "zkdl_srs_manifest_finalization_heap_v1", "datasets": finalization_rows},
            "bound_methods": {
                "additional_disk_peak": "source_derived_upper_bound",
                "host_peak": "source_derived_upper_bound",
                "finalization_host_peak": "srs_manifest_schema_python_heap_upper_bound",
            },
            "datasets": rows,
        }

    def test_remaining_capacity_accounts_for_future_retained_prefix(self) -> None:
        profile = self._profile()
        all_remaining = formal_runner._remaining_capacity_requirements(profile, ())
        self.assertEqual(all_remaining["host_bytes"], 500)
        self.assertEqual(all_remaining["device_bytes"], 0)
        self.assertEqual(all_remaining["finalization_host_bytes"], profile["finalization_host_peak_bytes_upper_bound"])
        self.assertEqual(all_remaining["disk_bytes"], 172)
        after_two = formal_runner._remaining_capacity_requirements(profile, ("cora", "citeseer"))
        self.assertEqual(after_two["host_bytes"], 500)
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

    def test_owner_sigkill_terminates_phase_descendants_and_releases_lock(self) -> None:
        runs = ROOT / "runs"
        runs.mkdir(exist_ok=True)
        session_id = uuid.uuid4().hex
        session = runs / f"formal_{session_id}"
        session.mkdir(mode=0o700)
        ready_path = session / "owner_ready.json"
        pids_path = session / "descendant_pids.json"
        lock_path = runs / ".formal.lock"
        helper = TESTS / "formal_local_owner_death_helper.py"
        owner = subprocess.Popen(
            [sys.executable, str(helper), "owner", str(ROOT), str(session), session_id,
             str(ready_path), str(pids_path), "formal_runner.py"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not pids_path.is_file():
                if owner.poll() is not None:
                    break
                time.sleep(0.05)
            self.assertTrue(pids_path.is_file(), "the phase-owned process group did not start")
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            pids = json.loads(pids_path.read_text(encoding="utf-8"))
            os.kill(owner.pid, signal.SIGKILL)
            owner.wait(timeout=5)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                tracked = [int(ready["phase_pid"]), int(pids["child"]), int(pids["grandchild"])]
                if not any(self._pid_is_live(pid) for pid in tracked):
                    break
                time.sleep(0.05)
            self.assertFalse(self._pid_is_live(int(ready["phase_pid"])), "phase survived its lifecycle owner")
            self.assertFalse(self._pid_is_live(int(pids["child"])), "phase child survived owner death")
            self.assertFalse(self._pid_is_live(int(pids["grandchild"])), "phase grandchild survived owner death")
            state = json.loads((session / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state.get("status"), "INTERRUPTED")
            self.assertEqual(state.get("interrupted_stage"), "cell:cora")
            lock_fd = os.open(lock_path, os.O_RDWR)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        finally:
            if owner.poll() is None:
                os.kill(owner.pid, signal.SIGKILL)
                owner.wait(timeout=5)
            shutil.rmtree(session, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
