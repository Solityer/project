#!/usr/bin/env python3
"""Targeted regressions for the baseline-local formal lifecycle and admission."""
from __future__ import annotations

import fcntl
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
sys.path.insert(0, str(SCRIPTS))
import formal_runner


class FormalLocalRunnerTests(unittest.TestCase):
    def _profile(self) -> dict[str, object]:
        rows: dict[str, dict[str, object]] = {}
        for index, dataset in enumerate(formal_runner.DATASETS, start=1):
            retained = index * 10
            rows[dataset] = {
                "additional_disk_peak_bytes": retained + index * 3,
                "cell_retained_artifact_bytes_upper_bound": retained,
                "host_peak_bytes": index * 100,
                "device_peak_bytes": 0,
                "phase_coverage": ["setup", "witness", "commitment", "prove", "verify"],
            }
        return {
            "schema": "zkdl_formal_capacity_profile_v1",
            "baseline": "zkdl",
            "backend": "cpu",
            "memory_accounting_kind": "host-only",
            "device": None,
            "session_id": "0" * 32,
            "fs_device_id": 1,
            "preexisting_baseline_artifact_bytes": 0,
            "finalization_disk_bytes_upper_bound": 7,
            "finalization_host_peak_bytes_upper_bound": 500,
            "bound_methods": {
                "additional_disk_peak": "source_derived_upper_bound",
                "host_peak": "source_derived_upper_bound",
            },
            "datasets": rows,
        }

    def test_remaining_capacity_accounts_for_future_retained_prefix(self) -> None:
        profile = self._profile()
        all_remaining = formal_runner._remaining_capacity_requirements(profile, ())
        self.assertEqual(all_remaining["host_bytes"], 500)
        self.assertEqual(all_remaining["device_bytes"], 0)
        self.assertEqual(all_remaining["finalization_host_bytes"], 500)
        self.assertEqual(all_remaining["disk_bytes"], 172)
        after_two = formal_runner._remaining_capacity_requirements(profile, ("cora", "citeseer"))
        self.assertEqual(after_two["host_bytes"], 500)
        self.assertEqual(after_two["finalization_host_bytes"], 500)
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
