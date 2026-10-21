#!/usr/bin/env python3
"""Read-only liveness checks for the zkGPT formal progress watcher."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import formal_progress
import stop_formal


class FormalProgressLivenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="zkgpt-progress-liveness-")
        self.root = Path(self.temporary.name)
        self.launch_id = "a" * 32
        self.launch_dir = self.root / "baseline-local" / "formal-detached" / self.launch_id
        self.launch_dir.mkdir(parents=True)
        (self.root / "scripts").mkdir(parents=True)
        self.status_path = self.launch_dir / "status.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_status(self, status: str, supervisor_pid: int = 2_000_000_000) -> dict:
        supervisor_script = self.root / "scripts" / "formal_detached_supervisor.py"
        runner = self.root / "scripts" / "run_formal.py"
        state = {
            "schema": "zkgpt-gat-detached-launch-v2", "launch_id": self.launch_id, "status": status,
            "root_path": str(self.root), "status_path": str(self.status_path),
            "console_path": str(self.launch_dir / "console.log"), "runner_path": str(runner),
            "supervisor_pid": supervisor_pid, "supervisor_start_ticks": 101,
            "supervisor_process_group": supervisor_pid, "supervisor_cwd": str(self.root),
            "supervisor_executable": sys.executable,
            "supervisor_argv": [sys.executable, str(supervisor_script), "--root", str(self.root), "--launch-id", self.launch_id],
            "child_pid": None, "child_start_ticks": None, "child_process_group": None,
            "child_cwd": None, "child_executable": None, "child_argv": None,
        }
        if status == "RUNNING":
            child_pid = supervisor_pid + 1
            state.update({
                "child_pid": child_pid, "child_start_ticks": 102, "child_process_group": child_pid,
                "child_cwd": str(self.root), "child_executable": sys.executable,
                "child_argv": [sys.executable, str(runner)],
            })
        self.status_path.write_text(json.dumps(state), encoding="utf-8")
        return state

    def test_dead_supervisor_reports_no_live_owner_without_mutating_status(self) -> None:
        self.write_status("RUNNING")
        before = self.status_path.read_bytes()
        health, _, reason = formal_progress.inspect_live_launch(self.root, self.launch_id, self.status_path)
        self.assertEqual(health, "NO_LIVE_OWNER", reason)
        self.assertEqual(self.status_path.read_bytes(), before)

    def test_live_running_and_terminal_status_are_read_only(self) -> None:
        state = self.write_status("RUNNING", os.getpid())

        def identity(pid: int) -> dict:
            role = "supervisor" if pid == state["supervisor_pid"] else "child"
            return {
                "start_ticks": state[f"{role}_start_ticks"],
                "process_group": state[f"{role}_process_group"],
                "cwd": state[f"{role}_cwd"], "executable": state[f"{role}_executable"],
                "argv": state[f"{role}_argv"],
            }

        before = self.status_path.read_bytes()
        with mock.patch.object(stop_formal, "process_identity", side_effect=identity):
            health, current, _ = formal_progress.inspect_live_launch(self.root, self.launch_id, self.status_path)
        self.assertEqual((health, current["status"]), ("LIVE", "RUNNING"))
        self.assertEqual(self.status_path.read_bytes(), before)
        self.write_status("SUBSET_FAILED")
        terminal_bytes = self.status_path.read_bytes()
        with mock.patch.object(stop_formal, "process_identity", side_effect=AssertionError("terminal state must not inspect processes")):
            health, current, _ = formal_progress.inspect_live_launch(self.root, self.launch_id, self.status_path)
        self.assertEqual((health, current["status"]), ("TERMINAL", "SUBSET_FAILED"))
        self.assertEqual(self.status_path.read_bytes(), terminal_bytes)


if __name__ == "__main__":
    unittest.main()
