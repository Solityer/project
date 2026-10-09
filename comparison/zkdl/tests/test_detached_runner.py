#!/usr/bin/env python3
"""Bounded lifecycle fixtures for the baseline-local detached supervisor."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import detached_supervisor


class DetachedRunnerTests(unittest.TestCase):
    def _launch_fixture(self, launch_dir: Path, child_code: str) -> subprocess.Popen[bytes]:
        (launch_dir / "console.log").write_bytes(b"")
        bootstrap = (f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); from pathlib import Path; "
                     "from detached_supervisor import run_supervisor; "
                     f"raise SystemExit(run_supervisor([sys.executable, '-c', {child_code!r}], "
                     f"Path({str(ROOT)!r}), Path({str(launch_dir)!r})))")
        with (launch_dir / "console.log").open("ab") as console:
            return subprocess.Popen(["nohup", "setsid", sys.executable, "-c", bootstrap], stdin=subprocess.DEVNULL,
                                    stdout=console, stderr=subprocess.STDOUT, close_fds=True)

    def _wait_exit(self, launch_dir: Path) -> dict[str, object]:
        exit_path = launch_dir / "exit.json"
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not exit_path.is_file():
            time.sleep(0.02)
        self.assertTrue(exit_path.is_file(), "detached supervisor did not write its final status")
        return json.loads(exit_path.read_text(encoding="utf-8"))

    def _wait_pid(self, launch_dir: Path) -> dict[str, object]:
        pid_path = launch_dir / "pid.json"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not pid_path.is_file():
            time.sleep(0.02)
        self.assertTrue(pid_path.is_file(), "detached supervisor did not publish its PID")
        return json.loads(pid_path.read_text(encoding="utf-8"))

    def test_detached_success_survives_hangup_and_writes_exact_status(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkdl-detached-success-") as raw:
            launch_dir = Path(raw)
            process = self._launch_fixture(launch_dir, "import time; print('fixture-success', flush=True); time.sleep(0.1)")
            try:
                pid = self._wait_pid(launch_dir)
                self.assertEqual(pid["supervisor_pid"], process.pid)
                self.assertEqual(pid["supervisor_session_id"], process.pid)
                os.kill(process.pid, signal.SIGHUP)
                self.assertEqual(process.wait(timeout=5), 0)
                result = self._wait_exit(launch_dir)
                self.assertEqual((result["status"], result["exit_code"], result["child_returncode"]), ("PASS", 0, 0))
                self.assertEqual(result["schema"], "zkdl_detached_launch_exit_v1")
                self.assertIn("fixture-success", (launch_dir / "console.log").read_text(encoding="utf-8"))
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=5)

    def test_detached_failure_preserves_numeric_child_exit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkdl-detached-failure-") as raw:
            launch_dir = Path(raw)
            process = self._launch_fixture(launch_dir, "raise SystemExit(23)")
            self.assertEqual(process.wait(timeout=5), 23)
            result = self._wait_exit(launch_dir)
            self.assertEqual((result["status"], result["exit_code"], result["child_returncode"]), ("FAILED", 23, 23))
            self.assertIsNone(result["signal"])

    def test_term_records_interruption_and_forwards_to_owned_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkdl-detached-signal-") as raw:
            launch_dir = Path(raw)
            process = self._launch_fixture(launch_dir, "import time; time.sleep(30)")
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not (launch_dir / "pid.json").is_file():
                    time.sleep(0.02)
                self.assertTrue((launch_dir / "pid.json").is_file(), "detached supervisor did not publish its PID")
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=6), 143)
                result = self._wait_exit(launch_dir)
                self.assertEqual((result["status"], result["exit_code"], result["signal"]), ("INTERRUPTED", 143, signal.SIGTERM))
                self.assertEqual(result["schema"], "zkdl_detached_launch_exit_v1")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    def test_baseline_wrapper_publishes_launch_pointer_and_runs_entry_fixture(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkdl-detached-wrapper-") as raw:
            root = Path(raw) / "zkdl"
            scripts = root / "scripts"
            runs = root / "runs"
            scripts.mkdir(parents=True)
            for name in ("run_five_detached.sh", "detached_supervisor.py"):
                shutil.copy2(SCRIPTS / name, scripts / name)
            entry = scripts / "run_five.sh"
            entry.write_text("#!/usr/bin/env bash\nprintf 'fixture-run-five\\n'\nexit 17\n", encoding="utf-8")
            entry.chmod(0o700)

            completed = subprocess.run([str(scripts / "run_five_detached.sh")], cwd=root,
                                        capture_output=True, text=True, timeout=8, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
            launch_id = Path(output["zkdl_detached_launch_dir"]).name
            self.assertEqual((runs / "latest_launch").read_text(encoding="utf-8").strip(), launch_id)
            launch_dir = runs / launch_id
            self.assertTrue(launch_dir.is_dir())
            self.assertTrue((launch_dir / "console.log").is_file())
            self.assertEqual(int((launch_dir / "supervisor.pid").read_text(encoding="ascii").strip()),
                             int(output["zkdl_detached_supervisor_pid"]))
            result = self._wait_exit(launch_dir)
            self.assertEqual((result["status"], result["exit_code"], result["child_returncode"]),
                             ("FAILED", 17, 17))
            self.assertEqual(result["schema"], "zkdl_detached_launch_exit_v1")
            self.assertIn("fixture-run-five", (launch_dir / "console.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
