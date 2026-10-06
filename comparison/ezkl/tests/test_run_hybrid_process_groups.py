"""Bounded lifecycle checks for formal prerequisite child process groups."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formal_common import ContractError  # noqa: E402
from run_hybrid_formal import run_prerequisite  # noqa: E402


class FormalPrerequisiteProcessGroupTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "process-group lifecycle check requires POSIX")
    def test_nonzero_leader_cleans_descendant_and_preserves_status_log(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-prerequisite-pg-") as temp:
            root = Path(temp)
            child_pid_file = root / "child.pid"
            log_path = root / "prerequisite.log"
            code = f"""\
import os, signal, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open({str(child_pid_file)!r}, 'w') as stream:
        stream.write(str(os.getpid()))
        stream.flush()
    time.sleep(30)
    os._exit(0)
print('prerequisite leader exiting 23', flush=True)
os._exit(23)
"""
            with self.assertRaisesRegex(ContractError, "exit code 23"):
                run_prerequisite(
                    [sys.executable, "-c", code],
                    "bounded process-group test",
                    timeout_seconds=20,
                    log_path=log_path,
                )

            self.assertIn("prerequisite leader exiting 23", log_path.read_text())
            child_pid = int(child_pid_file.read_text())
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                stat_path = Path(f"/proc/{child_pid}/stat")
                if not stat_path.exists():
                    break
                state = stat_path.read_text().split()[2]
                if state == "Z":
                    break
                time.sleep(0.05)
            else:
                self.fail(f"descendant process {child_pid} remained alive after leader failure")


if __name__ == "__main__":
    unittest.main()
