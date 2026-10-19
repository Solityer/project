#!/usr/bin/env python3
"""Bounded stub-only tests for the zkGPT SSH-safe formal wrapper."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DetachedFormalRunnerSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="zkgpt-detached-smoke-")
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        shutil.copy2(ROOT / "scripts" / "run_formal_detached.sh", scripts)
        shutil.copy2(ROOT / "scripts" / "formal_detached_supervisor.py", scripts)
        shutil.copy2(ROOT / "scripts" / "common.py", scripts)
        (scripts / "run_formal.py").write_text(
            "import os, signal, sys, time\n"
            "mode = os.environ.get('FORMAL_STUB_MODE', 'success')\n"
            "print('STUB_RUN_FORMAL mode=' + mode, flush=True)\n"
            "print('STUB_ARGS=' + repr(sys.argv[1:]), flush=True)\n"
            "if mode == 'failure': sys.exit(23)\n"
            "if mode in ('term', 'int', 'hup'):\n"
            "    sig = {'term': signal.SIGTERM, 'int': signal.SIGINT, 'hup': signal.SIGHUP}[mode]\n"
            "    def stop(number, _frame):\n"
            "        print('STUB_RECEIVED_SIGNAL=' + str(number), flush=True)\n"
            "        sys.exit(128 + number)\n"
            "    signal.signal(sig, stop)\n"
            "    print('STUB_READY', flush=True)\n"
            "    while True: time.sleep(0.05)\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )
        (scripts / "run_formal_detached.sh").chmod(0o700)
        self.launcher = scripts / "run_formal_detached.sh"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def launch(self, mode: str, datasets: tuple[str, ...] | None = None) -> tuple[str, Path]:
        command = [str(self.launcher)]
        if datasets is not None:
            command.extend(["--datasets", *datasets])
        completed = subprocess.run(
            command, cwd=self.root,
            env={**os.environ, "FORMAL_STUB_MODE": mode},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=3, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ZKGPT_FORMAL_LAUNCHED", completed.stdout)
        pointer = json.loads((self.root / "baseline-local/formal-detached/latest_launch.json").read_text())
        launch_id = pointer["launch_id"]
        return launch_id, self.root / "baseline-local/formal-detached" / launch_id

    def wait_terminal(self, status_path: Path) -> dict:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            try:
                status = json.loads(status_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.03)
                continue
            if status.get("status") in ("COMPLETE", "SUBSET_COMPLETE", "SUBSET_FAILED", "FAILED", "INTERRUPTED"):
                return status
            time.sleep(0.03)
        self.fail(f"detached child did not finish within 4 seconds: {status_path}")

    def test_success_and_failure_persist_actual_exit_codes(self) -> None:
        success_id, success_dir = self.launch("success")
        success = self.wait_terminal(success_dir / "status.json")
        self.assertEqual(success["launch_id"], success_id)
        self.assertEqual((success["status"], success["exit_code"]), ("COMPLETE", 0))
        self.assertEqual(success["requested_datasets"], ["cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"])
        self.assertFalse(success["datasets_selector_explicit"])
        self.assertIn("STUB_RUN_FORMAL mode=success", (success_dir / "console.log").read_text())

        failure_id, failure_dir = self.launch("failure")
        failure = self.wait_terminal(failure_dir / "status.json")
        self.assertEqual(failure["launch_id"], failure_id)
        self.assertEqual((failure["status"], failure["exit_code"]), ("FAILED", 23))

    def test_subset_is_canonical_and_failure_is_scoped(self) -> None:
        launch_id, launch_dir = self.launch("success", ("ogbn_arxiv", "ppi"))
        success = self.wait_terminal(launch_dir / "status.json")
        self.assertEqual(success["launch_id"], launch_id)
        self.assertEqual((success["status"], success["exit_code"]), ("SUBSET_COMPLETE", 0))
        self.assertEqual(success["requested_datasets"], ["ppi", "ogbn_arxiv"])
        self.assertTrue(success["datasets_selector_explicit"])
        self.assertEqual(
            success["child_argv"][1:],
            [str(self.root / "scripts" / "run_formal.py"), "--datasets", "ppi", "ogbn_arxiv"],
        )
        self.assertIn("STUB_ARGS=['--datasets', 'ppi', 'ogbn_arxiv']", (launch_dir / "console.log").read_text())

        failed_id, failed_dir = self.launch("failure", ("ppi", "ogbn_arxiv"))
        failed = self.wait_terminal(failed_dir / "status.json")
        self.assertEqual(failed["launch_id"], failed_id)
        self.assertEqual((failed["status"], failed["exit_code"]), ("SUBSET_FAILED", 23))

        for dataset in ("ppi", "ogbn_arxiv"):
            single_id, single_dir = self.launch("success", (dataset,))
            single = self.wait_terminal(single_dir / "status.json")
            self.assertEqual(single["launch_id"], single_id)
            self.assertEqual((single["status"], single["exit_code"]), ("SUBSET_COMPLETE", 0))
            self.assertEqual(single["requested_datasets"], [dataset])
            self.assertEqual(single["child_argv"][1:], [
                str(self.root / "scripts" / "run_formal.py"), "--datasets", dataset,
            ])

    def test_duplicate_selector_is_rejected_before_launch(self) -> None:
        completed = subprocess.run(
            [str(self.launcher), "--datasets", "ppi", "ppi"], cwd=self.root,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=3, check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertFalse((self.root / "baseline-local" / "formal-detached" / "latest_launch.json").exists())

    def test_term_int_and_hup_forward_to_the_formal_child(self) -> None:
        for mode, sig, expected_code in (
            ("term", signal.SIGTERM, 128 + signal.SIGTERM),
            ("int", signal.SIGINT, 128 + signal.SIGINT),
            ("hup", signal.SIGHUP, 128 + signal.SIGHUP),
        ):
            with self.subTest(signal=mode):
                launch_id, launch_dir = self.launch(mode)
                status_path = launch_dir / "status.json"
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        state = json.loads(status_path.read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        time.sleep(0.02)
                        continue
                    console_text = (launch_dir / "console.log").read_text() if (launch_dir / "console.log").exists() else ""
                    if state.get("status") == "RUNNING" and "STUB_READY" in console_text:
                        break
                    time.sleep(0.02)
                else:
                    self.fail(f"launch {launch_id} did not enter RUNNING")
                os.kill(state["supervisor_pid"], sig)
                terminal = self.wait_terminal(status_path)
                self.assertEqual(terminal["status"], "INTERRUPTED")
                self.assertEqual(terminal["exit_code"], expected_code)
                self.assertEqual(terminal["received_signal"], "SIG" + mode.upper())
                self.assertIn("STUB_RECEIVED_SIGNAL=", (launch_dir / "console.log").read_text())


if __name__ == "__main__":
    unittest.main()
