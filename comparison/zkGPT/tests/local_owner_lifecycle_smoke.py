#!/usr/bin/env python3
"""Exercise the zkGPT owner lock and SIGKILL parent-death cleanup contract."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_formal  # noqa: E402
from common import ChainError, formal_run_lock  # noqa: E402


def _process_state(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return None
    close = raw.rfind(")")
    return raw[close + 2:].split()[0] if close >= 0 else "?"


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _process_state(pid)
        if state is None or state == "Z":
            return True
        time.sleep(0.05)
    return False


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="zkgpt-local-owner-") as temporary:
        root = Path(temporary) / "baseline"
        session_id = "a" * 32
        session_dir = root / "results" / "formal" / session_id
        (root / "baseline-local").mkdir(parents=True)
        session_dir.mkdir(parents=True)
        status_path = session_dir / "session_status.json"
        status_path.write_text(json.dumps({
            "schema": "zkgpt-gat-formal-session-v1",
            "session_id": session_id,
            "status": "RUNNING",
            "active_stage": "phase:test",
            "completed_stages": [],
            "stage_status": {},
            "started_unix_ns": time.time_ns(),
        }), encoding="utf-8")
        fixture_dir = Path(temporary) / "fixtures"
        fixture_dir.mkdir()
        pid_path = fixture_dir / "processes.json"
        nested_pid_path = fixture_dir / "nested-process.pid"
        nested_script = fixture_dir / "nested.py"
        nested_script.write_text(
            "import os, time\n"
            f"from pathlib import Path\n"
            f"Path({str(nested_pid_path)!r}).write_text(str(os.getpid()))\n"
            "while True: time.sleep(1)\n",
            encoding="utf-8",
        )
        phase_script = fixture_dir / "phase.py"
        phase_script.write_text(
            "import json, os, sys\n"
            f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
            "from pathlib import Path\n"
            "import run_formal\n"
            f"root = Path({str(root)!r})\n"
            f"sid = {session_id!r}\n"
            f"nested_script = Path({str(nested_script)!r})\n"
            "nested_logs = root / 'baseline-local' / 'native-child-logs' / sid\n"
            "run_formal.ensure_directory(nested_logs, 'nested child logs', root)\n"
            f"Path({str(pid_path)!r}).write_text(json.dumps({{'phase': os.getpid()}}))\n"
            "run_formal._validate_owner_lease(root, 'test', sid, None)\n"
            "run_formal.install_process_signal_cleanup()\n"
            "run_formal._install_parent_death_watch(root, sid)\n"
            "run_formal.run_logged([sys.executable, str(nested_script)], root, nested_logs / 'child.stdout.log', nested_logs / 'child.stderr.log', 'nested-child')\n",
            encoding="utf-8",
        )
        owner_script = fixture_dir / "owner.py"
        owner_script.write_text(
            "import os, secrets, sys, time\n"
            f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
            "from pathlib import Path\n"
            "import run_formal\n"
            "from common import formal_run_lock\n"
            f"root = Path({str(root)!r})\n"
            f"sid = {session_id!r}\n"
            f"phase_script = Path({str(phase_script)!r})\n"
            f"pid_path = Path({str(pid_path)!r})\n"
            "with formal_run_lock(root) as lock:\n"
            "    info = os.fstat(lock.fileno())\n"
            "    token = secrets.token_hex(32)\n"
            "    pid = os.getpid()\n"
            "    start = run_formal._proc_start_ticks(pid)\n"
            "    run_formal._write_owner_record(root, {'schema': run_formal.LOCAL_OWNER_SCHEMA, 'baseline': 'zkgpt', 'status': 'RUNNING', 'session_id': sid, 'owner_pid': pid, 'owner_start_ticks': start, 'owner_token_sha256': __import__('hashlib').sha256(token.encode('ascii')).hexdigest(), 'lock_device': info.st_dev, 'lock_inode': info.st_ino, 'active_stage': 'test', 'active_dataset': None, 'stage_nonce': 'smoke-stage-nonce', 'completed_stages': [], 'started_unix_ns': time.time_ns()})\n"
            "    env = os.environ.copy()\n"
            "    env.update({'ZKGPT_LOCAL_OWNER_LOCK_FD': str(lock.fileno()), 'ZKGPT_LOCAL_OWNER_TOKEN': token, 'ZKGPT_LOCAL_OWNER_PID': str(pid), 'ZKGPT_LOCAL_OWNER_START_TICKS': str(start), 'ZKGPT_LOCAL_OWNER_STAGE': 'test', 'ZKGPT_LOCAL_OWNER_DATASET': '', 'ZKGPT_LOCAL_OWNER_STAGE_NONCE': 'smoke-stage-nonce', 'ZKGPT_LOCAL_OWNER_SESSION_ID': sid})\n"
            "    logs = root / 'baseline-local' / 'test-logs' / sid\n"
            "    run_formal.ensure_directory(logs, 'test phase logs', root)\n"
            "    run_formal._launch_owner_child(root, lock, [sys.executable, str(phase_script)], env, logs / 'phase.stdout.log', logs / 'phase.stderr.log', 'owner-smoke')\n"
            "    while True: time.sleep(1)\n",
            encoding="utf-8",
        )
        owner = subprocess.Popen([sys.executable, str(owner_script)])
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not pid_path.exists():
            if owner.poll() is not None:
                raise AssertionError(f"owner fixture exited early with {owner.returncode}")
            time.sleep(0.05)
        if not pid_path.exists():
            owner.kill()
            owner.wait()
            raise AssertionError("private phase did not reach its descendant-spawn point")
        processes = json.loads(pid_path.read_text(encoding="utf-8"))
        phase_pid = int(processes["phase"])
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not nested_pid_path.exists():
            time.sleep(0.05)
        if not nested_pid_path.exists():
            owner.kill()
            owner.wait()
            raise AssertionError("private phase did not launch its nested run_logged child")
        grandchild_pid = int(nested_pid_path.read_text(encoding="ascii"))
        try:
            with formal_run_lock(root):
                raise AssertionError("local formal lock allowed a concurrent owner")
        except ChainError:
            pass
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=3.0)
        if owner.returncode != -signal.SIGKILL:
            raise AssertionError(f"owner was not killed as expected: {owner.returncode}")
        if not _wait_gone(phase_pid) or not _wait_gone(grandchild_pid):
            raise AssertionError("private phase or its descendant survived owner SIGKILL")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "INTERRUPTED":
                break
            time.sleep(0.05)
        if status.get("status") != "INTERRUPTED":
            raise AssertionError(f"owner death did not persist INTERRUPTED: {status}")
        owner_record = run_formal._current_owner_record(root)
        if owner_record.get("status") != "INTERRUPTED":
            raise AssertionError("owner death did not mark the local owner record INTERRUPTED")
        owner_record.update({"status": "RUNNING", "active_stage": "cell",
                             "active_dataset": "ogbn_arxiv", "stage_nonce": "stale-owner-test"})
        run_formal._write_owner_record(root, owner_record)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update({"status": "RUNNING", "active_stage": "cell:ogbn_arxiv"})
        status_path.write_text(json.dumps(status), encoding="utf-8")
        with formal_run_lock(root):
            recovered = run_formal._recover_stale_owner(root)
            if recovered != session_id:
                raise AssertionError("stale local owner recovery did not identify its session")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "INTERRUPTED":
            raise AssertionError(f"stale local owner did not persist INTERRUPTED: {status}")
        if run_formal._current_owner_record(root).get("status") != "INTERRUPTED":
            raise AssertionError("stale recovery did not persist an interrupted local owner record")
        saved_root = run_formal.baseline_root
        saved_owner_env = {key: os.environ.pop(key, None) for key in (
            run_formal._OWNER_LOCK_FD_ENV, run_formal._OWNER_TOKEN_ENV,
            run_formal._OWNER_PID_ENV, run_formal._OWNER_START_ENV,
            run_formal._OWNER_STAGE_ENV, run_formal._OWNER_DATASET_ENV,
            run_formal._OWNER_STAGE_NONCE_ENV, run_formal._OWNER_SESSION_ENV,
        )}
        try:
            run_formal.baseline_root = lambda: root
            result = run_formal.main([
                "--internal-phase", "prepare", "--session-dir", str(session_dir),
            ])
            if result != 2:
                raise AssertionError("direct private phase invocation was accepted")
            unchanged = json.loads(status_path.read_text(encoding="utf-8"))
            if unchanged.get("status") != "INTERRUPTED":
                raise AssertionError("unauthorized private phase changed the session terminal state")
        finally:
            run_formal.baseline_root = saved_root
            for key, value in saved_owner_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        saved_ensure_directory = run_formal.ensure_directory
        try:
            def fail_owner_log_directory(*_args, **_kwargs):
                raise ChainError("injected owner log-directory failure")

            run_formal.baseline_root = lambda: root
            run_formal.ensure_directory = fail_owner_log_directory
            result = run_formal.main([])
            if result != 2:
                raise AssertionError("owner setup failure did not fail the official entrypoint")
        finally:
            run_formal.baseline_root = saved_root
            run_formal.ensure_directory = saved_ensure_directory
        failed_owner = run_formal._current_owner_record(root)
        if failed_owner.get("status") != "FAILED":
            raise AssertionError("official owner setup failure did not persist FAILED")
        failed_session_status = json.loads(
            (root / "results" / "formal" / failed_owner["session_id"] /
             "session_status.json").read_text(encoding="utf-8"))
        if failed_session_status.get("status") != "FAILED":
            raise AssertionError("official owner setup failure did not persist session FAILED")
        with formal_run_lock(root):
            pass
    print("local_owner_lifecycle_smoke status=PASS owner_sigkill=cleaned phase_and_grandchild=stopped lock=released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
