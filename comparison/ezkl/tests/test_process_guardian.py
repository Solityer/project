"""Bounded tests for the production EZKL process guardian."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import process_guardian
from process_guardian import ProcessGuardian, finish_guarded_process, spawn_guarded_process
import formal_common
import run_hybrid_formal


class ProcessGuardianTest(unittest.TestCase):
    def test_zombie_owner_identity_is_rejected(self) -> None:
        with mock.patch.object(process_guardian, "_proc_stat", return_value=(42, 9001, "Z")):
            self.assertFalse(process_guardian._owner_identity_matches(42, 9001))
        with mock.patch.object(process_guardian, "_proc_stat", return_value=(42, 9001, "S")):
            self.assertTrue(process_guardian._owner_identity_matches(42, 9001))

    def test_guardian_start_failure_marks_session_failed_and_releases_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-guardian-start-failure-") as temp:
            root = Path(temp)
            output_root = root / "runs" / "formal"
            output_root.mkdir(parents=True)
            # Path composition is written explicitly below to keep the temporary
            # session and lock wholly inside this test directory.
            session_file = output_root / ".sessions" / ("d" * 32 + ".json")
            lock_path = output_root / ".test-owner.lock"

            def begin_session(_output: Path, _matrix: Path, _resume_session: str | None = None) -> tuple[str, Path, int]:
                session_file.parent.mkdir(parents=True)
                session_file.write_text(json.dumps({"baseline": "ezkl", "session_id": "d" * 32, "status": "PREFLIGHT_STARTED", "stage": "session_created"}))
                fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return "d" * 32, session_file, fd

            relative_output = output_root.relative_to(root).as_posix()
            relative_matrix = (output_root / "matrix.json").relative_to(root).as_posix()
            args = run_hybrid_formal.argparse.Namespace(
                _from_ezkl_entry=True,
                output_root=relative_output,
                matrix=relative_matrix,
                binary=None,
                preflight_only=False,
                resume_session=None,
            )
            with mock.patch.object(run_hybrid_formal, "ROOT", root), mock.patch.object(formal_common, "ROOT", root):
                with mock.patch.object(run_hybrid_formal, "begin_formal_session", side_effect=begin_session):
                    with mock.patch.object(run_hybrid_formal.ProcessGuardian, "start", side_effect=OSError("injected guardian startup failure")):
                        with self.assertRaisesRegex(OSError, "injected guardian startup failure"):
                            run_hybrid_formal._main(args)
            document = json.loads(session_file.read_text())
            self.assertEqual(document.get("status"), "FAILED")
            self.assertEqual(document.get("stage"), "guardian_start")
            probe_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(probe_fd, fcntl.LOCK_UN)
            finally:
                os.close(probe_fd)

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "pidfd_open"), "requires Linux process supervision")
    def test_owner_sigkill_cleans_nested_tree_before_shared_lock_release(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-guardian-sigkill-") as temp:
            root = Path(temp)
            (root / "runs").mkdir()
            session_dir = root / "runs" / "formal" / ".sessions"
            session_dir.mkdir(parents=True)
            session_file = session_dir / "session.json"
            session_file.write_text(json.dumps({"baseline": "ezkl", "session_id": "a" * 32, "status": "RUNNING", "stage": "test"}))
            ready_file = root / "owner.json"
            pids_file = root / "stage.json"
            stage_child_file = root / "stage_child.pid"
            stage_code = f"import os,signal,subprocess,sys,time; fd=os.environ.get('EZKL_FORMAL_OWNER_LOCK_FD'); os.close(int(fd)) if fd is not None else None; signal.signal(signal.SIGTERM,signal.SIG_IGN); grandchild_code=\"import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); open({str(pids_file)!r},'w').write(str(os.getpid())); time.sleep(60)\"; child=subprocess.Popen([sys.executable,'-c',grandchild_code]); open({str(stage_child_file)!r},'w').write(str(child.pid)); time.sleep(60)"
            owner_code = f'''\
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
from process_guardian import ProcessGuardian, spawn_guarded_process
root = Path({str(root)!r})
lock_fd = os.open(root / "runs" / ".ezkl-formal.lock", os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
guardian = ProcessGuardian.start(root=root, session_id={"a" * 32!r}, session_file=root / "runs" / "formal" / ".sessions" / "session.json", lock_fd=lock_fd)
stage = spawn_guarded_process([sys.executable, "-c", {stage_code!r}], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
Path({str(ready_file)!r}).write_text(str(guardian.process.pid) + ":" + str(stage.pid))
while True:
    time.sleep(1)
'''
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SCRIPTS)
            owner_log = root / "owner.log"
            with owner_log.open("w") as stream:
                owner = subprocess.Popen([sys.executable, "-c", owner_code], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            lock_fd = os.open(root / "runs" / ".ezkl-formal.lock", os.O_CREAT | os.O_RDWR, 0o600)
            guardian_pid = None
            stage_pid = None
            stage_child_pid = None
            child_pid = None
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and (not ready_file.exists() or not pids_file.exists() or not stage_child_file.exists()):
                    time.sleep(0.05)
                self.assertTrue(ready_file.is_file(), f"formal owner did not start guardian and stage: {owner_log.read_text()}")
                self.assertTrue(pids_file.is_file(), "nested stage did not start its child")
                self.assertTrue(stage_child_file.is_file(), "nested process did not start its grandchild")
                guardian_pid, stage_pid = (int(value) for value in ready_file.read_text().split(":"))
                stage_child_pid = int(stage_child_file.read_text())
                child_pid = int(pids_file.read_text())
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.kill(owner.pid, signal.SIGKILL)
                owner.wait(timeout=5)
                deadline = time.monotonic() + 1.2
                while time.monotonic() < deadline:
                    self.assertTrue(_process_is_live(child_pid), "guardian released the lock before killing nested descendants")
                    self.assertTrue(_process_is_live(stage_child_pid), "guardian did not retain the full nested process tree")
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    time.sleep(0.05)
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and _process_is_live(child_pid):
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    time.sleep(0.05)
                self.assertFalse(_process_is_live(child_pid), "guardian failed to stop the nested child")
                self.assertFalse(_process_is_live(stage_child_pid), "guardian failed to stop the nested stage process")
                self.assertFalse(_process_is_live(stage_pid), "guardian failed to stop the registered stage group")
                with open(session_file, encoding="utf-8") as stream:
                    self.assertEqual(json.load(stream)["status"], "INTERRUPTED")
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.wait(timeout=5)
                for pid in (guardian_pid, stage_pid, stage_child_pid, child_pid):
                    if pid is not None and _process_is_live(pid):
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                if stage_pid is not None:
                    try:
                        os.killpg(stage_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                os.close(lock_fd)

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "pidfd_open"), "requires Linux process supervision")
    def test_failed_stage_cleans_descendant_and_guardian_shutdown_releases_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-guardian-failure-") as temp:
            root = Path(temp)
            (root / "runs").mkdir()
            session_dir = root / "runs" / "formal" / ".sessions"
            session_dir.mkdir(parents=True)
            session_file = session_dir / "session.json"
            session_file.write_text(json.dumps({"baseline": "ezkl", "session_id": "b" * 32, "status": "RUNNING", "stage": "test"}))
            lock_path = root / "runs" / ".ezkl-formal.lock"
            owner_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guardian = ProcessGuardian.start(root=root, session_id="b" * 32, session_file=session_file, lock_fd=owner_fd)
            child_pid_file = root / "child.pid"
            code = f'''\
import os, signal, time
pid = os.fork()
if pid == 0:
    open({str(child_pid_file)!r}, "w").write(str(os.getpid()))
    time.sleep(30)
    os._exit(0)
os._exit(23)
'''
            process = spawn_guarded_process([sys.executable, "-c", code], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                returncode = process.wait(timeout=5)
                self.assertEqual(returncode, 23)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and not child_pid_file.exists():
                    time.sleep(0.02)
                self.assertTrue(child_pid_file.is_file(), "failed stage did not start descendant")
                self.assertTrue(finish_guarded_process(process))
                self.assertFalse(_process_is_live(int(child_pid_file.read_text())))
                guardian.shutdown()
                os.close(owner_fd)
                owner_fd = -1
                probe_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(probe_fd, fcntl.LOCK_UN)
                finally:
                    os.close(probe_fd)
            finally:
                if guardian.process.poll() is None:
                    guardian.shutdown()
                if owner_fd >= 0:
                    os.close(owner_fd)

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "pidfd_open"), "requires Linux process supervision")
    def test_capture_cleans_tree_when_guardian_rpc_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-guardian-rpc-failure-") as temp:
            root = Path(temp)
            (root / "runs").mkdir()
            session_dir = root / "runs" / "formal" / ".sessions"
            session_dir.mkdir(parents=True)
            session_file = session_dir / "session.json"
            session_file.write_text(json.dumps({"baseline": "ezkl", "session_id": "c" * 32, "status": "RUNNING", "stage": "test"}))
            lock_path = root / "runs" / ".ezkl-formal.lock"
            owner_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guardian = ProcessGuardian.start(root=root, session_id="c" * 32, session_file=session_file, lock_fd=owner_fd)
            stage_pid_file = root / "stage.pid"
            child_pid_file = root / "child.pid"
            code = f'''\
import os, time
open({str(stage_pid_file)!r}, "w").write(str(os.getpid()))
pid = os.fork()
if pid == 0:
    open({str(child_pid_file)!r}, "w").write(str(os.getpid()))
    time.sleep(60)
    os._exit(0)
time.sleep(60)
'''

            def kill_guardian() -> None:
                time.sleep(0.5)
                if guardian.process.poll() is None:
                    guardian.process.kill()
                    guardian.process.wait()

            killer = threading.Thread(target=kill_guardian, daemon=True)
            killer.start()
            stage_pid: int | None = None
            child_pid: int | None = None
            try:
                with self.assertRaises(RuntimeError):
                    process_guardian.run_guarded_capture(
                        [sys.executable, "-c", code],
                        cwd=root,
                        env=os.environ.copy(),
                        timeout=30,
                    )
                killer.join(timeout=2)
                self.assertTrue(stage_pid_file.is_file())
                self.assertTrue(child_pid_file.is_file())
                stage_pid = int(stage_pid_file.read_text())
                child_pid = int(child_pid_file.read_text())
                self.assertFalse(_process_is_live(stage_pid), "capture left the stage leader alive after guardian loss")
                self.assertFalse(_process_is_live(child_pid), "capture left a descendant alive after guardian loss")
                probe_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(probe_fd)
            finally:
                killer.join(timeout=2)
                for pid in (stage_pid, child_pid):
                    if pid is not None and _process_is_live(pid):
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                os.environ.pop(process_guardian.GUARDIAN_SOCKET_ENV, None)
                os.environ.pop(process_guardian.LOCK_FD_ENV, None)
                process_guardian._ACTIVE_GUARDIAN = None
                try:
                    guardian.socket_path.unlink()
                except FileNotFoundError:
                    pass
                os.close(owner_fd)
            probe_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(probe_fd, fcntl.LOCK_UN)
            finally:
                os.close(probe_fd)


def _process_is_live(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state not in {"Z", "X"}


if __name__ == "__main__":
    unittest.main()
