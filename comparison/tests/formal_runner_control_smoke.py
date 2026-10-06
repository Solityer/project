#!/usr/bin/env python3
"""Isolated control-plane smoke; it cannot emit formal result artifacts."""

from __future__ import annotations

import os
import json
import ctypes
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import formal_runner  # noqa: E402


def _wait_for(path: Path, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return path.read_text(encoding="utf-8")
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _supervisor_case(scratch: Path, *, session_id: str, kill_owner: bool) -> None:
    previous_root = formal_runner.COMPARISON_ROOT
    formal_runner.COMPARISON_ROOT = scratch
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise AssertionError("cannot enable test subreaper")
    session_root = scratch / "runs" / "formal" / session_id
    ready_path = scratch / f"{session_id}.ready.json"
    done_path = scratch / f"{session_id}.done"
    leader_path = scratch / f"{session_id}.leader"
    grandchild_path = scratch / f"{session_id}.grandchild"
    owner_code = r"""
import fcntl, hashlib, json, os, pathlib, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
import formal_runner as f
root = pathlib.Path(sys.argv[2]); sid = sys.argv[3]
ready = pathlib.Path(sys.argv[4]); done = pathlib.Path(sys.argv[5])
leader = pathlib.Path(sys.argv[6]); grandchild = pathlib.Path(sys.argv[7])
f.COMPARISON_ROOT = root
session = f._session_root(sid); session.mkdir(parents=True, exist_ok=True)
lock_fd = f._acquire_lock()
token = 'b' * 64
owner = {'pid': os.getpid(), 'boot_id': f._read_boot_id(), 'start_ticks': f._proc_start_ticks(os.getpid())}
f.atomic_json(session / 'state.json', {
  'schema': 'comparison-formal-session-v1', 'session_id': sid, 'status': 'RUNNING',
  'stage': 'cell:ezkl:cora', 'lock': owner,
  'internal_owner_token_sha256': hashlib.sha256(token.encode()).hexdigest(),
})
env = dict(os.environ)
env.update({'COMPARISON_FORMAL_OWNER_TOKEN': token, 'COMPARISON_FORMAL_SESSION_ID': sid,
            'COMPARISON_FORMAL_ROOT_LOCK_FD': str(lock_fd)})
grandchild_code = "import os,time; os.setsid(); open(%r,'w').write(str(os.getpid())); time.sleep(60)" % str(grandchild)
leader_code = "import os,subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',sys.argv[1]]); open(sys.argv[2],'w').write(str(os.getpid())); time.sleep(60)"
command = [sys.executable, '-c', leader_code, grandchild_code, str(leader)]
supervisor_code = r'''import json, os, pathlib, sys
sys.path.insert(0, sys.argv[1])
import formal_runner as f
f.COMPARISON_ROOT = pathlib.Path(sys.argv[2]); sid=sys.argv[3]; stage=sys.argv[4]
state=f._read_json(f._session_root(sid)/'state.json','test owner state')
rc=f._supervise_owned_command(os.getppid(),stage,json.loads(sys.argv[5]),
    os.environ['COMPARISON_FORMAL_OWNER_TOKEN'],sid,
    int(os.environ['COMPARISON_FORMAL_ROOT_LOCK_FD']),state)
raise SystemExit(rc)
'''
log = open(root / 'stage-supervisor.log', 'ab', buffering=0)
child = subprocess.Popen([sys.executable, '-c', supervisor_code, sys.argv[1], str(root), sid,
      'cell:ezkl:cora', json.dumps(command)], cwd=sys.argv[1], env=env, stdin=subprocess.DEVNULL,
      stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True, pass_fds=(lock_fd,))
ready.write_text(json.dumps({'owner_pid': os.getpid(), 'supervisor_pid': child.pid,
      'owner_start_ticks': owner['start_ticks'], 'owner_boot_id': owner['boot_id']}))
rc = child.wait()
done.write_text(str(rc))
time.sleep(60)
"""
    root_for_worker = str(ROOT)
    owner = subprocess.Popen(
        [sys.executable, "-c", owner_code, root_for_worker, str(scratch), session_id,
         str(ready_path), str(done_path), str(leader_path), str(grandchild_path)],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    ready = json.loads(_wait_for(ready_path))
    state_path = session_root / "state.json"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if state_path.is_file() and formal_runner._read_json(state_path, "owner state").get("owned_stage"):
            break
        time.sleep(0.02)
    else:
        owner.kill()
        owner.wait(timeout=2)
        details = (scratch / "stage-supervisor.log").read_text(encoding="utf-8", errors="replace") if (scratch / "stage-supervisor.log").exists() else "missing log"
        raise AssertionError(f"stage supervisor did not register its owner lease: {details}")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (leader_path.is_file() and grandchild_path.is_file()):
        time.sleep(0.02)
    if not leader_path.is_file() or not grandchild_path.is_file():
        owner.kill()
        owner.wait(timeout=2)
        raise AssertionError("stage leader did not create its separate-session child")
    leader_pid = int(leader_path.read_text())
    grandchild_pid = int(grandchild_path.read_text())
    if kill_owner:
        owner.kill()
        owner.wait(timeout=2)
        try:
            supervisor_pid, _ = os.waitpid(int(ready["supervisor_pid"]), 0)
        except ChildProcessError:
            supervisor_pid = 0
        if supervisor_pid != int(ready["supervisor_pid"]):
            raise AssertionError("parent death did not leave a reapable supervisor")
    else:
        os.kill(leader_pid, signal.SIGKILL)
        rc = int(_wait_for(done_path))
        if rc == 0:
            raise AssertionError("SIGKILLed phase leader returned success")
        owner.terminate()
        owner.wait(timeout=2)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = formal_runner._proc_snapshot(grandchild_pid)
        if snapshot is None or snapshot["state"] == "Z":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("separate-session grandchild survived supervisor cleanup")
    descriptor = formal_runner._acquire_lock()
    try:
        if kill_owner:
            formal_runner._recover_interrupted_sessions()
            recovered = formal_runner._read_json(state_path, "recovered owner state")
            if recovered.get("status") != "FAILED":
                raise AssertionError("dead root owner was not recovered as FAILED")
    finally:
        os.close(descriptor)
        formal_runner.COMPARISON_ROOT = previous_root
        if libc.prctl(36, 0, 0, 0, 0) != 0:
            raise AssertionError("cannot restore test subreaper setting")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="comparison-runner-smoke-") as temporary:
        scratch = Path(temporary)

        # The launch helper really detaches, transfers the lock fd, and lets the
        # worker finish after its launching process has returned.
        previous_root = formal_runner.COMPARISON_ROOT
        formal_runner.COMPARISON_ROOT = scratch / "detached-run"
        lock_fd = formal_runner._acquire_lock()
        marker = scratch / "detached.done"
        code = (
            "import pathlib,time; time.sleep(0.4); "
            f"pathlib.Path({str(marker)!r}).write_text('done')"
        )
        try:
            child = formal_runner.spawn_detached_command(
                [sys.executable, "-c", code], scratch / "detached.log", lock_fd,
            )
            os.close(lock_fd)
            if child.poll() is not None:
                raise AssertionError("detached child exited before the launcher returned")
            try:
                formal_runner._acquire_lock()
            except formal_runner.RunnerError:
                pass
            else:
                raise AssertionError("detached worker did not retain the transferred flock")
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not marker.is_file() or marker.read_text() != "done":
                raise AssertionError("detached child did not finish independently")
            if child.wait(timeout=2) != 0:
                raise AssertionError("detached child returned failure")
            released_lock = formal_runner._acquire_lock()
            os.close(released_lock)
        finally:
            try:
                os.close(lock_fd)
            except OSError:
                pass
            formal_runner.COMPARISON_ROOT = previous_root

        # An owned flock prevents a second launch without trusting PID files.
        old_root = formal_runner.COMPARISON_ROOT
        formal_runner.COMPARISON_ROOT = scratch
        first = formal_runner._acquire_lock()
        try:
            try:
                formal_runner._acquire_lock()
            except formal_runner.RunnerError:
                pass
            else:
                raise AssertionError("duplicate formal lock acquisition succeeded")
        finally:
            os.close(first)
            formal_runner.COMPARISON_ROOT = old_root

        # A log is created only when a cell command starts; real child errors
        # remain in that log and propagate as their actual exit status.
        dataset_log = scratch / "logs" / "datasets" / "cora" / "ezkl.log"
        if dataset_log.exists():
            raise AssertionError("dataset log existed before that dataset began")
        status, output = formal_runner._run_stage(
            [sys.executable, "-c", "import sys; print('real-smoke-error'); sys.exit(23)"],
            dataset_log, "smoke:cell:cora", scratch, formal_runner.ROOT_CELL_LOG_LIMIT,
        )
        if status != 23 or "real-smoke-error" not in output or not dataset_log.is_file():
            raise AssertionError("stage failure was not preserved in its lazy log")
        if "FINAL_VALIDATION_OK" in output or list(scratch.rglob("final_validation.json")):
            raise AssertionError("control smoke produced an official success artifact")

        def fail_with_descendant(session_root: Path, *, hold_pipe: bool, name: str) -> tuple[int, str, float, Path]:
            survivor_path = scratch / f"{name}-survived"
            grandchild = (
                "import pathlib,time; time.sleep(8); "
                f"pathlib.Path({str(survivor_path)!r}).write_text('survived')"
            )
            launch = (
                "subprocess.Popen([sys.executable, '-c', sys.argv[1]])"
                if hold_pipe else
                "subprocess.Popen([sys.executable, '-c', sys.argv[1]], "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
            )
            leader = (
                "import subprocess,sys; "
                f"{launch}; "
                "print('leader-failed'); sys.exit(23)"
            )
            log = session_root / "logs" / f"{name}.log"
            started = time.monotonic()
            status, text = formal_runner._run_stage(
                [sys.executable, "-c", leader, grandchild],
                log, f"smoke:owned-process-group:{name}", session_root,
                formal_runner.ROOT_CELL_LOG_LIMIT,
            )
            return status, text, time.monotonic() - started, survivor_path

        # A failed leader must not strand a same-PGID descendant holding the
        # streaming log pipe. Run it through the worker so failure state and
        # inherited-lock release are exercised together.
        failure_root = scratch / "failure-run"
        old_root = formal_runner.COMPARISON_ROOT
        old_phase = formal_runner._run_phase
        formal_runner.COMPARISON_ROOT = failure_root

        def fail_prepare(baseline, phase, session_id, session_root, dataset=None, log_path=None,
                         *, owner_token=None, lock_fd=None):
            if phase != "prepare" or baseline != "zkgpt":
                raise AssertionError("control failure did not occur at the first prepare stage")
            status, output, elapsed, survivor_path = fail_with_descendant(
                session_root, hold_pipe=True, name="pipe-holding-grandchild",
            )
            if status != 23 or "leader-failed" not in output:
                raise AssertionError("owned-group cleanup changed the real child failure")
            if elapsed >= 5 or survivor_path.exists():
                raise AssertionError("failed stage left its pipe-holding descendant running")
            raise formal_runner.RunnerError("control child returned exit 23")

        owned_lock = formal_runner._acquire_lock()
        previous_env = {
            name: os.environ.get(name)
            for name in (formal_runner.OWNER_TOKEN_ENV, formal_runner.SESSION_ID_ENV,
                         formal_runner.ROOT_LOCK_FD_ENV)
        }
        os.environ[formal_runner.OWNER_TOKEN_ENV] = "a" * 64
        try:
            formal_runner._run_phase = fail_prepare
            worker_status = formal_runner._run_formal(
                "00000000000000000000000000000002", owned_lock,
            )
        finally:
            formal_runner._run_phase = old_phase
            formal_runner.COMPARISON_ROOT = old_root
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        failure_state = formal_runner._read_json(
            failure_root / "runs" / "formal" / "00000000000000000000000000000002" / "state.json",
            "control failure state",
        )
        if (worker_status != 2 or failure_state.get("status") != "FAILED" or
                failure_state.get("exit_code") != 2 or "exit 23" not in failure_state.get("failure", "")):
            raise AssertionError("worker did not persist the real failed-stage status")
        relock = formal_runner._acquire_lock()
        os.close(relock)
        if list(failure_root.rglob("final_validation.json")):
            raise AssertionError("failed worker emitted an official aggregate success artifact")

        # Also cover descendants that close/redirect the inherited log. The
        # stage may return on leader exit, but cleanup must still reap that
        # same owned group before releasing control.
        silent_status, silent_output, silent_elapsed, silent_survivor = fail_with_descendant(
            scratch, hold_pipe=False, name="silent-grandchild",
        )
        if (silent_status != 23 or "leader-failed" not in silent_output or
                silent_elapsed >= 5 or silent_survivor.exists()):
            raise AssertionError("failed stage left a silent owned descendant running")

        try:
            formal_runner._validate_global_admission(
                {baseline: {} for baseline in formal_runner.BASELINES},
                "00000000000000000000000000000001",
            )
        except formal_runner.RunnerError as exc:
            if "capacity profile" not in str(exc):
                raise AssertionError(f"unexpected fail-closed admission error: {exc}") from exc
        else:
            raise AssertionError("missing all-baseline capacity profile was admitted")

        _supervisor_case(
            scratch / "supervisor-rootkill", session_id="00000000000000000000000000000031",
            kill_owner=True,
        )
        _supervisor_case(
            scratch / "supervisor-stagekill", session_id="00000000000000000000000000000032",
            kill_owner=False,
        )

    print("FORMAL_RUNNER_CONTROL_SMOKE_OK detached=1 flock=1 lazy_log=1 failure_exit=23 owned_group_cleanup=pipe+silent stage_leader_sigkill_nested_setsid=1 root_sigkill_nested_setsid=1 stale_failed_recovery=1 lock_release=1 admission_fail_closed=1 no_formal_success=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
