#!/usr/bin/env python3
"""Run the three official detached five-dataset comparison chains serially.

This is a scheduler, not a replacement prover/runner. It uses exact launch IDs,
waits for terminal receipts and process/lock release, and never infers PASS from
an exit code alone. Uncertain process ownership halts the queue for safety.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS_REL = Path("runs/serial_comparisons")
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
BASELINES = ("ezkl", "zkdl", "zkgpt")
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
POLL_SECONDS = 5
START_GRACE_SECONDS = 90
DRAIN_GRACE_SECONDS = 180
LOCKS = {
    "ezkl": "ezkl/runs/.ezkl-formal.lock",
    "zkdl": "zkdl/runs/.formal.lock",
    "zkgpt": "zkGPT/baseline-local/formal-run.lock",
}
STOP_REQUESTED = False


class UnsafeTransition(RuntimeError):
    """Cannot prove it is safe to start another detached baseline."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_regular_dir(path: Path, *, create: bool = False) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise UnsafeTransition(f"not a real directory: {path}")
    if create and not path.exists():
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    if not path.is_dir() or path.is_symlink():
        raise UnsafeTransition(f"directory is missing/replaced: {path}")


def read_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        raise UnsafeTransition(f"JSON receipt is a symlink: {path}")
    if not path.exists():
        return None
    if not path.is_file():
        raise UnsafeTransition(f"JSON receipt is not a regular file: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsafeTransition(f"cannot read exact receipt {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise UnsafeTransition(f"JSON receipt is not an object: {path}")
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_regular_dir(path.parent)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise UnsafeTransition(f"state file was replaced: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def proc(pid: Any) -> tuple[int, int, str] | None:
    """Return (start_ticks, process_group_id, state), excluding exited processes."""
    if type(pid) is not int or pid <= 0:
        return None
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = value[value.rfind(")") + 2:].split()
        if len(fields) < 20 or fields[0] in ("Z", "X", "x"):
            return None
        return int(fields[19]), int(fields[2]), fields[0]
    except (OSError, ValueError, IndexError):
        return None


def alive(pid: Any, ticks: Any) -> bool:
    identity = proc(pid)
    if identity is None:
        return False
    try:
        return type(ticks) in (str, int) and identity[0] == int(ticks)
    except ValueError:
        return False


def group_has_live_members(pgid: Any) -> bool:
    if type(pgid) is not int or pgid <= 0:
        return False
    # A group leader can exit before other processes in its group; checking
    # its PID alone would incorrectly authorize the next GPU experiment.
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            identity = proc(int(entry.name))
            if identity is not None and identity[1] == pgid:
                return True
    return False


def all_native_locks_idle() -> bool:
    """Check pre-existing native flock files without replacing their inodes."""
    for name, relative in LOCKS.items():
        path = ROOT / relative
        if path.is_symlink():
            raise UnsafeTransition(f"{name} formal lock is a symlink: {path}")
        if not path.exists():
            continue
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise UnsafeTransition(f"{name} formal lock is not a regular file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    return True


def blank_state(sid: str) -> dict[str, Any]:
    return {
        "schema": "comparison.serial-formal.v1", "session_id": sid, "status": "STARTING",
        "started_at_utc": now(), "updated_at_utc": now(), "finished_at_utc": None,
        "supervisor": {}, "active_baseline": None, "failure": "",
        "baselines": {
            name: {"status": "PENDING", "launch_id": None, "native_session_id": None,
                   "started_at_utc": None, "finished_at_utc": None, "exit_code": None,
                   "datasets": {dataset: {"status": "PENDING", "detail": ""} for dataset in DATASETS}}
            for name in BASELINES
        },
    }


def commit(session_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = now()
    write_json(session_dir / "status.json", state)


def root_for(name: str) -> Path:
    return ROOT / ("zkGPT" if name == "zkgpt" else name)


def command_for(name: str) -> list[str]:
    if name == "ezkl":
        return [sys.executable, "scripts/ezkl_formal_control.py", "start"]
    if name == "zkdl":
        return ["bash", "scripts/run_five_detached.sh"]
    return ["bash", "scripts/run_formal_detached.sh"]


def verify_entrypoints() -> None:
    for baseline in BASELINES:
        parent = root_for(baseline)
        ensure_regular_dir(parent)
        entry = parent / command_for(baseline)[1]
        if entry.is_symlink() or not entry.is_file():
            raise UnsafeTransition(f"missing or symlinked official {baseline} launcher: {entry}")


def launch(name: str, record: dict[str, Any]) -> dict[str, Any]:
    cmd = command_for(name)
    print(f"[{now()}] LAUNCH {name}: {cmd!r}", flush=True)
    try:
        result = subprocess.run(cmd, cwd=root_for(name), stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UnsafeTransition(f"{name} launch status is uncertain; no other baseline started: {exc}") from exc
    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise UnsafeTransition(f"{name} launcher exited {result.returncode}; cannot exclude a detached child")
    if name == "ezkl":
        matches = re.findall(r"^session_id=([0-9a-f]{32})\s*$", result.stdout, re.M)
        if len(matches) != 1:
            raise UnsafeTransition("EZKL launcher did not return exactly one authenticated session ID")
        sid = matches[0]
        record["native_session_id"] = sid
        return {"name": name, "launch_id": sid, "dir": root_for(name) / "runs/formal/.sessions" / sid}
    if name == "zkdl":
        ids = re.findall(r"^zkdl_detached_launch=STARTED launch_id=(\S+)\s*$", result.stdout, re.M)
        dirs = re.findall(r"^zkdl_detached_launch_dir=(\S+)\s*$", result.stdout, re.M)
        pids = re.findall(r"^zkdl_detached_supervisor_pid=(\d+)\s*$", result.stdout, re.M)
        if len(ids) != 1 or len(dirs) != 1 or len(pids) != 1:
            raise UnsafeTransition("zkDL launcher did not return one exact launch identity")
        launch_dir = root_for(name) / "runs" / ids[0]
        if not re.fullmatch(r"launch_\d{8}T\d{6}Z_[A-Za-z0-9]{8}", ids[0]) or str(launch_dir) != dirs[0]:
            raise UnsafeTransition("zkDL reported launch directory differs from the canonical run path")
        return {"name": name, "launch_id": ids[0], "dir": launch_dir, "supervisor_pid": int(pids[0])}
    ids = re.findall(r"^ZKGPT_FORMAL_LAUNCHED launch_id=([0-9a-f]{32})\b", result.stdout, re.M)
    if len(ids) != 1:
        raise UnsafeTransition("zkGPT launcher did not return one exact launch identity")
    return {"name": name, "launch_id": ids[0],
            "dir": root_for(name) / "baseline-local/formal-detached" / ids[0]}


def snapshot(job: dict[str, Any], record: dict[str, Any]) -> tuple[bool, int | None, dict[str, tuple[str, str]]]:
    """Observe exact receipts. Return (terminal, exit_code, dataset snapshot)."""
    name, sid, directory = job["name"], job["launch_id"], job["dir"]
    # Detached startup may publish its directory after the launcher returns.
    # Once observed, any disappearance/replacement is a safety violation.
    if not job.get("observed_dir") and not directory.exists() and not directory.is_symlink():
        return False, None, {}
    ensure_regular_dir(directory)
    job["observed_dir"] = True
    values: dict[str, tuple[str, str]] = {}
    terminal, code = False, None
    if name == "ezkl":
        meta = read_json(directory / "controller.json")
        session = read_json(root_for(name) / "runs/formal/.sessions" / f"{sid}.json")
        if meta:
            if meta.get("session_id") != sid or meta.get("schema") != "ezkl.formal-controller.v1":
                raise UnsafeTransition("EZKL controller identity mismatch")
            job["supervisor"] = meta.get("control_process")
            job["child"] = meta.get("formal_owner_process")
            if isinstance(job["child"], dict):
                # The EZKL supervisor starts its formal owner in a fresh session;
                # its PID is therefore the process-group ID to check for stragglers.
                job["child"] = {**job["child"], "pgid": job["child"].get("pid")}
            if meta.get("status") in {"EXITED", "STOPPED", "LAUNCH_FAILED", "SUPERVISOR_FAILED", "CHILD_IDENTITY_UNAVAILABLE"}:
                terminal = True
                code = meta.get("exit_code")
                if type(code) is not int:
                    if meta.get("status") == "EXITED":
                        raise UnsafeTransition("EZKL EXITED metadata lacks an integer exit code")
                    code = 2  # Known terminal failure without a child exit status.
        if session:
            if session.get("baseline") != "ezkl" or session.get("session_id") != sid:
                raise UnsafeTransition("EZKL formal session identity mismatch")
            record["native_session_id"] = sid
            passed = session.get("completed_cells", {})
            failed = session.get("dataset_failures", {})
            if not isinstance(passed, dict) or not isinstance(failed, dict):
                raise UnsafeTransition("EZKL dataset progress schema is malformed")
            active = session.get("active_dataset")
            for dataset in DATASETS:
                if dataset in failed:
                    values[dataset] = ("FAILED", str(failed[dataset].get("error", ""))[:300])
                elif dataset in passed:
                    values[dataset] = ("PASS", "")
                elif active == dataset:
                    values[dataset] = ("RUNNING", str(session.get("stage", ""))[:150])
        job["session_status"] = session.get("status") if session else None
    elif name == "zkdl":
        exit_doc = read_json(directory / "exit.json")
        pid_doc = read_json(directory / "pid.json")
        if pid_doc:
            if pid_doc.get("schema") != "zkdl_detached_launch_pid_v1":
                raise UnsafeTransition("zkDL detached PID receipt has unexpected schema")
            job["supervisor"] = {"pid": pid_doc.get("supervisor_pid"), "start_ticks": pid_doc.get("supervisor_start_ticks")}
            job["child"] = {"pid": pid_doc.get("runner_pid"), "start_ticks": pid_doc.get("runner_start_ticks"),
                            "pgid": pid_doc.get("runner_process_group_id")}
        if exit_doc:
            if exit_doc.get("schema") != "zkdl_detached_launch_exit_v1" or exit_doc.get("status") not in {"PASS", "FAILED", "INTERRUPTED"}:
                raise UnsafeTransition("zkDL detached exit receipt is malformed")
            terminal, code = True, exit_doc.get("exit_code")
            if type(code) is not int:
                raise UnsafeTransition("zkDL terminal metadata lacks an integer exit code")
            if pid_doc is None and exit_doc.get("runner_pid") is not None:
                raise UnsafeTransition("zkDL exit receipt identifies a runner but lacks its authenticated PID metadata")
        link = read_json(directory / "formal_session.json")
        if link:
            session_id = link.get("session_id")
            sess = root_for(name) / "runs" / f"formal_{session_id}"
            if (not isinstance(session_id, str) or not HEX32.fullmatch(session_id)
                    or link.get("launch_id") != sid or link.get("session_dir") != str(sess)):
                raise UnsafeTransition("zkDL launch-to-formal-session mapping mismatch")
            ensure_regular_dir(sess)
            state = read_json(sess / "state.json")
            if state:
                if state.get("session_id") != session_id or state.get("baseline") != "zkdl":
                    raise UnsafeTransition("zkDL formal session state identity mismatch")
                record["native_session_id"] = session_id
                rows = state.get("dataset_status", {})
                if not isinstance(rows, dict):
                    raise UnsafeTransition("zkDL dataset state schema is malformed")
                for dataset in DATASETS:
                    row = rows.get(dataset, {})
                    if isinstance(row, dict):
                        cell_status = row.get("status", "PENDING")
                        if cell_status in {"PASS", "FAILED", "BLOCKED", "INTERRUPTED"}:
                            values[dataset] = (cell_status, str(row.get("failure", ""))[:300])
                        elif state.get("stage") == f"cell:{dataset}":
                            values[dataset] = ("RUNNING", str(row.get("stage", ""))[:150])
    else:
        doc = read_json(directory / "status.json")
        if doc:
            if doc.get("schema") != "zkgpt-gat-detached-launch-v2" or doc.get("launch_id") != sid:
                raise UnsafeTransition("zkGPT detached launch identity mismatch")
            job["supervisor"] = {"pid": doc.get("supervisor_pid"), "start_ticks": doc.get("supervisor_start_ticks")}
            job["child"] = {"pid": doc.get("child_pid"), "start_ticks": doc.get("child_start_ticks"),
                            "pgid": doc.get("child_process_group")}
            if doc.get("status") in {"COMPLETE", "FAILED", "INTERRUPTED"}:
                terminal, code = True, doc.get("exit_code")
                if type(code) is not int:
                    raise UnsafeTransition("zkGPT detached terminal receipt has no integer exit code")
        owner = read_json(root_for(name) / "baseline-local/formal-owner.json")
        child = job.get("child") or {}
        if owner and (owner.get("owner_pid") == child.get("pid") and owner.get("owner_start_ticks") == child.get("start_ticks")):
            sess_id = owner.get("session_id")
            if not isinstance(sess_id, str) or not HEX32.fullmatch(sess_id):
                raise UnsafeTransition("zkGPT native owner session ID is malformed")
            record["native_session_id"] = sess_id
        sess_id = record.get("native_session_id")
        if isinstance(sess_id, str) and HEX32.fullmatch(sess_id):
            state = read_json(root_for(name) / "results/formal" / sess_id / "session_status.json")
            if state:
                if state.get("session_id") != sess_id or state.get("schema") != "zkgpt-gat-formal-session-v1":
                    raise UnsafeTransition("zkGPT formal session status identity mismatch")
                stage_status = state.get("stage_status", {})
                failed = state.get("dataset_failures", {})
                if not isinstance(stage_status, dict) or not isinstance(failed, dict):
                    raise UnsafeTransition("zkGPT per-dataset status schema is malformed")
                for dataset in DATASETS:
                    mark = stage_status.get(f"cell:{dataset}")
                    if mark == "FAILED" or dataset in failed:
                        values[dataset] = ("FAILED", str(failed.get(dataset, ""))[:300])
                    elif mark == "PASS":
                        receipt = read_json(root_for(name) / "results/formal" / sess_id / "receipts" / f"{dataset}.json")
                        if receipt and receipt.get("session_id") == sess_id and receipt.get("dataset") == dataset and receipt.get("status") == "VERIFY_OK" and receipt.get("independent_verification") is True:
                            values[dataset] = ("PASS", "")
                        else:
                            values[dataset] = ("INCOMPLETE", "passing cell marker lacks its verification receipt")
                    elif str(state.get("active_stage", "")).endswith(f":{dataset}"):
                        values[dataset] = ("RUNNING", str(state.get("active_stage", ""))[:150])
    return terminal, code, values


def safe_to_transition(job: dict[str, Any]) -> bool:
    for key in ("supervisor", "child"):
        identity = job.get(key)
        if isinstance(identity, dict):
            if alive(identity.get("pid"), identity.get("start_ticks")):
                return False
            if key == "child" and group_has_live_members(identity.get("pgid")):
                return False
    return all_native_locks_idle()


def merge_progress(record: dict[str, Any], values: dict[str, tuple[str, str]]) -> bool:
    changed = False
    for dataset, (status, detail) in values.items():
        row = record["datasets"][dataset]
        if row["status"] != status or row["detail"] != detail:
            row.update({"status": status, "detail": detail})
            print(f"[{now()}] {record['label']}/{dataset}: {status}" + (f" ({detail})" if detail else ""), flush=True)
            changed = True
    return changed


def run_one(name: str, session_dir: Path, state: dict[str, Any]) -> None:
    record = state["baselines"][name]
    record["label"] = name
    record["status"] = "LAUNCHING"
    record["started_at_utc"] = now()
    state["active_baseline"] = name
    commit(session_dir, state)
    job = launch(name, record)
    record["launch_id"] = job["launch_id"]
    record["status"] = "RUNNING"
    record["launch_dir"] = str(job["dir"])
    record["native_log"] = str(job["dir"] / ("logs/formal_session.log" if name == "ezkl" else "console.log"))
    commit(session_dir, state)
    launch_time = time.monotonic()
    drained_since: float | None = None
    while True:
        if STOP_REQUESTED:
            raise UnsafeTransition(f"controller termination requested during {name}; native job {job['launch_id']} is not killed")
        terminal, exit_code, values = snapshot(job, record)
        if merge_progress(record, values):
            commit(session_dir, state)
        if terminal:
            if safe_to_transition(job):
                for row in record["datasets"].values():
                    if row["status"] in {"PENDING", "RUNNING"}:
                        row.update({"status": "NOT_RUN", "detail": "native run ended without a verified cell result"})
                record["status"] = "PASS" if exit_code == 0 and all(
                    r["status"] == "PASS" for r in record["datasets"].values()) else "FAILED"
                record["exit_code"] = exit_code
                record["finished_at_utc"] = now()
                print(f"[{now()}] BASELINE {name} {record['status']}; exit_code={exit_code}", flush=True)
                commit(session_dir, state)
                return
            if drained_since is None:
                drained_since = time.monotonic()
            elif time.monotonic() - drained_since > DRAIN_GRACE_SECONDS:
                raise UnsafeTransition(f"{name} terminal receipt exists but native process/group/lock did not drain")
        elif time.monotonic() - launch_time > START_GRACE_SECONDS:
            supervisor = job.get("supervisor")
            if not isinstance(supervisor, dict):
                raise UnsafeTransition(f"{name} detached supervisor failed to publish live identity or terminal receipt")
            if not alive(supervisor.get("pid"), supervisor.get("start_ticks")):
                raise UnsafeTransition(f"{name} supervisor vanished without a terminal receipt")
        time.sleep(POLL_SECONDS)


def run_session(session_dir: Path, state: dict[str, Any]) -> int:
    global STOP_REQUESTED
    def _request_stop(_sig: int, _frame: Any) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    lock_path = ROOT / RUNS_REL / ".controller.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        state.update({"status": "HALTED", "failure": "another serial controller owns the global lock", "finished_at_utc": now()})
        commit(session_dir, state)
        return 2
    try:
        self_stat = proc(os.getpid())
        state["supervisor"] = {"pid": os.getpid(), "start_ticks": self_stat[0] if self_stat else None}
        state["status"] = "RUNNING"
        commit(session_dir, state)
        verify_entrypoints()
        for name in BASELINES:
            if STOP_REQUESTED:
                raise UnsafeTransition("controller stop requested between baseline runs")
            if not all_native_locks_idle():
                raise UnsafeTransition("a native formal lock is held; refusing overlapping experiment")
            run_one(name, session_dir, state)
        state["active_baseline"] = None
        passed = sum(row["status"] == "PASS" for item in state["baselines"].values() for row in item["datasets"].values())
        state["status"] = "COMPLETE" if passed == len(DATASETS) * len(BASELINES) else "COMPLETE_WITH_FAILURES"
        state["finished_at_utc"] = now()
        commit(session_dir, state)
        print(f"[{now()}] ALL_BASELINES_FINISHED verified_cells={passed}/15 status={state['status']}", flush=True)
        return 0
    except Exception as exc:
        state["status"] = "HALTED"
        state["failure"] = f"{type(exc).__name__}: {exc}"
        state["finished_at_utc"] = now()
        active = state.get("active_baseline")
        if active:
            state["baselines"][active]["status"] = "UNCERTAIN_STOP"
        try:
            commit(session_dir, state)
        except Exception as write_exc:
            print(f"[{now()}] UNABLE_TO_SAVE_HALTED_STATE {write_exc}", file=sys.stderr, flush=True)
        print(f"[{now()}] SAFETY_HALT {state['failure']}", file=sys.stderr, flush=True)
        return 2
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)



def exact_signal(pid: Any, start_ticks: Any, expected_argv: list[str], expected_cwd: Path) -> bool:
    """SIGTERM only an authenticated process identity; never signal a reused PID."""
    if not alive(pid, start_ticks):
        return False
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise UnsafeTransition("Linux/Python pidfd signaling is unavailable; refusing a PID-only stop")

    def authenticate() -> bool:
        if not alive(pid, start_ticks):
            return False
        try:
            actual_argv = [os.fsdecode(part) for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
            actual_cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
        except OSError:
            return False
        if actual_argv != expected_argv or actual_cwd != expected_cwd:
            raise UnsafeTransition(f"process {pid} does not match the recorded session/argv/cwd; no signal sent")
        return True

    if not authenticate():
        return False
    try:
        pidfd = os.pidfd_open(pid, 0)
    except ProcessLookupError:
        return False
    try:
        if not authenticate():
            return False
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGTERM, None, 0)
        except ProcessLookupError:
            return False
    finally:
        os.close(pidfd)
    return True


def load_queue(sid: str) -> tuple[Path, dict[str, Any]]:
    path = session_dir(sid)
    doc = read_json(path / "status.json")
    if not doc or doc.get("session_id") != sid or doc.get("schema") != "comparison.serial-formal.v1":
        raise UnsafeTransition("serial controller state does not belong to the requested exact session")
    return path, doc


def _stop_controller(sid: str, deadline: float) -> dict[str, Any]:
    script = str(Path(__file__).resolve())
    argv = [sys.executable, script, "_run", "--session-id", sid]
    signaled = False
    while True:
        _path, doc = load_queue(sid)
        if doc["status"] in ("COMPLETE", "COMPLETE_WITH_FAILURES", "HALTED"):
            break
        ident = doc.get("supervisor")
        if isinstance(ident, dict) and alive(ident.get("pid"), ident.get("start_ticks")):
            if not signaled:
                signaled = exact_signal(ident["pid"], ident["start_ticks"], argv, ROOT)
                print(f"controller_stop_requested={signaled} pid={ident['pid']}", flush=True)
        elif doc["status"] not in ("STARTING",):
            raise UnsafeTransition("controller has no authenticated live PID and no terminal state")
        if time.monotonic() >= deadline:
            raise UnsafeTransition("controller did not reach a safe terminal state; native stop not yet attempted")
        time.sleep(0.5)

    ident = doc.get("supervisor")
    if isinstance(ident, dict) and alive(ident.get("pid"), ident.get("start_ticks")):
        if time.monotonic() >= deadline:
            raise UnsafeTransition("controller still alive after terminal state; native stop deferred")
        while alive(ident.get("pid"), ident.get("start_ticks")):
            if time.monotonic() >= deadline:
                raise UnsafeTransition("controller still alive after terminal state; native stop deferred")
            time.sleep(0.5)
    return doc


def _native_job(name: str, launch_id: str) -> dict[str, Any]:
    if name == "zkdl":
        if not re.fullmatch(r"launch_\d{8}T\d{6}Z_[A-Za-z0-9]{8}", launch_id):
            raise UnsafeTransition("zkDL native launch ID is invalid")
        relative = Path("runs") / launch_id
    else:
        if not HEX32.fullmatch(launch_id):
            raise UnsafeTransition(f"{name} native launch ID is invalid")
        relative = (Path("runs/formal/.sessions") / launch_id if name == "ezkl"
                    else Path("baseline-local/formal-detached") / launch_id)
    return {"name": name, "launch_id": launch_id, "dir": root_for(name) / relative}


def _run_native_stop(job: dict[str, Any]) -> None:
    name, sid, directory = job["name"], job["launch_id"], job["dir"]
    ensure_regular_dir(directory)
    if name == "ezkl":
        cmd = [sys.executable, "scripts/ezkl_formal_control.py", "stop", "--session-id", sid, "--wait-seconds", "30"]
    elif name == "zkdl":
        cmd = [sys.executable, "scripts/detached_control.py", "stop", "--launch-dir", str(directory)]
    else:
        status = read_json(directory / "status.json")
        if status is None:
            raise UnsafeTransition("zkGPT detached supervisor has no identity receipt; cannot stop safely")
        if status.get("schema") != "zkgpt-gat-detached-launch-v2" or status.get("launch_id") != sid:
            raise UnsafeTransition("zkGPT detached status identity mismatch")
        if status.get("status") in ("COMPLETE", "FAILED", "INTERRUPTED"):
            return
        pid, ticks = status.get("supervisor_pid"), status.get("supervisor_start_ticks")
        argv = status.get("supervisor_argv")
        cwd = status.get("supervisor_cwd")
        expected = str(root_for(name) / "scripts/formal_detached_supervisor.py")
        if (not isinstance(argv, list) or len(argv) != 6 or
                argv[1:] != [expected, "--root", str(root_for(name)), "--launch-id", sid] or
                cwd != str(root_for(name))):
            raise UnsafeTransition("zkGPT supervisor argv/root evidence is malformed")
        if not exact_signal(pid, ticks, argv, root_for(name)):
            raise UnsafeTransition("zkGPT supervisor disappeared without an authenticated terminal receipt")
        print(f"zkgpt_stop_requested=True launch_id={sid}", flush=True)
        return
    result = subprocess.run(cmd, cwd=root_for(name), capture_output=True, text=True, timeout=45, check=False)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    # zkDL returns 1 if the supervisor has not stopped within its 20s window.
    if result.returncode not in ((0, 1) if name == "zkdl" else (0,)):
        raise UnsafeTransition(f"{name} official native stop failed with code {result.returncode}")


def stop_session(sid: str, *, wait_seconds: int) -> int:
    if wait_seconds < 5 or wait_seconds > 3600:
        raise UnsafeTransition("--wait-seconds must be between 5 and 3600")
    deadline = time.monotonic() + wait_seconds
    doc = _stop_controller(sid, deadline)
    if doc["status"] in ("COMPLETE", "COMPLETE_WITH_FAILURES"):
        print(f"stop_status=ALREADY_FINISHED serial_session_id={sid}")
        return 0
    name = doc.get("active_baseline")
    if name is None:
        if not all_native_locks_idle():
            raise UnsafeTransition("controller stopped but an original native formal lock is still held")
        print(f"stop_status=QUEUE_STOPPED serial_session_id={sid}; no current native baseline")
        return 0
    if name not in BASELINES:
        raise UnsafeTransition("unknown active baseline in exact controller state")
    launch_id = doc["baselines"][name].get("launch_id")
    if not isinstance(launch_id, str):
        raise UnsafeTransition("controller halted during native launcher; exact launch ID unavailable, manual investigation required")
    job = _native_job(name, launch_id)
    _run_native_stop(job)
    while True:
        terminal, _code, _values = snapshot(job, {})
        if terminal and safe_to_transition(job):
            latest_path, latest = load_queue(sid)
            latest["native_stop_confirmed_at_utc"] = now()
            commit(latest_path, latest)
            print(f"stop_status=CONFIRMED serial_session_id={sid} native={name} launch_id={launch_id}")
            return 0
        if time.monotonic() >= deadline:
            raise UnsafeTransition(f"native stop not yet confirmed for {name}/{launch_id}; inspect exact native status/logs before restarting")
        time.sleep(1)

def session_dir(sid: str) -> Path:
    if not HEX32.fullmatch(sid):
        raise UnsafeTransition("session ID must be one exact 32-char lowercase hexadecimal token")
    path = ROOT / RUNS_REL / sid
    ensure_regular_dir(path)
    return path


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("start", help="start the complete serial queue and detach from SSH")
    for cmd in ("status", "logs", "_run"):
        sub.add_parser(cmd).add_argument("--session-id", required=True)
    stop_parser = sub.add_parser("stop", help="safely stop both the queue and its active native formal run")
    stop_parser.add_argument("--session-id", required=True)
    stop_parser.add_argument("--wait-seconds", type=int, default=180)
    args = parser.parse_args()
    try:
        if args.action == "start":
            root = ROOT / RUNS_REL
            ensure_regular_dir(ROOT / "runs", create=True)
            ensure_regular_dir(root, create=True)
            sid = uuid.uuid4().hex
            path = root / sid
            path.mkdir(mode=0o700)
            write_json(path / "status.json", blank_state(sid))
            log_file = path / "console.log"
            fd = os.open(log_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "wb") as log:
                subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_run", "--session-id", sid],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
            print(f"serial_session_id={sid}")
            print(f"serial_status={path / 'status.json'}")
            print(f"serial_log={log_file}")
            print(f"status_command=python3 scripts/serial_formal.py status --session-id {sid}")
            return 0
        if args.action == "stop":
            return stop_session(args.session_id, wait_seconds=args.wait_seconds)
        path = session_dir(args.session_id)
        if args.action == "logs":
            os.execvp("tail", ["tail", "-n", "80", "-F", str(path / "console.log")])
        doc = read_json(path / "status.json")
        if not doc or doc.get("session_id") != args.session_id or doc.get("schema") != "comparison.serial-formal.v1":
            raise UnsafeTransition("controller session status identity mismatch")
        if args.action == "_run":
            return run_session(path, doc)
        print(f"session={args.session_id} status={doc['status']} active={doc.get('active_baseline') or '-'}")
        if doc.get("failure"):
            print(f"failure={doc['failure']}")
        for baseline in BASELINES:
            entry = doc["baselines"][baseline]
            print(f"baseline={baseline} status={entry['status']} launch={entry['launch_id']} native_session={entry['native_session_id']}")
            if entry.get("native_log"):
                print(f"  native_log={entry['native_log']}")
            for dataset in DATASETS:
                row = entry["datasets"][dataset]
                print(f"  dataset={dataset} status={row['status']} detail={row['detail']}")
        print(f"log={path / 'console.log'}")
        return 0
    except (UnsafeTransition, OSError) as exc:
        print(f"serial-formal: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
