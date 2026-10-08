#!/usr/bin/env python3
"""Process tree used by the bounded local-owner death regression."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def proc_start_ticks(pid: int) -> str:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    return text[text.rfind(")") + 2:].split()[19]


def owner(root: Path, session: Path, session_id: str, ready: Path, pids: Path) -> None:
    lock_path = root / "runs" / ".formal.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_stat = os.fstat(lock_fd)
    token = "a" * 64
    state = {
        "schema": "zkdl_formal_runner_state_v1",
        "baseline": "zkdl",
        "session_id": session_id,
        "status": "RUNNING",
        "stage": "cell:cora",
        "owner": {
            "pid": os.getpid(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
            "start_ticks": proc_start_ticks(os.getpid()),
        },
        "lock": {"device": lock_stat.st_dev, "inode": lock_stat.st_ino},
        "internal_owner_token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
    }
    (session / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = dict(os.environ)
    env["ZKDL_FORMAL_OWNER_TOKEN"] = token
    env["ZKDL_FORMAL_LOCK_FD"] = str(lock_fd)
    phase = subprocess.Popen([sys.executable, __file__, "phase", str(root), str(session), session_id, str(pids)],
                             cwd=root, env=env, pass_fds=(lock_fd,), stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    ready.write_text(json.dumps({"phase_pid": phase.pid}), encoding="utf-8")
    while True:
        time.sleep(1)


def phase(root: Path, session: Path, session_id: str, pids: Path) -> None:
    sys.path.insert(0, str(root / "scripts"))
    import formal_phase

    for signum in (formal_phase.signal.SIGINT, formal_phase.signal.SIGTERM, formal_phase.signal.SIGHUP):
        formal_phase.signal.signal(signum, formal_phase._signal_handler)
    formal_phase._validate_local_owner(session, session_id, "cell", "cora")
    formal_phase._run([sys.executable, __file__, "descendant", str(pids)])


def descendant(pids: Path) -> None:
    grandchild = subprocess.Popen(["sleep", "300"])
    pids.write_text(json.dumps({"child": os.getpid(), "grandchild": grandchild.pid}), encoding="utf-8")
    print("local-owner-death-descendants-ready", flush=True)
    time.sleep(300)


def main() -> int:
    mode = sys.argv[1]
    if mode == "owner":
        _, _, root, session, session_id, ready, pids, *_ = sys.argv
        owner(Path(root), Path(session), session_id, Path(ready), Path(pids))
    elif mode == "phase":
        _, _, root, session, session_id, pids = sys.argv
        phase(Path(root), Path(session), session_id, Path(pids))
    elif mode == "descendant":
        descendant(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
