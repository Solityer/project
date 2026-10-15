import json
import os
import signal
from pathlib import Path

root = Path('/home/pzh/project/comparison/zkGPT')
launch_id = '7d55548fc8c647e48f12cfd536750b4f'
status_path = root / 'baseline-local/formal-detached' / launch_id / 'status.json'
status = json.loads(status_path.read_text(encoding='utf-8'))
expected = {
    'launch_id': launch_id,
    'status': 'RUNNING',
    'supervisor_pid': 1777759,
    'supervisor_start_ticks': 35773736,
    'supervisor_cwd': str(root),
    'supervisor_executable': '/usr/bin/python3.12',
    'supervisor_argv': ['python3', str(root / 'scripts/formal_detached_supervisor.py'), '--root', str(root), '--launch-id', launch_id],
    'child_pid': 1777764,
    'child_start_ticks': 35773738,
    'child_cwd': str(root),
    'child_executable': '/usr/bin/python3.12',
    'child_argv': ['/usr/bin/python3', str(root / 'scripts/run_formal.py')],
}
for key, value in expected.items():
    if status.get(key) != value:
        raise SystemExit(f'REFUSED metadata mismatch: {key}')


def proc_info(pid):
    base = Path('/proc') / str(pid)
    raw = (base / 'stat').read_text(encoding='ascii')
    tail = raw[raw.rfind(')') + 2:].split()
    return {
        'state': tail[0],
        'ppid': int(tail[1]),
        'pgrp': int(tail[2]),
        'start_ticks': int(tail[19]),
        'cwd': os.readlink(base / 'cwd'),
        'exe': os.readlink(base / 'exe'),
        'argv': [os.fsdecode(x) for x in (base / 'cmdline').read_bytes().split(b'\0') if x],
    }


def check_identity(pid, ppid, start_ticks, cwd, exe, argv):
    info = proc_info(pid)
    if (info['ppid'] != ppid or info['start_ticks'] != start_ticks or
            info['cwd'] != cwd or info['exe'] != exe or info['argv'] != argv):
        raise SystemExit(f'REFUSED live PID identity mismatch: {pid}')
    return info

supervisor = check_identity(1777759, 1, 35773736, str(root), '/usr/bin/python3.12', expected['supervisor_argv'])
owner = check_identity(1777764, 1777759, 35773738, str(root), '/usr/bin/python3.12', expected['child_argv'])
if owner['state'] != 'S':
    raise SystemExit(f'REFUSED owner state changed: {owner["state"]}')
if Path('/proc/1785224').exists():
    raise SystemExit('REFUSED original native PID 1785224 is occupied; re-check exact process identity')
child_tail = (Path('/proc/1785173/stat').read_text(encoding='ascii'))
child_tail = child_tail[child_tail.rfind(')') + 2:].split()
if int(child_tail[1]) != 1777764 or child_tail[0] != 'Z':
    raise SystemExit('REFUSED direct child 1785173 is not the expected zombie')
children = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        raw = (entry / 'stat').read_text(encoding='ascii')
        tail = raw[raw.rfind(')') + 2:].split()
        if int(tail[1]) == 1777764:
            children.append((int(entry.name), tail[0]))
    except (OSError, ValueError):
        continue
if children != [(1785173, 'Z')]:
    raise SystemExit(f'REFUSED unexpected owner children: {children}')
wchan = Path('/proc/1777764/wchan').read_text(encoding='ascii').strip()
if 'futex' not in wchan:
    raise SystemExit(f'REFUSED owner wait channel changed: {wchan}')
lock_fd = Path('/proc/1777764/fd/3')
lock_target = os.readlink(lock_fd)
if lock_target != str(root / 'baseline-local/formal-run.lock'):
    raise SystemExit('REFUSED owner FD 3 no longer references the original formal-run.lock')
if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
    raise SystemExit('REFUSED pidfd signaling is unavailable; no fallback signal used')
print(f'PIDFD_PRECHECK_OK launch_id={launch_id} owner_pid=1777764 start_ticks=35773738 wchan={wchan} lock_fd=3 original_native=gone', flush=True)
pidfd = os.pidfd_open(1777764, 0)
try:
    status2 = json.loads(status_path.read_text(encoding='utf-8'))
    for key, value in expected.items():
        if status2.get(key) != value:
            raise SystemExit(f'REFUSED metadata changed after pidfd open: {key}')
    current = check_identity(1777764, 1777759, 35773738, str(root), '/usr/bin/python3.12', expected['child_argv'])
    if current['state'] != 'S' or 'futex' not in Path('/proc/1777764/wchan').read_text(encoding='ascii').strip():
        raise SystemExit('REFUSED owner no longer matches deadlocked state')
    if os.readlink(lock_fd) != lock_target:
        raise SystemExit('REFUSED original lock FD changed after pidfd open')
    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
    print('PIDFD_SIGKILL_SENT pid=1777764 start_ticks=35773738 signal=SIGKILL', flush=True)
finally:
    os.close(pidfd)
