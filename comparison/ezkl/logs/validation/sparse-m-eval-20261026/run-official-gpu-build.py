"""Replay the exact complete official GPU release-build command from the last successful receipt."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[3]
previous_log = root / "logs/validation/sparse-m-commit-20261026/official-gpu-build-full.log"
output_log = Path(__file__).resolve().parent / "official-gpu-build-full.log"
line = previous_log.read_text(encoding="utf-8").splitlines()[0]
prefix = "OFFICIAL_GPU_FINAL_COMMAND="
if not line.startswith(prefix):
    raise SystemExit("prior successful official build log has no full command record")
command = json.loads(line[len(prefix):])
if command[:3] != ["bash", "/home/pzh/project/comparison/ezkl/scripts/build_ezkl.sh", "--gpu"]:
    raise SystemExit("prior full command is not the approved official GPU release build")
with output_log.open("w", encoding="utf-8") as log:
    command_record = prefix + json.dumps(command, separators=(",", ":")) + "\n"
    sys.stdout.write(command_record)
    sys.stdout.flush()
    log.write(command_record)
    log.flush()
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for item in process.stdout:
        sys.stdout.write(item)
        sys.stdout.flush()
        log.write(item)
        log.flush()
    code = process.wait()
    elapsed = time.monotonic() - started
    result = f"OFFICIAL_GPU_BUILD_EXIT_CODE={code}\nOFFICIAL_GPU_BUILD_ELAPSED_SECONDS={elapsed:.3f}\n"
    sys.stdout.write(result)
    sys.stdout.flush()
    log.write(result)
    log.flush()
raise SystemExit(code)
