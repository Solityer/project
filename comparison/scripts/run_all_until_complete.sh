#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/comparison/results"
HEARTBEAT="${RESULTS_DIR}/run_all_until_complete_heartbeat.json"
MARKER="${RESULTS_DIR}/current_rerun_marker.json"

mkdir -p "${RESULTS_DIR}"

write_json() {
  local phase="$1"
  local baseline="$2"
  local command="$3"
  local status="$4"
  python3 - "$HEARTBEAT" "$phase" "$baseline" "$command" "$status" <<'PY'
import datetime as dt
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
phase, baseline, command, status = sys.argv[2:6]

def grab(cmd):
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return (completed.stdout or completed.stderr).strip()

payload = {
    "timestamp": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "phase": phase,
    "baseline": baseline,
    "command": command,
    "status": status,
    "pid": os.getpid(),
    "host": socket.gethostname(),
    "available_memory": grab(["bash", "-lc", "free -h | awk '/Mem:/ {print $7}'"]),
    "available_disk": grab(["bash", "-lc", "df -h /home/pzh/project | awk 'NR==2 {print $4}'"]),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

if [[ ! -s "${MARKER}" || "${COMPARISON_FORCE_NEW_MARKER:-0}" == "1" ]]; then
python3 - "$MARKER" <<'PY'
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/home/pzh/project", text=True).strip()
stamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
payload = {
    "created_at": stamp,
    "git_head_at_start": git_head,
    "rerun_marker_id": "comparison-rerun-" + stamp.replace("-", "").replace(":", "").replace("Z", "Z"),
    "stale_policy": "proof/verify/run artifacts older than this marker cannot enter final comparison results",
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

run_step() {
  local baseline="$1"
  local dir="$2"
  local command="$3"
  write_json "before" "$baseline" "$command" "running"
  (cd "$dir" && bash -lc "$command")
  write_json "after" "$baseline" "$command" "passed"
}

run_step "ezkl" "${PROJECT_ROOT}/comparison/baselines/ezkl" "./scripts/run_until_complete.sh"
run_step "zkDL" "${PROJECT_ROOT}/comparison/baselines/zkdl" "./scripts/run_until_complete.sh"
run_step "zkGPT" "${PROJECT_ROOT}/comparison/baselines/zkGPT" "./scripts/run_until_complete.sh"
run_step "top-level-validation" "${PROJECT_ROOT}" "python3 comparison/scripts/validate_all_baselines.py"
run_step "top-level-collection" "${PROJECT_ROOT}" "python3 comparison/scripts/collect_all_comparison_results.py"

for output in \
  comparison/results/final_comparison_results.csv \
  comparison/results/final_comparison_results.json \
  comparison/results/final_comparison_audit.json \
  comparison/results/final_comparison_report.md; do
  test -s "${PROJECT_ROOT}/${output}"
done

write_json "complete" "all" "comparison/scripts/run_all_until_complete.sh" "passed"
echo "FINAL_STATUS=COMPARISON_BASELINES_FULLY_COMPLETED"
