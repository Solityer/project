#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(realpath -e -- "${BASH_SOURCE[0]}")"
ROOT="$(realpath -e -- "$(dirname -- "$SCRIPT_PATH")/..")"
cd -- "$ROOT"
if (($# != 0)); then
    echo "usage: $SCRIPT_PATH" >&2
    exit 2
fi
command -v nohup >/dev/null
command -v setsid >/dev/null
LAUNCH_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
LAUNCH_ROOT="$ROOT/baseline-local/formal-detached"
LAUNCH_DIR="$LAUNCH_ROOT/$LAUNCH_ID"
mkdir -p -- "$LAUNCH_DIR"
chmod 700 -- "$LAUNCH_ROOT" "$LAUNCH_DIR"
CONSOLE_REL="baseline-local/formal-detached/$LAUNCH_ID/console.log"
POINTER_TMP="$LAUNCH_ROOT/.latest_launch.$$.tmp"
printf '{"schema":"zkgpt-gat-detached-launch-v1","launch_id":"%s","console_path":"%s","status_path":"baseline-local/formal-detached/%s/status.json"}\n' \
    "$LAUNCH_ID" "$CONSOLE_REL" "$LAUNCH_ID" > "$POINTER_TMP"
chmod 600 -- "$POINTER_TMP"
mv -f -- "$POINTER_TMP" "$LAUNCH_ROOT/latest_launch.json"

nohup setsid python3 "$ROOT/scripts/formal_detached_supervisor.py" \
    --root "$ROOT" --launch-id "$LAUNCH_ID" \
    >> "$LAUNCH_DIR/console.log" 2>&1 < /dev/null &

printf 'ZKGPT_FORMAL_LAUNCHED launch_id=%s console=%s status=%s\n' \
    "$LAUNCH_ID" "$CONSOLE_REL" \
    "baseline-local/formal-detached/$LAUNCH_ID/status.json"
printf 'Progress: %s/scripts/formal_progress.sh\n' "$ROOT"
