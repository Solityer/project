#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$script_dir/.." && pwd -P)"
runs="$root/runs"
if [[ "$#" -ne 0 ]]; then
    echo "run-five-detached: usage: $0" >&2
    exit 2
fi
if [[ -L "$runs" ]]; then
    echo "run-five-detached: zkDL runs path is symlinked: $runs" >&2
    exit 2
fi
if [[ ! -e "$runs" ]]; then
    mkdir -m 700 -- "$runs"
fi
if [[ ! -d "$runs" ]]; then
    echo "run-five-detached: zkDL runs path is not a directory: $runs" >&2
    exit 2
fi
for command_name in nohup setsid python3 mktemp; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "run-five-detached: required system command is unavailable: $command_name" >&2
        exit 2
    fi
done
supervisor="$script_dir/detached_supervisor.py"
if [[ -L "$supervisor" || ! -f "$supervisor" ]]; then
    echo "run-five-detached: baseline-local supervisor is missing or symlinked: $supervisor" >&2
    exit 2
fi
launch_dir="$(mktemp -d -- "$runs/launch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")"
chmod 700 "$launch_dir"
launch_id="$(basename -- "$launch_dir")"
umask 077
: > "$launch_dir/console.log"
if [[ -L "$runs/latest_launch" || ( -e "$runs/latest_launch" && ! -f "$runs/latest_launch" ) ]]; then
    echo "run-five-detached: latest launch pointer is not a regular file: $runs/latest_launch" >&2
    exit 2
fi
latest_tmp="$(mktemp -- "$runs/.latest_launch.XXXXXXXX")"
printf '%s\n' "$launch_id" > "$latest_tmp"
mv -f -- "$latest_tmp" "$runs/latest_launch"
nohup setsid python3 "$supervisor" --launch-dir "$launch_dir" </dev/null >> "$launch_dir/console.log" 2>&1 &
supervisor_pid=$!
pid_tmp="$(mktemp -- "$launch_dir/.supervisor_pid.XXXXXXXX")"
printf '%s\n' "$supervisor_pid" > "$pid_tmp"
mv -f -- "$pid_tmp" "$launch_dir/supervisor.pid"
printf 'zkdl_detached_launch=STARTED launch_id=%s\n' "$launch_id"
printf 'zkdl_detached_launch_dir=%s\n' "$launch_dir"
printf 'zkdl_detached_supervisor_pid=%s\n' "$supervisor_pid"
printf 'zkdl_detached_console=%s\n' "$launch_dir/console.log"
printf 'zkdl_detached_exit_status=%s\n' "$launch_dir/exit.json"
