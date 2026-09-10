#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$script_dir/.." && pwd -P)"
config="${1:-}"
run_dir="${2:-}"
if [[ -z "$config" ]]; then
  echo "usage: $0 configs/<dataset>_secure.cfg [runs/<name>]" >&2
  exit 2
fi
case "$config" in
  /*) config_path="$config" ;;
  *) config_path="$root/$config" ;;
esac
config_path="$(realpath -- "$config_path")"
case "$config_path" in
  "$root/configs/"*.cfg) ;;
  *) echo "verify-single: config must be under baseline-local configs/" >&2; exit 2 ;;
esac
if [[ -n "$run_dir" ]]; then
  case "$run_dir" in
    /*) run_path="$run_dir" ;;
    *) run_path="$root/$run_dir" ;;
  esac
  run_path="$(realpath -m -- "$run_path")"
  case "$run_path" in
    "$root"/*) ;;
    *) echo "verify-single: run directory must be under baseline-local root/" >&2; exit 2 ;;
  esac
  exec "$root/gat_runner" verify --config "$config_path" --run-dir "$run_path"
fi
exec "$root/gat_runner" verify --config "$config_path"
