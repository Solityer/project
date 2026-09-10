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
if [[ -n "$run_dir" ]]; then
  exec "$root/gat_runner" verify --config "$config_path" --run-dir "$run_dir"
fi
exec "$root/gat_runner" verify --config "$config_path"
