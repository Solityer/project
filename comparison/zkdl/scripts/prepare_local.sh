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
  *) echo "prepare-local: config must be under baseline-local configs/" >&2; exit 2 ;;
esac

manifest_args=()
runner_args=()
if [[ -n "$run_dir" ]]; then
  case "$run_dir" in
    /*) run_path="$run_dir" ;;
    *) run_path="$root/$run_dir" ;;
  esac
  run_path="$(realpath -m -- "$run_path")"
  case "$run_path" in
    "$root"/*) ;;
    *) echo "prepare-local: run directory must be under baseline-local root/" >&2; exit 2 ;;
  esac
  mkdir -p -- "$run_path"
  manifest_args=(--output "$run_path/asset_manifest.json")
  runner_args=(--run-dir "$run_path")
fi

python3 "$script_dir/validate_resources.py" --config "$config_path"
python3 "$script_dir/asset_manifest.py" --config "$config_path" "${manifest_args[@]}"
"$root/gat_runner" prepare --config "$config_path" "${runner_args[@]}"
