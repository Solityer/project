#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$script_dir/.." && pwd -P)"
config="${1:-}"
backend="${2:-}"
run_dir="${3:-}"
if [[ -z "$config" || ( "$backend" != "cpu" && "$backend" != "gpu" ) ]]; then
  echo "usage: $0 configs/<dataset>_secure.cfg cpu|gpu [runs/<name>]" >&2
  exit 2
fi
if [[ "${ZKDL_ALLOW_FULL_DATASET_RUN:-0}" != "1" ]]; then
  echo "run-single: refusing full-dataset proving; set ZKDL_ALLOW_FULL_DATASET_RUN=1 after review" >&2
  exit 3
fi
case "$config" in
  /*) config_path="$config" ;;
  *) config_path="$root/$config" ;;
esac
config_path="$(realpath -- "$config_path")"
case "$config_path" in
  "$root/configs/"*.cfg) ;;
  *) echo "run-single: config must be under baseline-local configs/" >&2; exit 2 ;;
esac
if [[ -n "$run_dir" ]]; then
  "$script_dir/prepare_local.sh" "$config_path" "$run_dir"
  exec "$root/gat_runner" prove --config "$config_path" --backend "$backend" --run-dir "$run_dir"
fi
"$script_dir/prepare_local.sh" "$config_path"
exec "$root/gat_runner" prove --config "$config_path" --backend "$backend"
