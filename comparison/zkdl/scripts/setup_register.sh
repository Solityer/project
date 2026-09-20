#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$script_dir/.." && pwd -P)"
config="${1:-}"
if [[ -z "$config" ]]; then
  echo "usage: $0 configs/<dataset>_secure.cfg [--regenerate-srs]" >&2
  exit 2
fi
regenerate=()
if [[ "${2:-}" == "--regenerate-srs" ]]; then regenerate+=(--regenerate-srs); fi
case "$config" in
  /*) config_path="$config" ;;
  *) config_path="$root/$config" ;;
esac
config_path="$(realpath -- "$config_path")"
case "$config_path" in
  "$root/configs/"*.cfg) ;;
  *) echo "setup-register: config must be under baseline-local configs/" >&2; exit 2 ;;
esac
python3 "$script_dir/prepare_form_resources.py" \
  --config "$config_path" \
  --full-dataset "${regenerate[@]}"
python3 "$script_dir/validate_resources.py" --config "$config_path"
echo "setup-register: native local random SRS and resources validated; zkDL registration is ready"
